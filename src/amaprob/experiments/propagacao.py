"""Plug-in × plug-in recalibrado × propagação em função da taxa de quantidades ocultas, por regressor e imputador.

Cada regressor (regressão linear bayesiana e SVGP) e cada imputador (porção média, PPCA e FA, com e
sem média condicional) são ajustados uma vez no treino completo, com os hiperparâmetros adotados em
`configs/hiperparametros.json`. O braço recalibrado existe só com a regressão linear, e os
imputadores das 5 dobras dele também são ajustados uma vez por imputador. As máscaras definem o que
fica oculto no conjunto medido: a validação (padrão, em `results/`) ou, com `--conjunto teste`, o
teste (em `results/teste/`). Nada é ajustado com o conjunto medido; na validação, o teste nunca é
lido (Regra 6).

Com a regressão linear, grava também as métricas da Parte 1 (imputação nas células ocultas do
conjunto medido), que não dependem do regressor. A referência da taxa 0 por número de ocultas é
gravada por regressor.

Com `--regressores` e `--imputadores`, só os pares escolhidos rodam, e as linhas deles entram nos
CSVs existentes do mesmo conjunto; as dos outros pares e as referências da taxa 0 já gravadas ficam
como estavam. Vale porque os geradores dependem só da semente, da taxa e do fluxo, e não do
regressor nem do imputador.

Uso: uv run python -m amaprob.experiments.propagacao [--conjunto teste] [--regressores svgp] [--imputadores fa]
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from plotnine import (
    aes, facet_grid, geom_errorbar, geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual,
)

from amaprob.data.dataset import RAIZ, carregar_mascaras, carregar_refeicoes
from amaprob.data.splits import TESTE, VALIDACAO
from amaprob.evaluation.metricas import (
    avaliar_imputacao, avaliar_mistura, celulas_de_imputacao, imputacao_por_gaveta, intervalo, por_ocultas,
)
from amaprob.evaluation.tema import CORES, TEMA
from amaprob.experiments.taxa_zero import separar
from amaprob.imputation.base import Imputer, ocultar
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.imputation.fa import FA
from amaprob.imputation.ppca import PPCA
from amaprob.regression.base import ProbRegressor, montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagation import (
    N_DOBRAS,
    PLUGIN,
    PLUGIN_RECALIBRADO,
    PROPAGACAO,
    ajustar_dobras,
    amostras_em_gramas,
    dobras_por_pessoa,
    imputar_treino_cruzado,
    prever_bracos,
)
from amaprob.regression.svgp import SVGP

SAIDA = RAIZ / "results"
CONFIGURACAO = RAIZ / "configs" / "hiperparametros.json"
S = 100
REGRESSAO_LINEAR = "blr"
NOMES_DOS_CONJUNTOS = {VALIDACAO: "Validação", TESTE: "Teste"}


def ler_configuracao(arquivo: Path = CONFIGURACAO) -> dict[str, dict]:
    """Hiperparâmetros adotados na Parte 3; modelo ou botão sem entrada fica com o valor de partida."""
    return json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {}


def fabricas_de_imputadores(configuracao: dict[str, dict]) -> dict[str, Callable[[], Imputer]]:
    botoes = lambda nome: configuracao.get(nome, {})
    return {
        "porcao_media": lambda: PorcaoMedia(**botoes("porcao_media")),
        "ppca": lambda: PPCA(semente=0, **botoes("ppca")),
        "fa": lambda: FA(semente=0, **botoes("fa")),
        "porcao_condicional": lambda: PorcaoMedia(media_condicional=True, **botoes("porcao_condicional")),
        "ppca_condicional": lambda: PPCA(semente=0, media_condicional=True, **botoes("ppca_condicional")),
        "fa_condicional": lambda: FA(semente=0, media_condicional=True, **botoes("fa_condicional")),
    }


def fabricas_de_regressores(configuracao: dict[str, dict]) -> dict[str, Callable[[], ProbRegressor]]:
    return {
        REGRESSAO_LINEAR: RegressaoLinearBayesiana,
        "svgp": lambda: SVGP(semente=0, **configuracao.get("svgp", {})),
    }


def nomes_dos_imputadores(fabricas: dict[str, Callable[[], Imputer]]) -> dict[str, str]:
    base = {"porcao_media": "Porção média", "ppca": "PPCA", "fa": "FA", "porcao_condicional": "Porção média cond.",
            "ppca_condicional": "PPCA cond.", "fa_condicional": "FA cond."}
    com_q = {nome: getattr(fabrica(), "q", None) for nome, fabrica in fabricas.items()}
    return {nome: base[nome] if q is None else f"{base[nome]} (q = {q})" for nome, q in com_q.items()}


def nomes_dos_regressores(fabricas: dict[str, Callable[[], ProbRegressor]]) -> dict[str, str]:
    return {REGRESSAO_LINEAR: "Regressão linear bayesiana", "svgp": f"SVGP (M = {fabricas['svgp']().n_indutores})"}


IMPUTADORES = fabricas_de_imputadores(ler_configuracao())
NOMES_DOS_IMPUTADORES = nomes_dos_imputadores(IMPUTADORES)
REGRESSORES = fabricas_de_regressores(ler_configuracao())
NOMES_DOS_REGRESSORES = nomes_dos_regressores(REGRESSORES)
REFERENCIA_TAXA_ZERO = "referencia_taxa_0"
SEM_IMPUTADOR = "nenhum"
FLUXO_VALIDACAO = 0
FLUXO_IMPUTACAO = 1 + N_DOBRAS  # depois dos fluxos das dobras (1..N_DOBRAS)
NOMES_DOS_BRACOS = {PLUGIN: "Plug-in", PLUGIN_RECALIBRADO: "Plug-in recalibrado", PROPAGACAO: "Propagação"}
TABELAS_DA_PROPAGACAO = ("propagacao_metricas", "propagacao_por_ocultas")
TABELAS_DA_PARTE_1 = ("imputacao_metricas", "imputacao_por_gaveta", "imputadores_ajuste")


@dataclass(frozen=True)
class Conjunto:
    G: np.ndarray
    P: np.ndarray
    C: np.ndarray
    y: np.ndarray
    pessoas: np.ndarray
    meal_id: np.ndarray
    M: np.ndarray  # (sementes, taxas, N, K)


@dataclass(frozen=True)
class Ajustado:
    completo: Imputer
    dobras: list[Imputer]


def treino_e_medida(conjunto: str = VALIDACAO) -> tuple[Conjunto, Conjunto, np.ndarray, np.ndarray, np.ndarray]:
    """Treino (para todo ajuste) e o conjunto em que se mede; o teste só entra quando é pedido."""
    refeicoes = carregar_refeicoes()
    mascaras = carregar_mascaras(refeicoes)
    treino, validacao = separar(refeicoes)
    medida = validacao if conjunto == VALIDACAO else np.flatnonzero(refeicoes.conjunto == TESTE)
    recortes = [
        Conjunto(refeicoes.G[i], refeicoes.P[i], refeicoes.C[i], refeicoes.y[i], refeicoes.person_id[i],
                 refeicoes.meal_id[i], mascaras.M[:, :, i])
        for i in (treino, medida)
    ]
    return *recortes, mascaras.taxas, mascaras.sementes, refeicoes.gavetas


def treino_e_validacao() -> tuple[Conjunto, Conjunto, np.ndarray, np.ndarray, np.ndarray]:
    return treino_e_medida(VALIDACAO)


def pasta_de_saida(conjunto: str) -> Path:
    return SAIDA if conjunto == VALIDACAO else SAIDA / "teste"


def gerador(semente: int, indice_taxa: int, fluxo: int) -> np.random.Generator:
    return np.random.default_rng((int(semente), indice_taxa, fluxo))


def _diagnostico(imputador) -> dict[str, float]:
    no_piso = getattr(imputador, "gavetas_no_piso_", None)
    return {"iteracoes": getattr(imputador, "n_iteracoes_", np.nan),
            "teto_iteracoes": getattr(imputador, "max_iteracoes", np.nan),
            "gavetas_no_piso": np.nan if no_piso is None else len(no_piso)}


def _iteracoes(completo: Imputer, das_dobras: list[Imputer]) -> str:
    if not hasattr(completo, "n_iteracoes_"):
        return ""
    maior = max((i.n_iteracoes_ for i in das_dobras), default=None)
    teto = f" de {completo.max_iteracoes}"
    return f" | iterações: completo {completo.n_iteracoes_}{teto}" + ("" if maior is None else f", dobras até {maior}")


def ajustar_imputadores(
    tr: Conjunto, dobras: np.ndarray, nomes: list[str], com_dobras: bool,
) -> tuple[dict[str, Ajustado], pd.DataFrame]:
    sem_mascara = np.zeros_like(tr.P)
    ajustados, tempos = {}, []
    for nome in nomes:
        novo = IMPUTADORES[nome]
        inicio = time.perf_counter()
        completo = novo().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
        meio = time.perf_counter()
        das_dobras = ajustar_dobras(novo, tr.G, tr.P, tr.C, dobras) if com_dobras else []
        fim = time.perf_counter()
        ajustados[nome] = Ajustado(completo, das_dobras)
        tempos += [
            {"imputador": nome, "etapa": "treino completo", "segundos": meio - inicio, **_diagnostico(completo)},
            *({"imputador": nome, "etapa": f"dobra {f}", "segundos": np.nan, **_diagnostico(i)}
              for f, i in enumerate(das_dobras)),
            {"imputador": nome, "etapa": f"{N_DOBRAS} dobras", "segundos": fim - meio,
             "iteracoes": np.nan, "teto_iteracoes": np.nan, "gavetas_no_piso": np.nan},
        ]
        print(f"  {nome}: completo {meio - inicio:.1f} s | {N_DOBRAS} dobras {fim - meio:.1f} s"
              f"{_iteracoes(completo, das_dobras)}", flush=True)
    return ajustados, pd.DataFrame(tempos)


def ajustar_regressores(tr: Conjunto, nomes: list[str]) -> tuple[dict[str, ProbRegressor], pd.DataFrame]:
    Z = montar_Z(tr.G, tr.P, tr.C)
    ajustados, tempos = {}, []
    for nome in nomes:
        inicio = time.perf_counter()
        ajustados[nome] = REGRESSORES[nome]().fit(Z, tr.y)
        segundos = time.perf_counter() - inicio
        historico = getattr(ajustados[nome], "historico_elbo_", [np.nan])
        tempos.append({"regressor": nome, "segundos": segundos,
                       "epocas": getattr(ajustados[nome], "epocas", np.nan), "elbo_final": historico[-1]})
        print(f"  {nome}: {segundos:.1f} s", flush=True)
    return ajustados, pd.DataFrame(tempos)


def celulas_da_medida(imputador: Imputer, medida: Conjunto, M: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    return pd.concat([
        celulas_de_imputacao(medida.G[b], np.log1p(G_s), M[b])
        for b, G_s in amostras_em_gramas(imputador, medida.G, medida.P, M, medida.C, S, rng)
    ])


def medir(tr: Conjunto, medida: Conjunto, ajustados: dict[str, Ajustado], regressores: dict[str, ProbRegressor],
          dobras, taxas, sementes, gavetas):
    metricas, ocultas, imputacao, por_gaveta = [], [], [], []
    com_parte_1 = REGRESSAO_LINEAR in regressores
    Z_completo = montar_Z(medida.G, medida.P, medida.C)
    diretas = {}
    for nome_regressor, regressor in regressores.items():
        media, var = regressor.predict(Z_completo)
        diretas[nome_regressor] = (media, *intervalo(media[None], var[None]))

    for s, semente in enumerate(sementes):
        for t, taxa in enumerate(taxas):
            inicio = time.perf_counter()
            M = medida.M[s, t]
            n_ocultas = M.sum(axis=1)
            chave = {"taxa": taxa, "semente": int(semente)}
            for nome_regressor, (media, li, ls) in diretas.items():
                ocultas.append(por_ocultas(medida.y, media, li, ls, n_ocultas).assign(
                    regressor=nome_regressor, **chave, imputador=SEM_IMPUTADOR, braco=REFERENCIA_TAXA_ZERO))

            for nome, ajustado in ajustados.items():
                recalibrado = None
                if com_parte_1:
                    geradores = [gerador(semente, t, 1 + f) for f in range(N_DOBRAS)]
                    G_treino_imputado = imputar_treino_cruzado(ajustado.dobras, tr.G, tr.P, tr.M[s, t], tr.C, dobras, S, geradores)
                    recalibrado = RegressaoLinearBayesiana().fit(montar_Z(G_treino_imputado, tr.P, tr.C), tr.y)

                for nome_regressor, regressor in regressores.items():
                    bracos = prever_bracos(ajustado.completo, regressor, medida.G, medida.P, M, medida.C, S,
                                           gerador(semente, t, FLUXO_VALIDACAO),
                                           recalibrado if nome_regressor == REGRESSAO_LINEAR else None)
                    for braco, (medias, variancias) in bracos.items():
                        metricas.append({"regressor": nome_regressor, **chave, "braco": braco, "imputador": nome, "S": S,
                                         **avaliar_mistura(medida.y, medias, variancias)})
                        li, ls = intervalo(medias, variancias)
                        ocultas.append(por_ocultas(medida.y, medias.mean(axis=0), li, ls, n_ocultas)
                                       .assign(regressor=nome_regressor, **chave, imputador=nome, braco=braco))

                if com_parte_1 and taxa > 0:
                    celulas = celulas_da_medida(ajustado.completo, medida, M, gerador(semente, t, FLUXO_IMPUTACAO))
                    imputacao.append({"imputador": nome, **chave, **avaliar_imputacao(celulas)})
                    por_gaveta.append(imputacao_por_gaveta(celulas).assign(imputador=nome, **chave))

            print(f"  semente {semente}, taxa {taxa:.2f}: {time.perf_counter() - inicio:.1f} s", flush=True)

    tabelas = {
        "propagacao_metricas": pd.DataFrame(metricas),
        "propagacao_por_ocultas": _chaves_primeiro(pd.concat(ocultas), ["regressor", "imputador", "taxa", "semente", "braco"]),
    }
    if com_parte_1:
        por_gaveta = pd.concat(por_gaveta).assign(gaveta=lambda d: np.asarray(gavetas)[d.gaveta])
        tabelas["imputacao_metricas"] = pd.DataFrame(imputacao)
        tabelas["imputacao_por_gaveta"] = _chaves_primeiro(por_gaveta, ["imputador", "taxa", "semente", "gaveta"])
    return tabelas


def _chaves_primeiro(tabela: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    return tabela[chaves + [c for c in tabela.columns if c not in chaves]].reset_index(drop=True)


def juntar(novas: pd.DataFrame, arquivo: Path, chave: str, executados: list[str], imputadores: list[str]) -> pd.DataFrame:
    """Linhas existentes dos pares não executados, e as referências da taxa 0 já gravadas, mais as novas.

    `chave` é a coluna que identifica o que rodou: `regressor` nas tabelas da propagação e
    `imputador` nas da Parte 1 e dos tempos. Linhas gravadas antes de o regressor ser uma dimensão
    são da regressão linear.
    """
    if not arquivo.exists():
        return novas
    existentes = pd.read_csv(arquivo, float_precision="round_trip")  # o leitor padrão muda o último dígito
    if chave == "regressor" and "regressor" not in existentes:
        existentes.insert(0, "regressor", REGRESSAO_LINEAR)
    executado = existentes[chave].isin(executados)
    if "imputador" in existentes and chave != "imputador":
        executado &= existentes.imputador.isin(imputadores)
    mantidas = existentes[~executado]
    if "imputador" in mantidas and (mantidas.imputador == SEM_IMPUTADOR).any():
        com_referencia = mantidas.loc[mantidas.imputador == SEM_IMPUTADOR, "regressor"].unique()
        novas = novas[~((novas.imputador == SEM_IMPUTADOR) & novas.regressor.isin(com_referencia))]
    return pd.concat([mantidas, novas], ignore_index=True)


def resumir(metricas: pd.DataFrame) -> pd.DataFrame:
    valores = metricas.drop(columns=["semente", "S"])
    resumo = valores.groupby(["regressor", "imputador", "taxa", "braco"]).agg(["mean", "std"])
    resumo.columns = [f"{metrica}_{'media' if estatistica == 'mean' else 'desvio'}" for metrica, estatistica in resumo.columns]
    return resumo.reset_index()


def _rotular(tabela: pd.DataFrame) -> pd.DataFrame:
    """Nomes legíveis e na ordem das constantes; sem a coluna `imputador`, a linha vale para toda a fileira."""
    rotulada = tabela.assign(regressor=pd.Categorical(tabela.regressor.map(NOMES_DOS_REGRESSORES),
                                                      list(NOMES_DOS_REGRESSORES.values())))
    if "imputador" in tabela:
        rotulada["imputador"] = pd.Categorical(tabela.imputador.map(NOMES_DOS_IMPUTADORES),
                                               list(NOMES_DOS_IMPUTADORES.values()))
    return rotulada


def figura(resumo: pd.DataFrame, metrica: str, titulo: str, eixo_y: str, conjunto: str, nominal: float | None = None,
           taxa_zero: pd.DataFrame | None = None):
    """Um painel por regressor e imputador; linhas horizontais no nominal e na taxa 0 do regressor da linha."""
    dados = _rotular(resumo).assign(
        braco=pd.Categorical(resumo.braco.map(NOMES_DOS_BRACOS), list(NOMES_DOS_BRACOS.values())),
        centro=resumo[f"{metrica}_media"],
        inferior=resumo[f"{metrica}_media"] - resumo[f"{metrica}_desvio"],
        superior=resumo[f"{metrica}_media"] + resumo[f"{metrica}_desvio"],
    )
    grafico = ggplot(dados, aes(x="taxa", y="centro", color="braco"))
    referencias = []
    if nominal is not None:
        grafico += geom_hline(yintercept=nominal, color="#4D4D4D", linetype="dashed", size=0.8)
        referencias.append(f"tracejada: nominal ({100 * nominal:.0f}%)")
    if taxa_zero is not None:
        grafico += geom_hline(_rotular(taxa_zero), aes(yintercept="valor"), color="#4D4D4D", linetype="dotted", size=0.8)
        valores = "; ".join(f"{NOMES_DOS_REGRESSORES[r]} {100 * v:.1f}%".replace(".", ",")
                            for r, v in zip(taxa_zero.regressor, taxa_zero.valor))
        referencias.append(f"pontilhada: taxa 0 do regressor ({valores})")
    return (
        grafico
        + geom_errorbar(aes(ymin="inferior", ymax="superior"), width=0.03, size=0.8)
        + geom_line(size=1.2)
        + geom_point(size=3.5)
        + facet_grid("regressor ~ imputador")
        + scale_color_manual(values=CORES[: len(NOMES_DOS_BRACOS)])
        + TEMA
        + labs(title=titulo, subtitle=" · ".join(referencias),
               x="Taxa de quantidades ocultas", y=eixo_y, color="Braço",
               caption=f"{NOMES_DOS_CONJUNTOS[conjunto]}; média ± desvio entre 3 sementes de máscara. "
                       "Linhas: regressor; colunas: imputador.")
    )


def gravar(tabelas: dict[str, pd.DataFrame], saida: Path, conjunto: str) -> pd.DataFrame:
    """CSVs, resumo e figuras, todos dentro de `saida`."""
    for nome in TABELAS_DA_PROPAGACAO:
        tabelas[nome] = _chaves_primeiro(tabelas[nome], ["regressor"])
    resumo = resumir(tabelas["propagacao_metricas"])

    figuras = saida / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    for nome, tabela in tabelas.items():
        tabela.to_csv(saida / f"{nome}.csv", index=False)
    resumo.to_csv(saida / "propagacao_resumo.csv", index=False)

    picp_taxa_zero = (resumo.query("taxa == 0 and braco == @PLUGIN").groupby("regressor", sort=False).picp_media.first()
                      .rename("valor").reset_index())
    n_regressores, n_imputadores = resumo.regressor.nunique(), resumo.imputador.nunique()
    tamanho = {"width": 1 + 4 * n_imputadores, "height": 1 + 4.5 * n_regressores, "dpi": 200, "verbose": False,
               "limitsize": False}
    figura(resumo, "picp", "Cobertura do intervalo de 90%", "PICP", conjunto, nominal=0.9, taxa_zero=picp_taxa_zero
           ).save(figuras / "propagacao_picp.png", **tamanho)
    figura(resumo, "mpiw_g", "Largura do intervalo de 90%", "MPIW (g)", conjunto
           ).save(figuras / "propagacao_mpiw.png", **tamanho)
    return resumo


def main() -> None:
    argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentos.add_argument("--conjunto", choices=list(NOMES_DOS_CONJUNTOS), default=VALIDACAO)
    argumentos.add_argument("--regressores", nargs="+", choices=list(REGRESSORES), default=list(REGRESSORES))
    argumentos.add_argument("--imputadores", nargs="+", choices=list(IMPUTADORES), default=list(IMPUTADORES))
    escolha = argumentos.parse_args()
    regressores_executados, imputadores_executados = escolha.regressores, escolha.imputadores
    parcial = set(regressores_executados) != set(REGRESSORES) or set(imputadores_executados) != set(IMPUTADORES)
    com_parte_1 = REGRESSAO_LINEAR in regressores_executados
    saida = pasta_de_saida(escolha.conjunto)

    inicio = time.perf_counter()
    tr, medida, taxas, sementes, gavetas = treino_e_medida(escolha.conjunto)
    print(f"treino: {len(tr.y):,} refeições | {NOMES_DOS_CONJUNTOS[escolha.conjunto].lower()}: {len(medida.y):,} refeições | "
          f"S = {S} | saída: {saida.relative_to(RAIZ)} | "
          f"regressores: {', '.join(regressores_executados)} | imputadores: {', '.join(imputadores_executados)}", flush=True)
    dobras = dobras_por_pessoa(tr.pessoas, np.random.default_rng(0))

    print("ajuste dos regressores:", flush=True)
    regressores, tempos_regressores = ajustar_regressores(tr, regressores_executados)
    print("ajuste dos imputadores:", flush=True)
    ajustados, tempos_imputadores = ajustar_imputadores(tr, dobras, imputadores_executados, com_dobras=com_parte_1)
    print("curva por taxa:", flush=True)
    tabelas = medir(tr, medida, ajustados, regressores, dobras, taxas, sementes, gavetas)
    tabelas["regressores_ajuste"] = tempos_regressores
    if com_parte_1:
        tabelas["imputadores_ajuste"] = tempos_imputadores
    if parcial:
        chaves = {nome: ("imputador", imputadores_executados) if nome in TABELAS_DA_PARTE_1
                  else ("regressor", regressores_executados) for nome in tabelas}
        tabelas = {nome: juntar(tabela, saida / f"{nome}.csv", chaves[nome][0], chaves[nome][1], imputadores_executados)
                   for nome, tabela in tabelas.items()}
    resumo = gravar(tabelas, saida, escolha.conjunto)

    pd.set_option("display.width", 220)
    colunas = ["regressor", "imputador", "taxa", "braco"] + [f"{m}_media" for m in ("mae_g", "nll", "crps_g", "picp", "mpiw_g", "pct_meia_dose")]
    print("\n", resumo.query("regressor in @regressores_executados")[colunas].round(3).to_string(index=False))
    if com_parte_1:
        print("\n", tabelas["imputacao_metricas"].drop(columns="semente").groupby(["imputador", "taxa"]).mean().round(3).to_string())
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
