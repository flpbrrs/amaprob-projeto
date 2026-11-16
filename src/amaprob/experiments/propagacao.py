"""Plug-in × plug-in recalibrado × propagação em função da taxa de quantidades ocultas, por regressor e imputador.

Cada regressor (regressão linear bayesiana e SVGP) e cada imputador (porção média, PPCA e FA) são
ajustados uma vez no treino completo. O braço recalibrado existe só com a regressão linear, e os
imputadores das 5 dobras dele também são ajustados uma vez por imputador. As máscaras definem o que
fica oculto na validação. O teste nunca é lido (Regra 6).

Com a regressão linear, grava também as métricas da Parte 1 (imputação nas células ocultas da
validação), que não dependem do regressor. A referência da taxa 0 por número de ocultas é gravada
por regressor.

Com `--regressores` e `--imputadores`, só os pares escolhidos rodam, e as linhas deles entram nos
CSVs existentes; as dos outros pares e as referências da taxa 0 já gravadas ficam como estavam.
Vale porque os geradores dependem só da semente, da taxa e do fluxo, e não do regressor nem do
imputador.

Uso: uv run python -m amaprob.experiments.propagacao [--regressores svgp] [--imputadores fa]
"""

import argparse
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from plotnine import (
    aes, facet_grid, geom_errorbar, geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual,
)

from amaprob.data.dataset import RAIZ, carregar_mascaras, carregar_refeicoes
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
S = 100
Q = 5
M_INDUTORES = 256
IMPUTADORES = {
    "porcao_media": PorcaoMedia,
    "ppca": lambda: PPCA(q=Q, semente=0),
    "fa": lambda: FA(q=Q, semente=0),
    "porcao_condicional": lambda: PorcaoMedia(media_condicional=True),
    "ppca_condicional": lambda: PPCA(q=Q, semente=0, media_condicional=True),
    "fa_condicional": lambda: FA(q=Q, semente=0, media_condicional=True),
}
NOMES_DOS_IMPUTADORES = {
    "porcao_media": "Porção média",
    "ppca": f"PPCA (q = {Q})",
    "fa": f"FA (q = {Q})",
    "porcao_condicional": "Porção média cond.",
    "ppca_condicional": f"PPCA cond. (q = {Q})",
    "fa_condicional": f"FA cond. (q = {Q})",
}
REGRESSAO_LINEAR = "blr"
REGRESSORES = {
    REGRESSAO_LINEAR: RegressaoLinearBayesiana,
    "svgp": lambda: SVGP(n_indutores=M_INDUTORES, semente=0),
}
NOMES_DOS_REGRESSORES = {REGRESSAO_LINEAR: "Regressão linear bayesiana", "svgp": f"SVGP (M = {M_INDUTORES})"}
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


def treino_e_validacao() -> tuple[Conjunto, Conjunto, np.ndarray, np.ndarray, np.ndarray]:
    refeicoes = carregar_refeicoes()
    mascaras = carregar_mascaras(refeicoes)
    recortes = [
        Conjunto(refeicoes.G[i], refeicoes.P[i], refeicoes.C[i], refeicoes.y[i], refeicoes.person_id[i],
                 refeicoes.meal_id[i], mascaras.M[:, :, i])
        for i in separar(refeicoes)
    ]
    return *recortes, mascaras.taxas, mascaras.sementes, refeicoes.gavetas


def gerador(semente: int, indice_taxa: int, fluxo: int) -> np.random.Generator:
    return np.random.default_rng((int(semente), indice_taxa, fluxo))


def _diagnostico(imputador) -> dict[str, float]:
    no_piso = getattr(imputador, "gavetas_no_piso_", None)
    return {"iteracoes": getattr(imputador, "n_iteracoes_", np.nan),
            "gavetas_no_piso": np.nan if no_piso is None else len(no_piso)}


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
             "iteracoes": np.nan, "gavetas_no_piso": np.nan},
        ]
        print(f"  {nome}: completo {meio - inicio:.1f} s | {N_DOBRAS} dobras {fim - meio:.1f} s", flush=True)
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


def celulas_da_validacao(imputador: Imputer, va: Conjunto, M: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    return pd.concat([
        celulas_de_imputacao(va.G[b], np.log1p(G_s), M[b])
        for b, G_s in amostras_em_gramas(imputador, va.G, va.P, M, va.C, S, rng)
    ])


def medir(tr: Conjunto, va: Conjunto, ajustados: dict[str, Ajustado], regressores: dict[str, ProbRegressor],
          dobras, taxas, sementes, gavetas):
    metricas, ocultas, imputacao, por_gaveta = [], [], [], []
    com_parte_1 = REGRESSAO_LINEAR in regressores
    Z_completo = montar_Z(va.G, va.P, va.C)
    diretas = {}
    for nome_regressor, regressor in regressores.items():
        media, var = regressor.predict(Z_completo)
        diretas[nome_regressor] = (media, *intervalo(media[None], var[None]))

    for s, semente in enumerate(sementes):
        for t, taxa in enumerate(taxas):
            inicio = time.perf_counter()
            M = va.M[s, t]
            n_ocultas = M.sum(axis=1)
            chave = {"taxa": taxa, "semente": int(semente)}
            for nome_regressor, (media, li, ls) in diretas.items():
                ocultas.append(por_ocultas(va.y, media, li, ls, n_ocultas).assign(
                    regressor=nome_regressor, **chave, imputador=SEM_IMPUTADOR, braco=REFERENCIA_TAXA_ZERO))

            for nome, ajustado in ajustados.items():
                recalibrado = None
                if com_parte_1:
                    geradores = [gerador(semente, t, 1 + f) for f in range(N_DOBRAS)]
                    G_treino_imputado = imputar_treino_cruzado(ajustado.dobras, tr.G, tr.P, tr.M[s, t], tr.C, dobras, S, geradores)
                    recalibrado = RegressaoLinearBayesiana().fit(montar_Z(G_treino_imputado, tr.P, tr.C), tr.y)

                for nome_regressor, regressor in regressores.items():
                    bracos = prever_bracos(ajustado.completo, regressor, va.G, va.P, M, va.C, S,
                                           gerador(semente, t, FLUXO_VALIDACAO),
                                           recalibrado if nome_regressor == REGRESSAO_LINEAR else None)
                    for braco, (medias, variancias) in bracos.items():
                        metricas.append({"regressor": nome_regressor, **chave, "braco": braco, "imputador": nome, "S": S,
                                         **avaliar_mistura(va.y, medias, variancias)})
                        li, ls = intervalo(medias, variancias)
                        ocultas.append(por_ocultas(va.y, medias.mean(axis=0), li, ls, n_ocultas)
                                       .assign(regressor=nome_regressor, **chave, imputador=nome, braco=braco))

                if com_parte_1 and taxa > 0:
                    celulas = celulas_da_validacao(ajustado.completo, va, M, gerador(semente, t, FLUXO_IMPUTACAO))
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


def figura(resumo: pd.DataFrame, metrica: str, titulo: str, eixo_y: str, nominal: float | None = None,
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
               caption="Validação; média ± desvio entre 3 sementes de máscara. Linhas: regressor; colunas: imputador.")
    )


def main() -> None:
    argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentos.add_argument("--regressores", nargs="+", choices=list(REGRESSORES), default=list(REGRESSORES))
    argumentos.add_argument("--imputadores", nargs="+", choices=list(IMPUTADORES), default=list(IMPUTADORES))
    escolha = argumentos.parse_args()
    regressores_executados, imputadores_executados = escolha.regressores, escolha.imputadores
    parcial = set(regressores_executados) != set(REGRESSORES) or set(imputadores_executados) != set(IMPUTADORES)
    com_parte_1 = REGRESSAO_LINEAR in regressores_executados

    inicio = time.perf_counter()
    tr, va, taxas, sementes, gavetas = treino_e_validacao()
    print(f"treino: {len(tr.y):,} refeições | validação: {len(va.y):,} refeições | S = {S} | "
          f"regressores: {', '.join(regressores_executados)} | imputadores: {', '.join(imputadores_executados)}", flush=True)
    dobras = dobras_por_pessoa(tr.pessoas, np.random.default_rng(0))

    print("ajuste dos regressores:", flush=True)
    regressores, tempos_regressores = ajustar_regressores(tr, regressores_executados)
    print("ajuste dos imputadores:", flush=True)
    ajustados, tempos_imputadores = ajustar_imputadores(tr, dobras, imputadores_executados, com_dobras=com_parte_1)
    print("curva por taxa:", flush=True)
    tabelas = medir(tr, va, ajustados, regressores, dobras, taxas, sementes, gavetas)
    tabelas["regressores_ajuste"] = tempos_regressores
    if com_parte_1:
        tabelas["imputadores_ajuste"] = tempos_imputadores
    if parcial:
        chaves = {nome: ("imputador", imputadores_executados) if nome in TABELAS_DA_PARTE_1
                  else ("regressor", regressores_executados) for nome in tabelas}
        tabelas = {nome: juntar(tabela, SAIDA / f"{nome}.csv", chaves[nome][0], chaves[nome][1], imputadores_executados)
                   for nome, tabela in tabelas.items()}
    for nome in TABELAS_DA_PROPAGACAO:
        tabelas[nome] = _chaves_primeiro(tabelas[nome], ["regressor"])
    resumo = resumir(tabelas["propagacao_metricas"])

    figuras = SAIDA / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    for nome, tabela in tabelas.items():
        tabela.to_csv(SAIDA / f"{nome}.csv", index=False)
    resumo.to_csv(SAIDA / "propagacao_resumo.csv", index=False)

    picp_taxa_zero = (resumo.query("taxa == 0 and braco == @PLUGIN").groupby("regressor", sort=False).picp_media.first()
                      .rename("valor").reset_index())
    n_regressores, n_imputadores = resumo.regressor.nunique(), resumo.imputador.nunique()
    tamanho = {"width": 1 + 4 * n_imputadores, "height": 1 + 4.5 * n_regressores, "dpi": 200, "verbose": False,
               "limitsize": False}
    figura(resumo, "picp", "Cobertura do intervalo de 90%", "PICP", nominal=0.9, taxa_zero=picp_taxa_zero
           ).save(figuras / "propagacao_picp.png", **tamanho)
    figura(resumo, "mpiw_g", "Largura do intervalo de 90%", "MPIW (g)"
           ).save(figuras / "propagacao_mpiw.png", **tamanho)

    pd.set_option("display.width", 220)
    colunas = ["regressor", "imputador", "taxa", "braco"] + [f"{m}_media" for m in ("mae_g", "nll", "crps_g", "picp", "mpiw_g", "pct_meia_dose")]
    print("\n", resumo.query("regressor in @regressores_executados")[colunas].round(3).to_string(index=False))
    if com_parte_1:
        print("\n", tabelas["imputacao_metricas"].drop(columns="semente").groupby(["imputador", "taxa"]).mean().round(3).to_string())
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
