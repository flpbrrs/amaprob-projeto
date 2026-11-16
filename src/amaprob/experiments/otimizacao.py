"""Parte 3: hiperparâmetros por otimização bayesiana e por busca aleatória, só na validação.

Quatro buscas numa grade discreta: PPCA e FA com média condicional (q e α), porção média condicional
(α) e SVGP (M). Os imputadores são ajustados numa amostra do treino estratificada pela ocasião e
medidos pela NLL do carboidrato no braço de propagação com a regressão linear bayesiana (ajustada no
treino completo), na taxa 0,5 e semente 0. O SVGP é ajustado no treino completo e medido pela NLL
na taxa 0. O α do candidato é usado direto na amostra: é a precisão de um prior, e o ótimo dela não
depende do número de refeições.

Cada modelo tem a configuração de partida avaliada como referência, 12 avaliações aleatórias e 12
da otimização bayesiana, que começa pelas 4 primeiras aleatórias. A escolha (menor NLL; no SVGP, o
menor M a até 0,01 da menor NLL, com a partida entre os concorrentes) é confirmada no treino completo,
contra a de partida, nas 3 sementes da taxa 0,5; nos condicionais, também com o α multiplicado pela
razão entre as refeições do treino e as da amostra (o mesmo freio relativo aos dados), como
diagnóstico. Um imputador só adota a escolhida se ela melhorar a NLL em cada semente que não entrou
na escolha e a média delas em pelo menos 0,015, a oscilação entre máscaras; senão, fica a de partida. O arquivo de configuração é
derivado da confirmação. O teste nunca é lido (Regra 6).

Com `--modelos`, as linhas dos outros modelos nos CSVs ficam como estão.

Uso: uv run python -m amaprob.experiments.otimizacao [--modelos svgp]
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import itertools
import json
import time

import numpy as np
import pandas as pd
from plotnine import (
    aes, facet_wrap, geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual, scale_x_continuous,
)

from amaprob.data.dataset import OCASIOES, RAIZ
from amaprob.evaluation.metricas import avaliar, avaliar_mistura
from amaprob.evaluation.tema import CORES, TEMA
from amaprob.experiments.propagacao import FLUXO_VALIDACAO, S, Conjunto, gerador, treino_e_validacao
from amaprob.imputation.base import ocultar
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.imputation.fa import FA
from amaprob.imputation.ppca import PPCA
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagation import PROPAGACAO, prever_bracos
from amaprob.regression.svgp import SVGP
from amaprob.tuning.busca import (
    Avaliacao, Avaliador, busca_aleatoria, melhor_ate_aqui, menor_dentro_da_tolerancia, otimizacao_bayesiana,
)

SAIDA = RAIZ / "results"
CONFIGURACAO = RAIZ / "configs" / "hiperparametros.json"
SEMENTE = 0
N_AVALIACOES = 12
N_INICIAIS = 4
FRACAO_AMOSTRA = 0.25
TAXA_OBJETIVO = 0.5
TOLERANCIA_M = 0.01
QS = np.arange(1, 16)
ALFAS = 10.0 ** np.arange(-2, 3 + 1e-9, 0.25)
MS = np.arange(32, 513, 16)
METRICAS = ("nll", "crps_g", "picp", "mpiw_g", "mae_g")
MELHORA_MINIMA_NLL = 0.015  # oscilação da NLL entre máscaras para a mesma comparação
ALFA_CONVERTIDO = "escolhida_alfa_convertido"
FAIXAS = {"q": (QS.min(), QS.max()), "alfa": (np.log10(ALFAS.min()), np.log10(ALFAS.max())),
          "n_indutores": (np.log2(MS.min()), np.log2(MS.max()))}
SVGP_NOME = "svgp"
HERDAM_O_Q = {"ppca": "ppca_condicional", "fa": "fa_condicional"}


@dataclass(frozen=True)
class Modelo:
    nome: str
    rotulo: str
    candidatos: pd.DataFrame
    partida: dict
    construir: Callable[[dict], object]

    @property
    def pontos(self) -> np.ndarray:
        """Candidatos em [0, 1]: q linear, log₁₀ α e log₂ M."""
        colunas = []
        for botao, valores in self.candidatos.items():
            escala = {"q": valores, "alfa": np.log10(valores), "n_indutores": np.log2(valores)}[botao]
            piso, teto = FAIXAS[botao]
            colunas.append((escala - piso) / (teto - piso))
        return np.column_stack(colunas)

    def indice(self, params: dict) -> int:
        iguais = np.all([np.isclose(self.candidatos[b], v) for b, v in params.items()], axis=0)
        return int(np.flatnonzero(iguais)[0])

    def params(self, indice: int) -> dict:
        linha = self.candidatos.iloc[indice]
        return {b: (int(v) if b != "alfa" else float(v)) for b, v in linha.items()}


def grade(**botoes) -> pd.DataFrame:
    return pd.DataFrame(list(itertools.product(*botoes.values())), columns=list(botoes))


MODELOS = {
    m.nome: m for m in (
        Modelo("ppca_condicional", "PPCA cond. (q, α)", grade(q=QS, alfa=ALFAS), {"q": 5, "alfa": 1.0},
               lambda p: PPCA(q=p["q"], semente=0, media_condicional=True, alfa=p["alfa"])),
        Modelo("fa_condicional", "FA cond. (q, α)", grade(q=QS, alfa=ALFAS), {"q": 5, "alfa": 1.0},
               lambda p: FA(q=p["q"], semente=0, media_condicional=True, alfa=p["alfa"])),
        Modelo("porcao_condicional", "Porção média cond. (α)", grade(alfa=ALFAS), {"alfa": 1.0},
               lambda p: PorcaoMedia(media_condicional=True, alfa=p["alfa"])),
        Modelo(SVGP_NOME, "SVGP (M)", grade(n_indutores=MS), {"n_indutores": 256},
               lambda p: SVGP(n_indutores=p["n_indutores"], semente=0)),
    )
}


def amostra_estratificada(C: np.ndarray, fracao: float, rng: np.random.Generator) -> np.ndarray:
    """Índices de uma amostra com a mesma proporção de cada ocasião (colunas indicadoras no início de `C`)."""
    ocasiao = C[:, : len(OCASIOES)].argmax(axis=1)
    escolhidos = []
    for classe in np.unique(ocasiao):
        da_classe = np.flatnonzero(ocasiao == classe)
        escolhidos.append(rng.choice(da_classe, size=round(fracao * len(da_classe)), replace=False))
    return np.sort(np.concatenate(escolhidos))


def recortar(conjunto: Conjunto, indices: np.ndarray) -> Conjunto:
    return Conjunto(conjunto.G[indices], conjunto.P[indices], conjunto.C[indices], conjunto.y[indices],
                    conjunto.pessoas[indices], conjunto.meal_id[indices], conjunto.M[:, :, indices])


def ajustar_imputador(imputador, conjunto: Conjunto):
    sem_mascara = np.zeros_like(conjunto.P)
    return imputador.fit(ocultar(conjunto.G, sem_mascara), conjunto.P, sem_mascara, conjunto.C)


def medir_propagacao(imputador, regressor, va: Conjunto, s: int, t: int, semente: int) -> dict[str, float]:
    bracos = prever_bracos(imputador, regressor, va.G, va.P, va.M[s, t], va.C, S, gerador(semente, t, FLUXO_VALIDACAO))
    return _so_as_metricas(avaliar_mistura(va.y, *bracos[PROPAGACAO]))


def _so_as_metricas(todas: dict) -> dict[str, float]:
    return {m: float(todas[m]) for m in METRICAS}


def medir_imputador(modelo: Modelo, params: dict, ajuste: Conjunto, va: Conjunto, regressor, t: int) -> dict[str, float]:
    imputador = ajustar_imputador(modelo.construir(params), ajuste)
    return medir_propagacao(imputador, regressor, va, 0, t, SEMENTE)


def objetivo_do_imputador(modelo: Modelo, ajuste: Conjunto, va: Conjunto, regressor, t: int):
    return lambda indice: medir_imputador(modelo, modelo.params(indice), ajuste, va, regressor, t)


def objetivo_do_svgp(modelo: Modelo, tr: Conjunto, va: Conjunto):
    Z_treino, Z_validacao = montar_Z(tr.G, tr.P, tr.C), montar_Z(va.G, va.P, va.C)

    def objetivo(indice: int) -> dict[str, float]:
        regressor = modelo.construir(modelo.params(indice)).fit(Z_treino, tr.y)
        return _so_as_metricas(avaliar(va.y, *regressor.predict(Z_validacao)))
    return objetivo


def escolher(modelo: Modelo, avaliacoes: list[Avaliacao]) -> int:
    unicas = {a.candidato: a.nll for a in avaliacoes}
    candidatos, nlls = np.array(list(unicas)), np.array(list(unicas.values()))
    if modelo.nome == SVGP_NOME:
        M = modelo.candidatos.n_indutores.to_numpy()[candidatos]
        return int(candidatos[menor_dentro_da_tolerancia(M, nlls, TOLERANCIA_M)])
    return int(candidatos[np.argmin(nlls)])


def registrar(modelo: Modelo, metodo: str, avaliacoes: list[Avaliacao]) -> pd.DataFrame:
    return pd.DataFrame([
        {"modelo": modelo.nome, "metodo": metodo, "avaliacao": n, **modelo.params(a.candidato), **a.metricas,
         "segundos": a.segundos, "reaproveitada": a.reaproveitada, "melhor_ate_aqui": melhor}
        for n, (a, melhor) in enumerate(zip(avaliacoes, melhor_ate_aqui(avaliacoes)), start=1 if metodo != "partida" else 0)
    ])


def confirmar_imputador(modelo: Modelo, escolhida: dict, tr: Conjunto, va: Conjunto, regressor, t: int,
                        sementes: np.ndarray, razao_amostra: float, apenas: set[str] | None = None) -> pd.DataFrame:
    configuracoes = {"partida": modelo.partida, "escolhida": escolhida}
    if "alfa" in escolhida:
        configuracoes[ALFA_CONVERTIDO] = {**escolhida, "alfa": escolhida["alfa"] / razao_amostra}
    linhas, ajustados = [], {}
    for nome, params in configuracoes.items():
        if apenas is not None and nome not in apenas:
            continue
        chave = tuple(sorted(params.items()))
        if chave not in ajustados:
            inicio = time.perf_counter()
            ajustados[chave] = (ajustar_imputador(modelo.construir(params), tr), time.perf_counter() - inicio)
        imputador, segundos = ajustados[chave]
        for s, semente in enumerate(sementes):
            linhas.append({"modelo": modelo.nome, "configuracao": nome, **params, "semente": int(semente),
                           **medir_propagacao(imputador, regressor, va, s, t, int(semente)),
                           "segundos_ajuste": segundos})
        print(f"    confirmação {nome} {params}: ajuste {segundos:.1f} s", flush=True)
    return pd.DataFrame(linhas)


def confirmar_svgp(modelo: Modelo, partida: Avaliacao, escolhida: Avaliacao) -> pd.DataFrame:
    return pd.DataFrame([
        {"modelo": modelo.nome, "configuracao": nome, **modelo.params(a.candidato), "semente": np.nan,
         **a.metricas, "segundos_ajuste": a.segundos}
        for nome, a in (("partida", partida), ("escolhida", escolhida))
    ])


def juntar(novas: pd.DataFrame, arquivo, modelos: list[str]) -> pd.DataFrame:
    if not arquivo.exists():
        return novas
    existentes = pd.read_csv(arquivo, float_precision="round_trip")
    return pd.concat([existentes[~existentes.modelo.isin(modelos)], novas], ignore_index=True)


def params_da_linha(modelo: Modelo, linha: pd.Series) -> dict:
    return {b: (int(linha[b]) if b != "alfa" else float(linha[b])) for b in modelo.candidatos.columns}


def adotada(modelo: Modelo, confirmacao: pd.DataFrame) -> tuple[str, dict]:
    """Imputadores: a escolhida só substitui a de partida se melhorar a NLL em cada semente que não
    entrou na escolha e a média delas em pelo menos `MELHORA_MINIMA_NLL`. SVGP: a escolhida (sem
    máscara, não há sementes novas; a partida já concorre na regra do menor M)."""
    linhas = confirmacao[confirmacao.modelo == modelo.nome]
    por_configuracao = {nome: linhas[linhas.configuracao == nome] for nome in ("partida", "escolhida")}
    if modelo.nome == SVGP_NOME:
        return "escolhida", params_da_linha(modelo, por_configuracao["escolhida"].iloc[0])
    nll_nas_novas = {nome: d[d.semente != SEMENTE].set_index("semente").nll for nome, d in por_configuracao.items()}
    melhora = nll_nas_novas["partida"] - nll_nas_novas["escolhida"]
    nome = "escolhida" if (melhora > 0).all() and melhora.mean() >= MELHORA_MINIMA_NLL else "partida"
    return nome, params_da_linha(modelo, por_configuracao[nome].iloc[0])


def gravar_configuracao(confirmacao: pd.DataFrame) -> dict:
    configuracao = {}
    for nome in (m for m in MODELOS if m in set(confirmacao.modelo)):
        origem, params = adotada(MODELOS[nome], confirmacao)
        configuracao[nome] = params
        print(f"  {nome}: adotada a {origem} {params}", flush=True)
    for sem_media, condicional in HERDAM_O_Q.items():
        if condicional in configuracao:
            configuracao[sem_media] = {"q": configuracao[condicional]["q"]}
    CONFIGURACAO.parent.mkdir(parents=True, exist_ok=True)
    CONFIGURACAO.write_text(json.dumps(configuracao, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return configuracao


def figura(avaliacoes: pd.DataFrame):
    rotulos = {m.nome: m.rotulo for m in MODELOS.values()}
    ordem = [rotulos[m] for m in MODELOS if m in set(avaliacoes.modelo)]
    rotular = lambda d: d.assign(modelo=pd.Categorical(d.modelo.map(rotulos), ordem))
    nomes = {"aleatoria": "Busca aleatória", "bayesiana": "Otimização bayesiana"}
    curvas = rotular(avaliacoes.query("metodo != 'partida'")).assign(
        metodo=lambda d: pd.Categorical(d.metodo.map(nomes), list(nomes.values())))
    partida = rotular(avaliacoes.query("metodo == 'partida'"))
    return (
        ggplot(curvas, aes(x="avaliacao", y="melhor_ate_aqui", color="metodo"))
        + geom_hline(partida, aes(yintercept="nll"), color="#4D4D4D", linetype="dashed", size=0.8)
        + geom_line(size=1.2)
        + geom_point(size=3)
        + facet_wrap("modelo", scales="free_y", ncol=2)
        + scale_color_manual(values=CORES[:2])
        + scale_x_continuous(breaks=list(range(1, N_AVALIACOES + 1)))
        + TEMA
        + labs(title="Melhor NLL pelo número de avaliações",
               subtitle="tracejada: configuração de partida (q = 5, α = 1, M = 256)",
               x="Avaliação", y="Melhor NLL até ali", color="Método",
               caption="Validação. Imputadores: propagação com a regressão linear, taxa 0,5, semente 0, ajuste em 25% do treino. "
                       "SVGP: taxa 0, treino completo.")
    )


def buscar(modelo: Modelo, avaliador: Avaliador) -> tuple[Avaliacao, list[Avaliacao], list[Avaliacao]]:
    inicio = time.perf_counter()
    partida = avaliador(modelo.indice(modelo.partida))
    print(f"    partida {modelo.partida}: NLL {partida.nll:.4f} ({partida.segundos:.1f} s)", flush=True)
    aleatoria = busca_aleatoria(len(modelo.candidatos), avaliador, N_AVALIACOES, np.random.default_rng(SEMENTE))
    print(f"    aleatória: melhor NLL {min(a.nll for a in aleatoria):.4f} ({time.perf_counter() - inicio:.0f} s até aqui)", flush=True)
    iniciais = [a.candidato for a in aleatoria[:N_INICIAIS]]
    bayesiana = otimizacao_bayesiana(modelo.pontos, avaliador, N_AVALIACOES, iniciais, SEMENTE)
    print(f"    bayesiana: melhor NLL {min(a.nll for a in bayesiana):.4f} ({time.perf_counter() - inicio:.0f} s até aqui)", flush=True)
    return partida, aleatoria, bayesiana


def main() -> None:
    argumentos = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argumentos.add_argument("--modelos", nargs="+", choices=list(MODELOS), default=list(MODELOS))
    argumentos.add_argument("--so-confirmacao", action="store_true",
                            help=f"refaz só as linhas {ALFA_CONVERTIDO} da confirmação, a partir da escolhida gravada")
    escolha = argumentos.parse_args()
    executados = escolha.modelos

    inicio = time.perf_counter()
    tr, va, taxas, sementes, _ = treino_e_validacao()
    t = int(np.flatnonzero(np.isclose(taxas, TAXA_OBJETIVO))[0])
    indices_amostra = amostra_estratificada(tr.C, FRACAO_AMOSTRA, np.random.default_rng(SEMENTE))
    amostra = recortar(tr, indices_amostra)
    razao_amostra = len(indices_amostra) / len(tr.y)
    print(f"treino: {len(tr.y):,} | amostra: {len(indices_amostra):,} | validação: {len(va.y):,} | "
          f"modelos: {', '.join(executados)}", flush=True)
    regressor = RegressaoLinearBayesiana().fit(montar_Z(tr.G, tr.P, tr.C), tr.y)

    arquivo_confirmacao = SAIDA / "otimizacao_confirmacao.csv"
    if escolha.so_confirmacao:
        for nome in (n for n in executados if n != SVGP_NOME):
            existentes = pd.read_csv(arquivo_confirmacao, float_precision="round_trip")
            linha = existentes[(existentes.modelo == nome) & (existentes.configuracao == "escolhida")].iloc[0]
            escolhida = params_da_linha(MODELOS[nome], linha)
            print(f"{nome}: {ALFA_CONVERTIDO} de {escolhida}", flush=True)
            nova = confirmar_imputador(MODELOS[nome], escolhida, tr, va, regressor, t, sementes, razao_amostra,
                                       apenas={ALFA_CONVERTIDO})
            mantidas = existentes[~((existentes.modelo == nome) & (existentes.configuracao == ALFA_CONVERTIDO))]
            pd.concat([mantidas, nova], ignore_index=True).to_csv(arquivo_confirmacao, index=False)
        gravar_configuracao(pd.read_csv(arquivo_confirmacao, float_precision="round_trip"))
        return

    for nome in executados:
        modelo = MODELOS[nome]
        print(f"{nome} ({len(modelo.candidatos)} candidatos):", flush=True)
        if nome == SVGP_NOME:
            avaliador = Avaliador(objetivo_do_svgp(modelo, tr, va))
        else:
            avaliador = Avaliador(objetivo_do_imputador(modelo, amostra, va, regressor, t))
        partida, aleatoria, bayesiana = buscar(modelo, avaliador)
        escolhida_indice = escolher(modelo, [partida] + aleatoria + bayesiana)
        escolhida = modelo.params(escolhida_indice)
        print(f"    escolhida: {escolhida}", flush=True)
        if nome == SVGP_NOME:
            confirmacao = confirmar_svgp(modelo, partida, avaliador(escolhida_indice))
        else:
            confirmacao = confirmar_imputador(modelo, escolhida, tr, va, regressor, t, sementes, razao_amostra)

        registro = pd.concat([registrar(modelo, "partida", [partida]), registrar(modelo, "aleatoria", aleatoria),
                              registrar(modelo, "bayesiana", bayesiana)], ignore_index=True)
        tabelas = {"otimizacao_avaliacoes": registro, "otimizacao_confirmacao": confirmacao}
        for arquivo, tabela in tabelas.items():
            juntar(tabela, SAIDA / f"{arquivo}.csv", [nome]).to_csv(SAIDA / f"{arquivo}.csv", index=False)
        resumo = confirmacao.groupby("configuracao", sort=False)[["nll", "crps_g", "picp", "mpiw_g"]].mean()
        print(resumo.round(4).to_string(), flush=True)

    print("configuração adotada:", flush=True)
    gravar_configuracao(pd.read_csv(arquivo_confirmacao, float_precision="round_trip"))
    avaliacoes = pd.read_csv(SAIDA / "otimizacao_avaliacoes.csv")
    figuras = SAIDA / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    n_modelos = avaliacoes.modelo.nunique()
    figura(avaliacoes).save(figuras / "otimizacao_convergencia.png", width=12, height=1.5 + 4 * ((n_modelos + 1) // 2),
                            dpi=200, verbose=False)
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
