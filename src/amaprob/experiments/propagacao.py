"""Plug-in × plug-in recalibrado × propagação em função da taxa de quantidades ocultas.

Imputador (porção média) e regressão linear bayesiana ajustados uma vez no treino completo; as
máscaras definem o que fica oculto na validação. O teste nunca é lido (Regra 6).

Uso: uv run python -m amaprob.experiments.propagacao
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from plotnine import aes, geom_errorbar, geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual

from amaprob.data.dataset import RAIZ, carregar_mascaras, carregar_refeicoes
from amaprob.evaluation.metricas import avaliar_mistura, intervalo, por_ocultas
from amaprob.evaluation.tema import CORES, TEMA
from amaprob.experiments.taxa_zero import separar
from amaprob.imputation.base import ocultar
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagation import (
    N_DOBRAS,
    PLUGIN,
    PLUGIN_RECALIBRADO,
    PROPAGACAO,
    dobras_por_pessoa,
    imputar_treino_cruzado,
    prever_bracos,
)

SAIDA = RAIZ / "results"
IMPUTADOR = "porcao_media"
S_CANDIDATOS = (100, 500)
TAXA_DA_ESCOLHA_DE_S = 0.5
VARIACAO_MAXIMA_CRPS = 0.01
VARIACAO_MAXIMA_PICP = 0.005
FLUXO_VALIDACAO = 0
NOMES_DOS_BRACOS = {PLUGIN: "Plug-in", PLUGIN_RECALIBRADO: "Plug-in recalibrado", PROPAGACAO: "Propagação"}


@dataclass(frozen=True)
class Conjunto:
    G: np.ndarray
    P: np.ndarray
    C: np.ndarray
    y: np.ndarray
    pessoas: np.ndarray
    meal_id: np.ndarray
    M: np.ndarray  # (sementes, taxas, N, K)


def treino_e_validacao() -> tuple[Conjunto, Conjunto, np.ndarray, np.ndarray]:
    refeicoes = carregar_refeicoes()
    mascaras = carregar_mascaras(refeicoes)
    recortes = [
        Conjunto(refeicoes.G[i], refeicoes.P[i], refeicoes.C[i], refeicoes.y[i], refeicoes.person_id[i],
                 refeicoes.meal_id[i], mascaras.M[:, :, i])
        for i in separar(refeicoes)
    ]
    return *recortes, mascaras.taxas, mascaras.sementes


def ajustar_no_treino_completo(tr: Conjunto) -> tuple[PorcaoMedia, RegressaoLinearBayesiana]:
    sem_mascara = np.zeros_like(tr.P)
    imputador = PorcaoMedia().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
    return imputador, RegressaoLinearBayesiana().fit(montar_Z(tr.G, tr.P, tr.C), tr.y)


def gerador(semente: int, indice_taxa: int, fluxo: int) -> np.random.Generator:
    return np.random.default_rng((int(semente), indice_taxa, fluxo))


def escolher_S(imputador, regressor, va: Conjunto, taxas, sementes) -> tuple[int, pd.DataFrame]:
    t = int(np.flatnonzero(taxas == TAXA_DA_ESCOLHA_DE_S)[0])
    linhas = []
    for S in S_CANDIDATOS:
        inicio = time.perf_counter()
        medias, variancias = prever_bracos(
            imputador, regressor, va.G, va.P, va.M[0, t], va.C, S, gerador(sementes[0], t, FLUXO_VALIDACAO)
        )[PROPAGACAO]
        meio = time.perf_counter()
        metricas = avaliar_mistura(va.y, medias, variancias)
        fim = time.perf_counter()
        linhas.append({"S": S, "crps_g": metricas["crps_g"], "picp": metricas["picp"],
                       "segundos_amostras_e_previsao": meio - inicio, "segundos_metricas": fim - meio,
                       "segundos": fim - inicio})
        print(f"  S = {S}: CRPS {metricas['crps_g']:.4f} g | PICP {metricas['picp']:.4f} | {fim - inicio:.1f} s", flush=True)

    comparacao = pd.DataFrame(linhas)
    menor, maior = comparacao.iloc[0], comparacao.iloc[-1]
    estavel = (abs(menor.crps_g - maior.crps_g) / maior.crps_g < VARIACAO_MAXIMA_CRPS
               and abs(menor.picp - maior.picp) < VARIACAO_MAXIMA_PICP)
    S = int(menor.S if estavel else maior.S)
    return S, comparacao.assign(adotado=comparacao.S == S)


def medir(tr: Conjunto, va: Conjunto, imputador, regressor, dobras, S: int, taxas, sementes):
    metricas, ocultas = [], []
    for s, semente in enumerate(sementes):
        for t, taxa in enumerate(taxas):
            inicio = time.perf_counter()
            geradores = [gerador(semente, t, 1 + f) for f in range(N_DOBRAS)]
            G_treino_imputado = imputar_treino_cruzado(PorcaoMedia, tr.G, tr.P, tr.M[s, t], tr.C, dobras, S, geradores)
            recalibrado = RegressaoLinearBayesiana().fit(montar_Z(G_treino_imputado, tr.P, tr.C), tr.y)
            bracos = prever_bracos(imputador, regressor, va.G, va.P, va.M[s, t], va.C, S,
                                   gerador(semente, t, FLUXO_VALIDACAO), recalibrado)

            chave = {"taxa": taxa, "semente": int(semente)}
            n_ocultas = va.M[s, t].sum(axis=1)
            for braco, (medias, variancias) in bracos.items():
                metricas.append({**chave, "braco": braco, "imputador": IMPUTADOR, "S": S,
                                 **avaliar_mistura(va.y, medias, variancias)})
                li, ls = intervalo(medias, variancias)
                ocultas.append(por_ocultas(va.y, medias.mean(axis=0), li, ls, n_ocultas).assign(**chave, braco=braco))
            print(f"  semente {semente}, taxa {taxa:.2f}: {time.perf_counter() - inicio:.1f} s", flush=True)

    colunas_chave = ["taxa", "semente", "braco"]
    ocultas = pd.concat(ocultas)
    return pd.DataFrame(metricas), ocultas[colunas_chave + [c for c in ocultas.columns if c not in colunas_chave]]


def resumir(metricas: pd.DataFrame) -> pd.DataFrame:
    valores = metricas.drop(columns=["semente", "imputador", "S"])
    resumo = valores.groupby(["taxa", "braco"]).agg(["mean", "std"])
    resumo.columns = [f"{metrica}_{'media' if estatistica == 'mean' else 'desvio'}" for metrica, estatistica in resumo.columns]
    return resumo.reset_index()


def figura(resumo: pd.DataFrame, metrica: str, titulo: str, eixo_y: str, referencias: dict[str, float]):
    dados = resumo.assign(
        braco=pd.Categorical(resumo.braco.map(NOMES_DOS_BRACOS), list(NOMES_DOS_BRACOS.values())),
        centro=resumo[f"{metrica}_media"],
        inferior=resumo[f"{metrica}_media"] - resumo[f"{metrica}_desvio"],
        superior=resumo[f"{metrica}_media"] + resumo[f"{metrica}_desvio"],
    )
    grafico = ggplot(dados, aes(x="taxa", y="centro", color="braco"))
    for valor, tipo in zip(referencias.values(), ("dashed", "dotted")):
        grafico += geom_hline(yintercept=valor, color="#4D4D4D", linetype=tipo, size=0.8)
    return (
        grafico
        + geom_errorbar(aes(ymin="inferior", ymax="superior"), width=0.03, size=0.8)
        + geom_line(size=1.2)
        + geom_point(size=3.5)
        + scale_color_manual(values=CORES[: len(NOMES_DOS_BRACOS)])
        + TEMA
        + labs(title=titulo, subtitle=" · ".join(f"{tipo}: {nome}" for nome, tipo in zip(referencias, ("tracejada", "pontilhada"))),
               x="Taxa de quantidades ocultas", y=eixo_y, color="Braço",
               caption="Validação; média ± desvio entre 3 sementes de máscara. Imputador: porção média.")
    )


def main() -> None:
    tr, va, taxas, sementes = treino_e_validacao()
    print(f"treino: {len(tr.y):,} refeições | validação: {len(va.y):,} refeições", flush=True)
    imputador, regressor = ajustar_no_treino_completo(tr)

    print("escolha de S:", flush=True)
    S, comparacao_S = escolher_S(imputador, regressor, va, taxas, sementes)
    print(f"S adotado: {S}", flush=True)

    dobras = dobras_por_pessoa(tr.pessoas, np.random.default_rng(0))
    print("curva por taxa:", flush=True)
    metricas, ocultas = medir(tr, va, imputador, regressor, dobras, S, taxas, sementes)
    resumo = resumir(metricas)

    figuras = SAIDA / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    comparacao_S.to_csv(SAIDA / "propagacao_S.csv", index=False)
    metricas.to_csv(SAIDA / "propagacao_metricas.csv", index=False)
    ocultas.to_csv(SAIDA / "propagacao_por_ocultas.csv", index=False)
    resumo.to_csv(SAIDA / "propagacao_resumo.csv", index=False)

    picp_taxa_zero = resumo.query("taxa == 0 and braco == @PLUGIN").picp_media.item()
    figura(resumo, "picp", "Cobertura do intervalo de 90%", "PICP",
           {"nominal (90%)": 0.9, f"taxa 0 ({100 * picp_taxa_zero:.1f}%)".replace(".", ","): picp_taxa_zero}
           ).save(figuras / "propagacao_picp.png", dpi=200, verbose=False)
    figura(resumo, "mpiw_g", "Largura do intervalo de 90%", "MPIW (g)", {}
           ).save(figuras / "propagacao_mpiw.png", dpi=200, verbose=False)

    pd.set_option("display.width", 220)
    colunas = ["taxa", "braco"] + [f"{m}_media" for m in ("mae_g", "nll", "crps_g", "picp", "mpiw_g", "pct_meia_dose")]
    print("\n", resumo[colunas].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
