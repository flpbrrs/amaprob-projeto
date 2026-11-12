"""Plug-in × plug-in recalibrado × propagação em função da taxa de quantidades ocultas, por imputador.

Cada imputador (porção média e PPCA) e a regressão linear bayesiana são ajustados uma vez no
treino completo; os imputadores das 5 dobras do braço recalibrado também, uma vez por imputador.
As máscaras definem o que fica oculto na validação. O teste nunca é lido (Regra 6).

Grava também as métricas da Parte 1 (imputação nas células ocultas da validação) e a referência
da taxa 0 por número de ocultas.

Uso: uv run python -m amaprob.experiments.propagacao
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from plotnine import (
    aes, facet_wrap, geom_errorbar, geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual,
)

from amaprob.data.dataset import RAIZ, carregar_mascaras, carregar_refeicoes
from amaprob.evaluation.metricas import (
    avaliar_imputacao, avaliar_mistura, celulas_de_imputacao, imputacao_por_gaveta, intervalo, por_ocultas,
)
from amaprob.evaluation.tema import CORES, TEMA
from amaprob.experiments.taxa_zero import separar
from amaprob.imputation.base import Imputer, ocultar
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.imputation.ppca import PPCA
from amaprob.regression.base import montar_Z
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

SAIDA = RAIZ / "results"
S = 100
Q_PPCA = 5
IMPUTADORES = {"porcao_media": PorcaoMedia, "ppca": lambda: PPCA(q=Q_PPCA, semente=0)}
NOMES_DOS_IMPUTADORES = {"porcao_media": "Porção média", "ppca": f"PPCA (q = {Q_PPCA})"}
REFERENCIA_TAXA_ZERO = "referencia_taxa_0"
SEM_IMPUTADOR = "nenhum"
FLUXO_VALIDACAO = 0
FLUXO_IMPUTACAO = 1 + N_DOBRAS  # depois dos fluxos das dobras (1..N_DOBRAS)
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


def _iteracoes(imputador) -> float:
    return getattr(imputador, "n_iteracoes_", np.nan)


def ajustar_imputadores(tr: Conjunto, dobras: np.ndarray) -> tuple[dict[str, Ajustado], pd.DataFrame]:
    sem_mascara = np.zeros_like(tr.P)
    ajustados, tempos = {}, []
    for nome, novo in IMPUTADORES.items():
        inicio = time.perf_counter()
        completo = novo().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
        meio = time.perf_counter()
        das_dobras = ajustar_dobras(novo, tr.G, tr.P, tr.C, dobras)
        fim = time.perf_counter()
        ajustados[nome] = Ajustado(completo, das_dobras)
        tempos += [
            {"imputador": nome, "etapa": "treino completo", "segundos": meio - inicio, "iteracoes": _iteracoes(completo)},
            *({"imputador": nome, "etapa": f"dobra {f}", "segundos": np.nan, "iteracoes": _iteracoes(i)}
              for f, i in enumerate(das_dobras)),
            {"imputador": nome, "etapa": f"{N_DOBRAS} dobras", "segundos": fim - meio, "iteracoes": np.nan},
        ]
        print(f"  {nome}: completo {meio - inicio:.1f} s | {N_DOBRAS} dobras {fim - meio:.1f} s", flush=True)
    return ajustados, pd.DataFrame(tempos)


def celulas_da_validacao(imputador: Imputer, va: Conjunto, M: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    return pd.concat([
        celulas_de_imputacao(va.G[b], np.log1p(G_s), M[b])
        for b, G_s in amostras_em_gramas(imputador, va.G, va.P, M, va.C, S, rng)
    ])


def medir(tr: Conjunto, va: Conjunto, ajustados: dict[str, Ajustado], regressor, dobras, taxas, sementes, gavetas):
    metricas, ocultas, imputacao, por_gaveta = [], [], [], []
    media_direta, var_direta = regressor.predict(montar_Z(va.G, va.P, va.C))
    li_direta, ls_direta = intervalo(media_direta[None], var_direta[None])

    for s, semente in enumerate(sementes):
        for t, taxa in enumerate(taxas):
            inicio = time.perf_counter()
            M = va.M[s, t]
            n_ocultas = M.sum(axis=1)
            chave = {"taxa": taxa, "semente": int(semente)}
            ocultas.append(por_ocultas(va.y, media_direta, li_direta, ls_direta, n_ocultas)
                           .assign(**chave, imputador=SEM_IMPUTADOR, braco=REFERENCIA_TAXA_ZERO))

            for nome, ajustado in ajustados.items():
                geradores = [gerador(semente, t, 1 + f) for f in range(N_DOBRAS)]
                G_treino_imputado = imputar_treino_cruzado(ajustado.dobras, tr.G, tr.P, tr.M[s, t], tr.C, dobras, S, geradores)
                recalibrado = RegressaoLinearBayesiana().fit(montar_Z(G_treino_imputado, tr.P, tr.C), tr.y)
                bracos = prever_bracos(ajustado.completo, regressor, va.G, va.P, M, va.C, S,
                                       gerador(semente, t, FLUXO_VALIDACAO), recalibrado)
                for braco, (medias, variancias) in bracos.items():
                    metricas.append({**chave, "braco": braco, "imputador": nome, "S": S,
                                     **avaliar_mistura(va.y, medias, variancias)})
                    li, ls = intervalo(medias, variancias)
                    ocultas.append(por_ocultas(va.y, medias.mean(axis=0), li, ls, n_ocultas)
                                   .assign(**chave, imputador=nome, braco=braco))

                if taxa > 0:
                    celulas = celulas_da_validacao(ajustado.completo, va, M, gerador(semente, t, FLUXO_IMPUTACAO))
                    imputacao.append({"imputador": nome, **chave, **avaliar_imputacao(celulas)})
                    por_gaveta.append(imputacao_por_gaveta(celulas).assign(imputador=nome, **chave))

            print(f"  semente {semente}, taxa {taxa:.2f}: {time.perf_counter() - inicio:.1f} s", flush=True)

    ocultas = pd.concat(ocultas)
    por_gaveta = pd.concat(por_gaveta).assign(gaveta=lambda d: np.asarray(gavetas)[d.gaveta])
    return (
        pd.DataFrame(metricas),
        _chaves_primeiro(ocultas, ["imputador", "taxa", "semente", "braco"]),
        pd.DataFrame(imputacao),
        _chaves_primeiro(por_gaveta, ["imputador", "taxa", "semente", "gaveta"]),
    )


def _chaves_primeiro(tabela: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    return tabela[chaves + [c for c in tabela.columns if c not in chaves]].reset_index(drop=True)


def resumir(metricas: pd.DataFrame) -> pd.DataFrame:
    valores = metricas.drop(columns=["semente", "S"])
    resumo = valores.groupby(["imputador", "taxa", "braco"]).agg(["mean", "std"])
    resumo.columns = [f"{metrica}_{'media' if estatistica == 'mean' else 'desvio'}" for metrica, estatistica in resumo.columns]
    return resumo.reset_index()


def figura(resumo: pd.DataFrame, metrica: str, titulo: str, eixo_y: str, referencias: dict[str, float]):
    dados = resumo.assign(
        braco=pd.Categorical(resumo.braco.map(NOMES_DOS_BRACOS), list(NOMES_DOS_BRACOS.values())),
        imputador=pd.Categorical(resumo.imputador.map(NOMES_DOS_IMPUTADORES), list(NOMES_DOS_IMPUTADORES.values())),
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
        + facet_wrap("imputador")
        + scale_color_manual(values=CORES[: len(NOMES_DOS_BRACOS)])
        + TEMA
        + labs(title=titulo, subtitle=" · ".join(f"{tipo}: {nome}" for nome, tipo in zip(referencias, ("tracejada", "pontilhada"))),
               x="Taxa de quantidades ocultas", y=eixo_y, color="Braço",
               caption="Validação; média ± desvio entre 3 sementes de máscara. Um painel por imputador.")
    )


def main() -> None:
    inicio = time.perf_counter()
    tr, va, taxas, sementes, gavetas = treino_e_validacao()
    print(f"treino: {len(tr.y):,} refeições | validação: {len(va.y):,} refeições | S = {S}", flush=True)
    regressor = RegressaoLinearBayesiana().fit(montar_Z(tr.G, tr.P, tr.C), tr.y)
    dobras = dobras_por_pessoa(tr.pessoas, np.random.default_rng(0))

    print("ajuste dos imputadores:", flush=True)
    ajustados, tempos = ajustar_imputadores(tr, dobras)
    print("curva por taxa:", flush=True)
    metricas, ocultas, imputacao, por_gaveta = medir(tr, va, ajustados, regressor, dobras, taxas, sementes, gavetas)
    resumo = resumir(metricas)

    figuras = SAIDA / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    tempos.to_csv(SAIDA / "imputadores_ajuste.csv", index=False)
    metricas.to_csv(SAIDA / "propagacao_metricas.csv", index=False)
    ocultas.to_csv(SAIDA / "propagacao_por_ocultas.csv", index=False)
    resumo.to_csv(SAIDA / "propagacao_resumo.csv", index=False)
    imputacao.to_csv(SAIDA / "imputacao_metricas.csv", index=False)
    por_gaveta.to_csv(SAIDA / "imputacao_por_gaveta.csv", index=False)

    picp_taxa_zero = resumo.query("taxa == 0 and braco == @PLUGIN").picp_media.iloc[0]
    tamanho = {"width": 12, "height": 5.5, "dpi": 200, "verbose": False}
    figura(resumo, "picp", "Cobertura do intervalo de 90%", "PICP",
           {"nominal (90%)": 0.9, f"taxa 0 ({100 * picp_taxa_zero:.1f}%)".replace(".", ","): picp_taxa_zero}
           ).save(figuras / "propagacao_picp.png", **tamanho)
    figura(resumo, "mpiw_g", "Largura do intervalo de 90%", "MPIW (g)", {}
           ).save(figuras / "propagacao_mpiw.png", **tamanho)

    pd.set_option("display.width", 220)
    colunas = ["imputador", "taxa", "braco"] + [f"{m}_media" for m in ("mae_g", "nll", "crps_g", "picp", "mpiw_g", "pct_meia_dose")]
    print("\n", resumo[colunas].round(3).to_string(index=False))
    print("\n", imputacao.drop(columns="semente").groupby(["imputador", "taxa"]).mean().round(3).to_string())
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
