"""Regressão linear bayesiana heterocedástica contra a homocedástica, na validação.

O modelo está em `regression/blr_heterocedastica.py`. O λ do ridge em γ é escolhido entre
{0,1; 1; 10} pela NLL na validação com as gramas completas (taxa 0).

Medidas, todas na validação (o teste não é lido, Regra 6):

- taxa 0: métricas globais, cobertura por quintil do carboidrato previsto e métricas por segmento
  de tamanho, ocasião e local (segmentos de `experiments/segmentos.py`);
- taxas 0,5 e 1,0, FA condicional, 3 sementes: plug-in, propagação S = 100 e analítica, com as
  métricas globais e a cobertura por quintil da previsão. As duas regressões usam as mesmas
  amostras da imputação (mesmo gerador do experimento oficial), então a BLR (regressão linear bayesiana) reproduz o oficial.

Saídas: results/hetero_*.csv.

Uso: uv run python -m amaprob.experiments.heterocedastica [--taxas 0.5 1.0] [--sementes 0 1 2]
"""

import argparse
import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import RAIZ
from amaprob.data.splits import VALIDACAO
from amaprob.evaluation.metricas import avaliar, avaliar_mistura, intervalo
from amaprob.experiments.propagacao import FLUXO_VALIDACAO, IMPUTADORES, S, gerador, treino_e_medida
from amaprob.experiments.propagacao_analitica import BLOCO_DA_ANALITICA, prever_analitico
from amaprob.experiments.segmentos import (
    agregar_por_segmento, avaliar_linhas, carregar_segmentos, montar_segmentos_da_medida,
)
from amaprob.imputation.base import ocultar
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.blr_heterocedastica import (
    RegressaoLinearBayesianaHeterocedastica, prever_com_entrada_incerta_heterocedastica,
)
from amaprob.regression.propagation import PLUGIN, PROPAGACAO, prever_bracos
from amaprob.regression.propagacao_analitica import momentos_em_gramas

SAIDA = RAIZ / "results"
LAMBDAS = (0.1, 1.0, 10.0)
IMPUTADOR = "fa_condicional"
TAXAS = (0.5, 1.0)
DIMENSOES = ("tamanho", "ocasiao", "local")
HOMOCEDASTICA, HETEROCEDASTICA = "blr", "blr_hetero"


def prever_analitico_heterocedastico(imputador, modelo, G, P, M, C):
    media, variancia = np.empty(len(G)), np.empty(len(G))
    for inicio in range(0, len(G), BLOCO_DA_ANALITICA):
        b = slice(inicio, inicio + BLOCO_DA_ANALITICA)
        media_log, cov_log = momentos_ocultos(imputador, ocultar(G[b], M[b]), P[b], M[b], C[b])
        Eg, Cg = momentos_em_gramas(G[b], P[b], M[b], media_log, cov_log)
        media[b], variancia[b] = prever_com_entrada_incerta_heterocedastica(modelo, Eg, Cg, P[b], C[b])
    return media, variancia


def cobertura_por_quintil(y, medias, variancias) -> pd.DataFrame:
    """PICP por quintil da média prevista; vale para a gaussiana (S = 1) e para a mistura."""
    medias, variancias = np.atleast_2d(medias), np.atleast_2d(variancias)
    li, ls = intervalo(medias, variancias)
    media = medias.mean(axis=0)
    return pd.DataFrame({"quintil": pd.qcut(media, 5, labels=False) + 1, "media": media,
                         "dentro": (li <= y) & (y <= ls), "largura": ls - li}).groupby("quintil").agg(
        media_min_g=("media", "min"), media_max_g=("media", "max"), n=("dentro", "size"),
        picp=("dentro", "mean"), mpiw_g=("largura", "mean"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--taxas", nargs="+", type=float, default=list(TAXAS))
    parser.add_argument("--sementes", nargs="+", type=int, default=None)
    escolha = parser.parse_args()

    inicio = time.perf_counter()
    tempos = []
    tr, medida, taxas, sementes, gavetas = treino_e_medida(VALIDACAO)
    K = len(gavetas)
    Z_tr, Z_va = montar_Z(tr.G, tr.P, tr.C), montar_Z(medida.G, medida.P, medida.C)

    comeco = time.perf_counter()
    homocedastica = RegressaoLinearBayesiana().fit(Z_tr, tr.y)
    tempos.append({"etapa": "ajuste blr", "segundos": time.perf_counter() - comeco})

    # Escolha de λ pela NLL na validação, taxa 0.
    candidatos, historicos, escolha_lambda = {}, [], []
    for lam in LAMBDAS:
        comeco = time.perf_counter()
        modelo = RegressaoLinearBayesianaHeterocedastica(n_gavetas=K, lam=lam).fit(Z_tr, tr.y)
        tempos.append({"etapa": f"ajuste blr_hetero λ = {lam}", "segundos": time.perf_counter() - comeco})
        candidatos[lam] = modelo
        historicos.append(pd.DataFrame(modelo.historico_).assign(lam=lam))
        escolha_lambda.append({"lam": lam, **avaliar(medida.y, *modelo.predict(Z_va)),
                               "convergiu_ruido": modelo.convergiu_ruido_, "alpha": modelo.alpha_})
        print(f"  λ = {lam}: NLL na validação {escolha_lambda[-1]['nll']:.4f}", flush=True)
    escolha_lambda = pd.DataFrame(escolha_lambda)
    lam = float(escolha_lambda.loc[escolha_lambda.nll.idxmin(), "lam"])
    heterocedastica = candidatos[lam]
    escolha_lambda["escolhido"] = escolha_lambda.lam == lam
    regressores = {HOMOCEDASTICA: homocedastica, HETEROCEDASTICA: heterocedastica}

    coeficientes = pd.DataFrame({
        "variavel": ["intercepto", *[f"presenca: {g}" for g in gavetas],
                     *[f"ocasiao: {o}" for o in ("cafe", "almoco", "lanche", "jantar", "ceia")],
                     *[f"tamanho: {t}" for t in ("1", "2", "3", "4", "5+")]],
        "gamma": np.r_[heterocedastica.gamma0_, heterocedastica.gamma_],
    })

    # Taxa 0.
    segmentos = montar_segmentos_da_medida(medida, carregar_segmentos())
    globais, quintis, por_segmento = [], [], []
    for nome, modelo in regressores.items():
        media, variancia = modelo.predict(Z_va)
        globais.append({"modelo": nome, "taxa": 0.0, "braco": "taxa_zero", "semente": -1, **avaliar(medida.y, media, variancia),
                        "variancia_media_g2": float(variancia.mean()),
                        "variancia_p10_g2": float(np.quantile(variancia, 0.1)),
                        "variancia_p90_g2": float(np.quantile(variancia, 0.9))})
        quintis.append(cobertura_por_quintil(medida.y, media, variancia).assign(modelo=nome, taxa=0.0, braco="taxa_zero", semente=-1))
        linhas = avaliar_linhas(medida.y, media, variancia).assign(variancia_g2=variancia, residuo2=(medida.y - media) ** 2)
        tabela = agregar_por_segmento(linhas, segmentos, DIMENSOES).assign(modelo=nome)
        extra = []
        for dimensao in DIMENSOES:
            extra.append(linhas.assign(valor=segmentos[dimensao].astype(str).to_numpy()).groupby("valor").agg(
                variancia_prevista_g2=("variancia_g2", "mean"), variancia_residuo_g2=("residuo2", "mean"),
            ).reset_index().assign(segmento=dimensao))
        por_segmento.append(tabela.merge(pd.concat(extra), on=["segmento", "valor"]))
    print("taxa 0:", pd.DataFrame(globais)[["modelo", "mae_g", "nll", "crps_g", "picp", "mpiw_g"]].round(3).to_string(index=False), flush=True)

    # Taxas 0,5 e 1,0 com a FA condicional.
    comeco = time.perf_counter()
    sem_mascara = np.zeros_like(tr.P)
    imputador = IMPUTADORES[IMPUTADOR]().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
    tempos.append({"etapa": f"ajuste {IMPUTADOR}", "segundos": time.perf_counter() - comeco})
    for t, taxa in enumerate(taxas):
        if not any(np.isclose(taxa, x) for x in escolha.taxas):
            continue
        for s, semente in enumerate(sementes):
            if escolha.sementes is not None and int(semente) not in escolha.sementes:
                continue
            M = medida.M[s, t]
            for nome, modelo in regressores.items():
                comeco = time.perf_counter()
                bracos = prever_bracos(imputador, modelo, medida.G, medida.P, M, medida.C, S,
                                       gerador(semente, t, FLUXO_VALIDACAO))
                if nome == HOMOCEDASTICA:
                    analitica = prever_analitico(imputador, modelo, medida.G, medida.P, M, medida.C)
                else:
                    analitica = prever_analitico_heterocedastico(imputador, modelo, medida.G, medida.P, M, medida.C)
                medidos = {PLUGIN: bracos[PLUGIN], f"propagacao_S{S}": bracos[PROPAGACAO],
                           "analitica": (analitica[0][None], analitica[1][None])}
                for braco, (medias, variancias) in medidos.items():
                    chave = {"modelo": nome, "taxa": float(taxa), "braco": braco, "semente": int(semente)}
                    globais.append({**chave, **avaliar_mistura(medida.y, medias, variancias)})
                    quintis.append(cobertura_por_quintil(medida.y, medias, variancias).assign(**chave))
                segundos = time.perf_counter() - comeco
                tempos.append({"etapa": f"{nome}, taxa {taxa}, semente {semente}", "segundos": segundos})
                print(f"  {nome}, taxa {taxa:.1f}, semente {semente}: {segundos:.1f} s", flush=True)

    gravar = {
        "lambda": escolha_lambda, "historico": pd.concat(historicos, ignore_index=True),
        "coeficientes": coeficientes, "metricas": pd.DataFrame(globais),
        "quintis": pd.concat(quintis, ignore_index=True), "segmentos": pd.concat(por_segmento, ignore_index=True),
        "tempos": pd.DataFrame(tempos),
    }
    for nome, tabela in gravar.items():
        tabela.to_csv(SAIDA / f"hetero_{nome}.csv", index=False)
    print(f"λ escolhido: {lam} | tempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
