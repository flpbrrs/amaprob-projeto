"""BLR (regressão linear bayesiana) homocedástica, heterocedástica log-linear e heterocedástica aditiva, lado a lado, na validação.

Modelos em `regression/blr.py`, `regression/blr_heterocedastica.py` (λ = 0,1, o escolhido em
`experiments/heterocedastica.py`; os três λ empataram) e `regression/blr_heterocedastica_aditiva.py`.

Medidas (o teste não é lido, Regra 6):

- taxa 0: métricas globais, cobertura por quintil do carboidrato previsto e por segmento de
  tamanho, ocasião e local;
- taxas 0,5 e 1,0, FA condicional, 3 sementes: plug-in, propagação S = 100 e analítica, com
  métricas globais, por quintil da previsão e por segmento de tamanho, ocasião e local. Os três
  modelos usam as mesmas amostras da imputação (mesmo gerador do experimento oficial).

Saídas: results/hetero_aditiva_*.csv.

Uso: uv run python -m amaprob.experiments.heterocedastica_aditiva [--taxas 0.5 1.0] [--sementes 0 1 2]
"""

import argparse
import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import RAIZ
from amaprob.data.splits import VALIDACAO
from amaprob.evaluation.metricas import avaliar_mistura
from amaprob.experiments.heterocedastica import (
    IMPUTADOR, TAXAS, cobertura_por_quintil, prever_analitico_heterocedastico,
)
from amaprob.experiments.propagacao import FLUXO_VALIDACAO, IMPUTADORES, S, gerador, treino_e_medida
from amaprob.experiments.propagacao_analitica import prever_analitico
from amaprob.experiments.segmentos import (
    agregar_por_segmento, avaliar_linhas, avaliar_linhas_mistura, carregar_segmentos, montar_segmentos_da_medida,
)
from amaprob.imputation.base import ocultar
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.blr_heterocedastica import RegressaoLinearBayesianaHeterocedastica
from amaprob.regression.blr_heterocedastica_aditiva import RegressaoLinearBayesianaHeterocedasticaAditiva
from amaprob.regression.propagation import PLUGIN, PROPAGACAO, prever_bracos

SAIDA = RAIZ / "results"
LAMBDA_LOG_LINEAR = 0.1
DIMENSOES = ("tamanho", "ocasiao", "local")
HOMOCEDASTICA, LOG_LINEAR, ADITIVA = "blr", "log_linear", "aditiva"


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

    fabricas = {
        HOMOCEDASTICA: lambda: RegressaoLinearBayesiana(),
        LOG_LINEAR: lambda: RegressaoLinearBayesianaHeterocedastica(n_gavetas=K, lam=LAMBDA_LOG_LINEAR),
        ADITIVA: lambda: RegressaoLinearBayesianaHeterocedasticaAditiva(n_gavetas=K),
    }
    regressores, historicos = {}, []
    for nome, fabrica in fabricas.items():
        comeco = time.perf_counter()
        regressores[nome] = fabrica().fit(Z_tr, tr.y)
        tempos.append({"etapa": f"ajuste {nome}", "segundos": time.perf_counter() - comeco})
        if hasattr(regressores[nome], "historico_"):
            historicos.append(pd.DataFrame(regressores[nome].historico_).assign(modelo=nome))

    aditiva = regressores[ADITIVA]
    coeficientes = pd.DataFrame({
        "variavel": ["v0", *[f"presenca: {g}" for g in gavetas],
                     *[f"ocasiao: {o}" for o in ("cafe", "almoco", "lanche", "jantar", "ceia")]],
        "coeficiente_g2": aditiva.coeficientes_,
    })

    segmentos = montar_segmentos_da_medida(medida, carregar_segmentos())
    globais, quintis, por_segmento = [], [], []

    def registrar(chave: dict, medias: np.ndarray, variancias: np.ndarray) -> None:
        globais.append({**chave, **avaliar_mistura(medida.y, medias, variancias),
                        "variancia_media_g2": float((variancias.mean(axis=0) + medias.var(axis=0)).mean())})
        quintis.append(cobertura_por_quintil(medida.y, medias, variancias).assign(**chave))
        if len(medias) == 1:
            linhas = avaliar_linhas(medida.y, medias[0], variancias[0])
        else:
            linhas = avaliar_linhas_mistura(medida.y, medias, variancias)
        linhas = linhas.assign(variancia_g2=variancias.mean(axis=0) + medias.var(axis=0),
                               residuo2=(medida.y - medias.mean(axis=0)) ** 2)
        tabela = agregar_por_segmento(linhas, segmentos, DIMENSOES)
        extra = pd.concat([
            linhas.assign(valor=segmentos[d].astype(str).to_numpy()).groupby("valor").agg(
                variancia_prevista_g2=("variancia_g2", "mean"), variancia_residuo_g2=("residuo2", "mean"),
            ).reset_index().assign(segmento=d)
            for d in DIMENSOES
        ])
        por_segmento.append(tabela.merge(extra, on=["segmento", "valor"]).assign(**chave))

    for nome, modelo in regressores.items():
        media, variancia = modelo.predict(Z_va)
        registrar({"modelo": nome, "taxa": 0.0, "braco": "taxa_zero", "semente": -1}, media[None], variancia[None])
    taxa_zero = pd.DataFrame(globais)[["modelo", "mae_g", "nll", "crps_g", "picp", "mpiw_g"]]
    print("taxa 0:\n", taxa_zero.round(3).to_string(index=False), flush=True)
    print(pd.concat(quintis).pivot_table(index="modelo", columns="quintil", values="picp").round(3).to_string(), flush=True)

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
                prever = prever_analitico if nome == HOMOCEDASTICA else prever_analitico_heterocedastico
                media_ana, var_ana = prever(imputador, modelo, medida.G, medida.P, M, medida.C)
                medidos = {PLUGIN: bracos[PLUGIN], f"propagacao_S{S}": bracos[PROPAGACAO],
                           "analitica": (media_ana[None], var_ana[None])}
                for braco, (medias, variancias) in medidos.items():
                    registrar({"modelo": nome, "taxa": float(taxa), "braco": braco, "semente": int(semente)},
                              medias, variancias)
                segundos = time.perf_counter() - comeco
                tempos.append({"etapa": f"{nome}, taxa {taxa}, semente {semente}", "segundos": segundos})
                print(f"  {nome}, taxa {taxa:.1f}, semente {semente}: {segundos:.1f} s", flush=True)

    gravar = {
        "metricas": pd.DataFrame(globais), "quintis": pd.concat(quintis, ignore_index=True),
        "segmentos": pd.concat(por_segmento, ignore_index=True), "coeficientes": coeficientes,
        "historico": pd.concat(historicos, ignore_index=True), "tempos": pd.DataFrame(tempos),
    }
    for nome, tabela in gravar.items():
        tabela.to_csv(SAIDA / f"hetero_aditiva_{nome}.csv", index=False)
    print(f"tempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
