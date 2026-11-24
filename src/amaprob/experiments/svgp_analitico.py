"""SVGP com entrada incerta: plug-in, analítica-SVGP e propagação S = 100, na validação.

O braço analítico está em `regression/svgp_entrada_incerta.py`. Escopo: FA condicional, taxas 0,5
e 1,0, 3 sementes. O SVGP é o de `configs/hiperparametros.json` (M = 288), ajustado uma vez no
treino completo; plug-in e propagação saem do caminho oficial (`prever_bracos`, mesmo gerador).
O teste não é lido (Regra 6).

`--estimar` mede o tempo do braço analítico num bloco de refeições e para.

Saídas: results/svgp_analitico_{metricas,por_ocultas,tempos}.csv, regravadas a cada (taxa,
semente).

Uso: uv run python -m amaprob.experiments.svgp_analitico [--estimar] [--taxas 0.5 1.0] [--sementes 0 1 2]
"""

import argparse
import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import RAIZ
from amaprob.data.splits import VALIDACAO
from amaprob.evaluation.metricas import avaliar_mistura, intervalo, por_ocultas
from amaprob.experiments.propagacao import (
    FLUXO_VALIDACAO, IMPUTADORES, S, gerador, ler_configuracao, treino_e_medida,
)
from amaprob.imputation.base import ocultar
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.regression.base import montar_Z
from amaprob.regression.propagation import PLUGIN, PROPAGACAO, prever_bracos
from amaprob.regression.propagacao_analitica import momentos_em_gramas
from amaprob.regression.svgp import SVGP
from amaprob.regression.svgp_entrada_incerta import extrair, prever_com_entrada_incerta_svgp

SAIDA = RAIZ / "results"
IMPUTADOR = "fa_condicional"
TAXAS = (0.5, 1.0)
BLOCO_MOMENTOS = 4_096
ANALITICA = "analitica"


def prever_analitico_svgp(imputador, parametros, G, P, M, C):
    media, variancia = np.empty(len(G)), np.empty(len(G))
    for inicio in range(0, len(G), BLOCO_MOMENTOS):
        b = slice(inicio, inicio + BLOCO_MOMENTOS)
        media_log, cov_log = momentos_ocultos(imputador, ocultar(G[b], M[b]), P[b], M[b], C[b])
        Eg, Cg = momentos_em_gramas(G[b], P[b], M[b], media_log, cov_log)
        media[b], variancia[b] = prever_com_entrada_incerta_svgp(parametros, Eg, Cg, P[b], C[b], M[b])
    return media, variancia


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--estimar", action="store_true")
    parser.add_argument("--taxas", nargs="+", type=float, default=list(TAXAS))
    parser.add_argument("--sementes", nargs="+", type=int, default=None)
    escolha = parser.parse_args()

    inicio = time.perf_counter()
    tempos = []
    tr, medida, taxas, sementes, _ = treino_e_medida(VALIDACAO)
    comeco = time.perf_counter()
    svgp = SVGP(semente=0, **ler_configuracao().get("svgp", {})).fit(montar_Z(tr.G, tr.P, tr.C), tr.y)
    parametros = extrair(svgp)
    tempos.append({"etapa": f"ajuste do SVGP (M = {svgp.n_indutores})", "segundos": time.perf_counter() - comeco})
    print(f"SVGP: {tempos[-1]['segundos']:.0f} s", flush=True)
    comeco = time.perf_counter()
    sem_mascara = np.zeros_like(tr.P)
    imputador = IMPUTADORES[IMPUTADOR]().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
    tempos.append({"etapa": f"ajuste de {IMPUTADOR}", "segundos": time.perf_counter() - comeco})

    if escolha.estimar:
        # Um bloco de refeições da taxa 1,0, semente 0, e outro da taxa 0,5 (mais grupos de H).
        for taxa in TAXAS:
            t = int(np.flatnonzero(np.isclose(taxas, taxa))[0])
            M = medida.M[0, t]
            n = 2_000
            comeco = time.perf_counter()
            prever_analitico_svgp(imputador, parametros, medida.G[:n], medida.P[:n], M[:n], medida.C[:n])
            segundos = time.perf_counter() - comeco
            por_rodada = segundos * len(medida.y) / n
            print(f"taxa {taxa}: {segundos:.1f} s para {n} refeições -> {por_rodada / 60:.1f} min por rodada", flush=True)
            tempos.append({"etapa": f"estimativa, taxa {taxa}, {n} refeições", "segundos": segundos,
                           "minutos_por_rodada": por_rodada / 60})
        pd.DataFrame(tempos).to_csv(SAIDA / "svgp_analitico_estimativa.csv", index=False)
        return

    metricas, ocultas = [], []
    for t, taxa in enumerate(taxas):
        if not any(np.isclose(taxa, x) for x in escolha.taxas):
            continue
        for s, semente in enumerate(sementes):
            if escolha.sementes is not None and int(semente) not in escolha.sementes:
                continue
            M = medida.M[s, t]
            n_ocultas = M.sum(axis=1)
            comeco = time.perf_counter()
            bracos = prever_bracos(imputador, svgp, medida.G, medida.P, M, medida.C, S,
                                   gerador(semente, t, FLUXO_VALIDACAO))
            tempo_amostras = time.perf_counter() - comeco
            comeco = time.perf_counter()
            media, variancia = prever_analitico_svgp(imputador, parametros, medida.G, medida.P, M, medida.C)
            tempo_analitica = time.perf_counter() - comeco
            medidos = {PLUGIN: bracos[PLUGIN], f"propagacao_S{S}": bracos[PROPAGACAO],
                       ANALITICA: (media[None], variancia[None])}
            for braco, (medias, variancias) in medidos.items():
                chave = {"regressor": "svgp", "imputador": IMPUTADOR, "taxa": float(taxa),
                         "semente": int(semente), "braco": braco}
                metricas.append({**chave, **avaliar_mistura(medida.y, medias, variancias)})
                li, ls = intervalo(medias, variancias)
                ocultas.append(por_ocultas(medida.y, medias.mean(axis=0), li, ls, n_ocultas).assign(**chave))
            tempos += [{"etapa": f"plug-in e propagação, taxa {taxa}, semente {semente}", "segundos": tempo_amostras},
                       {"etapa": f"analítica, taxa {taxa}, semente {semente}", "segundos": tempo_analitica}]
            pd.DataFrame(metricas).to_csv(SAIDA / "svgp_analitico_metricas.csv", index=False)
            pd.concat(ocultas, ignore_index=True).to_csv(SAIDA / "svgp_analitico_por_ocultas.csv", index=False)
            pd.DataFrame(tempos).to_csv(SAIDA / "svgp_analitico_tempos.csv", index=False)
            print(f"  taxa {taxa}, semente {semente}: amostras {tempo_amostras:.0f} s, analítica {tempo_analitica:.0f} s",
                  flush=True)
    print(f"tempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
