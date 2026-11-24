"""Perguntas ao paciente com erro de relato e regra de parada.

Estende `experiments/pergunta.py` em duas direções, com as estratégias ótima e maior carboidrato:

- erro de relato: o paciente responde a quantidade com erro multiplicativo, g_relatada =
  g_verdadeira · exp(τ ε), ε ~ N(0, 1), τ ∈ {0; 0,2; 0,4}. O modelo trata a resposta como exata
  (o aplicativo não conhece o erro), então o erro entra na imputação das outras ocultas e na
  regressão. Com τ = 0,2, a resposta erra em torno de 20% (desvio em log). O ε de cada célula é
  sorteado uma vez por (semente, taxa), antes da simulação, e vale para as duas estratégias;
- regra de parada: pergunta enquanto a largura do intervalo de 90% da refeição for maior que L g,
  L ∈ {60, 80, 100}, e para quando ela cai a L ou quando acabam as ocultas. A largura é a do braço
  analítico, 2 · 1,645 · σ, calculável antes de cada pergunta.

A escolha gulosa não depende de L: a ordem das perguntas de uma refeição é a mesma para qualquer
limite. Por isso cada (τ, estratégia, taxa, semente) é simulado uma vez até esgotar as ocultas,
guardando a preditiva de cada refeição em cada passo, e a regra de parada é um corte nessa
sequência. A mesma sequência dá a curva por número fixo de perguntas (0 a 3), como em
`experiments/pergunta.py`; refeições que esgotam as ocultas ficam no último estado.

Regra 2: a escolha lê só o modelo e as células observadas (P & ~M). A grama verdadeira de uma oculta
entra apenas quando ela é escolhida, já com o erro de relato. Mede na validação, taxas 0,5 e 1,0,
3 sementes, com a FA condicional e a regressão linear bayesiana ajustadas uma vez no treino
completo. O teste não é lido (Regra 6).

Saídas: results/pergunta_parada_metricas.csv (por τ e L), results/pergunta_relato_metricas.csv
(por τ e número fixo de perguntas) e results/pergunta_parada_tempos.csv.

Uso: uv run python -m amaprob.experiments.pergunta_parada [--taxas 0.5 1.0] [--sementes 0 1 2]
"""

import argparse
import time

import numpy as np
import pandas as pd

from amaprob.data.splits import VALIDACAO
from amaprob.evaluation.metricas import Z_INTERVALO, avaliar_mistura
from amaprob.experiments.pergunta import IMPUTADOR, PERGUNTAS, SAIDA, escolher, prever
from amaprob.experiments.propagacao import IMPUTADORES, treino_e_medida
from amaprob.imputation.base import ocultar
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana

TAXAS = (0.5, 1.0)
ERROS_DE_RELATO = (0.0, 0.2, 0.4)
LIMITES_G = (60.0, 80.0, 100.0)
ESTRATEGIAS = ("otima", "maior_carboidrato")
FLUXO_RELATO = 200  # depois dos fluxos da validação (0), da imputação e das perguntas (100 + passo)


def fator_de_relato(forma: tuple[int, int], tau: float, semente: int, indice_taxa: int) -> np.ndarray:
    """exp(τ ε) por célula; o mesmo ε para todo τ, para que só a escala do erro mude entre eles."""
    rng = np.random.default_rng((int(semente), int(indice_taxa), FLUXO_RELATO))
    return np.exp(tau * rng.standard_normal(forma))


def simular_ate_esgotar(imputador, blr, G, P, C, M0, estrategia: str, fator: np.ndarray):
    """Preditiva de cada refeição antes de cada pergunta, até revelar todas as ocultas.

    Devolve médias e variâncias (N, passos + 1), com NaN depois que a refeição esgota as ocultas, e
    o número de ocultas de cada refeição. Só as refeições que ainda têm ocultas são recalculadas.
    """
    # Gramas com NaN nas ocultas: elas só ganham valor quando são perguntadas.
    G_relatado = np.where(M0, np.nan, np.asarray(G, dtype=np.float64))
    M = M0.copy()
    n_ocultas = M0.sum(axis=1)
    passos = int(n_ocultas.max())
    medias = np.full((len(G), passos + 1), np.nan)
    variancias = np.full((len(G), passos + 1), np.nan)
    ativas = np.arange(len(G))
    for passo in range(passos + 1):
        media, variancia, Eg, reducao = prever(imputador, blr, G_relatado[ativas], P[ativas], M[ativas], C[ativas])
        medias[ativas, passo] = media
        variancias[ativas, passo] = variancia
        com_oculta = M[ativas].sum(axis=1) > 0
        ativas, Eg, reducao = ativas[com_oculta], Eg[com_oculta], reducao[com_oculta]
        if len(ativas) == 0:
            break
        k = escolher(estrategia, reducao, Eg, blr, M[ativas], 0, 0, passo)
        G_relatado[ativas, k] = G[ativas, k] * fator[ativas, k]
        M[ativas, k] = False
    return medias, variancias, n_ocultas


def estado_apos(medias, variancias, n_ocultas, perguntas: np.ndarray):
    """Média e variância depois de `perguntas` (por refeição), limitadas ao número de ocultas."""
    passo = np.minimum(perguntas, n_ocultas)
    linhas = np.arange(len(medias))
    return medias[linhas, passo], variancias[linhas, passo], passo


def perguntas_ate_o_limite(variancias, n_ocultas, limite_g: float) -> np.ndarray:
    """Primeiro passo em que a largura do intervalo de 90% fica ≤ L; senão, todas as ocultas."""
    largura = 2 * Z_INTERVALO * np.sqrt(variancias)
    passos = np.arange(variancias.shape[1])[None, :]
    alcancou = (largura <= limite_g) & (passos <= n_ocultas[:, None])
    return np.where(alcancou.any(axis=1), alcancou.argmax(axis=1), n_ocultas)


def medir(y, media, variancia) -> dict:
    return avaliar_mistura(y, media[None], variancia[None])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--taxas", nargs="+", type=float, default=list(TAXAS))
    parser.add_argument("--sementes", nargs="+", type=int, default=None)
    escolha = parser.parse_args()

    inicio = time.perf_counter()
    tr, medida, taxas, sementes, _ = treino_e_medida(VALIDACAO)
    indices = [int(np.flatnonzero(np.isclose(taxas, x))[0]) for x in escolha.taxas]
    sem_mascara = np.zeros_like(tr.P)
    comeco = time.perf_counter()
    imputador = IMPUTADORES[IMPUTADOR]().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
    blr = RegressaoLinearBayesiana().fit(montar_Z(tr.G, tr.P, tr.C), tr.y)
    print(f"ajustes: {time.perf_counter() - comeco:.1f} s", flush=True)

    parada, fixas, tempos = [], [], []
    for t in indices:
        for s, semente in enumerate(sementes):
            if escolha.sementes is not None and int(semente) not in escolha.sementes:
                continue
            M0 = medida.M[s, t]
            sel = M0.sum(axis=1) > 0  # só refeições com pelo menos uma oculta, como em pergunta.py
            G, P, C, y, M0 = medida.G[sel], medida.P[sel], medida.C[sel], medida.y[sel], M0[sel]
            for tau in ERROS_DE_RELATO:
                fator = fator_de_relato(G.shape, tau, int(semente), t)
                for estrategia in ESTRATEGIAS:
                    comeco = time.perf_counter()
                    medias, variancias, n_ocultas = simular_ate_esgotar(imputador, blr, G, P, C, M0, estrategia, fator)
                    segundos = time.perf_counter() - comeco
                    chave = {"tau": tau, "estrategia": estrategia, "taxa": float(taxas[t]), "semente": int(semente)}
                    tempos.append({**chave, "segundos": segundos, "passos": medias.shape[1] - 1})

                    for n in PERGUNTAS:
                        media, variancia, feitas = estado_apos(medias, variancias, n_ocultas, np.full(len(y), n))
                        fixas.append({**chave, "n_perguntas": n, "n_refeicoes": len(y),
                                      "ocultas_restantes": float((n_ocultas - feitas).mean()),
                                      **medir(y, media, variancia)})

                    for limite in LIMITES_G:
                        feitas = perguntas_ate_o_limite(variancias, n_ocultas, limite)
                        media, variancia, _ = estado_apos(medias, variancias, n_ocultas, feitas)
                        largura = 2 * Z_INTERVALO * np.sqrt(variancia)
                        alcancou = largura <= limite
                        dentro = np.abs(y - media) <= largura / 2
                        parada.append({
                            **chave, "limite_g": limite, "n_refeicoes": len(y),
                            "perguntas_media": float(feitas.mean()),
                            "perguntas_p90": float(np.quantile(feitas, 0.9)),
                            "fracao_sem_pergunta": float((feitas == 0).mean()),
                            "fracao_alcancou_limite": float(alcancou.mean()),
                            "fracao_esgotou_ocultas": float((feitas == n_ocultas).mean()),
                            "picp_das_que_alcancaram": float(dentro[alcancou].mean()) if alcancou.any() else np.nan,
                            **medir(y, media, variancia),
                        })
                    print(f"  taxa {taxas[t]:.2f}, semente {semente}, τ = {tau}, {estrategia}: {segundos:.1f} s", flush=True)

    pd.DataFrame(parada).to_csv(SAIDA / "pergunta_parada_metricas.csv", index=False)
    pd.DataFrame(fixas).to_csv(SAIDA / "pergunta_relato_metricas.csv", index=False)
    pd.DataFrame(tempos).to_csv(SAIDA / "pergunta_parada_tempos.csv", index=False)
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
