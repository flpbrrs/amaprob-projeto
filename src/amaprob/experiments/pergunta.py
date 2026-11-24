"""Qual quantidade perguntar ao paciente: simulação de perguntas sequenciais.

Numa refeição com várias gavetas ocultas, o aplicativo pode pedir uma quantidade ao paciente. A
pergunta útil é a que mais reduz a incerteza do carboidrato. Com o braço analítico, a redução
esperada da variância preditiva ao revelar a gaveta k sai em forma fechada
(`regression/pergunta.py`).

O experimento compara quatro estratégias de escolha:

- ótima: maior redução prevista da variância;
- maior carboidrato esperado: maior |m_k| Eg_k / s_k, o efeito da gaveta na média da previsão;
- maior porção esperada: maior Eg_k;
- aleatória: uniforme entre as ocultas, com gerador próprio por (semente, taxa, pergunta).

Cada pergunta é respondida com a grama verdadeira e o imputador recondiciona sem reajustar: a
máscara perde a célula revelada e `momentos_ocultos` devolve a condicional com mais uma observação.
Refeições com menos ocultas que perguntas param quando acabam as ocultas, e todas entram nas
métricas de todos os passos.

Regra 2: a escolha usa só as células observadas (P & ~M) e o modelo. A grama verdadeira de uma
célula oculta entra apenas depois de ela ser escolhida, e as outras ocultas nunca são lidas. O teste
`test_escolha_nao_le_as_outras_ocultas` troca essas gramas por lixo para verificar.

Mede na validação, taxas 0,5 e 1,0, 3 sementes, com a FA condicional do repositório e a regressão
linear bayesiana, ambas ajustadas uma vez no treino completo. O teste não é lido (Regra 6).

Saídas novas: results/pergunta_metricas.csv, results/pergunta_reducao.csv,
results/pergunta_gavetas.csv e results/pergunta_tempos.csv.

Uso: uv run python -m amaprob.experiments.pergunta [--taxas 0.5 1.0] [--sementes 0 1 2]
"""

import argparse
import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import RAIZ
from amaprob.data.splits import VALIDACAO
from amaprob.evaluation.metricas import avaliar_mistura
from amaprob.experiments.propagacao import IMPUTADORES, treino_e_medida
from amaprob.imputation.base import ocultar
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.pergunta import reducao_por_gaveta
from amaprob.regression.propagacao_analitica import momentos_em_gramas, prever_com_entrada_incerta

SAIDA = RAIZ / "results"
IMPUTADOR = "fa_condicional"
TAXAS = (0.5, 1.0)
PERGUNTAS = (0, 1, 2, 3)
ESTRATEGIAS = ("otima", "maior_carboidrato", "maior_porcao", "aleatoria")
FLUXO_PERGUNTA = 100  # depois dos fluxos da validação (0) e da imputação (1 + N_DOBRAS)
BLOCO = 4_096  # refeições por bloco: a covariância em gramas é N × K × K


def prever(imputador, blr, G: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray):
    """Média, variância, média da entrada e redução por gaveta, em blocos de refeições."""
    media, variancia = np.empty(len(G)), np.empty(len(G))
    entrada = np.empty_like(G, dtype=np.float64)
    reducao = np.zeros_like(G, dtype=np.float64)
    for inicio in range(0, len(G), BLOCO):
        b = slice(inicio, inicio + BLOCO)
        X = ocultar(G[b], M[b])
        media_log, cov_log = momentos_ocultos(imputador, X, P[b], M[b], C[b])
        Eg, Cg = momentos_em_gramas(G[b], P[b], M[b], media_log, cov_log)
        media[b], variancia[b] = prever_com_entrada_incerta(blr, Eg, Cg, P[b], C[b])
        reducao[b] = reducao_por_gaveta(blr, Eg, Cg)
        entrada[b] = Eg
    return media, variancia, entrada, reducao


def escolher(estrategia: str, reducao: np.ndarray, Eg: np.ndarray, blr, M: np.ndarray,
             semente: int, indice_taxa: int, passo: int) -> np.ndarray:
    """Gaveta a perguntar em cada refeição, entre as ocultas. Lê só o modelo e o que está observado."""
    K = M.shape[1]
    if estrategia == "aleatoria":
        # Uniforme entre as ocultas da refeição: sorteia o posto da oculta escolhida.
        rng = np.random.default_rng((int(semente), int(indice_taxa), FLUXO_PERGUNTA + passo))
        contagem = M.sum(axis=1)
        posto = np.cumsum(M, axis=1) - 1
        alvo = np.minimum(np.floor(rng.random(len(M)) * contagem).astype(int), contagem - 1)
        return np.argmax(M & (posto == alvo[:, None]), axis=1)

    if estrategia == "otima":
        nota = reducao
    elif estrategia == "maior_carboidrato":
        nota = np.abs(blr.m_[:K])[None, :] * Eg / blr.desvio_z_[:K][None, :]
    elif estrategia == "maior_porcao":
        nota = Eg
    else:
        raise ValueError(f"estratégia desconhecida: {estrategia}")
    return np.argmax(np.where(M, nota, -np.inf), axis=1)


def simular(imputador, blr, G, P, C, y, M0, estrategia, semente, indice_taxa):
    """Perguntas sequenciais até o teto.

    Devolve as métricas e os tempos por passo, mais o registro da primeira pergunta: a
    gaveta escolhida, a redução prevista, a variância antes e a realizada.
    """
    M = M0.copy()
    linhas, tempos = [], []
    k_primeiro = prevista = var_antes = var_um = None
    for passo in range(max(PERGUNTAS) + 1):
        comeco = time.perf_counter()
        media, variancia, Eg, reducao = prever(imputador, blr, G, P, M, C)
        segundos = time.perf_counter() - comeco
        linhas.append({
            "n_perguntas": passo, "n_refeicoes": len(G),
            "ocultas_restantes": float(M.sum(axis=1).mean()),
            **avaliar_mistura(y, media[None], variancia[None]),
        })
        tempos.append({"estrategia": estrategia, "n_perguntas": passo, "segundos": segundos})
        if passo == 1:
            var_um = variancia.copy()
        if passo >= max(PERGUNTAS):
            break
        k = escolher(estrategia, reducao, Eg, blr, M, semente, indice_taxa, passo)
        if passo == 0:
            k_primeiro = k
            prevista = reducao[np.arange(len(G)), k]
            var_antes = variancia.copy()
        M[np.arange(len(G)), k] = False  # célula revelada passa a observada na próxima pergunta

    detalhe = {"k": k_primeiro, "prevista": prevista, "realizada": var_antes - var_um}
    return linhas, tempos, detalhe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--taxas", nargs="+", type=float, default=list(TAXAS))
    parser.add_argument("--sementes", nargs="+", type=int, default=None)
    escolha = parser.parse_args()

    inicio = time.perf_counter()
    tr, medida, taxas, sementes, gavetas = treino_e_medida(VALIDACAO)
    indices = [int(np.flatnonzero(np.isclose(taxas, x))[0]) for x in escolha.taxas]
    print(f"treino: {len(tr.y):,} | validação: {len(medida.y):,} | imputador: {IMPUTADOR} | "
          f"taxas: {escolha.taxas} | estratégias: {', '.join(ESTRATEGIAS)}", flush=True)

    sem_mascara = np.zeros_like(tr.P)
    comeco = time.perf_counter()
    imputador = IMPUTADORES[IMPUTADOR]().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
    ajuste_imputador = time.perf_counter() - comeco
    comeco = time.perf_counter()
    blr = RegressaoLinearBayesiana().fit(montar_Z(tr.G, tr.P, tr.C), tr.y)
    ajuste_blr = time.perf_counter() - comeco
    print(f"ajustes: {IMPUTADOR} {ajuste_imputador:.1f} s | regressão {ajuste_blr:.1f} s", flush=True)

    linhas, tempos, reducao, tabelas_de_gavetas = [], [], [], []
    for t in indices:
        for s, semente in enumerate(sementes):
            if escolha.sementes is not None and int(semente) not in escolha.sementes:
                continue
            M0 = medida.M[s, t]
            sel = M0.sum(axis=1) > 0  # só refeições com pelo menos uma oculta
            G, P, C, y = medida.G[sel], medida.P[sel], medida.C[sel], medida.y[sel]
            M0 = M0[sel]
            escolhida = None
            for estrategia in ESTRATEGIAS:
                comeco = time.perf_counter()
                metricas, tempo, detalhe = simular(
                    imputador, blr, G, P, C, y, M0, estrategia, int(semente), t
                )
                for m in metricas:
                    linhas.append({"estrategia": estrategia, "taxa": float(taxas[t]),
                                   "semente": int(semente), **m})
                tempos += [{"taxa": float(taxas[t]), "semente": int(semente), **x} for x in tempo]
                if estrategia == "otima":
                    escolhida = detalhe
                print(f"  taxa {taxas[t]:.2f}, semente {semente}, {estrategia}: "
                      f"{time.perf_counter() - comeco:.1f} s", flush=True)

            reducao.append(pd.DataFrame({
                "taxa": float(taxas[t]), "semente": int(semente),
                "prevista": escolhida["prevista"], "realizada": escolhida["realizada"],
            }))
            contagem = np.bincount(escolhida["k"], minlength=len(gavetas))
            tabelas_de_gavetas.append(pd.DataFrame({
                "taxa": float(taxas[t]), "semente": int(semente), "gaveta": gavetas,
                "frequencia": 100 * contagem / contagem.sum(), "n": int(contagem.sum()),
            }))

    metricas = pd.DataFrame(linhas)
    metricas.to_csv(SAIDA / "pergunta_metricas.csv", index=False)
    pd.DataFrame(tempos).to_csv(SAIDA / "pergunta_tempos.csv", index=False)
    pd.concat(reducao, ignore_index=True).to_csv(SAIDA / "pergunta_reducao.csv", index=False)
    pd.concat(tabelas_de_gavetas, ignore_index=True).to_csv(
        SAIDA / "pergunta_gavetas.csv", index=False
    )

    resumo = metricas.groupby(["estrategia", "taxa", "n_perguntas"])[
        ["mpiw_g", "picp", "nll", "crps_g", "mae_g", "pct_meia_dose"]
    ].agg(["mean", "std"])
    pd.set_option("display.width", 240)
    print("\n", resumo.round(2).to_string())
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
