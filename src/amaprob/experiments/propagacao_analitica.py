"""Braço de propagação analítica: plug-in, forma fechada e propagação por amostragem, lado a lado.

Os três braços preveem o carboidrato da refeição a partir do registro incompleto, com a mesma
regressão linear bayesiana ajustada no treino completo e o mesmo imputador:

- plug-in: as ocultas entram pela média das S amostras em gramas e a previsão é uma só;
- analítica: as ocultas entram como variável aleatória, com a média e a covariância da condicional
  do imputador convertidas para gramas, e a preditiva sai em forma fechada
  (`regression/propagacao_analitica.py`), sem gerador aleatório;
- propagação: prevê para cada amostra e combina as S preditivas na mistura (1/S) Σ N(μ_s, σ²_s).

A propagação é medida com S = 100, o valor do experimento oficial, e também com S = 10 e 30, para
mostrar quanto da diferença em relação à forma fechada é erro de Monte Carlo. Cada S tem fluxo
próprio de gerador; o fluxo de S = 100 é o mesmo do experimento oficial, o que permite conferir as
linhas de plug-in e propagação contra `results/propagacao_metricas.csv`.

Mede na validação, com as máscaras persistidas e as 3 sementes; a taxa 0 fica fora porque não há
oculta e a analítica coincide com o plug-in. O teste não é lido (Regra 6). Nada é ajustado no
conjunto medido.

As tabelas são novas e não mexem nos CSVs do experimento oficial. Rodar de novo com
`--imputadores` menor junta as linhas aos arquivos existentes, como no experimento oficial.

Rodadas em paralelo usam `--sufixo`, que põe o nome no fim de cada arquivo para os processos não
disputarem o mesmo CSV; a junção fica em `experiments/juntar_propagacao_analitica.py`, que confere
colunas e chaves antes de gravar. `--taxas` e `--sementes` restringem a rodada, para reconferir uma
combinação isolada.

Uso: uv run python -m amaprob.experiments.propagacao_analitica [--imputadores fa_condicional porcao_media]
             [--taxas 0.5] [--sementes 0] [--sufixo nome]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from amaprob.data.dataset import RAIZ
from amaprob.data.splits import VALIDACAO
from amaprob.evaluation.metricas import avaliar_mistura, intervalo, por_ocultas
from amaprob.experiments.propagacao import FLUXO_VALIDACAO, IMPUTADORES, S, gerador, treino_e_medida
from amaprob.imputation.base import ocultar
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagation import PLUGIN, PROPAGACAO, prever_bracos
from amaprob.regression.propagacao_analitica import momentos_em_gramas, prever_com_entrada_incerta

SAIDA = RAIZ / "results"
ANALITICA = "analitica"
AMOSTRAS_DE_COMPARACAO = (10, 30)
BLOCO_DA_ANALITICA = 4_096
COLUNAS_DO_RESUMO = ("mae_g", "nll", "crps_g", "picp", "mpiw_g", "pct_meia_dose")


def nome_da_propagacao(n_amostras: int) -> str:
    return f"propagacao_S{n_amostras}"


def prever_analitico(
    imputador, blr, G: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Média e variância pela forma fechada, em blocos de refeições (N × K × K)."""
    media = np.empty(len(G))
    variancia = np.empty(len(G))
    for inicio in range(0, len(G), BLOCO_DA_ANALITICA):
        b = slice(inicio, inicio + BLOCO_DA_ANALITICA)
        X = ocultar(G[b], M[b])
        media_log, cov_log = momentos_ocultos(imputador, X, P[b], M[b], C[b])
        Eg, Cg = momentos_em_gramas(G[b], P[b], M[b], media_log, cov_log)
        media[b], variancia[b] = prever_com_entrada_incerta(blr, Eg, Cg, P[b], C[b])
    return media, variancia


def _medir(braco: str, n_amostras: int, medias: np.ndarray, variancias: np.ndarray, y: np.ndarray,
           n_ocultas: np.ndarray) -> tuple[dict, pd.DataFrame]:
    metricas = {"braco": braco, "S": n_amostras, **avaliar_mistura(y, medias, variancias)}
    li, ls = intervalo(medias, variancias)
    return metricas, por_ocultas(y, medias.mean(axis=0), li, ls, n_ocultas)


def _juntar(novas: pd.DataFrame, arquivo) -> pd.DataFrame:
    """Mantém as linhas dos imputadores não executados nesta chamada."""
    if not arquivo.exists():
        return novas
    existentes = pd.read_csv(arquivo, float_precision="round_trip")
    return pd.concat(
        [existentes[~existentes.imputador.isin(novas.imputador.unique())], novas], ignore_index=True
    )


def resumir(metricas: pd.DataFrame) -> pd.DataFrame:
    valores = metricas.drop(columns=["semente", "S"])
    resumo = valores.groupby(["imputador", "taxa", "braco"]).agg(["mean", "std"])
    resumo.columns = [f"{metrica}_{'media' if estatistica == 'mean' else 'desvio'}"
                      for metrica, estatistica in resumo.columns]
    return resumo.reset_index()


def arquivo(nome: str, sufixo: str) -> Path:
    """CSV da tabela, com o sufixo da rodada paralela quando houver."""
    return SAIDA / f"{nome}{f'_{sufixo}' if sufixo else ''}.csv"


def gravar(metricas: list[dict], ocultas: list[pd.DataFrame], tempos: list[dict], sufixo: str = "") -> None:
    """Grava as três tabelas, juntando aos arquivos existentes o que não foi executado agora."""
    tabelas = {
        "propagacao_analitica_metricas": pd.DataFrame(metricas),
        "propagacao_analitica_por_ocultas": pd.concat(ocultas, ignore_index=True),
        "propagacao_analitica_tempos": pd.DataFrame(tempos),
    }
    for nome, tabela in tabelas.items():
        _juntar(tabela, arquivo(nome, sufixo)).to_csv(arquivo(nome, sufixo), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--imputadores", nargs="+", choices=list(IMPUTADORES), default=list(IMPUTADORES))
    parser.add_argument("--taxas", nargs="+", type=float, default=None, help="subconjunto das taxas medidas")
    parser.add_argument("--sementes", nargs="+", type=int, default=None, help="subconjunto das sementes de máscara")
    parser.add_argument("--sufixo", default="", help="vai no nome dos CSVs, para rodadas em paralelo")
    escolha = parser.parse_args()

    inicio = time.perf_counter()
    tr, medida, taxas, sementes, _ = treino_e_medida(VALIDACAO)
    indices = np.flatnonzero(taxas > 0)  # taxa 0: analítica igual ao plug-in, sem ocultas
    print(f"treino: {len(tr.y):,} refeições | validação: {len(medida.y):,} refeições | "
          f"S = {S} (e {', '.join(str(s) for s in AMOSTRAS_DE_COMPARACAO)}) | "
          f"imputadores: {', '.join(escolha.imputadores)}"
          + (f" | taxas: {escolha.taxas}" if escolha.taxas else "")
          + (f" | sementes: {escolha.sementes}" if escolha.sementes else "")
          + (f" | sufixo: {escolha.sufixo}" if escolha.sufixo else ""), flush=True)

    regressor = RegressaoLinearBayesiana().fit(montar_Z(tr.G, tr.P, tr.C), tr.y)
    sem_mascara = np.zeros_like(tr.P)

    metricas, ocultas, tempos = [], [], []
    for nome in escolha.imputadores:
        comeco = time.perf_counter()
        imputador = IMPUTADORES[nome]().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
        segundos = time.perf_counter() - comeco
        tempos.append({"imputador": nome, "etapa": "ajuste", "taxa": np.nan, "semente": np.nan,
                       "segundos": segundos})
        print(f"  {nome}: ajuste {segundos:.1f} s", flush=True)

        for s, semente in enumerate(sementes):
            if escolha.sementes is not None and int(semente) not in escolha.sementes:
                continue
            for t in indices:
                if escolha.taxas is not None and not any(np.isclose(taxas[t], x) for x in escolha.taxas):
                    continue
                taxa, y = float(taxas[t]), medida.y
                M = medida.M[s, t]
                n_ocultas = M.sum(axis=1)
                chave = {"imputador": nome, "taxa": taxa, "semente": int(semente)}

                comeco = time.perf_counter()
                bracos = prever_bracos(imputador, regressor, medida.G, medida.P, M, medida.C, S,
                                       gerador(semente, t, FLUXO_VALIDACAO))
                segundos = time.perf_counter() - comeco  # inclui o plug-in, que sai das mesmas amostras
                tempos.append({**chave, "etapa": nome_da_propagacao(S), "segundos": segundos})
                for braco, n_amostras, rotulo in ((PLUGIN, 1, "plugin"), (PROPAGACAO, S, nome_da_propagacao(S))):
                    medias, variancias = bracos[braco]
                    metrica, tabela = _medir(rotulo, n_amostras, medias, variancias, y, n_ocultas)
                    metricas.append({**chave, **metrica})
                    ocultas.append(tabela.assign(**chave, braco=rotulo, S=n_amostras))

                comeco = time.perf_counter()
                media, variancia = prever_analitico(imputador, regressor, medida.G, medida.P, M, medida.C)
                segundos = time.perf_counter() - comeco
                tempos.append({**chave, "etapa": ANALITICA, "segundos": segundos})
                metrica, tabela = _medir(ANALITICA, 1, media[None], variancia[None], y, n_ocultas)
                metricas.append({**chave, **metrica})
                ocultas.append(tabela.assign(**chave, braco=ANALITICA, S=1))

                for n_amostras in AMOSTRAS_DE_COMPARACAO:
                    comeco = time.perf_counter()
                    comparacao = prever_bracos(imputador, regressor, medida.G, medida.P, M, medida.C, n_amostras,
                                               gerador(semente, t, FLUXO_VALIDACAO + n_amostras))
                    segundos = time.perf_counter() - comeco
                    rotulo = nome_da_propagacao(n_amostras)
                    tempos.append({**chave, "etapa": rotulo, "segundos": segundos})
                    medias, variancias = comparacao[PROPAGACAO]
                    metrica, tabela = _medir(rotulo, n_amostras, medias, variancias, y, n_ocultas)
                    metricas.append({**chave, **metrica})
                    ocultas.append(tabela.assign(**chave, braco=rotulo, S=n_amostras))

                print(f"  {nome}, semente {semente}, taxa {taxa:.2f}: ok ({time.perf_counter() - comeco:.1f} s "
                      f"no último braço)", flush=True)

        gravar(metricas, ocultas, tempos, escolha.sufixo)

    resumo = resumir(pd.DataFrame(metricas))
    pd.set_option("display.width", 260)
    colunas = ["imputador", "taxa", "braco"] + [f"{m}_media" for m in COLUNAS_DO_RESUMO[:5]]
    print("\n", resumo[colunas].round(3).to_string(index=False))
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
