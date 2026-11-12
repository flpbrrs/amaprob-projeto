"""Métricas para preditivas gaussianas (média e variância por refeição), em gramas.

- MAE (erro absoluto médio): erro da média prevista.
- NLL (log-verossimilhança negativa média): penaliza tanto erro quanto incerteza mal calibrada.
- CRPS (continuous ranked probability score): distância entre a distribuição prevista e o valor
  observado, na unidade do alvo; reduz-se ao erro absoluto quando a previsão é determinística.
- PICP (cobertura) e MPIW (largura média) do intervalo central de 90%.
- Percentual de meia dose: refeições com erro de até 7,5 g, meia unidade de insulina na relação
  1 U : 15 g, o passo de uma caneta comum (ver docs/piso-de-erro.md).
"""

from statistics import NormalDist

import numpy as np
import pandas as pd
import scoringrules as sr
from scipy.special import logsumexp, ndtr

NIVEL = 0.90
Z_INTERVALO = NormalDist().inv_cdf(0.5 + NIVEL / 2)  # ≈ 1,645
TOLERANCIA_MEIA_DOSE_G = 7.5
ORCAMENTO_BLOCO_BYTES = 256 * 2**20
ITERACOES_BISSECCAO = 60
CLASSES_OCULTAS = ("0", "1", "2", "3", "4+")


def avaliar(y: np.ndarray, media: np.ndarray, var: np.ndarray) -> dict:
    y, media, var = (np.asarray(a, dtype=np.float64) for a in (y, media, var))
    desvio = np.sqrt(var)
    erro = np.abs(y - media)
    meia_largura = Z_INTERVALO * desvio
    return {
        "mae_g": erro.mean(),
        "nll": np.mean(0.5 * np.log(2 * np.pi * var) + (y - media) ** 2 / (2 * var)),
        "crps_g": np.mean(sr.crps_normal(y, media, desvio)),
        "picp": np.mean(erro <= meia_largura),
        "mpiw_g": np.mean(2 * meia_largura),
        "pct_meia_dose": 100 * np.mean(erro <= TOLERANCIA_MEIA_DOSE_G),
    }


def _blocos(n: int, tamanho: int):
    for inicio in range(0, n, max(1, tamanho)):
        yield slice(inicio, inicio + max(1, tamanho))


def intervalo(medias: np.ndarray, variancias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo central de 90% da mistura (1/S) Σ_s N(μ_s, σ²_s), com médias e variâncias (S, N).

    A mistura não tem quantil em forma fechada: ele sai por bissecção na CDF, que é crescente.
    """
    medias = np.atleast_2d(np.asarray(medias, dtype=np.float64))
    desvios = np.sqrt(np.atleast_2d(np.asarray(variancias, dtype=np.float64)))
    S, N = medias.shape
    if S == 1:
        return medias[0] - Z_INTERVALO * desvios[0], medias[0] + Z_INTERVALO * desvios[0]

    limites = {q: np.empty(N) for q in ((1 - NIVEL) / 2, (1 + NIVEL) / 2)}
    for b in _blocos(N, ORCAMENTO_BLOCO_BYTES // (4 * 8 * S)):
        mu, sd = medias[:, b], desvios[:, b]
        piso, teto = (mu - 8 * sd).min(axis=0), (mu + 8 * sd).max(axis=0)
        for q, saida in limites.items():
            lo, hi = piso.copy(), teto.copy()
            for _ in range(ITERACOES_BISSECCAO):
                meio = (lo + hi) / 2
                abaixo = ndtr((meio - mu) / sd).mean(axis=0) < q
                lo, hi = np.where(abaixo, meio, lo), np.where(abaixo, hi, meio)
            saida[b] = (lo + hi) / 2
    li, ls = limites.values()
    return li, ls


def avaliar_mistura(y: np.ndarray, medias: np.ndarray, variancias: np.ndarray) -> dict:
    """Métricas de `avaliar` para a mistura de pesos iguais, com médias e variâncias (S, N).

    A NLL soma as log-densidades por logsumexp para não dar log(0) quando y está longe de todos
    os componentes.
    """
    y = np.asarray(y, dtype=np.float64)
    medias = np.atleast_2d(np.asarray(medias, dtype=np.float64))
    variancias = np.atleast_2d(np.asarray(variancias, dtype=np.float64))
    S, N = medias.shape
    desvios = np.sqrt(variancias)
    media = medias.mean(axis=0)
    erro = np.abs(y - media)

    log_densidade = -0.5 * np.log(2 * np.pi * variancias) - (y - medias) ** 2 / (2 * variancias)
    nll = -np.mean(logsumexp(log_densidade, axis=0) - np.log(S))

    crps = np.empty(N)
    for b in _blocos(N, ORCAMENTO_BLOCO_BYTES // (8 * S * S)):
        crps[b] = sr.crps_mixnorm(y[b], medias[:, b], desvios[:, b], m_axis=0)

    li, ls = intervalo(medias, variancias)
    return {
        "mae_g": erro.mean(),
        "nll": nll,
        "crps_g": crps.mean(),
        "picp": np.mean((li <= y) & (y <= ls)),
        "mpiw_g": np.mean(ls - li),
        "pct_meia_dose": 100 * np.mean(erro <= TOLERANCIA_MEIA_DOSE_G),
    }


def por_ocultas(
    y: np.ndarray, media: np.ndarray, li: np.ndarray, ls: np.ndarray, n_ocultas: np.ndarray
) -> pd.DataFrame:
    """Recebe os limites do intervalo, e não a variância, para servir à gaussiana e à mistura."""
    y, media, li, ls = (np.asarray(a, dtype=np.float64) for a in (y, media, li, ls))
    classe = np.asarray(CLASSES_OCULTAS)[np.minimum(np.asarray(n_ocultas, dtype=int), len(CLASSES_OCULTAS) - 1)]
    erro = np.abs(y - media)
    tabela = pd.DataFrame({
        "n_ocultas": pd.Categorical(classe, CLASSES_OCULTAS),
        "dentro": (li <= y) & (y <= ls),
        "largura": ls - li,
        "erro": erro,
        "meia_dose": 100 * (erro <= TOLERANCIA_MEIA_DOSE_G),
    }).groupby("n_ocultas", observed=True).agg(
        n=("dentro", "size"), picp=("dentro", "mean"), mpiw_g=("largura", "mean"),
        mae_g=("erro", "mean"), pct_meia_dose=("meia_dose", "mean"),
    )
    return tabela.reset_index().assign(n_ocultas=lambda t: t.n_ocultas.astype(str))


def cobertura_por_faixa(y: np.ndarray, media: np.ndarray, var: np.ndarray, n_faixas: int = 5) -> pd.DataFrame:
    """PICP por quintil da média prevista: a global pode esconder sobra nas refeições pequenas
    e falta nas grandes."""
    dentro = np.abs(np.asarray(y) - media) <= Z_INTERVALO * np.sqrt(var)
    faixa = pd.qcut(media, n_faixas, labels=False)
    tabela = pd.DataFrame({"faixa": faixa, "media": media, "dentro": dentro}).groupby("faixa").agg(
        media_min_g=("media", "min"), media_max_g=("media", "max"), n=("dentro", "size"), picp=("dentro", "mean")
    )
    return tabela.reset_index()


def celulas_de_imputacao(G: np.ndarray, amostras_log: np.ndarray, M: np.ndarray) -> pd.DataFrame:
    """Métricas por célula oculta, a partir das amostras (S, N, K) em log1p(gramas).

    Única leitura das gramas verdadeiras de M fora do mascaramento (Regra 2: só na avaliação).
    Tudo sai das amostras, para servir a qualquer imputador pelo contrato: a NLL é a da gaussiana
    com a média e a variância das amostras da célula, e o intervalo vai dos quantis empíricos.
    """
    M = np.asarray(M, dtype=bool)
    linhas, gavetas = np.nonzero(M)
    amostras = np.asarray(amostras_log, dtype=np.float64)[:, linhas, gavetas]
    verdade_g = np.asarray(G, dtype=np.float64)[linhas, gavetas]
    verdade_log = np.log1p(verdade_g)

    media_log, var_log = amostras.mean(axis=0), amostras.var(axis=0)
    li, ls = np.quantile(amostras, [(1 - NIVEL) / 2, (1 + NIVEL) / 2], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):  # amostras constantes: variância 0
        nll = 0.5 * np.log(2 * np.pi * var_log) + (verdade_log - media_log) ** 2 / (2 * var_log)
    return pd.DataFrame({
        "gaveta": gavetas,
        "erro_g": np.abs(np.expm1(amostras).mean(axis=0) - verdade_g),
        "erro_log": np.abs(media_log - verdade_log),
        "nll_log": nll,
        "crps_log": sr.crps_ensemble(verdade_log, amostras, m_axis=0) if len(linhas) else np.empty(0),
        "dentro": (li <= verdade_log) & (verdade_log <= ls),
        "largura_g": np.expm1(ls) - np.expm1(li),
    })


_RESUMO_IMPUTACAO = {
    "n_celulas": ("erro_g", "size"), "mae_g": ("erro_g", "mean"), "mae_log": ("erro_log", "mean"),
    "nll_log": ("nll_log", "mean"), "crps_log": ("crps_log", "mean"), "picp": ("dentro", "mean"),
    "mpiw_g": ("largura_g", "mean"),
}


def avaliar_imputacao(celulas: pd.DataFrame) -> dict:
    return celulas.assign(todas=0).groupby("todas").agg(**_RESUMO_IMPUTACAO).iloc[0].to_dict()


def imputacao_por_gaveta(celulas: pd.DataFrame) -> pd.DataFrame:
    return celulas.groupby("gaveta").agg(**_RESUMO_IMPUTACAO).reset_index()
