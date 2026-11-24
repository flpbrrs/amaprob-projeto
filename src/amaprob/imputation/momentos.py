"""Momentos da condicional das quantidades ocultas, em log1p(gramas).

Complementa `Imputer.sample`: em vez de sortear as células ocultas, devolve a média e a covariância
da condicional gaussiana x_H | x_O de cada refeição. É a entrada do braço de propagação analítica
da regressão (`regression/propagacao_analitica.py`).

A conta é a de `sample`, sem o sorteio. Para o PPCA e a FA, com z | x_O ~ N(m, Σ_z) e W_k a linha k
dos pesos,

    E[x_k | x_O] = μ_k + W_k m
    Cov(x_a, x_b | x_O) = W_a Σ_z W_bᵀ + [ruído]_ab

em que o ruído é σ² I no PPCA, diag(ψ) na FA e diag(desvio²) na porção média. A porção média não
tem latente: a condicional é a própria marginal, diagonal.

Dependência de métodos internos. O contrato `Imputer` só expõe amostras, e a condicional exata
precisa do posterior do latente. Por isso este módulo lê `_passo_e`, `_media`, `W_`, `sigma2_`,
`psi_`, `media_`, `desvio_`, `coeficientes_media_` e `padronizacao_` dos imputadores. Mudança nesses
internos aparece nos testes de `tests/test_propagacao_analitica.py`.

Regra 2: X só é lido nas células observadas (P & ~M), com o mesmo `X_O = np.where(O, X, 0)` de
`sample`. Célula oculta com lixo no lugar do NaN produz o mesmo resultado.

Células fora de M saem com média NaN e covariância 0. O truncamento em 0 das amostras não entra
aqui: a condicional é gaussiana no log, e o truncamento é uma diferença declarada do braço
analítico.
"""

import numpy as np

from amaprob.imputation import media_condicional as mc
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.imputation.fa import FA
from amaprob.imputation.ppca import PPCA, observados

# Refeições por bloco: a covariância é N × K × K e não cabe inteira para a validação (K = 29).
BLOCO = 2_048


def momentos_ocultos(
    imputador, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Média (N, K) e covariância (N, K, K) de x | x_O nas células de M, em log1p(gramas)."""
    M = np.asarray(M, dtype=bool)
    P = np.asarray(P, dtype=bool)
    O = observados(P, M)
    X_O = np.where(O, np.asarray(X, dtype=np.float64), 0.0)

    N, K = X_O.shape
    media_log = np.full((N, K), np.nan)
    cov_log = np.zeros((N, K, K))
    for inicio in range(0, N, BLOCO):
        b = slice(inicio, inicio + BLOCO)
        _bloco(imputador, X_O[b], O[b], M[b], P[b], C[b], media_log[b], cov_log[b])
    return media_log, cov_log


def _bloco(
    imputador, X_O: np.ndarray, O: np.ndarray, M: np.ndarray, P: np.ndarray, C: np.ndarray,
    media_log: np.ndarray, cov_log: np.ndarray,
) -> None:
    if isinstance(imputador, PorcaoMedia):
        U = mc.montar_u(P, C, imputador.padronizacao_) if imputador.media_condicional else None
        media = imputador.media_[None, :] if U is None else U @ imputador.coeficientes_media_.T
        media_log[:] = np.where(M, np.broadcast_to(media, X_O.shape), np.nan)
        linhas, colunas = np.nonzero(M)
        cov_log[linhas, colunas, colunas] = imputador.desvio_[colunas] ** 2
        return

    U = mc.montar_u(P, C, imputador.padronizacao_) if imputador.media_condicional else None
    posterior = imputador._passo_e(X_O, O, U)
    if isinstance(imputador, PPCA):
        covariancia_z, ruido = imputador.sigma2_ * posterior.A_inv, np.full(X_O.shape[1], imputador.sigma2_)
    elif isinstance(imputador, FA):
        covariancia_z, ruido = posterior.A_inv, imputador.psi_
    else:
        raise TypeError(f"imputador sem momentos condicionais: {type(imputador).__name__}")

    media = np.broadcast_to(imputador._media(U), X_O.shape) + posterior.media_z @ imputador.W_.T
    media_log[:] = np.where(M, media, np.nan)

    pesos = imputador.W_ * M[:, :, None]  # gaveta fora de M com peso 0: linha e coluna nulas
    covariancia = np.einsum("nki,nij,nlj->nkl", pesos, covariancia_z, pesos)
    diagonal = np.arange(X_O.shape[1])
    covariancia[:, diagonal, diagonal] += np.where(M, ruido, 0.0)
    covariancia *= M[:, :, None] & M[:, None, :]
    cov_log[:] = covariancia
