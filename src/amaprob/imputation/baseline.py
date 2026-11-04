"""Imputador de referência: porção média por gaveta.

Modelo: para cada gaveta k, X_k = log1p(gramas_k) ~ N(μ_k, σ_k²), independente das outras gavetas
e do contexto. μ_k e σ_k são a média e o desvio de X_k nas células presentes e observadas
(P = 1, M = 0) do conjunto de ajuste. As amostras são truncadas em 0 (gramas não negativas).

Não usa coocorrência nem contexto: é o ponto de comparação para PPCA e FA, que exploram a
correlação entre gavetas.
"""

import numpy as np


class PorcaoMedia:
    def fit(self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray) -> "PorcaoMedia":
        observado = np.asarray(P, dtype=bool) & ~np.asarray(M, dtype=bool)
        valores = np.where(observado, X, np.nan)
        self.media_ = np.nanmean(valores, axis=0)
        self.desvio_ = np.nanstd(valores, axis=0)
        return self

    def sample(
        self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        M = np.asarray(M, dtype=bool)
        base = np.where(np.asarray(P, dtype=bool) & ~M, X, 0.0)  # observados copiados, ausentes em 0
        amostras = np.repeat(base[None], n_samples, axis=0)
        linhas, colunas = np.nonzero(M)
        sorteio = rng.normal(self.media_[colunas], self.desvio_[colunas], size=(n_samples, len(colunas)))
        amostras[:, linhas, colunas] = np.maximum(sorteio, 0.0)
        return amostras
