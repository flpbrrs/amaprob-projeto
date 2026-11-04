"""Contrato dos imputadores probabilísticos (Parte 1) e construção da entrada sem os valores ocultos.

Regra 2: os imputadores nunca recebem as gramas verdadeiras. `ocultar` é o único ponto que as lê
antes de um imputador e apaga (NaN) toda célula de M. Gavetas ausentes (P = 0) valem 0.
"""

from typing import Protocol

import numpy as np


def ocultar(G: np.ndarray, M: np.ndarray) -> np.ndarray:
    """X = log1p(gramas), com NaN nas células ocultas."""
    X = np.log1p(np.asarray(G, dtype=np.float64))
    X[np.asarray(M, dtype=bool)] = np.nan
    return X


class Imputer(Protocol):
    def fit(self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray) -> "Imputer": ...

    def sample(
        self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Retorna (n_samples, N, K) em log1p(gramas). Observados copiados; só X[M] é amostrado."""
        ...
