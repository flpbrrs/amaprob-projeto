"""Contrato dos regressores probabilísticos (Parte 2) e montagem da entrada Z.

Todo regressor devolve, por refeição, a média e a variância de uma preditiva gaussiana. Na
propagação da incerteza da imputação, cada amostra s gera (μ_s, σ²_s) e a preditiva final é a
mistura (1/S) Σ_s N(μ_s, σ²_s).
"""

from typing import Protocol

import numpy as np


def montar_Z(G: np.ndarray, P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Z = [gramas por gaveta | presença | contexto]. Gramas na escala original, não em log1p."""
    return np.c_[G, P, C].astype(np.float64)


class ProbRegressor(Protocol):
    def fit(self, Z: np.ndarray, y: np.ndarray) -> "ProbRegressor": ...

    def predict(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Média e variância da preditiva gaussiana, cada uma de forma (N,)."""
        ...
