"""Imputador de referência: porção média por gaveta.

Modelo: para cada gaveta k, X_k = log1p(gramas_k) ~ N(μ_k, σ_k²), independente das outras gavetas
e do contexto. μ_k e σ_k são a média e o desvio de X_k nas células presentes e observadas
(P = 1, M = 0) do conjunto de ajuste. As amostras são truncadas em 0 (gramas não negativas).

Não usa coocorrência nem contexto: é o ponto de comparação para PPCA e FA, que exploram a
correlação entre gavetas.

Média condicional (opção `media_condicional`): controle para a média condicional do PPCA e da FA.
μ_k vira b_kᵀ u_n, com u_n = [1; φ(P_n); φ(C_n)], por regressão ridge de X_k em u nas células
observadas, com a mesma priori N(0, 1/α) nos coeficientes de P e C (a escala da penalização é a
variância observada da gaveta); σ_k² é o resíduo médio dessa regressão. É o PPCA/FA com média
condicional sem os fatores: separa o ganho de usar P e C do ganho da correlação entre gavetas.
"""

import numpy as np

from amaprob.imputation import media_condicional as mc


class PorcaoMedia:
    def __init__(self, media_condicional: bool = False, alfa: float = mc.ALFA):
        self.media_condicional = media_condicional
        self.alfa = alfa

    def fit(self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray) -> "PorcaoMedia":
        observado = np.asarray(P, dtype=bool) & ~np.asarray(M, dtype=bool)
        valores = np.where(observado, X, np.nan)
        self.media_ = np.nanmean(valores, axis=0)
        self.desvio_ = np.nanstd(valores, axis=0)
        if self.media_condicional:
            X_O = np.where(observado, X, 0.0)
            tem_observado = observado.any(axis=0)
            self.padronizacao_ = mc.ajustar_padronizacao(P, C)
            U = mc.montar_u(P, C, self.padronizacao_)
            blocos = mc.blocos_fixos(U, X_O, observado)
            self.coeficientes_media_ = mc.coeficientes_iniciais(blocos, self.desvio_**2, tem_observado, self.alfa)
            residuo = ((X_O - U @ self.coeficientes_media_.T) ** 2 * observado).sum(axis=0)
            self.desvio_ = np.where(tem_observado, np.sqrt(residuo / np.maximum(observado.sum(axis=0), 1)), self.desvio_)
        return self

    def sample(
        self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        M = np.asarray(M, dtype=bool)
        base = np.where(np.asarray(P, dtype=bool) & ~M, X, 0.0)  # observados copiados, ausentes em 0
        amostras = np.repeat(base[None], n_samples, axis=0)
        linhas, colunas = np.nonzero(M)
        if self.media_condicional:
            media = (mc.montar_u(P, C, self.padronizacao_) @ self.coeficientes_media_.T)[linhas, colunas]
        else:
            media = self.media_[colunas]
        sorteio = rng.normal(media, self.desvio_[colunas], size=(n_samples, len(colunas)))
        amostras[:, linhas, colunas] = np.maximum(sorteio, 0.0)
        return amostras
