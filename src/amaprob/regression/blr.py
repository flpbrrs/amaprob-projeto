"""Regressão linear bayesiana conjugada, com hiperparâmetros pela evidência.

Referência: Bishop, *Pattern Recognition and Machine Learning* (2006), §3.3 e §3.5.

Modelo, sobre as colunas padronizadas φ(z) e o alvo centrado t = y − ȳ:

    w ~ N(0, α⁻¹ I)                  prior isotrópico nos pesos (precisão α)
    t | w ~ N(φᵀw, β⁻¹)              ruído gaussiano homocedástico (precisão β)

Posterior dos pesos (conjugado, forma fechada):

    S⁻¹ = α I + β ΦᵀΦ,      m = β S Φᵀt

Preditiva para uma nova refeição:

    y* ~ N(φ*ᵀm + ȳ,  1/β + φ*ᵀ S φ*)

O primeiro termo da variância é o ruído; o segundo, a incerteza sobre os pesos, que cresce longe
dos dados de ajuste e encolhe com N.

Evidência (verossimilhança marginal, com w integrado). Seus pontos estacionários em α e β dão
as reestimações de ponto fixo (Bishop, eq. 3.92 e 3.95), com λᵢ os autovalores de βΦᵀΦ:

    γ = Σᵢ λᵢ / (α + λᵢ)              número efetivo de parâmetros determinados pelos dados
    α ← γ / mᵀm
    1/β ← ‖t − Φm‖² / (N − γ)

Padronizar as colunas faz o prior isotrópico tratá-las na mesma escala. Centrar o alvo equivale
a um intercepto sem penalização. Colunas constantes no ajuste ficam com φ = 0.
"""

import numpy as np


class RegressaoLinearBayesiana:
    def __init__(self, tol: float = 1e-6, max_iter: int = 200):
        self.tol = tol
        self.max_iter = max_iter

    def _phi(self, Z: np.ndarray) -> np.ndarray:
        return (np.asarray(Z, dtype=np.float64) - self.media_z_) / self.desvio_z_

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "RegressaoLinearBayesiana":
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.media_z_ = Z.mean(axis=0)
        desvio = Z.std(axis=0)
        self.desvio_z_ = np.where(desvio > 0, desvio, 1.0)  # coluna constante: φ = 0 após centrar
        self.media_y_ = y.mean()
        Phi, t = self._phi(Z), y - self.media_y_
        N, d = Phi.shape

        PtP, Ptt = Phi.T @ Phi, Phi.T @ t
        autovalores = np.linalg.eigvalsh(PtP)  # de ΦᵀΦ; os de βΦᵀΦ são β vezes estes
        alpha, beta = 1.0, 1.0 / t.var()
        for self.n_iter_ in range(1, self.max_iter + 1):
            m = beta * np.linalg.solve(alpha * np.eye(d) + beta * PtP, Ptt)
            lam = beta * autovalores
            gamma = np.sum(lam / (alpha + lam))
            alpha_novo = gamma / (m @ m)
            beta_novo = (N - gamma) / np.sum((t - Phi @ m) ** 2)
            mudou = max(abs(alpha_novo - alpha) / alpha, abs(beta_novo - beta) / beta)
            alpha, beta = alpha_novo, beta_novo
            if mudou < self.tol:
                break

        self.alpha_, self.beta_, self.gamma_ = alpha, beta, gamma
        self.S_ = np.linalg.inv(alpha * np.eye(d) + beta * PtP)
        self.m_ = beta * self.S_ @ Ptt
        return self

    def predict(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Phi = self._phi(Z)
        media = Phi @ self.m_ + self.media_y_
        variancia = 1.0 / self.beta_ + np.einsum("ij,jk,ik->i", Phi, self.S_, Phi)
        return media, variancia
