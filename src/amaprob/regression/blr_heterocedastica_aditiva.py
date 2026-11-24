"""Regressão linear bayesiana heterocedástica com variância ADITIVA do ruído.

Variante de `blr_heterocedastica.py`, que fica como está. Lá, log σ²(h) é linear em h, então os
efeitos das gavetas se multiplicam e σ² explode nas refeições com muitas gavetas (no tamanho 5+, a
variância prevista é 429 g² contra 225 g² de resíduo). Aqui, os erros de itens independentes somam
variância:

    σ²(h) = v₀ + Σ_k P_k v_k + Σ_o O_o u_o,      v₀, v_k, u_o ≥ 0

com P as presenças das K gavetas e O as 5 indicadoras de ocasião, todas sempre observadas. O número
de gavetas não entra como indicadora: a soma das presenças já faz σ² crescer com o tamanho.

Positividade por restrição de caixa: os coeficientes c = (v₀, v, u) são otimizados direto, por
L-BFGS-B com c ≥ 0 (v₀ ≥ 1e-6, para σ² > 0). Uma reparametrização por softplus ou exp
(c = softplus(θ)) foi testada e descartada: quando um coeficiente encosta em 0 no começo da busca,
∂c/∂θ = logística(θ) → 0 e ele não volta, mesmo que o ótimo seja positivo (em dados sintéticos, um
efeito real de 5 g² ficou preso em 0 e foi absorvido por v₀). Com a caixa, o gradiente na fronteira
não some.

(a) Pesos com σ² fixos: idêntico a `blr_heterocedastica.py` (regressão linear bayesiana, BLR, ponderada por 1/σ², evidência
para α).

(b) Ruído com m e S fixos. Com A_n = [1 | P_n | O_n], σ²_n = A_n c e v_n = σ²_n + q_n, em que
q_n = φ_nᵀ S φ_n:

    L(c) = ½ Σ_n [log v_n + r_n² / v_n]
    ∂L/∂σ²_n = ½ (1/v_n − r_n²/v_n²)
    ∂L/∂c = Aᵀ ∂L/∂σ²

Sem ridge: com a positividade e N ≈ 213 mil, os coeficientes ficam bem determinados (no modelo em
log, o λ não mudou a NLL na 4ª casa). A coluna de 1 e as indicadoras de ocasião são colineares
(elas somam 1); a caixa mantém tudo não negativo, e só a soma v₀ + u_o é identificável, que é o que
entra na preditiva.

Início: v₀ = 1/β da BLR homocedástica e os demais coeficientes em 1 g².
"""

import numpy as np
from scipy.optimize import minimize

from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.blr_heterocedastica import N_OCASIOES, RegressaoLinearBayesianaHeterocedastica


PISO_V0 = 1e-6


def variaveis_aditivas(Z: np.ndarray, n_gavetas: int) -> np.ndarray:
    """A = [1 | presença (K) | ocasião (5)], lido das colunas de P e C em Z."""
    Z = np.asarray(Z, dtype=np.float64)
    presenca = (Z[:, n_gavetas:2 * n_gavetas] > 0).astype(np.float64)
    ocasiao = Z[:, 2 * n_gavetas:2 * n_gavetas + N_OCASIOES]
    return np.c_[np.ones(len(Z)), presenca, ocasiao]


class RegressaoLinearBayesianaHeterocedasticaAditiva(RegressaoLinearBayesianaHeterocedastica):
    def __init__(self, n_gavetas: int, n_iteracoes: int = 5, tol: float = 1e-6, max_iter_alpha: int = 200):
        super().__init__(n_gavetas=n_gavetas, lam=0.0, n_iteracoes=n_iteracoes, tol=tol,
                         max_iter_alpha=max_iter_alpha)

    def _ajustar_ruido_aditivo(self, A: np.ndarray, r2: np.ndarray, q: np.ndarray, inicio: np.ndarray) -> np.ndarray:
        def objetivo(c):
            v = A @ c + q
            valor = 0.5 * np.sum(np.log(v) + r2 / v)
            d_sigma2 = 0.5 * (1.0 / v - r2 / v ** 2)
            return valor, A.T @ d_sigma2

        limites = [(PISO_V0, None)] + [(0.0, None)] * (A.shape[1] - 1)
        resultado = minimize(objetivo, inicio, jac=True, method="L-BFGS-B", bounds=limites,
                             options={"maxiter": 2000})
        self.convergiu_ruido_ = bool(resultado.success)
        return resultado.x

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "RegressaoLinearBayesianaHeterocedasticaAditiva":
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        desvio = Z.std(axis=0)
        self.desvio_z_ = np.where(desvio > 0, desvio, 1.0)
        A = variaveis_aditivas(Z, self.n_gavetas)

        homocedastica = RegressaoLinearBayesiana(tol=self.tol).fit(Z, y)
        c = np.r_[1.0 / homocedastica.beta_, np.ones(A.shape[1] - 1)]
        alpha = homocedastica.alpha_
        self.historico_ = []
        for iteracao in range(1, self.n_iteracoes + 1):
            alpha = self._ajustar_pesos(Z, y, A @ c, alpha)
            Phi = (Z - self.media_z_) / self.desvio_z_
            residuo = y - self.media_y_ - Phi @ self.m_
            q = np.einsum("ij,jk,ik->i", Phi, self.S_, Phi)
            c = self._ajustar_ruido_aditivo(A, residuo ** 2, q, c)
            sigma2 = A @ c
            v = sigma2 + q
            self.historico_.append({
                "iteracao": iteracao, "alpha": alpha,
                "nll_treino": float(np.mean(0.5 * np.log(2 * np.pi * v) + residuo ** 2 / (2 * v))),
                "sigma2_min": float(sigma2.min()), "sigma2_max": float(sigma2.max()),
            })
        self.coeficientes_ = c
        self._ajustar_pesos(Z, y, A @ self.coeficientes_, alpha)
        return self

    def variancia_ruido(self, Z: np.ndarray) -> np.ndarray:
        return variaveis_aditivas(Z, self.n_gavetas) @ self.coeficientes_
