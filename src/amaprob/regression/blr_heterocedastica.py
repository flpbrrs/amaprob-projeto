"""Regressão linear bayesiana com ruído heterocedástico, ajustada por mínimos quadrados generalizados
iterados.

Motivação: com ruído homocedástico (`blr.py`), a variância preditiva é quase constante (≈ 190 g²),
mas a variância do resíduo vai de ~117 a ~318 g² entre segmentos, e a cobertura do intervalo de 90%
cai de ~98% nas refeições de previsão pequena para 82–88% nas de previsão grande.

Modelo, sobre as colunas padronizadas φ(z) e o alvo centrado t = y − ȳ:

    w ~ N(0, α⁻¹ I)
    t_n | w ~ N(φ_nᵀ w, σ²(h_n)),      log σ²(h) = γ₀ + γᵀ h

h usa só variáveis sempre observadas: presença das K gavetas, ocasião (5 indicadoras) e número de
gavetas presentes (indicadoras de 1, 2, 3, 4 e 5+). Assim σ² não depende das quantidades, que podem
estar imputadas, e os três braços da propagação continuam exatos: na analítica, o 1/β da lei da
variância total vira σ²(h), uma constante para cada refeição.

(a) Pesos, com σ²_n fixos. Com D = diag(1/σ²_n), o posterior conjugado é

    S⁻¹ = α I + Φᵀ D Φ,      m = S Φᵀ D t

(é o de `blr.py` com β Φᵀ Φ trocado por Φᵀ D Φ). O intercepto entra centrando y e as colunas pela
média ponderada por D: assim 1ᵀ D Φ = 0 e o intercepto se desacopla dos pesos, como no caso
homocedástico. A escala das colunas continua o desvio não ponderado, só para o prior isotrópico.
A evidência em α, com σ² fixos, tem o mesmo ponto fixo de Bishop (eq. 3.92), com λᵢ os autovalores
de Φᵀ D Φ:

    γ_ef = Σᵢ λᵢ / (α + λᵢ),      α ← γ_ef / mᵀ m

Com σ²_n = 1/β constante, (a) é exatamente a atualização de α de `blr.py` no β final dela; por
isso, partindo do σ² constante da BLR (regressão linear bayesiana), a primeira iteração reproduz m, S e α.

(b) Ruído, com m e S fixos. O resíduo no treino r_n = t_n − φ_nᵀ m tem variância aproximada
v_n = σ²(h_n) + q_n, com q_n = φ_nᵀ S φ_n a incerteza dos pesos. γ sai da máxima verossimilhança
gaussiana do resíduo, com ridge em γ (não em γ₀):

    L(γ₀, γ) = ½ Σ_n [log v_n + r_n² / v_n] + λ ‖γ‖²

Com η_n = γ₀ + γᵀ h_n e s_n = exp(η_n), ∂L/∂η_n = ½ s_n (1/v_n − r_n²/v_n²), e o gradiente em
(γ₀, γ) é [1 | H]ᵀ ∂L/∂η + 2λ [0, γ]. Minimiza-se com L-BFGS. A soma é sobre as N refeições do
treino, então λ está na escala da verossimilhança total: λ ∈ {0,1; 1; 10} regulariza pouco com
N ≈ 213 mil e serve para estabilizar indicadoras raras (ceia, gavetas pouco frequentes).

As duas etapas se alternam por um número fixo de iterações (5 por padrão), partindo de σ² = 1/β da
BLR homocedástica.

Preditiva: y* ~ N(φ*ᵀ m + ȳ_D,  σ²(h*) + φ*ᵀ S φ*).
"""

from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagacao_analitica import prever_com_entrada_incerta

N_OCASIOES = 5
TAMANHOS = 5  # indicadoras de 1, 2, 3, 4 e 5+ gavetas presentes


def variaveis_do_ruido(Z: np.ndarray, n_gavetas: int) -> np.ndarray:
    """h = [presença (K) | ocasião (5) | tamanho (5)], lido das colunas de P e C em Z."""
    Z = np.asarray(Z, dtype=np.float64)
    presenca = Z[:, n_gavetas:2 * n_gavetas] > 0
    ocasiao = Z[:, 2 * n_gavetas:2 * n_gavetas + N_OCASIOES]
    tamanho = np.clip(presenca.sum(axis=1), 1, TAMANHOS)
    indicadoras_de_tamanho = tamanho[:, None] == np.arange(1, TAMANHOS + 1)[None, :]
    return np.c_[presenca, ocasiao, indicadoras_de_tamanho].astype(np.float64)


class RegressaoLinearBayesianaHeterocedastica:
    def __init__(self, n_gavetas: int, lam: float = 1.0, n_iteracoes: int = 5,
                 tol: float = 1e-6, max_iter_alpha: int = 200):
        self.n_gavetas = n_gavetas
        self.lam = lam
        self.n_iteracoes = n_iteracoes
        self.tol = tol
        self.max_iter_alpha = max_iter_alpha

    # (a) pesos com σ² fixos
    def _ajustar_pesos(self, Z: np.ndarray, y: np.ndarray, variancia_ruido: np.ndarray, alpha: float) -> float:
        peso = 1.0 / variancia_ruido
        self.media_z_ = (peso[:, None] * Z).sum(axis=0) / peso.sum()
        self.media_y_ = float((peso * y).sum() / peso.sum())
        Phi = (Z - self.media_z_) / self.desvio_z_
        t = y - self.media_y_
        PtDP = Phi.T @ (peso[:, None] * Phi)
        PtDt = Phi.T @ (peso * t)
        autovalores = np.linalg.eigvalsh(PtDP)
        d = Phi.shape[1]
        for self.n_iter_alpha_ in range(1, self.max_iter_alpha + 1):
            m = np.linalg.solve(alpha * np.eye(d) + PtDP, PtDt)
            gamma_efetivo = np.sum(autovalores / (alpha + autovalores))
            alpha_novo = gamma_efetivo / (m @ m)
            mudou = abs(alpha_novo - alpha) / alpha
            alpha = alpha_novo
            if mudou < self.tol:
                break
        self.alpha_, self.gamma_efetivo_ = alpha, gamma_efetivo
        self.S_ = np.linalg.inv(alpha * np.eye(d) + PtDP)
        self.m_ = self.S_ @ PtDt
        return alpha

    # (b) ruído com m e S fixos
    def _ajustar_ruido(self, H: np.ndarray, r2: np.ndarray, q: np.ndarray, inicio: np.ndarray) -> np.ndarray:
        H1 = np.c_[np.ones(len(H)), H]
        penalizado = np.r_[0.0, np.ones(H.shape[1])]

        def objetivo(theta):
            s = np.exp(H1 @ theta)
            v = s + q
            valor = 0.5 * np.sum(np.log(v) + r2 / v) + self.lam * np.sum(penalizado * theta ** 2)
            d_eta = 0.5 * s * (1.0 / v - r2 / v ** 2)
            return valor, H1.T @ d_eta + 2 * self.lam * penalizado * theta

        resultado = minimize(objetivo, inicio, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.convergiu_ruido_ = bool(resultado.success)
        return resultado.x

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "RegressaoLinearBayesianaHeterocedastica":
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        desvio = Z.std(axis=0)
        self.desvio_z_ = np.where(desvio > 0, desvio, 1.0)
        H = variaveis_do_ruido(Z, self.n_gavetas)

        homocedastica = RegressaoLinearBayesiana(tol=self.tol).fit(Z, y)
        theta = np.r_[np.log(1.0 / homocedastica.beta_), np.zeros(H.shape[1])]
        alpha = homocedastica.alpha_
        self.historico_ = []
        for iteracao in range(1, self.n_iteracoes + 1):
            variancia_ruido = np.exp(theta[0] + H @ theta[1:])
            alpha = self._ajustar_pesos(Z, y, variancia_ruido, alpha)
            Phi = (Z - self.media_z_) / self.desvio_z_
            residuo = y - self.media_y_ - Phi @ self.m_
            q = np.einsum("ij,jk,ik->i", Phi, self.S_, Phi)
            theta = self._ajustar_ruido(H, residuo ** 2, q, theta)
            v = np.exp(theta[0] + H @ theta[1:]) + q
            self.historico_.append({
                "iteracao": iteracao, "alpha": alpha,
                "nll_treino": float(np.mean(0.5 * np.log(2 * np.pi * v) + residuo ** 2 / (2 * v))),
                "sigma2_min": float(np.exp(theta[0] + H @ theta[1:]).min()),
                "sigma2_max": float(np.exp(theta[0] + H @ theta[1:]).max()),
            })
        self.gamma0_, self.gamma_ = float(theta[0]), theta[1:]
        # Com o σ² final, os pesos são reajustados para ficarem coerentes com o ruído da preditiva.
        self._ajustar_pesos(Z, y, np.exp(theta[0] + H @ theta[1:]), alpha)
        return self

    def variancia_ruido(self, Z: np.ndarray) -> np.ndarray:
        return np.exp(self.gamma0_ + variaveis_do_ruido(Z, self.n_gavetas) @ self.gamma_)

    def predict(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = np.asarray(Z, dtype=np.float64)
        Phi = (Z - self.media_z_) / self.desvio_z_
        media = Phi @ self.m_ + self.media_y_
        variancia = self.variancia_ruido(Z) + np.einsum("ij,jk,ik->i", Phi, self.S_, Phi)
        return media, variancia

    def sem_ruido(self) -> SimpleNamespace:
        """Os atributos que `prever_com_entrada_incerta` lê, com 1/β = 0, para somar σ²(h) por fora."""
        return SimpleNamespace(media_z_=self.media_z_, desvio_z_=self.desvio_z_, m_=self.m_, S_=self.S_,
                               media_y_=self.media_y_, beta_=np.inf)


def prever_com_entrada_incerta_heterocedastica(modelo, Eg, Cg, P, C):
    """Lei da variância total de `propagacao_analitica.py`, com 1/β trocado por σ²(h).

    h só depende de P e C, então σ²(h) é constante dada a refeição e sai da esperança sem mudar.
    """
    media, variancia_sem_ruido = prever_com_entrada_incerta(modelo.sem_ruido(), Eg, Cg, P, C)
    return media, variancia_sem_ruido + modelo.variancia_ruido(montar_Z(Eg, P, C))
