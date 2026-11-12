"""PPCA (Análise de Componentes Principais Probabilística) por EM com dados faltantes.

Referências: Tipping e Bishop, "Probabilistic Principal Component Analysis" (1999); Ilin e Raiko,
"Practical Approaches to Principal Component Analysis in the Presence of Missing Values" (2010).

Modelo, para cada refeição, em x = log1p(gramas) ∈ R^K:

    z ~ N(0, I_q)                      fator latente de dimensão q
    x = W z + μ + ε,  ε ~ N(0, σ² I)   W de K × q; ruído isotrópico (igual para toda gaveta)

Gavetas ausentes como condicionantes: a verossimilhança de cada refeição n usa só o conjunto O_n
das gavetas observadas (P = 1 e M = 0). Integrando z, a marginal dos observados é

    x_O ~ N(μ_O, C_O),   C_O = W_O W_Oᵀ + σ² I

Gavetas ausentes não entram como zero: o modelo descreve o tamanho das porções dado que a gaveta
está presente, e "estar presente" fica em quais coordenadas entram na conta.

Passo E. Com r_n = x_O − μ_O e A_n = σ² I_q + W_Oᵀ W_O (q × q), a posterior do latente é

    z_n | x_O ~ N(E[z_n], σ² A_n⁻¹),   E[z_n] = A_n⁻¹ W_Oᵀ r_n

Passo M. Com z̃ = [z; 1], cada gaveta k tem sua própria regressão de x_k em z̃, somada só nas
refeições em que k foi observada. Maximizar a esperança da log-verossimilhança completa dá

    [w_k; μ_k] = (Σ_{n: k∈O_n} E[z̃_n z̃_nᵀ])⁻¹ Σ_{n: k∈O_n} x_nk E[z̃_n]

    σ² = (1 / Σ_n |O_n|) Σ_n Σ_{k∈O_n} E[(x_nk − w̃_kᵀ z̃_n)²]
       = (1 / Σ_n |O_n|) Σ_n Σ_{k∈O_n} (x_nk² − 2 x_nk w̃_kᵀ E[z̃_n] + w̃_kᵀ E[z̃_n z̃_nᵀ] w̃_k)

μ é atualizado junto com W: com faltantes, a média simples dos observados não é a de máxima
verossimilhança. Uma soma única de E[zzᵀ] sobre todas as refeições só vale sem faltantes.

Log-verossimilhança dos observados, só com matrizes q × q (lema do determinante e Woodbury):

    log|C_O| = (|O_n| − q) log σ² + log|A_n|
    r_nᵀ C_O⁻¹ r_n = (r_nᵀ r_n − r_nᵀ W_O A_n⁻¹ W_Oᵀ r_n) / σ²

O EM garante que ela não diminui entre iterações.

Amostragem das ocultas H_n: z ~ N(E[z_n], σ² A_n⁻¹) e x_H = μ_H + W_H z + σ v, v ~ N(0, I). É a
condicional exata x_H | x_O. Sem observados, a posterior é a priori e x_H segue a marginal
N(μ_H, W_H W_Hᵀ + σ² I). As amostras são truncadas em 0, como no baseline.
"""

from dataclasses import dataclass

import numpy as np


def observados(P: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.asarray(P, dtype=bool) & ~np.asarray(M, dtype=bool)


@dataclass
class _Posterior:
    A: np.ndarray  # (N, q, q)
    A_inv: np.ndarray  # (N, q, q)
    media_z: np.ndarray  # (N, q)
    log_verossimilhanca: float


class PPCA:
    def __init__(self, q: int = 5, semente: int = 0, tolerancia: float = 1e-6, max_iteracoes: int = 500):
        self.q = q
        self.semente = semente
        self.tolerancia = tolerancia
        self.max_iteracoes = max_iteracoes

    def fit(self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray) -> "PPCA":
        O = observados(P, M)
        X_O = np.where(O, X, 0.0)
        tem_observado = O.any(axis=0)
        n_por_gaveta = np.maximum(O.sum(axis=0), 1)

        rng = np.random.default_rng(self.semente)
        self.media_ = X_O.sum(axis=0) / n_por_gaveta
        self.sigma2_ = float(np.mean(((X_O - self.media_) ** 2 * O).sum(axis=0)[tem_observado] / n_por_gaveta[tem_observado]))
        self.W_ = rng.normal(0.0, 0.1, size=(X_O.shape[1], self.q)) * tem_observado[:, None]

        posterior = self._passo_e(X_O, O)
        historico = [posterior.log_verossimilhanca]
        for _ in range(self.max_iteracoes):
            self._passo_m(X_O, O, posterior, tem_observado)
            posterior = self._passo_e(X_O, O)
            historico.append(posterior.log_verossimilhanca)
            if historico[-1] - historico[-2] <= self.tolerancia * abs(historico[-2]):
                break

        self.historico_log_verossimilhanca_ = np.array(historico)
        self.n_iteracoes_ = len(historico) - 1
        return self

    def _passo_e(self, X_O: np.ndarray, O: np.ndarray) -> _Posterior:
        N, K = X_O.shape
        q, sigma2 = self.q, self.sigma2_
        produtos_W = (self.W_[:, :, None] * self.W_[:, None, :]).reshape(K, q * q)  # w_k w_kᵀ por gaveta
        A = sigma2 * np.eye(q) + (O @ produtos_W).reshape(N, q, q)  # σ²I + W_Oᵀ W_O
        R = (X_O - self.media_) * O
        b = R @ self.W_  # W_Oᵀ r_n
        A_inv = np.linalg.inv(A)
        media_z = np.einsum("nij,nj->ni", A_inv, b)

        n_obs = O.sum(axis=1)
        log_det_C = (n_obs - q) * np.log(sigma2) + np.linalg.slogdet(A)[1]
        forma_quadratica = ((R**2).sum(axis=1) - (b * media_z).sum(axis=1)) / sigma2
        log_verossimilhanca = -0.5 * np.sum(n_obs * np.log(2 * np.pi) + log_det_C + forma_quadratica)
        return _Posterior(A, A_inv, media_z, float(log_verossimilhanca))

    def _passo_m(self, X_O: np.ndarray, O: np.ndarray, posterior: _Posterior, tem_observado: np.ndarray) -> None:
        N, K = X_O.shape
        q = self.q
        Ez = posterior.media_z
        Ezz = self.sigma2_ * posterior.A_inv + Ez[:, :, None] * Ez[:, None, :]

        Ez_aum = np.c_[Ez, np.ones(N)]  # E[z̃]
        Ezz_aum = np.empty((N, q + 1, q + 1))  # E[z̃ z̃ᵀ] = [[E[zzᵀ], E[z]], [E[z]ᵀ, 1]]
        Ezz_aum[:, :q, :q] = Ezz
        Ezz_aum[:, :q, q] = Ez
        Ezz_aum[:, q, :q] = Ez
        Ezz_aum[:, q, q] = 1.0

        soma_Ezz = (O.T.astype(np.float64) @ Ezz_aum.reshape(N, -1)).reshape(K, q + 1, q + 1)
        soma_xEz = X_O.T @ Ez_aum  # X_O já é 0 fora de O
        w_aum = np.zeros((K, q + 1))
        w_aum[tem_observado] = np.linalg.solve(soma_Ezz[tem_observado], soma_xEz[tem_observado, :, None])[..., 0]

        soma_x2 = np.sum(X_O**2)
        cruzado = np.sum(soma_xEz * w_aum)
        quadratico = np.einsum("ki,kij,kj->", w_aum, soma_Ezz, w_aum)
        self.W_ = w_aum[:, :q]
        self.media_ = np.where(tem_observado, w_aum[:, q], self.media_)
        self.sigma2_ = float((soma_x2 - 2 * cruzado + quadratico) / O.sum())

    def sample(
        self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        M = np.asarray(M, dtype=bool)
        O = observados(P, M)
        X_O = np.where(O, X, 0.0)
        posterior = self._passo_e(X_O, O)

        # z = E[z] + σ L⁻ᵀ u, com A = L Lᵀ, tem covariância σ² A⁻¹.
        L_inv_T = np.swapaxes(np.linalg.inv(np.linalg.cholesky(posterior.A)), 1, 2)
        u = rng.standard_normal((n_samples, len(X_O), self.q))
        z = posterior.media_z + np.sqrt(self.sigma2_) * np.einsum("nij,snj->sni", L_inv_T, u)

        linhas, colunas = np.nonzero(M)
        ocultos = (
            self.media_[colunas]
            + np.einsum("shq,hq->sh", z[:, linhas], self.W_[colunas])
            + np.sqrt(self.sigma2_) * rng.standard_normal((n_samples, len(linhas)))
        )
        amostras = np.repeat(X_O[None], n_samples, axis=0)
        amostras[:, linhas, colunas] = np.maximum(ocultos, 0.0)
        return amostras
