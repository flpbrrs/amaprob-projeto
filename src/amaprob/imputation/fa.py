"""FA (Análise Fatorial) por EM com dados faltantes.

Referências: Rubin e Thayer, "EM Algorithms for ML Factor Analysis" (1982); Ghahramani e Hinton,
"The EM Algorithm for Mixtures of Factor Analyzers" (1996); Ilin e Raiko (2010) para os faltantes.

Modelo, para cada refeição, em x = log1p(gramas) ∈ R^K:

    z ~ N(0, I_q)                      fator latente de dimensão q
    x = W z + μ + ε,  ε ~ N(0, Ψ)      Ψ = diag(ψ_1, …, ψ_K): um ruído por gaveta

É o PPCA com o ruído isotrópico σ² I trocado por Ψ diagonal. Gavetas ausentes como condicionantes,
como no PPCA: a verossimilhança de cada refeição n usa só o conjunto O_n das gavetas observadas
(P = 1 e M = 0), com marginal

    x_O ~ N(μ_O, C_O),   C_O = W_O W_Oᵀ + Ψ_O

Passo E. Com r_n = x_O − μ_O e A_n = I_q + W_Oᵀ Ψ_O⁻¹ W_O (q × q), a posterior do latente é

    z_n | x_O ~ N(E[z_n], A_n⁻¹),   E[z_n] = A_n⁻¹ W_Oᵀ Ψ_O⁻¹ r_n

Com Ψ = σ² I, A_n é a matriz do PPCA dividida por σ², e as duas posteriores coincidem.

Passo M. Com z̃ = [z; 1], a regressão de x_k em z̃ por gaveta é a mesma do PPCA, porque o peso
1/ψ_k é constante dentro da gaveta e se cancela:

    [w_k; μ_k] = (Σ_{n: k∈O_n} E[z̃_n z̃_nᵀ])⁻¹ Σ_{n: k∈O_n} x_nk E[z̃_n]

O ruído passa a ser o resíduo esperado de cada gaveta, e não a média sobre todas:

    ψ_k = (1 / n_k) Σ_{n: k∈O_n} (x_nk² − 2 x_nk w̃_kᵀ E[z̃_n] + w̃_kᵀ E[z̃_n z̃_nᵀ] w̃_k)

Piso no ruído. Na FA, o ψ_k de máxima verossimilhança pode ir a 0 (caso de Heywood): a gaveta
passa a ser explicada só pelos fatores, e o intervalo dela fica estreito demais. As porções da
POF vêm em degraus, o que favorece esse colapso. Por isso ψ_k ≥ 0,1% da variância observada da
gaveta (e nunca abaixo de um mínimo absoluto). O piso é baixo de propósito: um piso de 1% já
prende gavetas normais cujos fatores explicam quase toda a variância, e alarga o intervalo delas. Gavetas sem observados ficam com w_k = 0 e
ψ_k = 1: finitas, para que 0 · ∞ não espalhe NaN pelas refeições.

Log-verossimilhança dos observados, só com matrizes q × q (lema do determinante e Woodbury):

    log|C_O| = Σ_{k∈O_n} log ψ_k + log|A_n|
    r_nᵀ C_O⁻¹ r_n = r_nᵀ Ψ_O⁻¹ r_n − b_nᵀ A_n⁻¹ b_n,   b_n = W_Oᵀ Ψ_O⁻¹ r_n

O EM garante que ela não diminui entre iterações enquanto o piso não é ativado.

Amostragem das ocultas H_n: z ~ N(E[z_n], A_n⁻¹) e x_H = μ_H + W_H z + √ψ_H ⊙ v, v ~ N(0, I).
É a condicional exata x_H | x_O. Sem observados, a posterior é a priori e x_H segue a marginal
N(μ_H, W_H W_Hᵀ + Ψ_H). As amostras são truncadas em 0, como no baseline.

Média condicional (opção `media_condicional`): a mesma do PPCA (derivação em `ppca.py`), μ_k vira
b_kᵀ u_n, com u_n = [1; φ(P_n); φ(C_n)] e priori N(0, 1/α) nos coeficientes de P e C. Como o
ruído é por gaveta, a penalização entra na regressão de cada gaveta escalada pelo ψ_k da iteração
anterior,

    [w_k; b_k] = (Σ_{n: k∈O_n} E[z̃_n z̃_nᵀ] + ψ_k α D)⁻¹ Σ_{n: k∈O_n} x_nk E[z̃_n]

e depois ψ_k sai do resíduo esperado, com o mesmo piso. O EM maximiza a log-verossimilhança menos
(α/2) Σ_k ‖b_k sem o intercepto‖².
"""

from dataclasses import dataclass

import numpy as np

from amaprob.imputation import media_condicional as mc
from amaprob.imputation.ppca import observados

PISO_RELATIVO = 0.001
PISO_ABSOLUTO = 1e-6
RUIDO_SEM_OBSERVADOS = 1.0


@dataclass
class _Posterior:
    A: np.ndarray  # (N, q, q)
    A_inv: np.ndarray  # (N, q, q)
    media_z: np.ndarray  # (N, q)
    log_verossimilhanca: float


class FA:
    def __init__(self, q: int = 5, semente: int = 0, tolerancia: float = 1e-6, max_iteracoes: int = 500,
                 media_condicional: bool = False, alfa: float = mc.ALFA):
        self.q = q
        self.semente = semente
        self.tolerancia = tolerancia
        self.max_iteracoes = max_iteracoes
        self.media_condicional = media_condicional
        self.alfa = alfa

    def fit(self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray) -> "FA":
        O = observados(P, M)
        X_O = np.where(O, X, 0.0)
        tem_observado = O.any(axis=0)
        n_por_gaveta = np.maximum(O.sum(axis=0), 1)

        rng = np.random.default_rng(self.semente)
        self.media_ = X_O.sum(axis=0) / n_por_gaveta
        variancia = ((X_O - self.media_) ** 2 * O).sum(axis=0) / n_por_gaveta
        self.piso_ = np.where(tem_observado, np.maximum(PISO_RELATIVO * variancia, PISO_ABSOLUTO), RUIDO_SEM_OBSERVADOS)
        self.psi_ = np.where(tem_observado, np.maximum(variancia, self.piso_), RUIDO_SEM_OBSERVADOS)
        self.W_ = rng.normal(0.0, 0.1, size=(X_O.shape[1], self.q)) * tem_observado[:, None]
        U = blocos = None
        if self.media_condicional:
            self.padronizacao_ = mc.ajustar_padronizacao(P, C)
            U = mc.montar_u(P, C, self.padronizacao_)
            blocos = mc.blocos_fixos(U, X_O, O)
            self.coeficientes_media_ = mc.coeficientes_iniciais(blocos, self.psi_, tem_observado, self.alfa)
            residuo = ((X_O - self._media(U)) ** 2 * O).sum(axis=0) / n_por_gaveta
            self.psi_ = np.where(tem_observado, np.maximum(residuo, self.piso_), RUIDO_SEM_OBSERVADOS)

        posterior = self._passo_e(X_O, O, U)
        verossimilhancas, objetivos = [posterior.log_verossimilhanca], [self._objetivo(posterior)]
        for _ in range(self.max_iteracoes):
            self._passo_m(X_O, O, posterior, tem_observado, n_por_gaveta, U, blocos)
            posterior = self._passo_e(X_O, O, U)
            verossimilhancas.append(posterior.log_verossimilhanca)
            objetivos.append(self._objetivo(posterior))
            if objetivos[-1] - objetivos[-2] <= self.tolerancia * abs(objetivos[-2]):
                break

        self.historico_log_verossimilhanca_ = np.array(verossimilhancas)
        self.historico_objetivo_ = np.array(objetivos)
        self.n_iteracoes_ = len(objetivos) - 1
        self.gavetas_no_piso_ = np.flatnonzero(tem_observado & (self.psi_ <= self.piso_))
        return self

    def _media(self, U: np.ndarray | None) -> np.ndarray:
        """μ (K,) com a média fixa; B u_n por refeição (N, K) com a média condicional."""
        return self.media_ if U is None else U @ self.coeficientes_media_.T

    def _objetivo(self, posterior: _Posterior) -> float:
        if not self.media_condicional:
            return posterior.log_verossimilhanca
        return posterior.log_verossimilhanca - mc.penalizacao(self.coeficientes_media_, self.alfa)

    def _passo_e(self, X_O: np.ndarray, O: np.ndarray, U: np.ndarray | None = None) -> _Posterior:
        N, K = X_O.shape
        q, psi = self.q, self.psi_
        W_sobre_psi = self.W_ / psi[:, None]  # linhas de Ψ⁻¹ W
        produtos_W = (self.W_[:, :, None] * W_sobre_psi[:, None, :]).reshape(K, q * q)  # w_k w_kᵀ / ψ_k
        A = np.eye(q) + (O @ produtos_W).reshape(N, q, q)  # I + W_Oᵀ Ψ_O⁻¹ W_O
        R = (X_O - self._media(U)) * O
        b = R @ W_sobre_psi  # W_Oᵀ Ψ_O⁻¹ r_n
        A_inv = np.linalg.inv(A)
        media_z = np.einsum("nij,nj->ni", A_inv, b)

        n_obs = O.sum(axis=1)
        log_det_C = O @ np.log(psi) + np.linalg.slogdet(A)[1]
        forma_quadratica = (R**2 / psi).sum(axis=1) - (b * media_z).sum(axis=1)
        log_verossimilhanca = -0.5 * np.sum(n_obs * np.log(2 * np.pi) + log_det_C + forma_quadratica)
        return _Posterior(A, A_inv, media_z, float(log_verossimilhanca))

    def _passo_m(
        self, X_O: np.ndarray, O: np.ndarray, posterior: _Posterior, tem_observado: np.ndarray, n_por_gaveta: np.ndarray,
        U: np.ndarray | None = None, blocos: mc.BlocosFixos | None = None,
    ) -> None:
        N, K = X_O.shape
        q = self.q
        Ez = posterior.media_z
        Ezz = posterior.A_inv + Ez[:, :, None] * Ez[:, None, :]

        if U is not None:
            W, B, residuo = mc.passo_m(Ez, Ezz, U, X_O, O, blocos, self.psi_, tem_observado, self.alfa)
            self.W_, self.coeficientes_media_ = W, B
            self.psi_ = np.where(tem_observado, np.maximum(residuo / n_por_gaveta, self.piso_), RUIDO_SEM_OBSERVADOS)
            return

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

        soma_x2 = (X_O**2).sum(axis=0)
        cruzado = (soma_xEz * w_aum).sum(axis=1)
        quadratico = np.einsum("ki,kij,kj->k", w_aum, soma_Ezz, w_aum)
        residuo = (soma_x2 - 2 * cruzado + quadratico) / n_por_gaveta
        self.W_ = w_aum[:, :q]
        self.media_ = np.where(tem_observado, w_aum[:, q], self.media_)
        self.psi_ = np.where(tem_observado, np.maximum(residuo, self.piso_), RUIDO_SEM_OBSERVADOS)

    def sample(
        self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray, n_samples: int, rng: np.random.Generator
    ) -> np.ndarray:
        M = np.asarray(M, dtype=bool)
        O = observados(P, M)
        X_O = np.where(O, X, 0.0)
        U = mc.montar_u(P, C, self.padronizacao_) if self.media_condicional else None
        posterior = self._passo_e(X_O, O, U)

        # z = E[z] + L⁻ᵀ u, com A = L Lᵀ, tem covariância A⁻¹.
        L_inv_T = np.swapaxes(np.linalg.inv(np.linalg.cholesky(posterior.A)), 1, 2)
        u = rng.standard_normal((n_samples, len(X_O), self.q))
        z = posterior.media_z + np.einsum("nij,snj->sni", L_inv_T, u)

        linhas, colunas = np.nonzero(M)
        ocultos = (
            np.broadcast_to(self._media(U), X_O.shape)[linhas, colunas]
            + np.einsum("shq,hq->sh", z[:, linhas], self.W_[colunas])
            + np.sqrt(self.psi_[colunas]) * rng.standard_normal((n_samples, len(linhas)))
        )
        amostras = np.repeat(X_O[None], n_samples, axis=0)
        amostras[:, linhas, colunas] = np.maximum(ocultos, 0.0)
        return amostras
