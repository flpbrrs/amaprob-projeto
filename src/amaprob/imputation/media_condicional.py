"""Média de cada gaveta como função linear da presença e do contexto, para o PPCA e a FA.

A derivação completa está no topo de `ppca.py`. Resumo, com u_n = [1; φ(P_n); φ(C_n)]:

    x_nk = w_kᵀ z_n + b_kᵀ u_n + ε_nk

Passo M de cada gaveta k (ECM: coeficientes com o ruído s_k da iteração anterior, depois o ruído):

    [w_k; b_k] = (Σ_{n: k∈O_n} E[z̃_n z̃_nᵀ] + s_k α D)⁻¹ Σ_{n: k∈O_n} x_nk E[z̃_n],   z̃_n = [z_n; u_n]

D penaliza os coeficientes de P e C, e não o intercepto nem W. Os blocos Σ u uᵀ e Σ x u não mudam
entre iterações e são calculados uma vez, só nas linhas observadas de cada gaveta.
"""

from dataclasses import dataclass

import numpy as np

ALFA = 1.0


@dataclass(frozen=True)
class Padronizacao:
    media: np.ndarray
    desvio: np.ndarray


@dataclass(frozen=True)
class BlocosFixos:
    linhas: list[np.ndarray]  # linhas observadas de cada gaveta
    uu: np.ndarray  # (K, d, d): Σ u uᵀ
    xu: np.ndarray  # (K, d): Σ x u
    x2: np.ndarray  # (K,): Σ x²


def ajustar_padronizacao(P: np.ndarray, C: np.ndarray) -> Padronizacao:
    colunas = np.c_[np.asarray(P, dtype=np.float64), np.asarray(C, dtype=np.float64)]
    desvio = colunas.std(axis=0)
    return Padronizacao(colunas.mean(axis=0), np.where(desvio > 0, desvio, 1.0))


def montar_u(P: np.ndarray, C: np.ndarray, padronizacao: Padronizacao) -> np.ndarray:
    colunas = np.c_[np.asarray(P, dtype=np.float64), np.asarray(C, dtype=np.float64)]
    return np.c_[np.ones(len(colunas)), (colunas - padronizacao.media) / padronizacao.desvio]


def blocos_fixos(U: np.ndarray, X_O: np.ndarray, O: np.ndarray) -> BlocosFixos:
    linhas = [np.flatnonzero(O[:, k]) for k in range(O.shape[1])]
    uu = np.stack([U[i].T @ U[i] for i in linhas])
    xu = np.stack([X_O[i, k] @ U[i] for k, i in enumerate(linhas)])
    return BlocosFixos(linhas, uu, xu, (X_O**2).sum(axis=0))


def _penalizados(d: int) -> np.ndarray:
    return np.r_[0.0, np.ones(d - 1)]


def coeficientes_iniciais(blocos: BlocosFixos, escala: np.ndarray, tem_observado: np.ndarray, alfa: float) -> np.ndarray:
    """Regressão ridge de x_k em u, sem o latente, nas linhas observadas de cada gaveta."""
    K, d = blocos.xu.shape
    B = np.zeros((K, d))
    for k in np.flatnonzero(tem_observado):
        B[k] = np.linalg.solve(blocos.uu[k] + escala[k] * alfa * np.diag(_penalizados(d)), blocos.xu[k])
    return B


def passo_m(
    Ez: np.ndarray, Ezz: np.ndarray, U: np.ndarray, X_O: np.ndarray, O: np.ndarray, blocos: BlocosFixos,
    escala: np.ndarray, tem_observado: np.ndarray, alfa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W (K, q), B (K, d) e a soma dos resíduos quadráticos esperados de cada gaveta (K,)."""
    N, q = Ez.shape
    K, d = O.shape[1], U.shape[1]
    soma_zz = (O.T.astype(np.float64) @ Ezz.reshape(N, q * q)).reshape(K, q, q)
    soma_xz = X_O.T @ Ez  # X_O já é 0 fora de O
    penalizacao = np.diag(np.r_[np.zeros(q), _penalizados(d)])
    W, B, residuo = np.zeros((K, q)), np.zeros((K, d)), np.zeros(K)
    for k in np.flatnonzero(tem_observado):
        i = blocos.linhas[k]
        zu = Ez[i].T @ U[i]
        G = np.block([[soma_zz[k], zu], [zu.T, blocos.uu[k]]])  # Σ E[z̃ z̃ᵀ]
        lado_direito = np.r_[soma_xz[k], blocos.xu[k]]  # Σ x E[z̃]
        w = np.linalg.solve(G + escala[k] * alfa * penalizacao, lado_direito)
        residuo[k] = blocos.x2[k] - 2 * w @ lado_direito + w @ G @ w
        W[k], B[k] = w[:q], w[q:]
    return W, B, residuo


def penalizacao(B: np.ndarray, alfa: float) -> float:
    """(α/2) Σ_k ‖b_k sem o intercepto‖², o termo subtraído da log-verossimilhança."""
    return 0.5 * alfa * float(np.sum(B[:, 1:] ** 2))
