"""Modelo hierárquico v3 (variante por gaveta): o efeito da pessoa entra antes de condicionar.

Na v2 (`hierarquico_v2.py`), a correção Σ_novo = Σ_FA − T + V subtrai o τ² marginal de uma
covariância já condicionada às gavetas observadas. Quando a condicional encolhe abaixo de τ_k², a
matriz deixa de ser semidefinida positiva (até 32% das refeições projetadas, com 3,7 a 5,0% de
traço acrescentado; `results/hierarquico_v2_projecoes.csv`). Aqui a variação entre pessoas sai do
ruído da FA, e a pessoa entra no modelo gerador:

    x_n = b(u_n) + δ_p + W z_n + ε_n,   z_n ~ N(0, I),   ε_n ~ N(0, Ψ − T),   δ_p ~ N(δ̂_p, V_p)

com T = diag(τ_k²) e V_p = diag(v_pk), a posterior de δ a partir do histórico (`estimar_delta_v2`).
Integrando δ_p, cada refeição vê uma FA com ruído próprio

    ψ*_nk = max(ψ_k − τ_k², piso) + v_pk,   piso = 1e-6,

em TODAS as gavetas, observadas e ocultas. O passo E da FA (`fa.py`) vale com Ψ* no lugar de Ψ,
com os resíduos deslocados por δ̂:

    r_n = (x_O − δ̂_O) − b_O(u_n)
    A_n = I + W_Oᵀ Ψ*_O⁻¹ W_O,   E[z_n] = A_n⁻¹ W_Oᵀ Ψ*_O⁻¹ r_n,   Cov(z_n) = A_n⁻¹
    E[x_H] = b_H(u_n) + W_H E[z_n] + δ̂_H
    Cov(x_H) = W_H A_n⁻¹ W_Hᵀ + diag(ψ*_H)

Simplificação declarada: δ_p entra como ruído independente por refeição (v_pk somado a ψ*), sem a
correlação entre as gavetas observadas e ocultas da mesma pessoa que um δ_p comum criaria. É a
mesma aproximação (a) de `hierarquico.py`: só o histórico atualiza δ, não as observadas do próprio
dia avaliado.

A covariância das ocultas é semidefinida positiva por construção: A_n⁻¹ é positiva definida e
ψ*_H ≥ piso > 0. Nenhuma projeção é necessária.

Sem histórico, v_pk = τ_k² e ψ* = ψ exatamente nas gavetas sem piso: aritmeticamente
(ψ − τ²) + τ² pode diferir de ψ em 1 ulp, então nesse caso grava-se o próprio ψ (o mesmo cuidado
de `estimar_delta_v2`). Nas gavetas com τ_k² > ψ_k − piso, o piso aumenta o ruído para
piso + τ_k² > ψ_k mesmo sem histórico; elas são listadas por `gavetas_no_piso`.

Regra 2: X só é lido nas células observadas (P & ~M), por `ocultar`; lixo nas ocultas não muda
média nem covariância.
"""

import numpy as np

from amaprob.imputation import media_condicional as mc
from amaprob.imputation.base import ocultar
from amaprob.imputation.momentos import BLOCO
from amaprob.imputation.ppca import observados

PISO_PSI = 1e-6


def ruido_sem_pessoa(fa, tau2: np.ndarray) -> np.ndarray:
    """max(ψ_k − τ_k², piso): o ruído da FA sem a variação entre pessoas (K,)."""
    return np.maximum(fa.psi_ - tau2, PISO_PSI)


def gavetas_no_piso(fa, tau2: np.ndarray) -> np.ndarray:
    """Índices das gavetas em que ψ_k − τ_k² < piso (o piso é acionado)."""
    return np.flatnonzero(fa.psi_ - tau2 < PISO_PSI)


def psi_estrela(fa, v: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """ψ*_nk = max(ψ_k − τ_k², piso) + v_nk, com ψ* = ψ exato onde v = τ² e não há piso (N, K)."""
    base = ruido_sem_pessoa(fa, tau2)
    psi = base[None, :] + v
    sem_historico = (v == tau2[None, :]) & (fa.psi_ - tau2 >= PISO_PSI)[None, :]
    return np.where(sem_historico, fa.psi_[None, :], psi)


def condicionar_com_psi(fa, X_O: np.ndarray, O: np.ndarray, M: np.ndarray, U, psi: np.ndarray):
    """Passo E da FA com Ψ por refeição (cópia de `FA._passo_e` e de `momentos._bloco`).

    Devolve a média (N, K) e a covariância (N, K, K) das ocultas, NaN e 0 fora de M.
    """
    W = fa.W_
    W_sobre_psi = W[None, :, :] / psi[:, :, None]  # (N, K, q): linhas de Ψ*⁻¹ W
    A = np.eye(fa.q)[None] + np.einsum("nk,nki,kj->nij", O.astype(np.float64), W_sobre_psi, W)
    R = (X_O - fa._media(U)) * O
    b = np.einsum("nk,nki->ni", R, W_sobre_psi)
    A_inv = np.linalg.inv(A)
    media_z = np.einsum("nij,nj->ni", A_inv, b)

    media = np.broadcast_to(fa._media(U), X_O.shape) + media_z @ W.T
    media = np.where(M, media, np.nan)
    pesos = W * M[:, :, None]
    covariancia = np.einsum("nki,nij,nlj->nkl", pesos, A_inv, pesos)
    diagonal = np.arange(X_O.shape[1])
    covariancia[:, diagonal, diagonal] += np.where(M, psi, 0.0)
    covariancia *= M[:, :, None] & M[:, None, :]
    return media, covariancia


def momentos_com_pessoa_v3(
    fa, G: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray,
    delta: np.ndarray, v: np.ndarray, tau2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Média (N, K) e covariância (N, K, K) das ocultas em log1p(gramas), com Ψ* por refeição."""
    M = np.asarray(M, dtype=bool)
    P = np.asarray(P, dtype=bool)
    O = observados(P, M)
    X_O = np.where(O, ocultar(G, M) - delta, 0.0)
    psi = psi_estrela(fa, v, tau2)
    N, K = X_O.shape
    media_log = np.full((N, K), np.nan)
    cov_log = np.zeros((N, K, K))
    for inicio in range(0, N, BLOCO):
        b = slice(inicio, inicio + BLOCO)
        U = mc.montar_u(P[b], C[b], fa.padronizacao_) if fa.media_condicional else None
        media_log[b], cov_log[b] = condicionar_com_psi(fa, X_O[b], O[b], M[b], U, psi[b])
    linhas, colunas = np.nonzero(M)
    media_log[linhas, colunas] += delta[linhas, colunas]
    return media_log, cov_log


def menor_autovalor_das_ocultas(cov_log: np.ndarray, M: np.ndarray, bloco: int = 2_048) -> float:
    """Menor autovalor do bloco oculto sobre todas as refeições com ocultas.

    A diagonal fora de M vira +inf de fato (um valor grande) para não contar as direções vazias.
    """
    N, K = M.shape
    idx = np.arange(K)
    minimo = np.inf
    com_oculta = M.any(axis=1)
    for inicio in range(0, N, bloco):
        b = slice(inicio, inicio + bloco)
        sel = com_oculta[b]
        if not sel.any():
            continue
        A = cov_log[b][sel].copy()
        A[:, idx, idx] = np.where(M[b][sel], np.einsum("nii->ni", A), 1e6)
        minimo = min(minimo, float(np.linalg.eigvalsh(A).min()))
    return minimo
