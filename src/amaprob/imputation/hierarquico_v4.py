"""Modelo hierárquico v4: a redução da v2 limitada para que Σ_novo seja sempre válida.

A v2 (`hierarquico_v2.py`) corrige o bloco oculto da FA condicional por Σ_novo = Σ_FA − T + V e
projeta para semidefinida positiva quando a subtração passa do ponto (3,7 a 5,0% de traço
acrescentado). A v3 (`hierarquico_v3.py`) evita a projeção tirando τ² do ruído antes de
condicionar, mas em 13 das 29 gavetas τ_k² > ψ_k e o piso conta τ² duas vezes. Aqui a fórmula da
v2 é mantida, com a redução encolhida por um fator c por refeição:

    Σ_novo = Σ_FA − c·D,   D = T − V   (bloco oculto H)
    c = min(1; ρ / λ_max),   ρ = 0,9,   λ_max = maior autovalor de Σ_FA^(−1/2) D Σ_FA^(−1/2)

- por gaveta: D = diag(τ_k² − v_pk);
- fator único: D = (τ² − v_p)·11ᵀ, de posto 1.

D é semidefinida positiva porque a posterior nunca passa do prior (V ≤ T). Escrevendo
Σ_FA = L Lᵀ (Cholesky) e R = L⁻¹ D L⁻ᵀ, que tem autovalores em [0, λ_max]:

    Σ_novo − (1 − ρ)·Σ_FA = L (ρ I − c R) Lᵀ ⪰ 0,   porque c·λ_max ≤ ρ,

ou seja, Σ_novo ⪰ 0,1·Σ_FA: nenhuma direção perde mais de 90% da variância da FA, e Σ_novo é
positiva definida sem projeção. Como diag(D) ≥ 0, diag(Σ_novo) ≤ diag(Σ_FA).

λ_max sai do bloco oculto completado com a identidade fora de M (bloco ⊕ I, cujo Cholesky é o do
bloco ⊕ I) e com D nulo fora de M, o que não acrescenta autovalores além de zeros:
- por gaveta: S = L⁻¹ diag(√d), λ_max = maior autovalor de Sᵀ S;
- fator único: λ_max = d · 1_Hᵀ Σ_FA⁻¹ 1_H.

Casos exatos:
- sem histórico, V = T bit a bit (`estimar_delta_v2`), D = 0, λ_max = 0, c = 1 e
  Σ_novo = Σ_FA − 0 = Σ_FA sem diferença, em todas as gavetas;
- com c = 1 a conta é a da v2: Σ_FA − (τ² − v) é igual, em ponto flutuante, a Σ_FA + (v − τ²),
  porque a troca de sinal de uma subtração é exata. A v2 só difere quando projeta, o que não
  acontece com c = 1 (Σ_novo ⪰ 0,1·Σ_FA).

A média continua μ_novo = μ_FA + δ̂, como na v2. A amostragem é a da v2 (`amostrar_bloco`).

Regra 2: as gramas só são lidas por `ocultar`, que apaga as células de M.
"""

import numpy as np

from amaprob.imputation.base import ocultar
from amaprob.imputation.hierarquico import Variancias
from amaprob.imputation.momentos import momentos_ocultos

RHO = 0.9  # fração máxima da variância da FA que a redução pode tirar, em qualquer direção
BLOCO_V4 = 2_048
VARIANTES_V4 = ("gaveta", "fator")


def reducao(v: np.ndarray, M: np.ndarray, variancias: Variancias, variante: str) -> np.ndarray:
    """d_nk = τ² − v nas células ocultas, 0 fora de M (N, K).

    Por gaveta, d é a diagonal de D; no fator único, d_n é constante nas ocultas e D = d_n·11ᵀ.
    """
    if variante == "gaveta":
        return np.where(M, variancias.tau2[None, :] - v, 0.0)
    if variante == "fator":
        return np.where(M, variancias.tau2_fator - v, 0.0)
    raise ValueError(f"variante desconhecida: {variante!r}")


def lambda_max(cov_fa: np.ndarray, d: np.ndarray, M: np.ndarray, variante: str) -> np.ndarray:
    """Maior autovalor de Σ_FA^(−1/2) D Σ_FA^(−1/2) por refeição (N,); 0 sem ocultas."""
    N, K = M.shape
    idx = np.arange(K)
    saida = np.zeros(N)
    for inicio in range(0, N, BLOCO_V4):
        b = slice(inicio, inicio + BLOCO_V4)
        A = cov_fa[b].copy()
        A[:, idx, idx] = np.where(M[b], np.einsum("nii->ni", A), 1.0)
        if variante == "gaveta":
            L = np.linalg.cholesky(A)
            S = np.linalg.solve(L, np.sqrt(np.maximum(d[b], 0.0))[:, None, :] * np.eye(K)[None])
            saida[b] = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", S, S)).max(axis=1)
        else:
            h = M[b].astype(np.float64)
            w = np.linalg.solve(A, h[:, :, None])[:, :, 0]
            d_n = d[b].max(axis=1)  # constante nas ocultas
            saida[b] = d_n * np.einsum("nk,nk->n", h, w)
    return saida


def fator_c(lam: np.ndarray, rho: float = RHO) -> np.ndarray:
    """c = min(1; ρ/λ_max), com c = 1 exato quando λ_max ≤ ρ (inclusive λ_max = 0)."""
    return np.where(lam > rho, rho / np.where(lam > rho, lam, 1.0), 1.0)


def momentos_com_pessoa_v4(
    fa,
    G: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    delta: np.ndarray,
    v: np.ndarray,
    variancias: Variancias,
    variante: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Média (N, K) e covariância (N, K, K) em log de Σ_novo = Σ_FA − c·D, com c (N,) e Σ_FA.

    Σ_FA (a covariância condicional antes da correção) volta para as conferências.
    """
    M = np.asarray(M, dtype=bool)
    media, cov_fa = momentos_ocultos(fa, ocultar(G, M) - delta, P, M, C)
    linhas, colunas = np.nonzero(M)
    media[linhas, colunas] += delta[linhas, colunas]

    d = reducao(v, M, variancias, variante)
    c = fator_c(lambda_max(cov_fa, d, M, variante))
    cd = c[:, None] * d
    covariancia = cov_fa.copy()
    if variante == "gaveta":
        covariancia[linhas, colunas, colunas] -= cd[linhas, colunas]
    else:
        covariancia -= cd[:, :, None] * M[:, None, :]
    return media, covariancia, c, cov_fa


def menor_autovalor_folga(cov_novo: np.ndarray, cov_fa: np.ndarray, M: np.ndarray) -> float:
    """Menor autovalor de Σ_novo − (1 − ρ)·Σ_FA no bloco oculto, sobre todas as refeições.

    A diagonal fora de M é preenchida com 1, da ordem dos autovalores do bloco: um preenchimento
    grande (1e6) levaria o erro absoluto do `eigvalsh` a ~eps·1e6 ≈ 2e-10, acima da tolerância.
    """
    K = M.shape[1]
    idx = np.arange(K)
    minimo = np.inf
    com_oculta = np.flatnonzero(M.any(axis=1))
    for inicio in range(0, len(com_oculta), BLOCO_V4):
        sel = com_oculta[inicio:inicio + BLOCO_V4]
        A = cov_novo[sel] - (1.0 - RHO) * cov_fa[sel]
        A[:, idx, idx] = np.where(M[sel], np.einsum("nii->ni", A), 1.0)
        minimo = min(minimo, float(np.linalg.eigvalsh(A).min()))
    return minimo
