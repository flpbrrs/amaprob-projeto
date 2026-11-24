"""Braço analítico para o SVGP: preditiva com entrada gaussiana, pela aproximação por momentos.

Referências: Girard, Rasmussen, Quiñonero-Candela e Murray-Smith, "Gaussian Process Priors with
Uncertain Inputs" (NIPS 2003); Damianou e Lawrence, "Semi-described and semi-supervised learning
with Gaussian processes" (UAI 2015); Titsias e Lawrence (2010) para as estatísticas ψ.

O SVGP de `svgp.py` (que não muda) trabalha em φ = (Z − media_z) / desvio_z e t = (y − ȳ) / s_y:

    m(φ) = aᵀφ + b,   k(φ, φ') = σ_f² exp(−½ (φ − φ')ᵀ Λ⁻¹ (φ − φ')),   Λ = diag(ℓ²)

com q(u) em coordenadas brancas: u − m(Z_u) = L v, L Lᵀ = K_uu + jitter I, q(v) = N(m_v, S_v).
Como no GPyTorch (`VariationalStrategy`), com k_φ = k(Z_u, φ), a marginal de f em φ fixo é

    μ_f(φ) = aᵀφ + b + k_φᵀ A,                       A = L⁻ᵀ m_v
    σ²_f(φ) = σ_f² + jitter − k_φᵀ B k_φ,            B = L⁻ᵀ (I − S_v) L⁻¹

(B = K_uu⁻¹ − K_uu⁻¹ S_u K_uu⁻¹ com S_u = L S_v Lᵀ, e o GPyTorch também soma o jitter à variância
a priori do ponto de teste). A preditiva de t soma o ruído σ².

Entrada incerta: φ ~ N(μ, Σ), com Σ diferente de 0 só no bloco H das gramas ocultas (os momentos
log-normais em gramas do braço analítico, divididos por s sᵀ). Pela lei da variância total,

    E[t]   = E[μ_f(φ)]
    Var[t] = σ² + E[σ²_f(φ)] + Var[μ_f(φ)]

e as esperanças de k sob N(μ, Σ) têm forma fechada para o RBF (produto de gaussianas):

    ψ1_i  = E[k(φ, z_i)] = σ_f² |I + Λ⁻¹Σ|^(−1/2) exp(−½ (μ − z_i)ᵀ (Λ + Σ)⁻¹ (μ − z_i))
    Ψ2_ij = E[k(φ, z_i) k(φ, z_j)]
          = σ_f⁴ |I + 2Λ⁻¹Σ|^(−1/2) exp(−¼ (z_i − z_j)ᵀ Λ⁻¹ (z_i − z_j))
                 · exp(−½ (μ − z̄_ij)ᵀ (Λ/2 + Σ)⁻¹ (μ − z̄_ij)),     z̄_ij = (z_i + z_j)/2
    E[φ k(φ, z_i)] = ψ1_i (μ + Σ (Λ + Σ)⁻¹ (z_i − μ))

Daí, com c_i = aᵀ Σ (Λ + Σ)⁻¹ (z_i − μ) (a covariância entre a parte linear e k_i, por ψ1_i):

    E[t]   = aᵀμ + b + ψ1ᵀ A
    Var[t] = σ² + σ_f² + jitter − tr(B Ψ2) + aᵀΣa + Aᵀ Ψ2 A − (ψ1ᵀ A)² + 2 Σ_i A_i ψ1_i c_i

De volta: média · s_y + ȳ, variância · s_y².

Aproximação declarada: a preditiva é resumida pelos dois primeiros momentos (gaussiana), como no
braço analítico da regressão linear; a entrada é a gaussiana com os momentos da log-normal, e não a
log-normal em si (no SVGP, ao contrário da regressão linear, a média depende da forma da
distribuição, não só dos momentos).

Eficiência: os termos quadráticos com (Λ + Σ)⁻¹ e (Λ/2 + Σ)⁻¹ são separados em H e fora de H.
Fora de H a matriz é diagonal (Λ); em H é |H| × |H| por refeição. Os determinantes também só
dependem do bloco H. As refeições vão em blocos (Ψ2 é M × M por refeição); dentro do bloco, as
refeições são agrupadas pelo conjunto de ocultas, para que H tenha o mesmo tamanho no grupo.
"""

from dataclasses import dataclass

import numpy as np

from amaprob.regression.base import montar_Z

BLOCO_REFEICOES = 128


@dataclass(frozen=True)
class ParametrosSVGP:
    a: np.ndarray            # (d,) pesos da média linear
    b: float                 # viés da média linear
    lengthscale2: np.ndarray  # (d,) ℓ² (diagonal de Λ)
    sigma_f2: float          # escala do kernel
    ruido: float             # σ² da verossimilhança (escala de t)
    jitter: float
    Z_u: np.ndarray          # (M, d) pontos indutores em φ
    A: np.ndarray            # (M,) L⁻ᵀ m_v
    B: np.ndarray            # (M, M) L⁻ᵀ (I − S_v) L⁻¹
    media_z: np.ndarray
    desvio_z: np.ndarray
    media_y: float
    desvio_y: float


def kernel(X: np.ndarray, Y: np.ndarray, lengthscale2: np.ndarray, sigma_f2: float) -> np.ndarray:
    d2 = (((X[:, None, :] - Y[None, :, :]) ** 2) / lengthscale2).sum(axis=-1)
    return sigma_f2 * np.exp(-0.5 * d2)


def extrair(svgp) -> ParametrosSVGP:
    """Os parâmetros do SVGP ajustado, em float64, na forma das equações do topo."""
    import torch

    modelo = svgp.modelo_
    estrategia = modelo.variational_strategy
    with torch.no_grad():
        Z_u = estrategia.inducing_points.detach().numpy().astype(np.float64)
        q_v = estrategia.variational_distribution
        m_v = q_v.mean.detach().numpy().astype(np.float64)
        S_v = q_v.covariance_matrix.detach().numpy().astype(np.float64)
        a = modelo.mean_module.weights.detach().numpy().reshape(-1).astype(np.float64)
        b = float(modelo.mean_module.bias.detach().numpy().reshape(-1)[0])
        lengthscale2 = modelo.covar_module.base_kernel.lengthscale.detach().numpy().reshape(-1).astype(np.float64) ** 2
        sigma_f2 = float(modelo.covar_module.outputscale.detach().numpy())
        ruido = float(svgp.verossimilhanca_.noise.detach().numpy().reshape(-1)[0])
    jitter = float(estrategia.jitter_val)
    K_uu = kernel(Z_u, Z_u, lengthscale2, sigma_f2) + jitter * np.eye(len(Z_u))
    L = np.linalg.cholesky(K_uu)
    L_inv = np.linalg.solve(L, np.eye(len(Z_u)))
    A = L_inv.T @ m_v
    B = L_inv.T @ (np.eye(len(Z_u)) - S_v) @ L_inv
    return ParametrosSVGP(a, b, lengthscale2, sigma_f2, ruido, jitter, Z_u, A, (B + B.T) / 2,
                          svgp.media_z_, svgp.desvio_z_, float(svgp.media_y_), float(svgp.desvio_y_))


def prever_entrada_fixa(par: ParametrosSVGP, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preditiva em gramas com a entrada conhecida; deve coincidir com `SVGP.predict`."""
    phi = (np.asarray(Z, dtype=np.float64) - par.media_z) / par.desvio_z
    k = kernel(phi, par.Z_u, par.lengthscale2, par.sigma_f2)
    media = phi @ par.a + par.b + k @ par.A
    variancia = par.ruido + par.sigma_f2 + par.jitter - np.einsum("ni,ij,nj->n", k, par.B, k)
    return media * par.desvio_y + par.media_y, variancia * par.desvio_y ** 2


def estatisticas_psi(par: ParametrosSVGP, mu: np.ndarray, Sigma_H: np.ndarray, H: np.ndarray):
    """ψ1 (n, M), Ψ2 (n, M, M) e c (n, M) para n refeições com o mesmo conjunto oculto H.

    `mu` (n, d) é a média de φ e `Sigma_H` (n, |H|, |H|) a covariância no bloco H.
    """
    n, d = mu.shape
    lam = par.lengthscale2
    fora = np.setdiff1d(np.arange(d), H)
    e = mu[:, None, :] - par.Z_u[None, :, :]  # (n, M, d): μ − z_i
    e_fora, e_H = e[:, :, fora], e[:, :, H]

    # ψ1: (Λ + Σ)⁻¹ = Λ⁻¹ fora de H; em H, (Λ_H + Σ_H)⁻¹.
    lam_H = lam[H]
    S1 = np.diag(lam_H)[None] + Sigma_H                          # (n, h, h)
    S1_inv = np.linalg.inv(S1)
    quad1 = (e_fora ** 2 / lam[fora]).sum(-1) + np.einsum("nmi,nij,nmj->nm", e_H, S1_inv, e_H)
    det1 = np.linalg.slogdet(np.eye(len(H))[None] + Sigma_H / lam_H[None, None, :])[1]
    psi1 = par.sigma_f2 * np.exp(-0.5 * det1)[:, None] * np.exp(-0.5 * quad1)

    # c_i = aᵀ Σ (Λ + Σ)⁻¹ (z_i − μ): só o bloco H contribui.
    a_Sigma = np.einsum("i,nij->nj", par.a[H], Sigma_H)          # (n, h)
    c = -np.einsum("nj,njk,nmk->nm", a_Sigma, S1_inv, e_H)

    # Ψ2: termo dos indutores e termo de μ − z̄_ij com (Λ/2 + Σ)⁻¹.
    dz = par.Z_u[:, None, :] - par.Z_u[None, :, :]
    termo_indutores = np.exp(-0.25 * (dz ** 2 / lam).sum(-1))   # (M, M)
    S2_inv = np.linalg.inv(np.diag(lam_H / 2)[None] + Sigma_H)
    # (μ − z̄_ij) = (e_i + e_j)/2; forma quadrática pela expansão em e_i e e_j.
    G_fora = np.einsum("nik,njk->nij", e_fora / (lam[fora] / 2), e_fora)
    QE_H = np.einsum("nkl,nml->nmk", S2_inv, e_H)
    G_H = np.einsum("nik,njk->nij", e_H, QE_H)
    G = G_fora + G_H
    diag = np.einsum("nii->ni", G)
    quad2 = 0.25 * (diag[:, :, None] + diag[:, None, :] + 2 * G)
    det2 = np.linalg.slogdet(np.eye(len(H))[None] + 2 * Sigma_H / lam_H[None, None, :])[1]
    Psi2 = (par.sigma_f2 ** 2) * np.exp(-0.5 * det2)[:, None, None] * termo_indutores[None] * np.exp(-0.5 * quad2)
    return psi1, Psi2, c


def momentos_t(par: ParametrosSVGP, mu: np.ndarray, Sigma_H: np.ndarray, H: np.ndarray):
    """E[t] e Var[t] (escala padronizada) para refeições com o mesmo H."""
    psi1, Psi2, c = estatisticas_psi(par, mu, Sigma_H, H)
    psi1_A = psi1 @ par.A
    media = mu @ par.a + par.b + psi1_A
    a_H = par.a[H]
    variancia = (
        par.ruido + par.sigma_f2 + par.jitter
        - np.einsum("ij,nji->n", par.B, Psi2)
        + np.einsum("i,nij,j->n", a_H, Sigma_H, a_H)
        + np.einsum("i,nij,j->n", par.A, Psi2, par.A)
        - psi1_A ** 2
        + 2 * np.einsum("i,ni,ni->n", par.A, psi1, c)
    )
    return media, variancia


def prever_com_entrada_incerta_svgp(par: ParametrosSVGP, Eg: np.ndarray, Cg: np.ndarray,
                                    P: np.ndarray, C: np.ndarray, M: np.ndarray):
    """Média e variância da preditiva em gramas, com a entrada de média Eg e covariância Cg.

    Cg (N, K, K) é zero fora das ocultas M; as refeições são agrupadas pelo conjunto oculto.
    """
    K = Eg.shape[1]
    mu = (montar_Z(Eg, P, C) - par.media_z) / par.desvio_z
    s = par.desvio_z[:K]
    Sigma = Cg / (s[:, None] * s[None, :])
    M = np.asarray(M, dtype=bool)
    media, variancia = np.empty(len(Eg)), np.empty(len(Eg))
    chaves = np.packbits(M, axis=1)
    _, grupo = np.unique(chaves, axis=0, return_inverse=True)
    grupo = grupo.reshape(-1)
    for g in np.unique(grupo):
        linhas = np.flatnonzero(grupo == g)
        H = np.flatnonzero(M[linhas[0]])
        for inicio in range(0, len(linhas), BLOCO_REFEICOES):
            sel = linhas[inicio:inicio + BLOCO_REFEICOES]
            Sigma_H = Sigma[sel][:, H][:, :, H]
            media[sel], variancia[sel] = momentos_t(par, mu[sel], Sigma_H, H)
    return media * par.desvio_y + par.media_y, variancia * par.desvio_y ** 2
