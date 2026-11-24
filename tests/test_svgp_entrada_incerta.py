"""Testes do braço analítico do SVGP (entrada gaussiana, aproximação por momentos)."""

import numpy as np
import pytest

from amaprob.regression.base import montar_Z
from amaprob.regression.svgp import SVGP
from amaprob.regression.svgp_entrada_incerta import (
    ParametrosSVGP, estatisticas_psi, extrair, kernel, prever_com_entrada_incerta_svgp, prever_entrada_fixa,
)

K = 3


def _dados(n: int, semente: int):
    rng = np.random.default_rng(semente)
    P = rng.random((n, K)) < 0.7
    P[~P.any(axis=1), 0] = True
    G = np.where(P, rng.gamma(3.0, 40.0, (n, K)), 0.0)
    C = np.zeros((n, 13))
    C[np.arange(n), rng.integers(0, 5, n)] = 1.0
    C[:, 5:] = rng.standard_normal((n, 8))
    # Não linear nas gramas, para o kernel ter o que corrigir sobre a média linear.
    y = 0.3 * G[:, 0] + 20 * np.sin(G[:, 1] / 40) + 0.002 * G[:, 2] ** 2 + rng.normal(0, 3, n)
    return montar_Z(G, P, C), y, G, P, C


@pytest.fixture(scope="module")
def svgp():
    Z, y, *_ = _dados(3_000, 0)
    return SVGP(n_indutores=24, epocas=15, tamanho_lote=256, taxa_aprendizado=0.03).fit(Z, y)


def test_entrada_fixa_reproduz_o_predict(svgp):
    Z, *_ = _dados(500, 1)
    par = extrair(svgp)
    media, var = prever_entrada_fixa(par, Z)
    media_ref, var_ref = svgp.predict(Z)
    np.testing.assert_allclose(media, media_ref, rtol=0, atol=1e-8 * np.abs(media_ref).max())
    np.testing.assert_allclose(var, var_ref, rtol=1e-8)

    # Σ = 0 pelo caminho da entrada incerta (ψ1 = k, Ψ2 = k kᵀ) dá o mesmo.
    _, _, G, P, C = _dados(500, 1)
    M = np.zeros_like(P)
    media0, var0 = prever_com_entrada_incerta_svgp(par, G, np.zeros((len(G), K, K)), P, C, M)
    np.testing.assert_allclose(media0, media_ref, rtol=0, atol=1e-8 * np.abs(media_ref).max())
    np.testing.assert_allclose(var0, var_ref, rtol=1e-8)
    # Com ocultas declaradas mas covariância nula, também (o bloco H entra com Σ_H = 0).
    M[:, 0] = P[:, 0]
    media1, var1 = prever_com_entrada_incerta_svgp(par, G, np.zeros((len(G), K, K)), P, C, M)
    np.testing.assert_allclose(media1, media_ref, rtol=0, atol=1e-8 * np.abs(media_ref).max())
    np.testing.assert_allclose(var1, var_ref, rtol=1e-8)


def test_entrada_gaussiana_bate_com_monte_carlo(svgp):
    _, _, G, P, C = _dados(6, 2)
    P[:] = True
    G = np.where(P, np.maximum(G, 60.0), 0.0)
    M = np.zeros_like(P)
    M[:, :2] = True  # as duas primeiras gavetas ocultas
    rng = np.random.default_rng(5)
    Cg = np.zeros((len(G), K, K))
    for n in range(len(G)):
        A = rng.normal(0.0, 25.0, (2, 2))
        Cg[n, :2, :2] = A @ A.T + np.diag([200.0, 300.0])
    par = extrair(svgp)
    media, var = prever_com_entrada_incerta_svgp(par, G, Cg, P, C, M)

    for n in range(len(G)):
        amostras = np.tile(G[n], (5_000, 1))
        amostras[:, :2] = rng.multivariate_normal(G[n, :2], Cg[n, :2, :2], 5_000)
        Zs = montar_Z(amostras, np.tile(P[n], (5_000, 1)), np.tile(C[n], (5_000, 1)))
        m_s, v_s = svgp.predict(Zs)
        np.testing.assert_allclose(media[n], m_s.mean(), rtol=0.02, atol=0.02 * np.sqrt(var[n]))
        np.testing.assert_allclose(var[n], v_s.mean() + m_s.var(), rtol=0.02)


def test_psi1_e_psi2_batem_com_monte_carlo_em_2d():
    rng = np.random.default_rng(0)
    Z_u = rng.normal(0.0, 1.0, (5, 2))
    par = ParametrosSVGP(
        a=np.array([0.4, -0.7]), b=0.0, lengthscale2=np.array([0.8, 1.5]) ** 2, sigma_f2=1.3, ruido=0.1,
        jitter=0.0, Z_u=Z_u, A=np.zeros(5), B=np.zeros((5, 5)), media_z=np.zeros(2), desvio_z=np.ones(2),
        media_y=0.0, desvio_y=1.0,
    )
    mu = np.array([[0.3, -0.2]])
    Sigma = np.array([[[0.5, 0.2], [0.2, 0.4]]])
    H = np.array([0, 1])
    psi1, Psi2, c = estatisticas_psi(par, mu, Sigma, H)

    x = rng.multivariate_normal(mu[0], Sigma[0], 400_000)
    k = kernel(x, Z_u, par.lengthscale2, par.sigma_f2)  # (S, M)
    np.testing.assert_allclose(psi1[0], k.mean(axis=0), rtol=0.01)
    np.testing.assert_allclose(Psi2[0], k.T @ k / len(x), rtol=0.01)
    # c_i ψ1_i = Cov(aᵀφ, k_i).
    linear = x @ par.a
    cov = ((linear - linear.mean())[:, None] * (k - k.mean(axis=0))).mean(axis=0)
    np.testing.assert_allclose(c[0] * psi1[0], cov, rtol=0.03, atol=1e-3)

    # Só uma dimensão incerta (H = [0]): a outra fica fixa em μ.
    psi1_h, Psi2_h, _ = estatisticas_psi(par, mu, Sigma[:, :1, :1], np.array([0]))
    x1 = np.c_[rng.normal(mu[0, 0], np.sqrt(Sigma[0, 0, 0]), 400_000), np.full(400_000, mu[0, 1])]
    k1 = kernel(x1, Z_u, par.lengthscale2, par.sigma_f2)
    np.testing.assert_allclose(psi1_h[0], k1.mean(axis=0), rtol=0.01)
    np.testing.assert_allclose(Psi2_h[0], k1.T @ k1 / len(x1), rtol=0.01)
