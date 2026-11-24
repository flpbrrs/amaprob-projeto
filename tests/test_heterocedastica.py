"""Testes da regressão linear bayesiana heterocedástica."""

import numpy as np

from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.blr_heterocedastica import (
    RegressaoLinearBayesianaHeterocedastica, prever_com_entrada_incerta_heterocedastica, variaveis_do_ruido,
)
from amaprob.regression.propagacao_analitica import prever_com_entrada_incerta

K = 4


def _dados(n: int, semente: int, desvio_por_gaveta=(1.0, 1.0, 1.0, 1.0), desvio_por_ocasiao=(5.0, 5.0, 5.0, 5.0, 5.0)):
    """y linear nas gramas, com desvio do ruído = produto de um fator por gaveta presente e um por ocasião."""
    rng = np.random.default_rng(semente)
    P = rng.random((n, K)) < 0.5
    P[~P.any(axis=1), 0] = True
    G = np.where(P, rng.gamma(2.0, 40.0, (n, K)), 0.0)
    ocasiao = rng.integers(0, 5, n)
    C = np.zeros((n, 13))
    C[np.arange(n), ocasiao] = 1.0
    C[:, 5:] = rng.standard_normal((n, 8))
    log_desvio = np.log(np.asarray(desvio_por_ocasiao, dtype=float))[ocasiao] + (P * np.log(desvio_por_gaveta)).sum(axis=1)
    desvio = np.exp(log_desvio)
    y = 10.0 + G @ np.array([0.5, 0.1, 0.3, 0.0]) + desvio * rng.standard_normal(n)
    return montar_Z(G, P, C), y, desvio ** 2, G, P, C


def test_sigma_constante_reproduz_a_blr():
    Z, y, *_ = _dados(3_000, 0)
    blr = RegressaoLinearBayesiana().fit(Z, y)
    hetero = RegressaoLinearBayesianaHeterocedastica(n_gavetas=K)
    hetero.desvio_z_ = np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
    hetero._ajustar_pesos(Z, y, np.full(len(y), 1.0 / blr.beta_), alpha=1.0)

    np.testing.assert_allclose(hetero.alpha_, blr.alpha_, rtol=1e-5)
    np.testing.assert_allclose(hetero.m_, blr.m_, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(hetero.S_, blr.S_, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(hetero.media_y_, blr.media_y_)
    np.testing.assert_allclose(hetero.media_z_, blr.media_z_)


def test_ruido_homocedastico_da_preditiva_da_blr():
    """Sem heterocedasticidade nos dados, γ ≈ 0 e a preditiva coincide com a da BLR (regressão linear bayesiana)."""
    Z, y, *_ = _dados(20_000, 1)
    blr = RegressaoLinearBayesiana().fit(Z, y)
    hetero = RegressaoLinearBayesianaHeterocedastica(n_gavetas=K, lam=1.0).fit(Z, y)
    media_b, var_b = blr.predict(Z[:500])
    media_h, var_h = hetero.predict(Z[:500])
    np.testing.assert_allclose(media_h, media_b, atol=0.05)
    # Grupos pequenos (as 4 gavetas presentes: ~1/16 das refeições) oscilam por amostragem.
    np.testing.assert_allclose(var_h, var_b, rtol=0.1)
    np.testing.assert_allclose(var_h.mean(), var_b.mean(), rtol=0.02)


def test_gamma_recupera_variancia_conhecida():
    desvio_por_gaveta = (1.0, 2.0, 1.0, 1.5)
    desvio_por_ocasiao = (2.0, 2.0, 4.0, 2.0, 2.0)
    Z, y, variancia, *_ = _dados(60_000, 2, desvio_por_gaveta, desvio_por_ocasiao)
    hetero = RegressaoLinearBayesianaHeterocedastica(n_gavetas=K, lam=0.1).fit(Z, y)
    estimada = hetero.variancia_ruido(Z)
    # A parametrização tem colinearidade (ocasião e tamanho somam 1), mas σ²(h) é identificável.
    np.testing.assert_allclose(np.median(estimada / variancia), 1.0, atol=0.03)
    assert np.quantile(np.abs(np.log(estimada / variancia)), 0.95) < 0.1
    # E a variância por grupo (ocasião 2, quatro vezes maior) sai na ordem certa.
    H = variaveis_do_ruido(Z, K)
    razao = estimada[H[:, K + 2] == 1].mean() / estimada[H[:, K + 0] == 1].mean()
    real = variancia[H[:, K + 2] == 1].mean() / variancia[H[:, K + 0] == 1].mean()
    np.testing.assert_allclose(razao, real, rtol=0.1)


def test_analitica_troca_so_o_ruido():
    """Sem incerteza na entrada, a analítica coincide com o predict; com incerteza, soma o mesmo
    termo da entrada que a forma homocedástica."""
    Z, y, _, G, P, C = _dados(5_000, 3, (1.0, 2.0, 1.0, 1.5))
    hetero = RegressaoLinearBayesianaHeterocedastica(n_gavetas=K).fit(Z, y)
    n = 200
    Cg0 = np.zeros((n, K, K))
    media, var = prever_com_entrada_incerta_heterocedastica(hetero, G[:n], Cg0, P[:n], C[:n])
    media_ref, var_ref = hetero.predict(Z[:n])
    np.testing.assert_allclose(media, media_ref)
    np.testing.assert_allclose(var, var_ref)

    rng = np.random.default_rng(4)
    A = rng.normal(0.0, 10.0, (n, K, K)) * (P[:n, :, None] & P[:n, None, :])
    Cg = np.einsum("nij,nkj->nik", A, A)
    _, var_incerta = prever_com_entrada_incerta_heterocedastica(hetero, G[:n], Cg, P[:n], C[:n])
    _, var_entrada = prever_com_entrada_incerta(hetero.sem_ruido(), G[:n], Cg, P[:n], C[:n])
    _, var_sem_entrada = prever_com_entrada_incerta(hetero.sem_ruido(), G[:n], Cg0, P[:n], C[:n])
    np.testing.assert_allclose(var_incerta - var_ref, var_entrada - var_sem_entrada, rtol=1e-8, atol=1e-10)

    # Monte Carlo: amostrando a entrada gaussiana, a média das variâncias + variância das médias.
    i = 0
    amostras = rng.multivariate_normal(G[i], Cg[i], 200_000)
    Zs = montar_Z(amostras, np.tile(P[i], (len(amostras), 1)), np.tile(C[i], (len(amostras), 1)))
    medias_s, variancias_s = hetero.predict(Zs)
    np.testing.assert_allclose(variancias_s.mean() + medias_s.var(), var_incerta[i], rtol=0.01)
