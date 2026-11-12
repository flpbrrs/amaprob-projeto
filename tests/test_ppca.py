"""PPCA-EM: solução convergida, recuperação, monotonia do EM e amostragem pela condicional."""

import numpy as np
import pytest

from amaprob.imputation.ppca import PPCA

AJUSTE_APERTADO = {"tolerancia": 1e-12, "max_iteracoes": 5_000}


def _gerar(N, W, media, sigma, rng):
    z = rng.standard_normal((N, W.shape[1]))
    return z @ W.T + media + sigma * rng.standard_normal((N, len(media)))


def _sem_faltantes(X):
    return np.ones(X.shape, dtype=bool), np.zeros(X.shape, dtype=bool), np.zeros((len(X), 1))


def _parametros(q, W, media, sigma2):
    modelo = PPCA(q=q)
    modelo.W_, modelo.media_, modelo.sigma2_ = np.asarray(W, float), np.asarray(media, float), float(sigma2)
    return modelo


def _erro_relativo(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.fixture(scope="module")
def sintetico():
    rng = np.random.default_rng(3)
    K, q = 8, 2
    W = rng.normal(0, 1, size=(K, q)) * np.array([2.0, 1.2])
    media = rng.uniform(3, 5, size=K)
    X = _gerar(20_000, W, media, 0.4, rng)
    P = rng.random(X.shape) < 0.5
    return X, P, W, media, 0.4**2


def test_forma_fechada_sem_faltantes():
    rng = np.random.default_rng(0)
    W = np.c_[rng.normal(0, 1, 6) * 3.0, rng.normal(0, 1, 6) * 1.5]
    X = _gerar(5_000, W, np.full(6, 4.0), 0.5, rng)
    P, M, C = _sem_faltantes(X)
    modelo = PPCA(q=2, **AJUSTE_APERTADO).fit(X, P, M, C)

    autovalores, autovetores = np.linalg.eigh(np.cov(X, rowvar=False, bias=True))
    autovalores, autovetores = autovalores[::-1], autovetores[:, ::-1]
    sigma2 = autovalores[2:].mean()
    WWt = autovetores[:, :2] @ np.diag(autovalores[:2] - sigma2) @ autovetores[:, :2].T

    assert abs(modelo.sigma2_ - sigma2) / sigma2 < 1e-3
    assert _erro_relativo(modelo.W_ @ modelo.W_.T, WWt) < 1e-3
    np.testing.assert_allclose(modelo.media_, X.mean(axis=0), rtol=1e-6)


def test_recuperacao_com_metade_ausente(sintetico):
    X, P, W, media, sigma2 = sintetico
    modelo = PPCA(q=2, **AJUSTE_APERTADO).fit(np.where(P, X, 0.0), P, np.zeros_like(P), np.zeros((len(X), 1)))
    covariancia = W @ W.T + sigma2 * np.eye(len(media))
    ajustada = modelo.W_ @ modelo.W_.T + modelo.sigma2_ * np.eye(len(media))
    assert _erro_relativo(ajustada, covariancia) < 0.10
    assert _erro_relativo(modelo.media_, media) < 0.10


def test_log_verossimilhanca_monotona(sintetico):
    X, P, *_ = sintetico
    modelo = PPCA(q=2).fit(np.where(P, X, 0.0), P, np.zeros_like(P), np.zeros((len(X), 1)))
    historico = modelo.historico_log_verossimilhanca_
    assert modelo.n_iteracoes_ == len(historico) - 1 > 1
    assert np.all(np.diff(historico) >= -1e-8 * np.abs(historico[:-1]))


def test_ausentes_fora_da_verossimilhanca(sintetico):
    X, P, *_ = sintetico
    M, C = np.zeros_like(P), np.zeros((len(X), 1))
    X_zeros = np.where(P, X, 0.0)
    X_lixo = np.where(P, X, np.random.default_rng(1).uniform(-50, 50, size=X.shape))
    a, b = PPCA(q=2).fit(X_zeros, P, M, C), PPCA(q=2).fit(X_lixo, P, M, C)
    np.testing.assert_array_equal(a.W_, b.W_)
    np.testing.assert_array_equal(a.historico_log_verossimilhanca_, b.historico_log_verossimilhanca_)
    M_meia = P & (np.random.default_rng(2).random(P.shape) < 0.5)
    amostras = [m.sample(x, P[:200], M_meia[:200], C[:200], 5, np.random.default_rng(0))
                for m, x in ((a, X_zeros[:200]), (b, X_lixo[:200]))]
    np.testing.assert_array_equal(*amostras)


def test_reprodutivel_pela_semente(sintetico):
    X, P, *_ = sintetico
    args = (np.where(P, X, 0.0), P, np.zeros_like(P), np.zeros((len(X), 1)))
    np.testing.assert_array_equal(PPCA(q=2, semente=7).fit(*args).W_, PPCA(q=2, semente=7).fit(*args).W_)


PARAMETROS = {
    "W": [[1.0, 0.2], [0.8, -0.5], [0.0, 1.0], [-0.6, 0.4], [0.3, 0.9]],
    "media": [10.0, 9.0, 11.0, 10.5, 9.5],
    "sigma2": 0.3,
}


def _momentos_das_amostras(x, P, M, n=200_000):
    modelo = _parametros(2, **PARAMETROS)
    amostras = modelo.sample(x[None], P[None], M[None], np.zeros((1, 1)), n, np.random.default_rng(4))[:, 0]
    ocultas = amostras[:, M]
    return ocultas.mean(axis=0), np.cov(ocultas, rowvar=False)


def test_momentos_da_condicional():
    W, media = np.array(PARAMETROS["W"]), np.array(PARAMETROS["media"])
    covariancia = W @ W.T + PARAMETROS["sigma2"] * np.eye(5)
    P = np.ones(5, dtype=bool)
    M = np.array([True, False, False, True, False])
    o, h = ~M, M
    x = np.where(M, 0.0, np.array([10.4, 8.1, 12.0, 10.0, 10.3]))

    ganho = covariancia[np.ix_(h, o)] @ np.linalg.inv(covariancia[np.ix_(o, o)])
    media_condicional = media[h] + ganho @ (x[o] - media[o])
    cov_condicional = covariancia[np.ix_(h, h)] - ganho @ covariancia[np.ix_(o, h)]

    media_amostral, cov_amostral = _momentos_das_amostras(x, P, M)
    np.testing.assert_allclose(media_amostral, media_condicional, atol=0.01)
    np.testing.assert_allclose(cov_amostral, cov_condicional, atol=0.01)


def test_taxa_1_pela_marginal():
    W, media = np.array(PARAMETROS["W"]), np.array(PARAMETROS["media"])
    covariancia = W @ W.T + PARAMETROS["sigma2"] * np.eye(5)
    P = np.array([True, True, False, True, False])
    media_amostral, cov_amostral = _momentos_das_amostras(np.zeros(5), P, P.copy())
    np.testing.assert_allclose(media_amostral, media[P], atol=0.01)
    np.testing.assert_allclose(cov_amostral, covariancia[np.ix_(P, P)], atol=0.02)
