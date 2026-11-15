"""FA-EM: recuperação com ruído heterogêneo, FA × PPCA, monotonia, piso e amostragem pela condicional."""

import numpy as np
import pytest

from amaprob.imputation.fa import FA
from amaprob.imputation.ppca import PPCA

AJUSTE_APERTADO = {"tolerancia": 1e-12, "max_iteracoes": 5_000}


def _gerar(N, W, media, psi, rng):
    z = rng.standard_normal((N, W.shape[1]))
    return z @ W.T + media + np.sqrt(psi) * rng.standard_normal((N, len(media)))


def _parametros(q, W, media, psi):
    modelo = FA(q=q)
    modelo.W_, modelo.media_, modelo.psi_ = np.asarray(W, float), np.asarray(media, float), np.asarray(psi, float)
    return modelo


def _erro_relativo(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def _sem_mascara(P):
    return np.zeros_like(P), np.zeros((len(P), 1))


@pytest.fixture(scope="module")
def sintetico():
    rng = np.random.default_rng(3)
    K, q = 8, 2
    W = rng.normal(0, 1, size=(K, q)) * np.array([2.0, 1.2])
    media = rng.uniform(3, 5, size=K)
    psi = rng.uniform(0.1, 0.8, size=K)
    X = _gerar(20_000, W, media, psi, rng)
    P = rng.random(X.shape) < 0.5
    return np.where(P, X, 0.0), P, W, media, psi


def test_recuperacao_com_ruido_heterogeneo(sintetico):
    X, P, W, media, psi = sintetico
    modelo = FA(q=2, **AJUSTE_APERTADO).fit(X, P, *_sem_mascara(P))
    assert _erro_relativo(modelo.W_ @ modelo.W_.T + np.diag(modelo.psi_), W @ W.T + np.diag(psi)) < 0.10
    assert _erro_relativo(modelo.media_, media) < 0.10
    # O erro de estimação de ψ_k escala com a variância da gaveta, e não com ψ_k: uma gaveta quase
    # toda explicada pelos fatores tem ψ_k pequeno e relativamente impreciso, que encolhe com N.
    variancia_da_gaveta = (W**2).sum(axis=1) + psi
    assert np.all(np.abs(modelo.psi_ - psi) < 0.02 * variancia_da_gaveta)
    assert len(modelo.gavetas_no_piso_) == 0


def test_fa_supera_ppca_com_ruido_heterogeneo(sintetico):
    X, P, *_ = sintetico
    fa = FA(q=2).fit(X, P, *_sem_mascara(P))
    ppca = PPCA(q=2).fit(X, P, *_sem_mascara(P))
    assert fa.historico_log_verossimilhanca_[-1] > ppca.historico_log_verossimilhanca_[-1]


def test_log_verossimilhanca_monotona(sintetico):
    X, P, *_ = sintetico
    modelo = FA(q=2).fit(X, P, *_sem_mascara(P))
    historico = modelo.historico_log_verossimilhanca_
    assert modelo.n_iteracoes_ == len(historico) - 1 > 1
    assert np.all(np.diff(historico) >= -1e-8 * np.abs(historico[:-1]))


def test_ausentes_fora_da_verossimilhanca(sintetico):
    X_zeros, P, *_ = sintetico
    M, C = _sem_mascara(P)
    X_lixo = np.where(P, X_zeros, np.random.default_rng(1).uniform(-50, 50, size=P.shape))
    a, b = FA(q=2).fit(X_zeros, P, M, C), FA(q=2).fit(X_lixo, P, M, C)
    np.testing.assert_array_equal(a.W_, b.W_)
    np.testing.assert_array_equal(a.psi_, b.psi_)
    np.testing.assert_array_equal(a.historico_log_verossimilhanca_, b.historico_log_verossimilhanca_)
    M_meia = P & (np.random.default_rng(2).random(P.shape) < 0.5)
    amostras = [m.sample(x, P[:200], M_meia[:200], C[:200], 5, np.random.default_rng(0))
                for m, x in ((a, X_zeros[:200]), (b, X_lixo[:200]))]
    np.testing.assert_array_equal(*amostras)


def test_reprodutivel_pela_semente(sintetico):
    X, P, *_ = sintetico
    args = (X, P, *_sem_mascara(P))
    a, b = FA(q=2, semente=7).fit(*args), FA(q=2, semente=7).fit(*args)
    np.testing.assert_array_equal(a.W_, b.W_)
    np.testing.assert_array_equal(a.psi_, b.psi_)


def test_gaveta_constante_no_piso(sintetico):
    X, P, *_ = sintetico
    X = X.copy()
    X[:, 0] = np.where(P[:, 0], 4.0, 0.0)
    modelo = FA(q=2).fit(X, P, *_sem_mascara(P))
    assert 0 in modelo.gavetas_no_piso_
    assert np.all(np.isfinite(modelo.W_)) and np.all(np.isfinite(modelo.psi_)) and np.all(modelo.psi_ > 0)
    M = P[:200] & (np.random.default_rng(2).random(P[:200].shape) < 0.5)
    amostras = modelo.sample(X[:200], P[:200], M, np.zeros((200, 1)), 5, np.random.default_rng(0))
    assert np.all(np.isfinite(amostras))


def test_gaveta_sem_observados(sintetico):
    X, P, *_ = sintetico
    P = P.copy()
    P[:, 0] = False
    X = np.where(P, X, 0.0)
    modelo = FA(q=2).fit(X, P, *_sem_mascara(P))
    for parametro in (modelo.W_, modelo.psi_, modelo.media_, modelo.historico_log_verossimilhanca_):
        assert np.all(np.isfinite(parametro))
    M = P[:200] & (np.random.default_rng(2).random(P[:200].shape) < 0.5)
    amostras = modelo.sample(X[:200], P[:200], M, np.zeros((200, 1)), 5, np.random.default_rng(0))
    assert np.all(np.isfinite(amostras))


PARAMETROS = {
    "W": [[1.0, 0.2], [0.8, -0.5], [0.0, 1.0], [-0.6, 0.4], [0.3, 0.9]],
    "media": [10.0, 9.0, 11.0, 10.5, 9.5],
    "psi": [0.1, 0.5, 0.2, 0.8, 0.3],
}


def _covariancia():
    W = np.array(PARAMETROS["W"])
    return W @ W.T + np.diag(PARAMETROS["psi"])


def _momentos_das_amostras(x, P, M, n=200_000):
    modelo = _parametros(2, **PARAMETROS)
    amostras = modelo.sample(x[None], P[None], M[None], np.zeros((1, 1)), n, np.random.default_rng(4))[:, 0]
    ocultas = amostras[:, M]
    return ocultas.mean(axis=0), np.cov(ocultas, rowvar=False)


def test_momentos_da_condicional():
    media, covariancia = np.array(PARAMETROS["media"]), _covariancia()
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
    media, covariancia = np.array(PARAMETROS["media"]), _covariancia()
    P = np.array([True, True, False, True, False])
    media_amostral, cov_amostral = _momentos_das_amostras(np.zeros(5), P, P.copy())
    np.testing.assert_allclose(media_amostral, media[P], atol=0.01)
    np.testing.assert_allclose(cov_amostral, covariancia[np.ix_(P, P)], atol=0.02)
