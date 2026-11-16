"""Média condicional no PPCA, na FA e na porção média: redução ao modelo atual, recuperação, colinearidade, monotonia."""

import numpy as np
import pytest

from amaprob.imputation.baseline import PorcaoMedia
from amaprob.imputation.fa import FA
from amaprob.imputation.ppca import PPCA

MODELOS = {"ppca": PPCA, "fa": FA}


def _ruido(modelo):
    return modelo.sigma2_ if isinstance(modelo, PPCA) else modelo.psi_


@pytest.fixture(scope="module")
def sintetico():
    """Porções cuja média depende das outras gavetas presentes e de um contexto categórico e contínuo."""
    rng = np.random.default_rng(5)
    N, K, q = 20_000, 6, 2
    P = rng.random((N, K)) < 0.6
    categoria = rng.integers(0, 3, size=N)
    C = np.c_[np.eye(3)[categoria], rng.normal(size=N)]  # indicadoras que somam 1 + uma contínua
    W = rng.normal(0, 0.5, size=(K, q))
    intercepto = rng.uniform(3, 5, size=K)
    efeito_P = rng.normal(0, 0.4, size=(K, K)) * (1 - np.eye(K))
    efeito_C = np.c_[rng.normal(0, 0.4, size=(K, 3)), rng.normal(0, 0.3, size=K)]
    media = intercepto + P @ efeito_P.T + C @ efeito_C.T
    X = media + rng.standard_normal((N, q)) @ W.T + 0.3 * rng.standard_normal((N, K))
    return np.where(P, X, 0.0), P, C, media


@pytest.mark.parametrize("nome", MODELOS)
def test_reducao_ao_modelo_atual(nome):
    rng = np.random.default_rng(0)
    N, K = 3_000, 5
    X = np.abs(rng.normal(3, 1, size=(N, K)))
    P = np.ones((N, K), dtype=bool)
    M = rng.random((N, K)) < 0.4
    C = np.full((N, 2), 7.0)
    atual = MODELOS[nome](q=2).fit(X, P, M, C)
    condicional = MODELOS[nome](q=2, media_condicional=True).fit(X, P, M, C)
    np.testing.assert_allclose(condicional.W_, atual.W_, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(_ruido(condicional), _ruido(atual), rtol=1e-6)
    np.testing.assert_allclose(condicional.coeficientes_media_[:, 0], atual.media_, rtol=1e-6)
    np.testing.assert_allclose(condicional.coeficientes_media_[:, 1:], 0.0, atol=1e-9)
    amostras = [m.sample(X[:100], P[:100], M[:100], C[:100], 5, np.random.default_rng(1)) for m in (atual, condicional)]
    np.testing.assert_allclose(*amostras, rtol=1e-6)


@pytest.mark.parametrize("nome", MODELOS)
def test_recuperacao_e_taxa_1(sintetico, nome):
    X, P, C, media = sintetico
    M = P & (np.random.default_rng(1).random(P.shape) < 0.5)
    modelo = MODELOS[nome](q=2, media_condicional=True).fit(X, P, M, C)
    atual = MODELOS[nome](q=2).fit(X, P, M, C)

    U = np.c_[np.ones(len(P)), (np.c_[P, C] - modelo.padronizacao_.media) / modelo.padronizacao_.desvio]
    media_ajustada = U @ modelo.coeficientes_media_.T
    assert np.abs(media_ajustada - media)[P].mean() < 0.05

    teste = slice(0, 2_000)
    taxa_1 = P[teste]
    erros = []
    for m in (modelo, atual):
        amostras = m.sample(X[teste], P[teste], taxa_1, C[teste], 200, np.random.default_rng(2))
        erros.append(np.abs(amostras.mean(axis=0) - media[teste])[taxa_1].mean())
    assert erros[0] < 0.5 * erros[1]


@pytest.mark.parametrize("nome", MODELOS)
def test_colunas_colineares(sintetico, nome):
    X, P, C, _ = sintetico
    P = P.copy()
    P[:, 1] &= ~P[:, 0]  # gavetas 0 e 1 nunca aparecem juntas
    X = np.where(P, X, 0.0)
    M = P & (np.random.default_rng(3).random(P.shape) < 0.3)
    modelo = MODELOS[nome](q=2, media_condicional=True).fit(X, P, M, C)
    assert np.isfinite(modelo.coeficientes_media_).all() and np.isfinite(modelo.W_).all()
    assert np.isfinite(modelo.sample(X[:300], P[:300], M[:300], C[:300], 5, np.random.default_rng(0))).all()


@pytest.mark.parametrize("nome", MODELOS)
def test_ausentes_fora_da_verossimilhanca(sintetico, nome):
    X, P, C, _ = sintetico
    M = np.zeros_like(P)
    X_lixo = np.where(P, X, np.random.default_rng(4).uniform(-50, 50, size=X.shape))
    a = MODELOS[nome](q=2, media_condicional=True).fit(X, P, M, C)
    b = MODELOS[nome](q=2, media_condicional=True).fit(X_lixo, P, M, C)
    np.testing.assert_array_equal(a.coeficientes_media_, b.coeficientes_media_)
    np.testing.assert_array_equal(a.W_, b.W_)
    M_meia = P & (np.random.default_rng(5).random(P.shape) < 0.5)
    amostras = [m.sample(x[:200], P[:200], M_meia[:200], C[:200], 5, np.random.default_rng(0))
                for m, x in ((a, X), (b, X_lixo))]
    np.testing.assert_array_equal(*amostras)


@pytest.mark.parametrize("nome", MODELOS)
def test_reprodutivel_pela_semente(sintetico, nome):
    X, P, C, _ = sintetico
    args = (X, P, np.zeros_like(P), C)
    a = MODELOS[nome](q=2, semente=7, media_condicional=True).fit(*args)
    b = MODELOS[nome](q=2, semente=7, media_condicional=True).fit(*args)
    np.testing.assert_array_equal(a.coeficientes_media_, b.coeficientes_media_)
    np.testing.assert_array_equal(a.W_, b.W_)


@pytest.mark.parametrize("nome", MODELOS)
def test_objetivo_penalizado_monotono(sintetico, nome):
    X, P, C, _ = sintetico
    M = P & (np.random.default_rng(6).random(P.shape) < 0.3)
    modelo = MODELOS[nome](q=2, media_condicional=True).fit(X, P, M, C)
    historico = modelo.historico_objetivo_
    assert modelo.n_iteracoes_ == len(historico) - 1 > 1
    assert np.all(np.diff(historico) >= -1e-8 * np.abs(historico[:-1]))


def test_porcao_condicional_reduz_a_porcao_media():
    rng = np.random.default_rng(0)
    N, K = 3_000, 5
    X = np.abs(rng.normal(3, 1, size=(N, K)))
    P = np.ones((N, K), dtype=bool)
    M = rng.random((N, K)) < 0.4
    C = np.full((N, 2), 7.0)
    atual = PorcaoMedia().fit(X, P, M, C)
    condicional = PorcaoMedia(media_condicional=True).fit(X, P, M, C)
    np.testing.assert_allclose(condicional.coeficientes_media_[:, 0], atual.media_, rtol=1e-9)
    np.testing.assert_allclose(condicional.desvio_, atual.desvio_, rtol=1e-9)
    amostras = [m.sample(X[:100], P[:100], M[:100], C[:100], 5, np.random.default_rng(1)) for m in (atual, condicional)]
    np.testing.assert_allclose(*amostras, rtol=1e-9)


def test_porcao_condicional_recupera_a_media(sintetico):
    X, P, C, media = sintetico
    M = P & (np.random.default_rng(1).random(P.shape) < 0.5)
    modelo = PorcaoMedia(media_condicional=True).fit(X, P, M, C)
    U = np.c_[np.ones(len(P)), (np.c_[P, C] - modelo.padronizacao_.media) / modelo.padronizacao_.desvio]
    assert np.abs(U @ modelo.coeficientes_media_.T - media)[P].mean() < 0.05
    teste = slice(0, 2_000)
    erros = [np.abs(m.sample(X[teste], P[teste], P[teste], C[teste], 200, np.random.default_rng(2)).mean(axis=0)
                    - media[teste])[P[teste]].mean()
             for m in (modelo, PorcaoMedia().fit(X, P, M, C))]
    assert erros[0] < 0.5 * erros[1]
