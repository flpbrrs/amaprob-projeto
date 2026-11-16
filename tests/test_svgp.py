"""SVGP: reta, parte não linear contra a regressão linear, incerteza fora dos dados, semente e coluna constante."""

import numpy as np
import pytest

from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.svgp import SVGP

AJUSTE_RAPIDO = {"n_indutores": 32, "epocas": 40, "tamanho_lote": 256, "taxa_aprendizado": 0.05}


def _nll(y, media, variancia):
    return np.mean(0.5 * np.log(2 * np.pi * variancia) + 0.5 * (y - media) ** 2 / variancia)


def _dados_lineares(n, desvio_ruido=2.0, semente=0):
    rng = np.random.default_rng(semente)
    Z = rng.normal(size=(n, 3)) * [1.0, 10.0, 0.1] + 5.0
    media_verdadeira = Z @ [3.0, -0.5, 20.0] + 7.0
    return Z, media_verdadeira + rng.normal(0, desvio_ruido, size=n), media_verdadeira


def _dados_nao_lineares(n, semente):
    rng = np.random.default_rng(semente)
    Z = rng.uniform(-3, 3, size=(n, 2))
    return Z, 2 * Z[:, 0] + 3 * np.sin(2 * Z[:, 1]) + rng.normal(0, 0.3, size=n)


@pytest.fixture(scope="module")
def reta():
    Z, y, media_verdadeira = _dados_lineares(4_000)
    return Z, y, media_verdadeira, SVGP(**AJUSTE_RAPIDO).fit(Z, y)


def test_formas_e_variancia_positiva(reta):
    Z, _, _, modelo = reta
    media, variancia = modelo.predict(Z[:10])
    assert media.shape == variancia.shape == (10,)
    assert (variancia > 0).all()
    assert len(modelo.historico_elbo_) == AJUSTE_RAPIDO["epocas"]


def test_recupera_reta_e_ruido(reta):
    Z, _, media_verdadeira, modelo = reta
    media, variancia = modelo.predict(Z)
    assert np.abs(media - media_verdadeira).mean() < 0.3
    assert abs(np.sqrt(modelo.variancia_ruido_) - 2.0) < 0.2
    assert np.allclose(np.median(variancia), 4.0, rtol=0.2)


def test_captura_o_que_a_reta_nao_captura():
    Z, y = _dados_nao_lineares(3_000, semente=1)
    Z_separado, y_separado = _dados_nao_lineares(1_000, semente=2)
    nll_svgp = _nll(y_separado, *SVGP(**AJUSTE_RAPIDO).fit(Z, y).predict(Z_separado))
    nll_reta = _nll(y_separado, *RegressaoLinearBayesiana().fit(Z, y).predict(Z_separado))
    assert nll_svgp < nll_reta


def test_variancia_maior_longe_dos_dados():
    rng = np.random.default_rng(3)
    z = rng.uniform(0, 1, size=(500, 1))
    y = 2 * z[:, 0] + rng.normal(0, 0.1, size=500)
    modelo = SVGP(**AJUSTE_RAPIDO).fit(z, y)
    _, var_dentro = modelo.predict(z)
    _, var_longe = modelo.predict(np.array([[100.0]]))
    assert var_longe[0] > var_dentro.max()


def test_mesma_semente_mesmas_previsoes():
    Z, y, _ = _dados_lineares(1_000)
    rapido = {**AJUSTE_RAPIDO, "epocas": 5}
    a = SVGP(**rapido, semente=7).fit(Z, y).predict(Z[:50])
    b = SVGP(**rapido, semente=7).fit(Z, y).predict(Z[:50])
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_coluna_constante():
    Z, y, _ = _dados_lineares(1_000)
    Z_const = np.c_[Z, np.full(len(Z), 3.0)]
    media, variancia = SVGP(**{**AJUSTE_RAPIDO, "epocas": 5}).fit(Z_const, y).predict(Z_const)
    assert np.isfinite(media).all() and np.isfinite(variancia).all()
