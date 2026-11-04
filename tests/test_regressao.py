"""Entrada Z e regressão linear bayesiana, com dados sintéticos."""

import numpy as np
import pytest

from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana


def _dados_lineares(n, d=3, desvio_ruido=2.0, semente=0):
    rng = np.random.default_rng(semente)
    Z = rng.normal(size=(n, d)) * [1.0, 10.0, 0.1][:d] + 5.0
    w = np.array([3.0, -0.5, 20.0])[:d]
    media_verdadeira = Z @ w + 7.0
    return Z, media_verdadeira + rng.normal(0, desvio_ruido, size=n), media_verdadeira


def test_montar_Z():
    N = 4
    Z = montar_Z(np.ones((N, 29)), np.ones((N, 29), dtype=bool), np.zeros((N, 13)))
    assert Z.shape == (N, 71) and Z.dtype == np.float64


def test_formas_e_variancia_positiva():
    Z, y, _ = _dados_lineares(500)
    media, var = RegressaoLinearBayesiana().fit(Z, y).predict(Z[:10])
    assert media.shape == var.shape == (10,)
    assert (var > 0).all()


def test_recupera_reta_e_ruido():
    Z, y, media_verdadeira = _dados_lineares(20_000)
    modelo = RegressaoLinearBayesiana().fit(Z, y)
    media, var = modelo.predict(Z)
    assert np.abs(media - media_verdadeira).max() < 0.2
    assert abs(1 / np.sqrt(modelo.beta_) - 2.0) < 0.05
    assert np.allclose(var, 4.0, rtol=0.05)
    assert modelo.n_iter_ < modelo.max_iter  # convergiu


def test_variancia_maior_longe_dos_dados():
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 1, size=(10, 1))
    y = 2 * z[:, 0] + rng.normal(0, 0.1, size=10)
    modelo = RegressaoLinearBayesiana().fit(z, y)
    _, var_dentro = modelo.predict(z)
    _, var_longe = modelo.predict(np.array([[100.0]]))
    assert var_longe[0] > var_dentro.max()


def test_coluna_constante():
    Z, y, _ = _dados_lineares(1_000)
    Z_const = np.c_[Z, np.full(len(Z), 3.0)]
    media_sem, _ = RegressaoLinearBayesiana().fit(Z, y).predict(Z)
    media_com, var_com = RegressaoLinearBayesiana().fit(Z_const, y).predict(Z_const)
    assert np.isfinite(var_com).all()
    np.testing.assert_allclose(media_com, media_sem, rtol=1e-6)
