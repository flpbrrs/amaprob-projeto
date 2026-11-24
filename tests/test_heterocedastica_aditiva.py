"""Testes da variante aditiva da variância do ruído."""

import numpy as np

from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.blr_heterocedastica import prever_com_entrada_incerta_heterocedastica
from amaprob.regression.blr_heterocedastica_aditiva import (
    RegressaoLinearBayesianaHeterocedasticaAditiva, variaveis_aditivas,
)

K = 4


def _dados(n: int, semente: int, v0: float, v_gaveta, u_ocasiao):
    rng = np.random.default_rng(semente)
    P = rng.random((n, K)) < 0.5
    P[~P.any(axis=1), 0] = True
    G = np.where(P, rng.gamma(2.0, 40.0, (n, K)), 0.0)
    ocasiao = rng.integers(0, 5, n)
    C = np.zeros((n, 13))
    C[np.arange(n), ocasiao] = 1.0
    C[:, 5:] = rng.standard_normal((n, 8))
    variancia = v0 + P @ np.asarray(v_gaveta, float) + np.asarray(u_ocasiao, float)[ocasiao]
    y = 10.0 + G @ np.array([0.5, 0.1, 0.3, 0.0]) + np.sqrt(variancia) * rng.standard_normal(n)
    return montar_Z(G, P, C), y, variancia, G, P, C


def test_recupera_variancia_aditiva_conhecida():
    Z, y, variancia, *_ = _dados(60_000, 0, 20.0, (5.0, 80.0, 0.0, 30.0), (0.0, 0.0, 60.0, 0.0, 10.0))
    modelo = RegressaoLinearBayesianaHeterocedasticaAditiva(n_gavetas=K).fit(Z, y)
    estimada = modelo.variancia_ruido(Z)
    assert np.all(estimada > 0)
    np.testing.assert_allclose(modelo.coeficientes_[1:1 + K], [5.0, 80.0, 0.0, 30.0], atol=4.0)
    np.testing.assert_allclose(np.median(estimada / variancia), 1.0, atol=0.03)
    assert np.quantile(np.abs(estimada / variancia - 1), 0.95) < 0.12
    # A gaveta de maior variância (a segunda) é a de maior coeficiente.
    assert np.argmax(modelo.coeficientes_[1:1 + K]) == 1


def test_variancia_constante_da_a_blr():
    Z, y, *_ = _dados(20_000, 1, 25.0, (0.0,) * K, (0.0,) * 5)
    blr = RegressaoLinearBayesiana().fit(Z, y)
    modelo = RegressaoLinearBayesianaHeterocedasticaAditiva(n_gavetas=K).fit(Z, y)
    _, var_b = blr.predict(Z[:500])
    _, var_a = modelo.predict(Z[:500])
    np.testing.assert_allclose(var_a.mean(), var_b.mean(), rtol=0.03)
    np.testing.assert_allclose(var_a, var_b, rtol=0.15)


def test_analitica_sem_incerteza_coincide_com_predict():
    Z, y, _, G, P, C = _dados(5_000, 2, 20.0, (5.0, 80.0, 0.0, 30.0), (0.0,) * 5)
    modelo = RegressaoLinearBayesianaHeterocedasticaAditiva(n_gavetas=K).fit(Z, y)
    n = 200
    media, var = prever_com_entrada_incerta_heterocedastica(modelo, G[:n], np.zeros((n, K, K)), P[:n], C[:n])
    media_ref, var_ref = modelo.predict(Z[:n])
    np.testing.assert_allclose(media, media_ref)
    np.testing.assert_allclose(var, var_ref)
    assert variaveis_aditivas(Z, K).shape == (len(Z), 1 + K + 5)
