"""Contrato dos imputadores e não vazamento de X[M] (Regra 2), sobre todos os imputadores."""

import numpy as np
import pytest

from amaprob.data.mascaras import gerar_mascaras
from amaprob.imputation.base import ocultar
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.imputation.ppca import PPCA

N, K, S = 2_000, 6, 20
IMPUTADORES = {"porcao_media": PorcaoMedia, "ppca": lambda: PPCA(q=2)}


@pytest.fixture(params=list(IMPUTADORES), scope="module")
def novo_imputador(request):
    return IMPUTADORES[request.param]


@pytest.fixture(scope="module")
def dados():
    rng = np.random.default_rng(5)
    P = rng.random((N, K)) < 0.5
    G = np.where(P, np.exp(rng.normal(4.0, 0.6, size=(N, K))), 0.0)  # porções ~ 55 g, log-normais
    M = gerar_mascaras(P, taxas=(0.5,), sementes=(0,))[0, 0]
    C = rng.random((N, 3))
    return G, P, M, C


def _imputar(novo_imputador, G, P, M, C, semente=0):
    X = ocultar(G, M)
    return novo_imputador().fit(X, P, M, C).sample(X, P, M, C, S, np.random.default_rng(semente))


def test_ocultar(dados):
    G, P, M, _ = dados
    X = ocultar(G, M)
    np.testing.assert_array_equal(np.isnan(X), M)
    np.testing.assert_allclose(X[~M], np.log1p(G[~M]))


def test_forma_e_observados_preservados(dados, novo_imputador):
    G, P, M, C = dados
    amostras = _imputar(novo_imputador, G, P, M, C)
    assert amostras.shape == (S, N, K)
    observado = P & ~M
    assert np.all(amostras[:, observado] == np.log1p(G[observado]))
    assert np.all(amostras[:, ~P] == 0.0)


def test_ocultos_preenchidos(dados, novo_imputador):
    G, P, M, C = dados
    ocultos = _imputar(novo_imputador, G, P, M, C)[:, M]
    assert np.isfinite(ocultos).all() and (ocultos >= 0).all()


def test_reprodutivel_pelo_gerador(dados, novo_imputador):
    G, P, M, C = dados
    np.testing.assert_array_equal(_imputar(novo_imputador, G, P, M, C, semente=3), _imputar(novo_imputador, G, P, M, C, semente=3))


def test_nenhum_vazamento(dados, novo_imputador):
    G, P, M, C = dados
    # Mesmas gramas observadas; valores ocultos verdadeiros trocados por outros quaisquer.
    G_alterado = G.copy()
    G_alterado[M] = np.random.default_rng(9).uniform(1, 2_000, size=M.sum())
    np.testing.assert_array_equal(_imputar(novo_imputador, G, P, M, C), _imputar(novo_imputador, G_alterado, P, M, C))


def test_imputador_ignora_x_m_mesmo_preenchido(dados, novo_imputador):
    # Segunda camada: mesmo que X[M] chegasse preenchido (com valores errados), a saída do
    # imputador não muda. Protege contra um imputador que leia X sem respeitar M.
    G, P, M, C = dados
    X_apagado = ocultar(G, M)
    X_preenchido = np.log1p(G)
    X_preenchido[M] = np.random.default_rng(9).uniform(0, 8, size=M.sum())
    saidas = []
    for X in (X_apagado, X_preenchido):
        modelo = novo_imputador().fit(X, P, M, C)
        saidas.append(modelo.sample(X, P, M, C, S, np.random.default_rng(0)))
    np.testing.assert_array_equal(saidas[0], saidas[1])


def test_parametros_pelos_observados(dados):
    G, P, M, C = dados
    X = ocultar(G, M)
    modelo = PorcaoMedia().fit(X, P, M, C)
    observado = P & ~M
    for k in range(K):
        valores = np.log1p(G[observado[:, k], k])
        assert np.isclose(modelo.media_[k], valores.mean())
        assert np.isclose(modelo.desvio_[k], valores.std())


def test_distribuicao_das_amostras(dados):
    G, P, M, C = dados
    X = ocultar(G, M)
    modelo = PorcaoMedia().fit(X, P, M, C)
    i, k = np.argwhere(M)[0]
    amostras = modelo.sample(X[i : i + 1], P[i : i + 1], M[i : i + 1], C[i : i + 1], 20_000, np.random.default_rng(1))
    valores = amostras[:, 0, k]
    assert abs(valores.mean() - modelo.media_[k]) < 0.02
    assert abs(valores.std() - modelo.desvio_[k]) < 0.02
