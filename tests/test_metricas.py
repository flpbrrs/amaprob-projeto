"""Métricas com casos de valor conhecido."""

import numpy as np
import pytest

from scipy.special import ndtr

from amaprob.evaluation.metricas import (
    Z_INTERVALO,
    avaliar,
    avaliar_imputacao,
    avaliar_mistura,
    celulas_de_imputacao,
    cobertura_por_faixa,
    imputacao_por_gaveta,
    intervalo,
    por_ocultas,
)


def test_mae_e_meia_dose():
    m = avaliar([10, 20, 30], [12, 20, 20], [1, 1, 1])
    assert m["mae_g"] == pytest.approx(4.0)
    assert m["pct_meia_dose"] == pytest.approx(200 / 3)


def test_normal_padrao_em_zero():
    m = avaliar([0.0], [0.0], [1.0])
    assert m["nll"] == pytest.approx(0.5 * np.log(2 * np.pi))  # ≈ 0,9189
    assert m["crps_g"] == pytest.approx(2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))  # 2φ(0) − 1/√π ≈ 0,2337


def test_escala_do_crps():
    y, media, desvio = np.array([1.0, -2.0, 0.5]), np.array([0.0, 0.3, 1.0]), np.array([1.0, 2.0, 0.5])
    base = avaliar(y, media, desvio**2)["crps_g"]
    assert avaliar(10 * y, 10 * media, (10 * desvio) ** 2)["crps_g"] == pytest.approx(10 * base)


def test_picp_e_mpiw():
    m = avaliar([0.0, 1.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert m["picp"] == pytest.approx(2 / 3)
    assert m["mpiw_g"] == pytest.approx(2 * Z_INTERVALO)
    assert Z_INTERVALO == pytest.approx(1.6448536, abs=1e-6)


def test_cobertura_por_faixa():
    rng = np.random.default_rng(0)
    media = rng.uniform(10, 100, size=1_003)
    y = media + rng.normal(size=media.size)
    tabela = cobertura_por_faixa(y, media, np.ones_like(media))
    assert len(tabela) == 5
    assert tabela.n.sum() == media.size
    assert (tabela.media_max_g.to_numpy()[:-1] <= tabela.media_min_g.to_numpy()[1:]).all()
    assert tabela.picp.between(0.8, 1.0).all()


def _gaussianas(n=500, semente=0):
    rng = np.random.default_rng(semente)
    media = rng.uniform(10, 100, size=n)
    var = rng.uniform(4, 400, size=n)
    y = media + rng.normal(size=n) * np.sqrt(var)
    return y, media, var


def _assert_metricas_iguais(obtido, esperado):
    assert obtido.keys() == esperado.keys()
    for chave in esperado:
        assert obtido[chave] == pytest.approx(esperado[chave], rel=1e-9), chave


def test_mistura_um_componente():
    y, media, var = _gaussianas()
    _assert_metricas_iguais(avaliar_mistura(y, media[None], var[None]), avaliar(y, media, var))


def test_mistura_componentes_iguais():
    y, media, var = _gaussianas()
    S = 7
    _assert_metricas_iguais(avaliar_mistura(y, np.tile(media, (S, 1)), np.tile(var, (S, 1))), avaliar(y, media, var))


def test_mistura_quantis():
    medias = np.array([[0.0, 10.0], [50.0, 12.0]])
    variancias = np.array([[1.0, 4.0], [9.0, 1.0]])
    li, ls = intervalo(medias, variancias)
    cdf = lambda x: ndtr((x - medias) / np.sqrt(variancias)).mean(axis=0)
    np.testing.assert_allclose(cdf(li), 0.05, atol=1e-6)
    np.testing.assert_allclose(cdf(ls), 0.95, atol=1e-6)


def test_mistura_observacao_longe():
    m = avaliar_mistura([1_000.0], [[0.0], [10.0]], [[1.0], [4.0]])
    assert np.isfinite(m["nll"])
    assert m["nll"] > 0


def test_por_ocultas_classes():
    rng = np.random.default_rng(0)
    n = 200
    y, media = rng.normal(size=n), np.zeros(n)
    tabela = por_ocultas(y, media, media - 1, media + 1, rng.integers(0, 4, size=n))
    assert tabela.n.sum() == n
    assert "4+" not in set(tabela.n_ocultas)

    tabela = por_ocultas(y, media, media - 1, media + 1, rng.integers(0, 9, size=n))
    assert tabela.n.sum() == n
    assert list(tabela.n_ocultas) == ["0", "1", "2", "3", "4+"]


def test_por_ocultas_caso_conhecido():
    y = np.array([0.0, 5.0, 0.0])
    zeros = np.zeros(3)
    tabela = por_ocultas(y, zeros, zeros - 1, zeros + 1, [0, 0, 5]).set_index("n_ocultas")
    assert tabela.loc["0", "n"] == 2
    assert tabela.loc["0", "picp"] == pytest.approx(0.5)
    assert tabela.loc["4+", "n"] == 1
    assert tabela.loc["4+", "picp"] == pytest.approx(1.0)


def test_imputacao_caso_conhecido():
    G = np.array([[10.0, 7.0], [3.0, 100.0]])
    M = np.array([[True, False], [False, True]])
    amostras = np.zeros((4, 2, 2))
    amostras[:, 0, 0], amostras[:, 1, 1] = np.log1p(20), np.log1p(100)
    m = avaliar_imputacao(celulas_de_imputacao(G, amostras, M))
    assert m["n_celulas"] == 2
    assert m["mae_g"] == pytest.approx(5.0)
    assert m["mae_log"] == pytest.approx(abs(np.log1p(20) - np.log1p(10)) / 2)


def test_imputacao_so_celulas_ocultas():
    rng = np.random.default_rng(0)
    G = rng.uniform(1, 200, size=(50, 4))
    M = rng.random((50, 4)) < 0.4
    amostras = rng.normal(4, 0.5, size=(30, 50, 4))
    G_alterado = np.where(M, G, rng.uniform(1, 200, size=G.shape))
    pd_a, pd_b = (celulas_de_imputacao(g, amostras, M) for g in (G, G_alterado))
    assert avaliar_imputacao(pd_a) == avaliar_imputacao(pd_b)


def test_imputacao_amostras_gaussianas():
    mu, sigma, verdade_log = 4.0, 0.5, 4.3
    amostras = np.random.default_rng(1).normal(mu, sigma, size=(200_000, 1, 1))
    m = avaliar_imputacao(celulas_de_imputacao(np.array([[np.expm1(verdade_log)]]), amostras, np.ones((1, 1), bool)))
    z = (verdade_log - mu) / sigma
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + z**2 / 2
    crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    largura = np.expm1(mu + Z_INTERVALO * sigma) - np.expm1(mu - Z_INTERVALO * sigma)
    assert m["nll_log"] == pytest.approx(nll, abs=0.01)
    assert m["crps_log"] == pytest.approx(crps, abs=0.005)
    assert m["mpiw_g"] == pytest.approx(largura, rel=0.01)
    assert m["picp"] == 1.0


def test_imputacao_cobertura_conhecida():
    amostras = np.linspace(0, 1, 101)[:, None, None] * np.ones((1, 1, 3))  # intervalo [0,05; 0,95] em log
    G = np.expm1(np.array([[0.5, 0.9, 2.0]]))
    m = avaliar_imputacao(celulas_de_imputacao(G, amostras, np.ones((1, 3), bool)))
    assert m["picp"] == pytest.approx(2 / 3)


def test_imputacao_por_gaveta_soma_celulas():
    rng = np.random.default_rng(2)
    M = rng.random((40, 5)) < 0.3
    M[:, 4] = False
    celulas = celulas_de_imputacao(rng.uniform(1, 50, size=(40, 5)), rng.normal(3, 0.4, size=(20, 40, 5)), M)
    tabela = imputacao_por_gaveta(celulas)
    assert set(tabela.gaveta) == set(np.flatnonzero(M.any(axis=0)))
    assert tabela.n_celulas.sum() == M.sum()
