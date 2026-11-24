"""Testes das perguntas com erro de relato e regra de parada."""

import numpy as np

from amaprob.experiments.pergunta import simular
from amaprob.experiments.pergunta_parada import (
    estado_apos, fator_de_relato, perguntas_ate_o_limite, simular_ate_esgotar,
)
from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from test_pergunta import _regressao, _sinteticos


def _modelos():
    G, P, M, C = _sinteticos()
    vazio = np.zeros_like(P)
    imputador = FA(q=2).fit(ocultar(G, vazio), P, vazio, C)
    blr = _regressao(k=G.shape[1])[0]
    y = G.sum(axis=1)
    return imputador, blr, G, P, M, C, y


def test_sem_erro_reproduz_a_simulacao_de_pergunta():
    imputador, blr, G, P, M, C, y = _modelos()
    um = np.ones(G.shape)
    for estrategia in ("otima", "maior_carboidrato"):
        medias, variancias, n_ocultas = simular_ate_esgotar(imputador, blr, G, P, C, M, estrategia, um)
        linhas, _, _ = simular(imputador, blr, G, P, C, y, M, estrategia, 0, 0)
        for linha in linhas:
            n = linha["n_perguntas"]
            media, variancia, _ = estado_apos(medias, variancias, n_ocultas, np.full(len(y), n))
            np.testing.assert_allclose(np.abs(y - media).mean(), linha["mae_g"], rtol=1e-10)
            np.testing.assert_allclose(2 * 1.6448536269514722 * np.sqrt(variancia).mean(), linha["mpiw_g"], rtol=1e-10)


def test_ocultas_ainda_nao_perguntadas_nao_sao_lidas():
    """Lixo nas ocultas só pode mudar a preditiva a partir do passo em que a célula é perguntada."""
    imputador, blr, G, P, M, C, _ = _modelos()
    um = np.ones(G.shape)
    G_lixo = G.copy()
    G_lixo[M] = np.random.default_rng(9).uniform(1e5, 1e6, int(M.sum()))
    referencia = simular_ate_esgotar(imputador, blr, G, P, C, M, "otima", um)
    lixo = simular_ate_esgotar(imputador, blr, G_lixo, P, C, M, "otima", um)
    np.testing.assert_allclose(lixo[0][:, 0], referencia[0][:, 0])
    np.testing.assert_allclose(lixo[1][:, 0], referencia[1][:, 0])


def test_todas_as_ocultas_sao_perguntadas_uma_vez():
    imputador, blr, G, P, M, C, _ = _modelos()
    medias, variancias, n_ocultas = simular_ate_esgotar(imputador, blr, G, P, C, M, "otima", np.ones(G.shape))
    validos = np.isfinite(variancias)
    np.testing.assert_array_equal(validos.sum(axis=1), n_ocultas + 1)
    # Sem erro de relato, revelar uma quantidade nunca aumenta a variância esperada da previsão
    # na média das refeições.
    assert np.nanmean(variancias[:, 1]) < np.nanmean(variancias[:, 0])


def test_regra_de_parada():
    z = 1.6448536269514722
    largura = np.array([[120.0, 90.0, 70.0, np.nan], [50.0, 40.0, np.nan, np.nan], [130.0, 110.0, 105.0, 101.0]])
    variancias = (largura / (2 * z)) ** 2
    n_ocultas = np.array([2, 1, 3])
    np.testing.assert_array_equal(perguntas_ate_o_limite(variancias, n_ocultas, 80.0), [2, 0, 3])
    np.testing.assert_array_equal(perguntas_ate_o_limite(variancias, n_ocultas, 100.0), [1, 0, 3])


def test_fator_de_relato():
    np.testing.assert_array_equal(fator_de_relato((50, 4), 0.0, 0, 1), np.ones((50, 4)))
    a = np.log(fator_de_relato((20_000, 2), 0.2, 0, 1))
    b = np.log(fator_de_relato((20_000, 2), 0.4, 0, 1))
    np.testing.assert_allclose(b, 2 * a)  # mesmo ε, só a escala muda
    assert abs(a.std() - 0.2) < 0.005
