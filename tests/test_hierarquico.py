"""Testes do modelo hierárquico com efeito da pessoa sobre a FA condicional.

As três garantias do modelo: com τ² = 0 ele reproduz exatamente a FA sem pessoa; a posterior de δ
bate com a fórmula conjunta e some sem dado no histórico; e as células ocultas do dia avaliado,
trocadas por lixo, não mudam nada.
"""

import numpy as np
import pytest

from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from amaprob.imputation.hierarquico import (
    Variancias,
    amostrar_com_pessoa,
    estimar_delta,
    estimar_variancias,
    momentos_com_pessoa,
)
from amaprob.imputation.momentos import momentos_ocultos


def _fa(n: int = 400, k: int = 4, semente: int = 0):
    """FA condicional ajustada em dados sintéticos, sem máscara (sem o que imputar no ajuste)."""
    rng = np.random.default_rng(semente)
    P = rng.random((n, k)) < 0.6
    C = rng.normal(size=(n, 13))
    G = np.where(P, rng.gamma(2.0, 25.0, (n, k)), 0.0)
    M = np.zeros_like(P)
    fa = FA(q=3, semente=0, media_condicional=True, max_iteracoes=60).fit(ocultar(G, M), P, M, C)
    return fa, G, P, C, M


def test_com_tau_zero_reproduz_a_fa_sem_pessoa():
    fa, G, P, C, _ = _fa()
    k = G.shape[1]
    variancias = Variancias(
        tau2=np.zeros(k), sigma2=np.full(k, 0.5), tau2_fator=0.0, sigma2_fator=0.5
    )
    residuo = np.full((10, k), 0.1)
    delta, v = estimar_delta(residuo, ~np.isnan(residuo), np.arange(10) % 3, 3, variancias, "gaveta")
    assert np.all(delta == 0.0) and np.all(v == 0.0)

    M = P & (np.random.default_rng(1).random(P.shape) < 0.5)
    linhas = np.zeros_like(G)
    esperadas = fa.sample(ocultar(G, M), P, M, C, 7, np.random.default_rng(1))
    obtidas = amostrar_com_pessoa(
        fa, G, P, M, C, linhas, linhas, 7, np.random.default_rng(1), np.random.default_rng(2)
    )
    np.testing.assert_array_equal(obtidas, esperadas)

    media0, cov0 = momentos_ocultos(fa, ocultar(G, M), P, M, C)
    media1, cov1 = momentos_com_pessoa(fa, G, P, M, C, linhas, linhas)
    np.testing.assert_array_equal(media1, media0)
    np.testing.assert_array_equal(cov1, cov0)


def test_posterior_bate_com_a_formula_e_some_sem_dado_no_historico():
    k = 3
    residuo = np.array([
        [0.4, np.nan, np.nan],
        [np.nan, -0.2, np.nan],
        [0.1, 0.3, np.nan],
    ])
    observado = ~np.isnan(residuo)
    codigos = np.array([0, 0, 1])
    variancias = Variancias(
        tau2=np.array([0.04, 0.09, 0.12]), sigma2=np.array([0.25, 0.16, 0.36]),
        tau2_fator=0.05, sigma2_fator=0.30,
    )
    delta, v = estimar_delta(residuo, observado, codigos, 2, variancias, "gaveta")

    # pessoa 0, gaveta 0: n = 1, r̄ = 0.4
    v00 = 1.0 / (1.0 / 0.04 + 1.0 / 0.25)
    assert v[0, 0] == pytest.approx(v00)
    assert delta[0, 0] == pytest.approx(v00 * 0.4 / 0.25)

    # pessoa 0, gaveta 1: n = 1, r̄ = −0.2, pela conta por gaveta mesmo
    v01 = 1.0 / (1.0 / 0.09 + 1.0 / 0.16)
    assert v[0, 1] == pytest.approx(v01)
    assert delta[0, 1] == pytest.approx(v01 * (-0.2) / 0.16)

    # gaveta sem dado para a pessoa: o fator único é o valor de partida, e v = tau2
    v_f0 = 1.0 / (1.0 / 0.05 + 2.0 / 0.30)  # fatores 0.4 e −0.2 → n = 2, média 0.1
    assert delta[0, 2] == pytest.approx(v_f0 * 2 * 0.1 / 0.30)
    assert v[0, 2] == pytest.approx(0.12)

    # histórico vazio: δ̂ = 0 e v = tau2 (ou tau2_fator no fator único)
    vazio = np.zeros((0, k))
    for variante, esperado in (("fator", 0.05), ("gaveta", variancias.tau2)):
        d0, v0 = estimar_delta(vazio, vazio.astype(bool), np.zeros(0, dtype=int), 2, variancias, variante)
        np.testing.assert_array_equal(d0, np.zeros((2, k)))
        np.testing.assert_allclose(v0, np.broadcast_to(np.asarray(esperado), v0.shape))


def test_lixo_nas_ocultas_do_avaliado_nao_muda_nada():
    fa, G, P, C, _ = _fa(semente=1)
    k = G.shape[1]
    rng = np.random.default_rng(0)
    M = P & (rng.random(P.shape) < 0.5)
    delta = rng.normal(0.0, 0.2, (len(G), k))
    v = rng.uniform(0.01, 0.09, (len(G), k))

    limpas = amostrar_com_pessoa(
        fa, G, P, M, C, delta, v, 5, np.random.default_rng(2), np.random.default_rng(3)
    )
    G_lixo = G.copy()
    G_lixo[M] = 1e6  # ocultas do dia avaliado trocadas por lixo
    sujas = amostrar_com_pessoa(
        fa, G_lixo, P, M, C, delta, v, 5, np.random.default_rng(2), np.random.default_rng(3)
    )
    np.testing.assert_array_equal(sujas, limpas)

    media0, cov0 = momentos_com_pessoa(fa, G, P, M, C, delta, v)
    media1, cov1 = momentos_com_pessoa(fa, G_lixo, P, M, C, delta, v)
    np.testing.assert_array_equal(media1, media0)
    np.testing.assert_array_equal(cov1, cov0)

    # as observadas saem com o valor verdadeiro, sem δ̂
    np.testing.assert_array_equal(
        limpas[:, P & ~M], np.broadcast_to(np.log1p(G[P & ~M]), limpas[:, P & ~M].shape)
    )


def test_variancias_do_treino_sao_nao_negativas():
    rng = np.random.default_rng(0)
    n_pessoas, n_dias = 40, 2
    pessoa = np.repeat([f"p{i}" for i in range(n_pessoas)], n_dias)
    dia = np.tile([1, 2], n_pessoas)
    efeito = np.repeat(rng.normal(0.0, 1.0, n_pessoas), n_dias)
    r = (efeito + rng.normal(0.0, 0.3, len(efeito)))[:, None] * np.ones((1, 2))
    variancias = estimar_variancias(r, pessoa, dia)
    assert np.all(variancias.tau2 >= 0.0)
    assert np.all(variancias.sigma2 > 0.0)
    assert variancias.tau2_fator > 0.0 and variancias.sigma2_fator > 0.0
