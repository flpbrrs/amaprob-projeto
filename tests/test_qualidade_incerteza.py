"""Testes das métricas de qualidade da incerteza com previsões sintéticas perfeitamente calibradas."""

import numpy as np
import pytest

from amaprob.evaluation.metricas import intervalo
from amaprob.experiments.qualidade_incerteza import (
    NIVEIS, cobertura_por_nivel, histograma_pit, pit_mistura, por_decil_do_desvio, resumo_de_grupo,
    risco_cobertura,
)

N = 200_000


@pytest.fixture(scope="module")
def gaussiana():
    rng = np.random.default_rng(0)
    media = rng.normal(50.0, 20.0, N)
    desvio = rng.lognormal(2.5, 0.6, N)  # desvio heterogêneo, como na regressão
    y = media + desvio * rng.standard_normal(N)
    return y, media, desvio


@pytest.fixture(scope="module")
def mistura():
    """y sorteado da própria mistura de 5 componentes por refeição."""
    rng = np.random.default_rng(1)
    S, n = 5, 100_000
    medias = rng.normal(40.0, 15.0, (S, n))
    variancias = rng.lognormal(3.0, 0.5, (S, n))
    componente = rng.integers(0, S, n)
    y = medias[componente, np.arange(n)] + np.sqrt(variancias[componente, np.arange(n)]) * rng.standard_normal(n)
    return y, medias, variancias


def test_cobertura_empirica_igual_a_nominal(gaussiana):
    y, media, desvio = gaussiana
    tabela = cobertura_por_nivel(pit_mistura(y, media[None], desvio[None] ** 2))
    np.testing.assert_allclose(tabela.empirica, NIVEIS, atol=0.005)


def test_pit_uniforme_na_gaussiana_e_na_mistura(gaussiana, mistura):
    y, media, desvio = gaussiana
    _, desvio_max = histograma_pit(pit_mistura(y, media[None], desvio[None] ** 2))
    assert desvio_max < 0.004
    y, medias, variancias = mistura
    pit = pit_mistura(y, medias, variancias)
    _, desvio_max = histograma_pit(pit)
    assert desvio_max < 0.006
    np.testing.assert_allclose(cobertura_por_nivel(pit).empirica, NIVEIS, atol=0.007)


def test_cobertura_pela_pit_bate_com_o_intervalo_por_bisseccao(mistura):
    y, medias, variancias = mistura
    li, ls = intervalo(medias, variancias)
    pela_pit = np.abs(pit_mistura(y, medias, variancias) - 0.5) <= 0.45
    assert np.mean(pela_pit != ((li <= y) & (y <= ls))) < 1e-4  # só empates na borda da bissecção


def test_razao_rmse_desvio_perto_de_1_em_todo_decil(gaussiana):
    y, media, desvio = gaussiana
    dentro = np.abs(y - media) <= 1.6448536269514722 * desvio
    tabela = por_decil_do_desvio(y, media, desvio, dentro)
    assert len(tabela) == 10 and tabela.n.sum() == N
    np.testing.assert_allclose(tabela.razao_rmse_desvio, 1.0, atol=0.02)
    np.testing.assert_allclose(tabela.picp90, 0.90, atol=0.01)
    assert np.all(np.diff(tabela.desvio_medio) > 0)


def test_risco_cobertura_cresce_com_a_fracao_e_fecha_na_aleatoria(gaussiana):
    y, media, desvio = gaussiana
    tabela = risco_cobertura(y, media, desvio, np.random.default_rng(2), permutacoes=5)
    assert np.all(np.diff(tabela.mae_g) > 0)  # o desvio ordena o erro
    final = tabela.iloc[-1]
    assert final.mae_g == pytest.approx(final.mae_g_aleatoria)
    assert np.all(tabela.mae_g.iloc[:-1] < tabela.mae_g_aleatoria.iloc[:-1])


def test_resumo_de_grupo(gaussiana):
    y, media, desvio = gaussiana
    li, ls = media - 1.6448536269514722 * desvio, media + 1.6448536269514722 * desvio
    r = resumo_de_grupo(y, media, desvio, li, ls)
    assert r["n"] == N
    assert r["razao_rmse_desvio"] == pytest.approx(1.0, abs=0.01)
    assert r["picp90"] == pytest.approx(0.90, abs=0.003)
    assert resumo_de_grupo(y[:0], media[:0], desvio[:0], li[:0], ls[:0])["n"] == 0
