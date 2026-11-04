"""Geração das máscaras (Regra 4), sobre presença sintética."""

import numpy as np
import pytest

from amaprob.data.mascaras import SEMENTES, TAXAS, gerar_mascaras


@pytest.fixture(scope="module")
def P() -> np.ndarray:
    # 20.000 refeições × 29 gavetas, com ~3 gavetas presentes por refeição.
    return np.random.default_rng(42).random((20_000, 29)) < 3 / 29


@pytest.fixture(scope="module")
def M(P) -> np.ndarray:
    return gerar_mascaras(P)


def test_forma(P, M):
    assert M.shape == (len(SEMENTES), len(TAXAS), *P.shape)
    assert M.dtype == bool


def test_mascara_so_sobre_presentes(P, M):
    assert not (M & ~P).any()


@pytest.mark.parametrize("j", range(len(TAXAS)))
def test_taxa_observada(P, M, j):
    r, n = TAXAS[j], P.sum()
    desvio = np.sqrt(r * (1 - r) / n)
    for i in range(len(SEMENTES)):
        observada = M[i, j].sum() / n
        assert abs(observada - r) <= 4 * desvio + 1e-12


def test_aninhamento(M):
    for i in range(len(SEMENTES)):
        for j in range(len(TAXAS) - 1):
            assert not (M[i, j] & ~M[i, j + 1]).any()


def test_extremos(P, M):
    assert TAXAS[0] == 0.0 and TAXAS[-1] == 1.0
    assert not M[:, 0].any()
    assert all((M[i, -1] == P).all() for i in range(len(SEMENTES)))


def test_mesma_semente_identica(P, M):
    assert np.array_equal(gerar_mascaras(P), M)


def test_sementes_diferentes(M):
    meio = TAXAS.index(0.5)
    assert not np.array_equal(M[0, meio], M[1, meio])
    assert not np.array_equal(M[1, meio], M[2, meio])


def test_refeicao_de_uma_gaveta_mascaravel_em_025():
    # 10.000 refeições, cada uma com uma única gaveta presente.
    P = np.zeros((10_000, 29), dtype=bool)
    P[np.arange(10_000), np.arange(10_000) % 29] = True
    M = gerar_mascaras(P, taxas=(0.25,), sementes=(0,))[0, 0]
    observada = M.any(axis=1).mean()
    assert abs(observada - 0.25) <= 4 * np.sqrt(0.25 * 0.75 / 10_000)
