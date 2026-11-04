"""Conferência dos arquivos reais em data/processed/ (versão 20230713 da POF).

Pulado quando os arquivos não existem. Para gerá-los: uv run python -m amaprob.data.dataset
"""

import numpy as np
import pandas as pd
import pytest

from amaprob.data.dataset import ARQUIVO_MASCARAS, ARQUIVO_REFEICOES, carregar_mascaras, carregar_refeicoes
from amaprob.data.mascaras import SEMENTES, TAXAS
from amaprob.data.splits import TESTE, TREINO, VALIDACAO

pytestmark = pytest.mark.skipif(
    not (ARQUIVO_REFEICOES.exists() and ARQUIVO_MASCARAS.exists()),
    reason="dados processados ausentes; gere com `uv run python -m amaprob.data.dataset`",
)


@pytest.fixture(scope="module")
def refeicoes():
    return carregar_refeicoes()


@pytest.fixture(scope="module")
def mascaras(refeicoes):
    return carregar_mascaras(refeicoes)  # já confere o alinhamento


def test_dimensoes_e_presenca(refeicoes):
    assert refeicoes.G.shape == (304_670, 29)
    assert len(refeicoes.gavetas) == 29
    assert refeicoes.P.sum() == 941_320
    np.testing.assert_array_equal(refeicoes.P, refeicoes.G > 0)
    assert (refeicoes.y >= 10).all()


def test_contexto(refeicoes):
    assert refeicoes.C.shape == (304_670, 13)
    assert len(refeicoes.C_nomes) == 13
    assert np.isfinite(refeicoes.C).all()


def test_pessoas_por_conjunto(refeicoes):
    conjunto_por_pessoa = pd.Series(refeicoes.conjunto).groupby(refeicoes.person_id).nunique()
    assert (conjunto_por_pessoa == 1).all()
    pessoas = pd.Series(refeicoes.person_id).groupby(refeicoes.conjunto).nunique()
    assert pessoas.to_dict() == {TREINO: 32_299, VALIDACAO: 6_921, TESTE: 6_921}


def test_grade_das_mascaras(mascaras):
    np.testing.assert_array_equal(mascaras.taxas, TAXAS)
    np.testing.assert_array_equal(mascaras.sementes, SEMENTES)


def test_mascaras_so_sobre_presentes(refeicoes, mascaras):
    assert not (mascaras.M & ~refeicoes.P).any()


def test_taxa_observada(refeicoes, mascaras):
    observada = mascaras.M.sum(axis=(2, 3)) / refeicoes.P.sum()
    assert np.abs(observada - mascaras.taxas[None, :]).max() < 0.002
