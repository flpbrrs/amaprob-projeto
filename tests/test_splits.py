"""Split por indivíduo (Regra 3), com identificadores sintéticos."""

import numpy as np
import pandas as pd
import pytest

from amaprob.data.splits import TESTE, TREINO, VALIDACAO, dividir_por_pessoa


@pytest.fixture
def person_ids() -> pd.Series:
    # 1.000 pessoas com 1 a 8 refeições cada.
    rng = np.random.default_rng(123)
    repeticoes = rng.integers(1, 9, size=1000)
    return pd.Series(np.repeat([f"p{i}" for i in range(1000)], repeticoes))


def test_nenhuma_pessoa_em_dois_conjuntos(person_ids):
    conjunto = dividir_por_pessoa(person_ids)
    conjuntos_por_pessoa = conjunto.groupby(person_ids).nunique()
    assert (conjuntos_por_pessoa == 1).all()
    assert set(conjunto) == {TREINO, VALIDACAO, TESTE}


def test_fracoes_sobre_pessoas(person_ids):
    conjunto = dividir_por_pessoa(person_ids, fracoes=(0.70, 0.15, 0.15))
    pessoas_por_conjunto = person_ids.groupby(conjunto).nunique()
    assert pessoas_por_conjunto[TREINO] == 700
    assert pessoas_por_conjunto[VALIDACAO] == 150
    assert pessoas_por_conjunto[TESTE] == 150


def test_mesmo_rotulo_com_linhas_embaralhadas(person_ids):
    original = dividir_por_pessoa(person_ids, seed=0)
    embaralhado = person_ids.sample(frac=1.0, random_state=7)
    novo = dividir_por_pessoa(embaralhado, seed=0)
    rotulo_original = dict(zip(person_ids, original))
    assert all(rotulo_original[p] == r for p, r in zip(embaralhado, novo))
