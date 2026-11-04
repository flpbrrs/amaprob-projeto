"""Montagem, persistência e alinhamento do dataset e das máscaras, com itens sintéticos."""

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from amaprob.data.dataset import (
    Mascaras,
    Refeicoes,
    carregar_mascaras,
    carregar_refeicoes,
    montar_refeicoes,
    salvar_mascaras,
    salvar_refeicoes,
)
from amaprob.data.mascaras import SEMENTES, TAXAS, gerar_mascaras

GAVETAS = pd.Series({1: "X", 2: "Y", 3: "Z"}, name="gaveta")


def _item(pessoa, quadro, ocasiao, hora, codigo, gramas, carbo):
    upa, dom, uc, inf = pessoa
    return {
        "COD_UPA": upa, "NUM_DOM": dom, "NUM_UC": uc, "COD_INFORMANTE": inf, "QUADRO": quadro,
        "V9017": ocasiao, "V9015": hora, "V9001": codigo, "QTD": gramas, "CHOTOT": carbo,
    }


@pytest.fixture(scope="module")
def consumo() -> pd.DataFrame:
    a, b, c = (1, 1, 1, 1), (2, 1, 1, 1), (3, 1, 1, 1)
    return pd.DataFrame([
        _item(a, 72, 2, "12H", 1, 100.0, 30.0),  # refeição 1: X e Y, y = 42
        _item(a, 72, 2, "12H", 2, 50.0, 12.0),
        _item(a, 73, 1, "8H", 2, 80.0, 15.0),  # refeição 2: só Y
        _item(b, 72, 4, "20H", 1, 0.0, 0.0),  # refeição 3: X com 0 g continua presente
        _item(b, 72, 4, "20H", 3, 200.0, 50.0),
        _item(b, 72, 3, "16H", 3, 20.0, 5.0),  # descartada: menos de 10 g de carboidrato
        _item(c, 72, 6, "10H", 1, 300.0, 90.0),  # descartada: ocasião 6
    ])


@pytest.fixture(scope="module")
def refeicoes(consumo) -> Refeicoes:
    return montar_refeicoes(consumo, GAVETAS)


def test_montagem(refeicoes):
    assert list(refeicoes.gavetas) == ["X", "Y", "Z"]
    assert list(refeicoes.meal_id) == sorted(refeicoes.meal_id)
    np.testing.assert_array_equal(refeicoes.y, [42.0, 15.0, 50.0])
    np.testing.assert_array_equal(refeicoes.G, [[100, 50, 0], [0, 80, 0], [0, 0, 200]])
    np.testing.assert_array_equal(refeicoes.P, [[1, 1, 0], [0, 1, 0], [1, 0, 1]])
    assert list(refeicoes.person_id) == ["1-1-1-1", "1-1-1-1", "2-1-1-1"]
    assert refeicoes.conjunto[0] == refeicoes.conjunto[1]


def test_ida_e_volta_do_dataset(refeicoes, tmp_path):
    salvar_refeicoes(refeicoes, tmp_path / "r.npz")
    carregado = carregar_refeicoes(tmp_path / "r.npz")
    for campo in fields(Refeicoes):
        np.testing.assert_array_equal(getattr(carregado, campo.name), getattr(refeicoes, campo.name))


@pytest.fixture
def arquivo_mascaras(refeicoes, tmp_path):
    mascaras = Mascaras(M=gerar_mascaras(refeicoes.P), taxas=np.array(TAXAS), sementes=np.array(SEMENTES))
    salvar_mascaras(mascaras, refeicoes, tmp_path / "m.npz")
    return mascaras, tmp_path / "m.npz"


def test_ida_e_volta_das_mascaras(refeicoes, arquivo_mascaras):
    mascaras, path = arquivo_mascaras
    carregado = carregar_mascaras(refeicoes, path)
    np.testing.assert_array_equal(carregado.M, mascaras.M)
    np.testing.assert_array_equal(carregado.taxas, TAXAS)
    np.testing.assert_array_equal(carregado.sementes, SEMENTES)


def test_desalinhamento_por_ordem(refeicoes, arquivo_mascaras):
    _, path = arquivo_mascaras
    invertido = Refeicoes(**{c.name: getattr(refeicoes, c.name)[::-1] for c in fields(Refeicoes) if c.name != "gavetas"},
                          gavetas=refeicoes.gavetas)
    with pytest.raises(ValueError, match="meal_id"):
        carregar_mascaras(invertido, path)


def test_desalinhamento_por_gavetas(refeicoes, arquivo_mascaras):
    _, path = arquivo_mascaras
    with pytest.raises(ValueError, match="gavetas"):
        carregar_mascaras(replace(refeicoes, gavetas=np.array(["X", "Y", "W"])), path)


def test_desalinhamento_por_presenca(refeicoes, arquivo_mascaras):
    _, path = arquivo_mascaras
    outra_P = refeicoes.P.copy()
    outra_P[1, 0] = True
    with pytest.raises(ValueError, match="presença"):
        carregar_mascaras(replace(refeicoes, P=outra_P), path)
