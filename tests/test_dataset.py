"""Montagem, persistência e alinhamento do dataset e das máscaras, com itens sintéticos."""

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from amaprob.data.dataset import (
    C_NOMES,
    Mascaras,
    Refeicoes,
    carregar_mascaras,
    carregar_refeicoes,
    codificar_hora,
    montar_refeicoes,
    salvar_mascaras,
    salvar_refeicoes,
)
from amaprob.data.mascaras import SEMENTES, TAXAS, gerar_mascaras

GAVETAS = pd.Series({1: "X", 2: "Y", 3: "Z"}, name="gaveta")


UF_RENDA = {1: (23, 1000.0), 2: (35, -5.0), 3: (43, 500.0)}  # Ceará (NE), São Paulo (SE), RS (S)


def _item(pessoa, quadro, ocasiao, hora, codigo, gramas, carbo):
    upa, dom, uc, inf = pessoa
    uf, renda = UF_RENDA[upa]
    return {
        "COD_UPA": upa, "NUM_DOM": dom, "NUM_UC": uc, "COD_INFORMANTE": inf, "QUADRO": quadro,
        "V9017": ocasiao, "V9015": hora, "V9001": codigo, "QTD": gramas, "CHOTOT": carbo,
        "UF": uf, "RENDA_TOTAL": renda,
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


def test_contexto(refeicoes):
    C = pd.DataFrame(refeicoes.C, columns=refeicoes.C_nomes)
    assert C.shape == (3, 13) and list(C.columns) == list(C_NOMES)
    assert not C.isna().any().any()
    ocasioes = C.filter(like="ocasiao_")
    regioes = C.filter(like="regiao_")
    # Refeição 1: almoço de pessoa do Nordeste.
    assert ocasioes.iloc[0].to_dict() == {"ocasiao_cafe": 0, "ocasiao_almoco": 1, "ocasiao_lanche": 0,
                                          "ocasiao_jantar": 0, "ocasiao_ceia": 0}
    assert regioes.iloc[0].to_dict() == {"regiao_norte": 0, "regiao_nordeste": 1, "regiao_sudeste": 0,
                                         "regiao_sul": 0, "regiao_centro_oeste": 0}
    assert (ocasioes.sum(axis=1) == 1).all() and (regioes.sum(axis=1) == 1).all()
    np.testing.assert_allclose(C.log_renda, [np.log1p(1000), np.log1p(1000), 0.0])  # renda negativa -> 0
    np.testing.assert_allclose(C[["hora_sen", "hora_cos"]].iloc[0], [0.0, -1.0], atol=1e-12)  # 12H


def test_hora_ciclica():
    h = codificar_hora(pd.Series(["23H", "0H", "11H", "12H"]))
    assert np.isclose(np.linalg.norm(h[0] - h[1]), np.linalg.norm(h[2] - h[3]))


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
    por_coluna = {"gavetas", "C_nomes"}  # campos que não são por linha
    invertido = Refeicoes(
        **{c.name: getattr(refeicoes, c.name)[::-1] for c in fields(Refeicoes) if c.name not in por_coluna},
        **{nome: getattr(refeicoes, nome) for nome in por_coluna},
    )
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
