"""Configuração adotada de 29 gavetas, verificada sobre a classificação versionada do IBGE."""

from pathlib import Path

import pandas as pd
import pytest

from amaprob.data.gavetas import COD_ACUCAR, DIVIDIR_K29, FUNDIR_K29, GAVETA_ACUCAR, montar_gavetas

CLASSIFICACAO = Path(__file__).resolve().parents[1] / "configs/classificacao_ibge.csv"


@pytest.fixture(scope="module")
def classificacao() -> pd.DataFrame:
    return pd.read_csv(CLASSIFICACAO)


@pytest.fixture(scope="module")
def gavetas(classificacao) -> pd.Series:
    return montar_gavetas(classificacao, DIVIDIR_K29, FUNDIR_K29)


def test_29_gavetas_distintas(gavetas):
    assert gavetas.nunique() == 29


def test_perfis_de_bebidas(classificacao, gavetas):
    bebidas = classificacao[classificacao.grupo == "Bebidas"]
    perfil_por_subgrupo = bebidas.groupby("subgrupo").V9001.first().map(gavetas).to_dict()
    esperado = {
        "Água": "sem açúcar", "Café": "sem açúcar", "Chá": "sem açúcar",
        "Refrigerantes diet/light": "sem açúcar",
        "Refrescos/sucos industrializados diet/light": "sem açúcar", "Destiladas": "sem açúcar",
        "Sucos": "açucaradas", "Refrescos/sucos industrializados": "açucaradas",
        "Refrigerantes": "açucaradas", "Outras bebidas não-alcoólicas": "açucaradas",
        "Bebidas à base de soja": "açucaradas",
        "Bebidas lácteas": "lácteas",
        "Cerveja": "cerveja e vinho", "Vinho": "cerveja e vinho",
    }
    assert perfil_por_subgrupo == {sub: f"Bebidas / {perfil}" for sub, perfil in esperado.items()}
    # Todo código de Bebidas cai num dos 4 perfis.
    assert set(gavetas.loc[bebidas.V9001]) == {f"Bebidas / {p}" for p in set(esperado.values())}


def test_farinhas_e_massas_por_subgrupo(classificacao, gavetas):
    farinhas = classificacao[classificacao.grupo == "Farinhas e massas"]
    esperado = "Farinhas e massas / " + farinhas.subgrupo
    assert (gavetas.loc[farinhas.V9001].to_numpy() == esperado.to_numpy()).all()


def test_demais_grupos_inteiros_e_acucar_proprio(classificacao, gavetas):
    outros = classificacao[~classificacao.grupo.isin(DIVIDIR_K29) & (classificacao.V9001 != COD_ACUCAR)]
    assert (gavetas.loc[outros.V9001].to_numpy() == outros.grupo.to_numpy()).all()
    assert gavetas.loc[COD_ACUCAR] == GAVETA_ACUCAR
