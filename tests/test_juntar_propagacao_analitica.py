"""Testes da junção das tabelas do braço analítico."""

from pathlib import Path

import pandas as pd
import pytest

from amaprob.experiments.juntar_propagacao_analitica import TABELAS, juntar

NOME = "propagacao_analitica_metricas"
COLUNAS = ["imputador", "taxa", "semente", "braco", "S", "mae_g"]


def _linha(imputador: str, semente: int = 0, taxa: float = 0.5) -> dict:
    return {"imputador": imputador, "taxa": taxa, "semente": semente, "braco": "plugin", "S": 1, "mae_g": 1.0}


def _gravar(pasta: Path, sufixo: str, linhas: list[dict], colunas: list[str] = COLUNAS) -> None:
    nome = f"{NOME}{f'_{sufixo}' if sufixo else ''}.csv"
    pd.DataFrame(linhas)[colunas].to_csv(pasta / nome, index=False)


def test_junta_base_e_sufixos(tmp_path: Path):
    _gravar(tmp_path, "", [_linha("porcao_media")])
    _gravar(tmp_path, "a", [_linha("fa"), _linha("fa", semente=1)])
    juntas = juntar(NOME, ["a"], tmp_path)
    assert len(juntas) == 3
    assert set(juntas.imputador) == {"porcao_media", "fa"}


def test_falha_com_coluna_diferente(tmp_path: Path):
    _gravar(tmp_path, "", [_linha("porcao_media")])
    _gravar(tmp_path, "a", [_linha("fa")], COLUNAS[:-1])
    with pytest.raises(ValueError, match="colunas diferentes"):
        juntar(NOME, ["a"], tmp_path)


def test_falha_com_chave_repetida(tmp_path: Path):
    _gravar(tmp_path, "", [_linha("porcao_media")])
    _gravar(tmp_path, "a", [_linha("porcao_media")])
    with pytest.raises(ValueError, match="chave repetida"):
        juntar(NOME, ["a"], tmp_path)


def test_falha_com_sufixo_ausente(tmp_path: Path):
    _gravar(tmp_path, "", [_linha("porcao_media")])
    with pytest.raises(FileNotFoundError, match="ausentes"):
        juntar(NOME, ["nao_existe"], tmp_path)


def test_chaves_das_tabelas_cobrem_as_colunas_de_identificacao():
    assert TABELAS[NOME] == ["imputador", "taxa", "semente", "braco", "S"]
    assert "n_ocultas" in TABELAS["propagacao_analitica_por_ocultas"]
    assert "etapa" in TABELAS["propagacao_analitica_tempos"]
