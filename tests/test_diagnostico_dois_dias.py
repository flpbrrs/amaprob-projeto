"""Testes do diagnóstico dia 1 → dia 2: ICC, correlação e a montagem da matriz por pessoa e dia."""

import numpy as np
import pandas as pd
import pytest

from amaprob.experiments.diagnostico_dois_dias import (
    contar_pessoas_e_refeicoes, correlacao_dias, icc_dois_dias, matriz_dias, resumo_pares,
)


def test_icc_e_um_quando_o_dia_repete_e_negativa_quando_os_dias_trocam():
    repetido = np.array([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [7.0, 7.0]])
    assert icc_dois_dias(repetido) == pytest.approx(1.0)
    trocado = np.array([[1.0, 5.0], [5.0, 1.0], [2.0, 6.0], [6.0, 2.0]])
    assert icc_dois_dias(trocado) < 0  # entre-pessoas menor que dentro-pessoas
    assert np.isnan(icc_dois_dias(np.ones((3, 2))))  # sem variância: sem efeito, ICC indeterminada


def test_correlacao_dias_e_de_pearson():
    valores = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    assert correlacao_dias(valores) == pytest.approx(1.0)
    assert np.isnan(correlacao_dias(np.ones((3, 2))))


def test_matriz_dias_so_termina_quem_tem_os_dois_dias():
    sub = pd.DataFrame({
        "person_id": ["a", "a", "b"],
        "dia": [1, 2, 1],
        "residuo": [1.0, 3.0, 10.0],
    })
    matriz = matriz_dias(sub)
    assert matriz.shape == (1, 2)  # a só tem o dia 1 e fica de fora
    assert matriz[0].tolist() == [1.0, 3.0]


def test_resumo_pares_com_poucas_pessoas_nao_chuta_intervalo():
    sub = pd.DataFrame({
        "person_id": ["a", "a", "b", "b"],
        "dia": [1, 2, 1, 2],
        "residuo": [1.0, 2.0, 3.0, 4.0],
    })
    resumo = resumo_pares("fator_unico", "total", sub)
    assert resumo["n_pessoas"] == 2
    assert np.isnan(resumo["icc_li"]) and np.isnan(resumo["icc_ls"])


def test_contagem_de_pessoas_e_refeicoes_por_dia():
    tabela = pd.DataFrame({
        "person_id": ["a", "a", "b", "c", "c"],
        "dia": [1, 2, 1, 1, 2],
        "conjunto": ["treino"] * 3 + ["validacao"] * 2,
    })
    contagem = contar_pessoas_e_refeicoes(tabela).set_index("conjunto")
    assert contagem.loc["treino", "n_pessoas"] == 2
    assert contagem.loc["treino", "pessoas_2_dias"] == 1
    assert contagem.loc["treino", "pessoas_1_dia"] == 1
    assert contagem.loc["treino", "n_refeicoes"] == 3
    assert contagem.loc["validacao", "pessoas_2_dias"] == 1
