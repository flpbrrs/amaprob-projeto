"""Testes da análise segmentada: codificação, métricas por refeição e decomposição da variância."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from amaprob.evaluation.metricas import avaliar, avaliar_mistura
from amaprob.experiments.segmentos import (
    agregar_por_segmento, avaliar_linhas, avaliar_linhas_mistura, decompor_variancia,
    montar_segmentos_da_medida, quintis_por_posto, rotular_local, rotular_periodo,
)
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagacao_analitica import momentos_em_gramas, prever_com_entrada_incerta


def test_periodo_nos_quatro_intervalos():
    horas = pd.Series([0, 5, 6, 11, 12, 17, 18, 23])
    assert rotular_periodo(horas).tolist() == [
        "madrugada", "madrugada", "manha", "manha", "tarde", "tarde", "noite", "noite",
    ]


def test_local_dentro_fora_misto_e_sem_informacao():
    rotulo, fracao = rotular_local(
        pd.Series([10.0, 0.0, 10.0, 20.0, 0.0]),
        pd.Series([0.0, 10.0, 30.0, 20.0, 0.0]),
    )
    assert rotulo.tolist() == ["dentro", "fora", "misto", "misto", "nao_informado"]
    assert fracao.iloc[0] == 0.0
    assert fracao.iloc[1] == 1.0
    assert fracao.iloc[2] == pytest.approx(0.75)
    assert fracao.iloc[3] == pytest.approx(0.5)
    assert np.isnan(fracao.iloc[4])


def test_quintis_por_posto_tem_cinco_grupos_iguais_com_empates():
    valores = np.repeat([10.0, 20.0, 30.0], 10)
    quintil = quintis_por_posto(valores)
    assert quintil.nunique() == 5
    assert quintil.value_counts().to_dict() == {"Q1": 6, "Q2": 6, "Q3": 6, "Q4": 6, "Q5": 6}


def test_avaliar_linhas_concorda_com_avaliar():
    rng = np.random.default_rng(0)
    y = rng.normal(50, 20, 200)
    media = y + rng.normal(0, 5, 200)
    var = rng.uniform(1, 100, 200)
    linhas = avaliar_linhas(y, media, var)
    esperado = avaliar(y, media, var)
    for coluna in ("mae_g", "nll", "crps_g", "picp", "mpiw_g", "pct_meia_dose"):
        assert linhas[coluna].mean() == pytest.approx(esperado[coluna], rel=1e-10, abs=1e-10)


def test_avaliar_linhas_mistura_concorda_com_avaliar_mistura():
    rng = np.random.default_rng(1)
    y = rng.normal(50, 20, 60)
    medias = rng.normal(50, 10, (5, 60))
    variancias = rng.uniform(1, 50, (5, 60))
    linhas = avaliar_linhas_mistura(y, medias, variancias)
    esperado = avaliar_mistura(y, medias, variancias)
    for coluna in ("mae_g", "nll", "crps_g", "picp", "mpiw_g", "pct_meia_dose"):
        assert linhas[coluna].mean() == pytest.approx(esperado[coluna], rel=1e-10, abs=1e-10)


def test_agregar_por_segmento_conta_e_calcula_mae_relativo():
    linhas = pd.DataFrame({
        "y": [40.0, 60.0, 80.0, 100.0], "media": [40.0, 60.0, 80.0, 100.0],
        "mae_g": [10.0, 20.0, 30.0, 50.0], "nll": 0.0, "crps_g": 0.0,
        "picp": True, "mpiw_g": 1.0, "pct_meia_dose": 100.0,
    })
    segmentos = pd.DataFrame({"local": pd.Categorical(["dentro", "dentro", "fora", "fora"])})
    tabela = agregar_por_segmento(linhas, segmentos, dimensoes=("local",))
    dentro = tabela.query("valor == 'dentro'").iloc[0]
    assert dentro.n == 2
    assert dentro.carboidrato_medio_g == pytest.approx(50.0)
    assert dentro.mae_relativo == pytest.approx(15.0 / 50.0)


def test_decomposicao_por_gaveta_fecha_com_a_formula_fechada():
    rng = np.random.default_rng(2)
    N, K = 120, 4
    G = rng.uniform(0, 100, (N, K))
    P = np.ones((N, K), dtype=bool)
    M = np.ones((N, K), dtype=bool)
    C = rng.normal(size=(N, 2))
    y = 10 + G @ rng.normal(size=K) + rng.normal(0, 5, N)
    blr = RegressaoLinearBayesiana().fit(montar_Z(G, P, C), y)

    media_log = np.log1p(rng.uniform(10, 100, (N, K)))
    cov_log = np.zeros((N, K, K))
    diagonal = np.arange(K)
    cov_log[:, diagonal, diagonal] = 0.3
    cov_log += 0.02 * rng.normal(size=(N, K, K)) * 0.5
    cov_log = 0.5 * (cov_log + np.transpose(cov_log, (0, 2, 1)))

    contribuicao, total, entrada = decompor_variancia(blr, G, P, M, C, media_log, cov_log)
    Eg, Cg = momentos_em_gramas(G, P, M, media_log, cov_log)
    media, variancia = prever_com_entrada_incerta(blr, Eg, Cg, P, C)
    phi = (montar_Z(Eg, P, C) - blr.media_z_) / blr.desvio_z_
    base = 1.0 / blr.beta_ + np.einsum("ni,ij,nj->n", phi, blr.S_, phi)

    assert np.allclose(total, variancia, rtol=1e-10, atol=1e-10)
    assert np.allclose(entrada, contribuicao.sum(axis=1), rtol=1e-10, atol=1e-10)
    assert np.allclose(total - base, entrada, rtol=1e-10, atol=1e-10)


def test_montar_segmentos_da_medida_alinha_por_meal_id():
    from amaprob.experiments.propagacao import Conjunto

    tabela = pd.DataFrame({
        "local": ["dentro", "fora"], "fracao_fora": [0.0, 1.0], "ocasiao": ["cafe", "almoco"],
        "periodo": ["manha", "tarde"], "dia_semana": ["dia_util", "fim_de_semana"],
        "dia_atipico": ["tipico", "atipico"], "hora": [8, 14],
    }, index=pd.Index([10, 20], name="meal_id"))
    C = np.zeros((2, 13))
    C[0, 0] = 1
    C[1, 1] = 1
    C[0, 7] = 1
    C[1, 10] = 1
    C[:, 12] = [1.0, 2.0]
    medida = Conjunto(
        G=np.ones((2, 2)), P=np.ones((2, 2), dtype=bool), C=C, y=np.array([10.0, 80.0]),
        pessoas=np.array(["a", "b"]), meal_id=np.array([20, 10]), M=np.ones((1, 1, 2, 2), dtype=bool),
    )
    segmentos = montar_segmentos_da_medida(medida, tabela)
    assert segmentos.local.tolist() == ["fora", "dentro"]
    assert segmentos.ocasiao.tolist() == ["almoco", "cafe"]
    assert segmentos.regiao.tolist() == ["norte", "sul"]
    assert segmentos.quintil_carboidrato.nunique() == 2


def test_modulo_do_experimento_importa_e_responde_help():
    resultado = subprocess.run(
        [sys.executable, "-m", "amaprob.experiments.segmentos", "--help"],
        capture_output=True, text=True, timeout=600,
    )
    assert resultado.returncode == 0, resultado.stderr[-2_000:]
    assert "usage" in resultado.stdout
