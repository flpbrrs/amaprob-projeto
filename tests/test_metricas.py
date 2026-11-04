"""Métricas com casos de valor conhecido."""

import numpy as np
import pytest

from amaprob.evaluation.metricas import Z_INTERVALO, avaliar, cobertura_por_faixa


def test_mae_e_meia_dose():
    m = avaliar([10, 20, 30], [12, 20, 20], [1, 1, 1])
    assert m["mae_g"] == pytest.approx(4.0)
    assert m["pct_meia_dose"] == pytest.approx(200 / 3)


def test_normal_padrao_em_zero():
    m = avaliar([0.0], [0.0], [1.0])
    assert m["nll"] == pytest.approx(0.5 * np.log(2 * np.pi))  # ≈ 0,9189
    assert m["crps_g"] == pytest.approx(2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))  # 2φ(0) − 1/√π ≈ 0,2337


def test_escala_do_crps():
    y, media, desvio = np.array([1.0, -2.0, 0.5]), np.array([0.0, 0.3, 1.0]), np.array([1.0, 2.0, 0.5])
    base = avaliar(y, media, desvio**2)["crps_g"]
    assert avaliar(10 * y, 10 * media, (10 * desvio) ** 2)["crps_g"] == pytest.approx(10 * base)


def test_picp_e_mpiw():
    m = avaliar([0.0, 1.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert m["picp"] == pytest.approx(2 / 3)
    assert m["mpiw_g"] == pytest.approx(2 * Z_INTERVALO)
    assert Z_INTERVALO == pytest.approx(1.6448536, abs=1e-6)


def test_cobertura_por_faixa():
    rng = np.random.default_rng(0)
    media = rng.uniform(10, 100, size=1_003)
    y = media + rng.normal(size=media.size)
    tabela = cobertura_por_faixa(y, media, np.ones_like(media))
    assert len(tabela) == 5
    assert tabela.n.sum() == media.size
    assert (tabela.media_max_g.to_numpy()[:-1] <= tabela.media_min_g.to_numpy()[1:]).all()
    assert tabela.picp.between(0.8, 1.0).all()
