"""Métricas para preditivas gaussianas (média e variância por refeição), em gramas.

- MAE (erro absoluto médio): erro da média prevista.
- NLL (log-verossimilhança negativa média): penaliza tanto erro quanto incerteza mal calibrada.
- CRPS (continuous ranked probability score): distância entre a distribuição prevista e o valor
  observado, na unidade do alvo; reduz-se ao erro absoluto quando a previsão é determinística.
- PICP (cobertura) e MPIW (largura média) do intervalo central de 90%.
- Percentual de meia dose: refeições com erro de até 7,5 g, meia unidade de insulina na relação
  1 U : 15 g, o passo de uma caneta comum (ver docs/piso-de-erro.md).
"""

from statistics import NormalDist

import numpy as np
import pandas as pd
import scoringrules as sr

NIVEL = 0.90
Z_INTERVALO = NormalDist().inv_cdf(0.5 + NIVEL / 2)  # ≈ 1,645
TOLERANCIA_MEIA_DOSE_G = 7.5


def avaliar(y: np.ndarray, media: np.ndarray, var: np.ndarray) -> dict:
    y, media, var = (np.asarray(a, dtype=np.float64) for a in (y, media, var))
    desvio = np.sqrt(var)
    erro = np.abs(y - media)
    meia_largura = Z_INTERVALO * desvio
    return {
        "mae_g": erro.mean(),
        "nll": np.mean(0.5 * np.log(2 * np.pi * var) + (y - media) ** 2 / (2 * var)),
        "crps_g": np.mean(sr.crps_normal(y, media, desvio)),
        "picp": np.mean(erro <= meia_largura),
        "mpiw_g": np.mean(2 * meia_largura),
        "pct_meia_dose": 100 * np.mean(erro <= TOLERANCIA_MEIA_DOSE_G),
    }


def cobertura_por_faixa(y: np.ndarray, media: np.ndarray, var: np.ndarray, n_faixas: int = 5) -> pd.DataFrame:
    """PICP por quintil da média prevista: a global pode esconder sobra nas refeições pequenas
    e falta nas grandes."""
    dentro = np.abs(np.asarray(y) - media) <= Z_INTERVALO * np.sqrt(var)
    faixa = pd.qcut(media, n_faixas, labels=False)
    tabela = pd.DataFrame({"faixa": faixa, "media": media, "dentro": dentro}).groupby("faixa").agg(
        media_min_g=("media", "min"), media_max_g=("media", "max"), n=("dentro", "size"), picp=("dentro", "mean")
    )
    return tabela.reset_index()
