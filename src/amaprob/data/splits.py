"""Divisão treino / validação / teste por indivíduo (Regra 3: os dois dias de uma pessoa ficam juntos)."""

import numpy as np
import pandas as pd

TREINO, VALIDACAO, TESTE = "treino", "validacao", "teste"


def dividir_por_pessoa(
    person_ids: pd.Series, fracoes: tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 0
) -> pd.Series:
    """Rótulo do conjunto de cada linha, sorteado por pessoa com semente fixa."""
    pessoas = np.sort(person_ids.unique())
    sorteio = np.random.default_rng(seed).permutation(pessoas)
    n_treino = round(fracoes[0] * len(pessoas))
    n_validacao = round(fracoes[1] * len(pessoas))
    rotulos = np.full(len(pessoas), TESTE, dtype=object)
    rotulos[:n_treino] = TREINO
    rotulos[n_treino : n_treino + n_validacao] = VALIDACAO
    return person_ids.map(dict(zip(sorteio, rotulos)))
