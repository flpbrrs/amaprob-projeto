"""Montagem das refeições a partir dos itens da POF.

Definição: refeição = mesma pessoa, mesmo dia, mesma ocasião e mesmo
horário. Ficam só as refeições que pedem insulina: café da manhã, almoço, lanche, jantar e ceia,
com pelo menos 10 g de carboidrato.
"""

import pandas as pd

OCASIOES_COM_INSULINA = [1, 2, 3, 4, 5]  # V9017: café, almoço, lanche, jantar, ceia
CARBO_MINIMO_G = 10.0

_CHAVE_PESSOA = ["COD_UPA", "NUM_DOM", "NUM_UC", "COD_INFORMANTE"]
_CHAVE_REFEICAO = ["person_id", "QUADRO", "V9017", "V9015"]  # QUADRO 72/73 = dia 1/2 (a verificar)


def montar_itens_de_refeicao(consumo: pd.DataFrame, gavetas: pd.Series) -> pd.DataFrame:
    """Itens das refeições válidas, com person_id, meal_id e gaveta."""
    itens = consumo[consumo.V9017.isin(OCASIOES_COM_INSULINA)].copy()
    chave = [itens[c].astype(str) for c in _CHAVE_PESSOA]
    itens["person_id"] = chave[0].str.cat(chave[1:], sep="-")
    itens["meal_id"] = itens.groupby(_CHAVE_REFEICAO, sort=False).ngroup()
    carbo = itens.groupby("meal_id").CHOTOT.transform("sum")
    itens = itens[carbo >= CARBO_MINIMO_G]
    itens["gaveta"] = itens.V9001.map(gavetas)
    return itens


def gramas_por_gaveta(itens: pd.DataFrame) -> pd.DataFrame:
    """Matriz refeição × gaveta com a soma das gramas (0 quando a gaveta está ausente)."""
    return itens.pivot_table(index="meal_id", columns="gaveta", values="QTD", aggfunc="sum", fill_value=0.0)
