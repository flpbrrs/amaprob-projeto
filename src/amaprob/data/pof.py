"""Leitura dos microdados de consumo alimentar da POF 2017–2018 (Pesquisa de Orçamentos Familiares, IBGE).

Layout de largura fixa conferido no "Dicionário de variáveis" e no programa de leitura em R
distribuídos pelo IBGE (versão 20230713).

Regra 1: as colunas da medida caseira (V9005, V9007, GRAMATURA1, COD_UNIDADE_MEDIDA_FINAL)
não são lidas. QTD = V9005 × GRAMATURA1, então expô-las tornaria o problema determinístico.

Regra 2: CHOTOT e demais nutrientes são QTD × composição. Servem apenas como alvo (y);
usados como entrada, vazariam as quantidades mascaradas. Por isso só CHOTOT é lido.
"""

from pathlib import Path

import pandas as pd

# (nome, posição inicial 1-based, tamanho)
_CONSUMO_LAYOUT = [
    ("UF", 1, 2),
    ("ESTRATO_POF", 3, 4),
    ("TIPO_SITUACAO_REG", 7, 1),
    ("COD_UPA", 8, 9),
    ("NUM_DOM", 17, 2),
    ("NUM_UC", 19, 1),
    ("COD_INFORMANTE", 20, 2),
    ("QUADRO", 22, 2),
    ("SEQ", 24, 2),
    ("V9001", 32, 7),  # código do alimento
    ("V9015", 39, 3),  # horário ("8H")
    ("V9016", 42, 2),  # forma de preparação declarada
    ("V9017", 44, 1),  # ocasião de consumo
    ("V9018", 45, 1),  # local da refeição
    ("COD_PREPARACAO_FINAL", 60, 2),
    ("QTD", 69, 9),  # gramas consumidas
    ("COD_TBCA", 78, 6),
    ("CHOTOT", 126, 14),  # carboidrato total (g)
    ("PESO_FINAL", 504, 15),
    ("RENDA_TOTAL", 519, 10),
    ("DIA_SEMANA", 529, 15),
    ("DIA_ATIPICO", 544, 1),
]

_STRING_COLUMNS = {"V9015", "COD_TBCA", "DIA_SEMANA"}


def read_consumo_alimentar(path: Path) -> pd.DataFrame:
    """Lê CONSUMO_ALIMENTAR.txt, uma linha por item consumido."""
    names = [name for name, _, _ in _CONSUMO_LAYOUT]
    colspecs = [(start - 1, start - 1 + width) for _, start, width in _CONSUMO_LAYOUT]
    dtypes = {name: "string" for name in names if name in _STRING_COLUMNS}
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=dtypes, header=None)
    for col in _STRING_COLUMNS:
        df[col] = df[col].str.strip()
    return df
