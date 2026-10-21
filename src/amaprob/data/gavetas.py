"""Gavetas: agrupamento dos alimentos da POF em K grupos, a partir da classificação do IBGE.

A classificação do IBGE ("Classificação dos grupos de Consumo Alimentar.xlsx", na documentação
dos microdados) tem três níveis: grupo (20) > subgrupo (116) > item (1.593 códigos V9001).
Uma configuração de gavetas usa o grupo como padrão e desce ao subgrupo só nos grupos indicados.
"""

from pathlib import Path

import pandas as pd

# Açúcar de mesa (V9001 = 6906602): quase sempre registrado como adição a outro alimento.
# Vira gaveta própria: o paciente costuma saber que adoçou, mas não quanto.
COD_ACUCAR = 6906602
GAVETA_ACUCAR = "Açúcar"


def ler_classificacao_ibge(path: Path) -> pd.DataFrame:
    """Lê a planilha de classificação do IBGE. Requer openpyxl: `uv run --with openpyxl`.

    A planilha é hierárquica: linha com nome na 1ª coluna abre um grupo; linha sem código
    e com descrição abre um subgrupo; linha com código é um item.
    """
    raw = pd.read_excel(path, header=None, skiprows=2)
    rows, grupo, subgrupo = [], None, None
    for nome_grupo, codigo, descricao in raw.itertuples(index=False):
        if pd.notna(nome_grupo):
            grupo, subgrupo = str(nome_grupo).strip(), None
        elif pd.isna(codigo) and pd.notna(descricao):
            subgrupo = str(descricao).strip()
        elif pd.notna(codigo):
            rows.append((int(codigo), grupo, subgrupo, str(descricao).strip()))
    return pd.DataFrame(rows, columns=["V9001", "grupo", "subgrupo", "descricao"])


def montar_gavetas(
    classificacao: pd.DataFrame, dividir: set[str] | str, fundir: dict[str, str] | None = None
) -> pd.Series:
    """Retorna a gaveta de cada código V9001.

    `dividir` lista os grupos que descem ao nível de subgrupo; "todos" divide todos.
    `fundir` junta subgrupos de um grupo dividido numa gaveta comum: {subgrupo: nome da gaveta}.
    """
    if dividir == "todos":
        dividir = set(classificacao.grupo)
    subgrupo = classificacao.grupo + " / " + classificacao.subgrupo
    if fundir:
        subgrupo = classificacao.subgrupo.map(fundir).fillna(subgrupo)
    gaveta = classificacao.grupo.where(~classificacao.grupo.isin(dividir), subgrupo)
    gaveta = gaveta.where(classificacao.V9001 != COD_ACUCAR, GAVETA_ACUCAR)
    return pd.Series(gaveta.to_numpy(), index=classificacao.V9001, name="gaveta")


if __name__ == "__main__":
    # Gera configs/classificacao_ibge.csv, versionado para revisão:
    #   uv run --with openpyxl python -m amaprob.data.gavetas
    raiz = Path(__file__).resolve().parents[3]
    doc = raiz / "data/raw/pof2017_2018/Documentacao_20230713"
    planilha = next(doc.glob("Classifica*grupos*.xlsx"))
    ler_classificacao_ibge(planilha).to_csv(raiz / "configs/classificacao_ibge.csv", index=False)

