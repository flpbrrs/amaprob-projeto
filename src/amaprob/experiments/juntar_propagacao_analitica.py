"""Junção das tabelas do braço analítico gravadas por rodadas em paralelo.

Cada processo roda com `--sufixo` e grava os três CSVs com o nome no fim, para não disputar arquivo.
Este script lê o arquivo base (se existir) mais o de cada sufixo, confere as colunas e as chaves, e
grava nos arquivos base.

A conferência falha em vez de juntar quando:
- as colunas diferem entre dois arquivos, em nome ou em ordem;
- a chave se repete, o que indicaria duas rodadas cobrindo a mesma célula.

A chave é (imputador, taxa, semente, braço, S) nas métricas. Nas tabelas com mais de uma linha por
braço, a chave inclui o que as distingue: a classe de ocultas, na tabela por ocultas, e a etapa, na
tabela de tempos.

Uso: uv run python -m amaprob.experiments.juntar_propagacao_analitica --sufixos ppca ppca_condicional
"""

import argparse
from pathlib import Path

import pandas as pd

from amaprob.data.dataset import RAIZ

SAIDA = RAIZ / "results"
TABELAS = {
    "propagacao_analitica_metricas": ["imputador", "taxa", "semente", "braco", "S"],
    "propagacao_analitica_por_ocultas": ["imputador", "taxa", "semente", "braco", "n_ocultas", "S"],
    "propagacao_analitica_tempos": ["imputador", "etapa", "taxa", "semente"],
}


def _arquivo(nome: str, sufixo: str, pasta: Path) -> Path:
    return pasta / f"{nome}{f'_{sufixo}' if sufixo else ''}.csv"


def juntar(nome: str, sufixos: list[str], pasta: Path = SAIDA) -> pd.DataFrame:
    """Junta o arquivo base e os de cada sufixo, conferindo colunas e chaves."""
    ausentes = [_arquivo(nome, s, pasta).name for s in sufixos if not _arquivo(nome, s, pasta).exists()]
    if ausentes:
        raise FileNotFoundError(f"arquivos ausentes: {', '.join(ausentes)}")
    arquivos = [a for a in (_arquivo(nome, s, pasta) for s in ["", *sufixos]) if a.exists()]
    if not arquivos:
        raise FileNotFoundError(f"nenhum arquivo da tabela {nome} em {pasta}")

    partes = [pd.read_csv(a, float_precision="round_trip") for a in arquivos]
    colunas = list(partes[0].columns)
    for arquivo, parte in zip(arquivos[1:], partes[1:], strict=True):
        if list(parte.columns) != colunas:
            diferentes = sorted(set(colunas) ^ set(parte.columns))
            raise ValueError(f"colunas diferentes em {arquivo.name}: {diferentes}")

    juntas = pd.concat(partes, ignore_index=True)
    chave = juntas[TABELAS[nome]].copy()
    chave["taxa"] = chave["taxa"].round(6)  # o CSV guarda a taxa com ruído de ponto flutuante
    repetidas = chave.duplicated()
    if repetidas.any():
        exemplo = juntas.loc[repetidas, TABELAS[nome]].head(3).to_string(index=False)
        raise ValueError(f"chave repetida em {nome}:\n{exemplo}")
    return juntas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sufixos", nargs="+", required=True)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    escolha = parser.parse_args()

    for nome in TABELAS:
        juntas = juntar(nome, escolha.sufixos, escolha.saida)
        juntas.to_csv(escolha.saida / f"{nome}.csv", index=False)
        print(f"{nome}: {len(juntas)} linhas de {1 + len(escolha.sufixos)} arquivo(s)")


if __name__ == "__main__":
    main()
