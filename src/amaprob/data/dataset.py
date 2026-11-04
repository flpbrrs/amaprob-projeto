"""Dataset de refeições e máscaras persistidos em data/processed/ (Regras 3 e 4).

Dois arquivos .npz, lidos sem pickle:
- refeicoes_k29.npz: G (gramas por gaveta), P (presença), C (contexto) e nomes das suas colunas,
  y (carboidrato), meal_id, person_id, conjunto (treino/validacao/teste) e nomes das gavetas,
  todos na mesma ordem de linhas;
- mascaras_k29.npz: M (sementes × taxas × N × K), taxas, sementes, meal_id e gavetas.

As máscaras ficam em arquivo próprio para que o dataset possa ser regravado (por exemplo, com o
contexto C) sem tocar nelas. A carga confere o alinhamento entre os dois arquivos.

Preparação: uv run python -m amaprob.data.dataset
"""

import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from amaprob.data.gavetas import DIVIDIR_K29, FUNDIR_K29, montar_gavetas
from amaprob.data.mascaras import SEMENTES, TAXAS, gerar_mascaras
from amaprob.data.pof import read_consumo_alimentar
from amaprob.data.refeicoes import gramas_por_gaveta, montar_itens_de_refeicao
from amaprob.data.splits import dividir_por_pessoa

RAIZ = Path(__file__).resolve().parents[3]
ARQUIVO_REFEICOES = RAIZ / "data/processed/refeicoes_k29.npz"
ARQUIVO_MASCARAS = RAIZ / "data/processed/mascaras_k29.npz"
SEED_SPLIT = 0

# Contexto C: só campos do arquivo de consumo alimentar.
OCASIOES = {1: "cafe", 2: "almoco", 3: "lanche", 4: "jantar", 5: "ceia"}  # V9017
REGIOES = {1: "norte", 2: "nordeste", 3: "sudeste", 4: "sul", 5: "centro_oeste"}  # 1º dígito da UF
C_NOMES = (
    *(f"ocasiao_{nome}" for nome in OCASIOES.values()),
    "hora_sen",
    "hora_cos",
    *(f"regiao_{nome}" for nome in REGIOES.values()),
    "log_renda",
)


@dataclass(frozen=True)
class Refeicoes:
    """Uma linha por refeição, em ordem crescente de meal_id; K colunas de gavetas."""

    G: np.ndarray  # (N, K) gramas por gaveta, sem transformação
    P: np.ndarray  # (N, K) bool, gaveta presente na refeição
    C: np.ndarray  # (N, d) contexto: ocasião, hora cíclica, macrorregião, log da renda
    C_nomes: np.ndarray  # (d,) nomes das colunas de C
    y: np.ndarray  # (N,) carboidrato total da refeição (g)
    meal_id: np.ndarray  # (N,)
    person_id: np.ndarray  # (N,) texto
    conjunto: np.ndarray  # (N,) "treino", "validacao" ou "teste"
    gavetas: np.ndarray  # (K,) nomes das gavetas, na ordem das colunas


@dataclass(frozen=True)
class Mascaras:
    """M[s, t] é a máscara da semente sementes[s] na taxa taxas[t], alinhada às refeições."""

    M: np.ndarray  # (S, T, N, K) bool
    taxas: np.ndarray  # (T,)
    sementes: np.ndarray  # (S,)


def codificar_hora(v9015: pd.Series) -> np.ndarray:
    """Hora cheia ("12H") como ponto no círculo (seno, cosseno de 2π·h/24): 23H fica perto de 0H."""
    angulo = 2 * np.pi * v9015.str.rstrip("H").astype(int).to_numpy() / 24
    return np.c_[np.sin(angulo), np.cos(angulo)]


def montar_contexto(por_refeicao: pd.DataFrame) -> np.ndarray:
    """Matriz C (N × 13), uma linha por refeição, na ordem de C_NOMES.

    Ocasião e macrorregião viram indicadoras completas (sem categoria de referência); com prior
    gaussiano na regressão, a colinearidade com o intercepto não impede o ajuste. A renda
    (constante por pessoa) entra como log1p, com valores negativos levados a 0.
    """
    ocasiao = por_refeicao.V9017.to_numpy()[:, None] == np.array(list(OCASIOES))
    regiao = (por_refeicao.UF.to_numpy() // 10)[:, None] == np.array(list(REGIOES))
    renda = np.log1p(np.clip(por_refeicao.RENDA_TOTAL.to_numpy(dtype=np.float64), 0, None))
    return np.c_[ocasiao, codificar_hora(por_refeicao.V9015), regiao, renda].astype(np.float64)


def montar_refeicoes(consumo: pd.DataFrame, gavetas: pd.Series) -> Refeicoes:
    """Monta as matrizes por refeição a partir dos itens da POF e de um mapeamento V9001 → gaveta."""
    itens = montar_itens_de_refeicao(consumo, gavetas)
    G = gramas_por_gaveta(itens)  # linhas em meal_id crescente; colunas por ponto de código
    por_refeicao = itens.groupby("meal_id")
    person_id = por_refeicao.person_id.first().loc[G.index]
    # Presença pelos itens, não pelas gramas: uma gaveta com item de 0 g continua presente.
    contagem = pd.crosstab(itens.meal_id, itens.gaveta).reindex(index=G.index, columns=G.columns, fill_value=0)
    # Ocasião e hora são constantes na refeição por definição; UF e renda, por pessoa.
    primeiro = por_refeicao[["V9017", "V9015", "UF", "RENDA_TOTAL"]].first().loc[G.index]
    return Refeicoes(
        G=G.to_numpy(dtype=np.float64),
        P=contagem.to_numpy() > 0,
        C=montar_contexto(primeiro),
        C_nomes=np.array(C_NOMES),
        y=por_refeicao.CHOTOT.sum().loc[G.index].to_numpy(dtype=np.float64),
        meal_id=G.index.to_numpy(dtype=np.int64),
        person_id=person_id.to_numpy(dtype=str),
        conjunto=dividir_por_pessoa(person_id, seed=SEED_SPLIT).to_numpy(dtype=str),
        gavetas=G.columns.to_numpy(dtype=str),
    )


def salvar_refeicoes(refeicoes: Refeicoes, path: Path = ARQUIVO_REFEICOES) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **asdict(refeicoes))


def carregar_refeicoes(path: Path = ARQUIVO_REFEICOES) -> Refeicoes:
    with np.load(path, allow_pickle=False) as arquivo:
        return Refeicoes(**{campo.name: arquivo[campo.name] for campo in fields(Refeicoes)})


def salvar_mascaras(mascaras: Mascaras, refeicoes: Refeicoes, path: Path = ARQUIVO_MASCARAS) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **asdict(mascaras), meal_id=refeicoes.meal_id, gavetas=refeicoes.gavetas)


def carregar_mascaras(refeicoes: Refeicoes, path: Path = ARQUIVO_MASCARAS) -> Mascaras:
    """Carrega as máscaras e confere que foram geradas para estas refeições.

    Conferências: mesmo meal_id na mesma ordem; mesmas gavetas; máscara da taxa 1 igual a P.
    O meal_id sozinho não basta: trocar a configuração de gavetas mantém os meal_id e muda P.
    """
    with np.load(path, allow_pickle=False) as arquivo:
        mascaras = Mascaras(M=arquivo["M"], taxas=arquivo["taxas"], sementes=arquivo["sementes"])
        meal_id, gavetas = arquivo["meal_id"], arquivo["gavetas"]
    if not np.array_equal(meal_id, refeicoes.meal_id):
        raise ValueError(f"Máscaras desalinhadas: o meal_id de {path} não coincide, na ordem, com o do dataset.")
    if not np.array_equal(gavetas, refeicoes.gavetas):
        raise ValueError(f"Máscaras desalinhadas: as gavetas de {path} diferem das do dataset.")
    taxa_um = np.flatnonzero(mascaras.taxas == 1.0)
    if len(taxa_um) and not all(np.array_equal(m, refeicoes.P) for m in mascaras.M[:, taxa_um[0]]):
        raise ValueError(f"Máscaras desalinhadas: a máscara da taxa 1 em {path} difere da presença P do dataset.")
    return mascaras


def main() -> None:
    inicio = time.perf_counter()
    consumo = read_consumo_alimentar(RAIZ / "data/raw/pof2017_2018/Dados_20230713/CONSUMO_ALIMENTAR.txt")
    classificacao = pd.read_csv(RAIZ / "configs/classificacao_ibge.csv")
    refeicoes = montar_refeicoes(consumo, montar_gavetas(classificacao, DIVIDIR_K29, FUNDIR_K29))
    salvar_refeicoes(refeicoes)
    mascaras = Mascaras(M=gerar_mascaras(refeicoes.P), taxas=np.array(TAXAS), sementes=np.array(SEMENTES))
    salvar_mascaras(mascaras, refeicoes)
    duracao = time.perf_counter() - inicio

    N, K = refeicoes.G.shape
    presentes = refeicoes.P.sum()
    print(f"refeições: {N:,} | gavetas: {K} | células presentes: {presentes:,}")
    pessoas = pd.Series(refeicoes.person_id).groupby(refeicoes.conjunto).nunique()
    print("pessoas por conjunto:", pessoas.to_dict())
    taxa_obs = mascaras.M.sum(axis=(2, 3)) / presentes
    print("taxa observada (linhas = sementes, colunas = taxas):")
    print(pd.DataFrame(taxa_obs, index=SEMENTES, columns=TAXAS).round(5).to_string())
    for arq in (ARQUIVO_REFEICOES, ARQUIVO_MASCARAS):
        print(f"{arq.name}: {arq.stat().st_size / 2**20:.1f} MB")
    print(f"tempo total: {duracao:.1f} s")


if __name__ == "__main__":
    main()
