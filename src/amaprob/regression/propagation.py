"""Incerteza da imputação levada até a previsão de carboidrato: plug-in, plug-in recalibrado e propagação.

Os três braços saem das mesmas S amostras do imputador, convertidas para gramas por expm1:

- plug-in: prevê uma vez com a média das amostras em gramas;
- plug-in recalibrado: a mesma entrada, num regressor ajustado com o treino imputado da mesma forma;
- propagação: prevê para cada amostra e combina as S preditivas na mistura (1/S) Σ_s N(μ_s, σ²_s).

A média é feita em gramas, e não no log, porque expm1 da média do log subestima a média em gramas.
"""

from collections.abc import Callable, Iterator, Sequence

import numpy as np

from amaprob.evaluation.metricas import ORCAMENTO_BLOCO_BYTES
from amaprob.imputation.base import Imputer, ocultar
from amaprob.regression.base import ProbRegressor, montar_Z

PLUGIN = "plugin"
PLUGIN_RECALIBRADO = "plugin_recalibrado"
PROPAGACAO = "propagacao"
N_DOBRAS = 5
COPIAS_POR_PREVISAO = 3  # Z, φ(Z) e o produto intermediário da variância ficam na memória juntos


def amostras_em_gramas(
    imputador: Imputer, G: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray,
    n_amostras: int, rng: np.random.Generator,
) -> Iterator[tuple[slice, np.ndarray]]:
    largura_Z = 2 * G.shape[1] + C.shape[1]
    tamanho = max(1, ORCAMENTO_BLOCO_BYTES // (n_amostras * largura_Z * 8 * COPIAS_POR_PREVISAO))
    for inicio in range(0, len(G), tamanho):
        b = slice(inicio, inicio + tamanho)
        X = ocultar(G[b], M[b])
        yield b, np.expm1(imputador.sample(X, P[b], M[b], C[b], n_amostras, rng))


def prever_bracos(
    imputador: Imputer, regressor: ProbRegressor, G: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray,
    n_amostras: int, rng: np.random.Generator, recalibrado: ProbRegressor | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Médias e variâncias (S, N) por braço; S = 1 no plug-in e no recalibrado."""
    N = len(G)
    saida = {PLUGIN: (np.empty((1, N)), np.empty((1, N))), PROPAGACAO: (np.empty((n_amostras, N)), np.empty((n_amostras, N)))}
    if recalibrado is not None:
        saida[PLUGIN_RECALIBRADO] = (np.empty((1, N)), np.empty((1, N)))

    for b, G_s in amostras_em_gramas(imputador, G, P, M, C, n_amostras, rng):
        S, B, K = G_s.shape
        Z_plugin = montar_Z(G_s.mean(axis=0), P[b], C[b])
        Z_amostras = montar_Z(G_s.reshape(S * B, K), np.tile(P[b], (S, 1)), np.tile(C[b], (S, 1)))
        modelos = {PLUGIN: (regressor, Z_plugin), PROPAGACAO: (regressor, Z_amostras)}
        if recalibrado is not None:
            modelos[PLUGIN_RECALIBRADO] = (recalibrado, Z_plugin)
        for braco, (modelo, Z) in modelos.items():
            media, var = modelo.predict(Z)
            saida[braco][0][:, b] = media.reshape(-1, B)
            saida[braco][1][:, b] = var.reshape(-1, B)
    return saida


def momentos_da_mistura(medias: np.ndarray, variancias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return medias.mean(axis=0), variancias.mean(axis=0) + medias.var(axis=0)


def dobras_por_pessoa(pessoas: np.ndarray, rng: np.random.Generator, n_dobras: int = N_DOBRAS) -> np.ndarray:
    unicas, posicao_da_refeicao = np.unique(pessoas, return_inverse=True)
    dobra_da_pessoa = np.empty(len(unicas), dtype=int)
    dobra_da_pessoa[rng.permutation(len(unicas))] = np.arange(len(unicas)) % n_dobras
    return dobra_da_pessoa[posicao_da_refeicao]


def ajustar_dobras(
    novo_imputador: Callable[[], Imputer], G: np.ndarray, P: np.ndarray, C: np.ndarray, dobras: np.ndarray,
) -> list[Imputer]:
    """Um imputador por dobra, ajustado com o treino completo das outras dobras.

    Sem máscara, o ajuste não depende da taxa nem da semente: basta um por dobra para todo o
    experimento. Só a amostragem usa a máscara.
    """
    sem_mascara = np.zeros_like(P, dtype=bool)
    ajustados = []
    for f in range(int(dobras.max()) + 1):
        fora = dobras != f
        ajustados.append(novo_imputador().fit(ocultar(G[fora], sem_mascara[fora]), P[fora], sem_mascara[fora], C[fora]))
    return ajustados


def imputar_treino_cruzado(
    imputadores_das_dobras: Sequence[Imputer], G: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray,
    dobras: np.ndarray, n_amostras: int, geradores: Sequence[np.random.Generator],
) -> np.ndarray:
    """Gramas do treino com as células de M preenchidas pela média das amostras em gramas.

    Cada dobra é imputada pelo imputador ajustado sem ela (`ajustar_dobras`), para que nenhum
    imputador veja as gramas que vai preencher.
    """
    G_imputado = np.empty_like(G, dtype=np.float64)
    for f, (imputador, rng) in enumerate(zip(imputadores_das_dobras, geradores, strict=True)):
        dentro = dobras == f
        linhas = np.flatnonzero(dentro)
        for b, G_s in amostras_em_gramas(imputador, G[dentro], P[dentro], M[dentro], C[dentro], n_amostras, rng):
            G_imputado[linhas[b]] = G_s.mean(axis=0)
    return G_imputado
