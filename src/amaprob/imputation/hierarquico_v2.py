"""Contabilidade de variância do modelo hierárquico: Σ_novo = Σ_FA − T + V.

A FA condicional foi ajustada em todas as refeições, de modo que a covariância condicional Σ_FA
que ela devolve já contém a variação entre pessoas τ_k². Somar a posterior v_pk por cima (a
versão em `hierarquico.py`) contava τ² duas vezes: sem histórico o modelo ficava com a variância
da FA mais τ², o que infla a média em gramas pela log-normal e afasta a cobertura da referência.
Aqui a covariância das células ocultas do dia avaliado é corrigida, em log1p(gramas):

    Σ_novo = Σ_FA − T + V     (bloco oculto de cada refeição)

- por gaveta: T = diag(τ_k²) e V = diag(v_pk);
- fator único: T = τ²·11ᵀ e V = v_p·11ᵀ, porque o fator é comum às gavetas da refeição;
- sem pessoa (variante `nenhum`): T = V = 0, logo Σ_novo = Σ_FA.

A média continua μ_novo = μ_FA + δ̂, como em `hierarquico.py`. Sem histórico, v = τ² exatamente:
`estimar_delta` devolveria 1/(1/τ²), que pode diferir de τ² em 1 ulp, então `estimar_delta_v2`
grava o próprio prior onde não há dado; assim V = T bit a bit e Σ_novo = Σ_FA, a igualdade
exata exigida pela conferência do H2 na taxa 1,0. Com histórico, v < τ² e o intervalo encolhe.

Se Σ_novo não for semidefinida positiva numa refeição (a subtração de T pode deixar a diagonal
negativa quando τ² grande), os autovalores abaixo de PISO_AUTOVALOR são elevados a esse piso;
contam-se separadamente as refeições com autovalores negativas (não PSD) e as que precisaram do
piso. A projeção acontece uma vez, aqui, antes da amostragem e da análise, para que os dois
braços usem a mesma matriz.

`amostrar_bloco` sorteia as ocultas de N(μ_novo, Σ_novo) por refeição (Cholesky da matriz com a
diagonal fora de M preenchida com 1, o que a deixa bloco-diagonal, bloco oculto ⊕ identidade,
sem alterar a marginal das ocultas), trunca em 0 e devolve em log1p(gramas), o mesmo recorte e
a mesma escala de `FA.sample`. Esse caminho difere do `fa.sample` oficial, que sortearia z e o
ruído da FA por conta própria; a comparação entre os dois caminhos é feita na rodada
(`experiments/hierarquico_v2.py`), com a NLL e a PICP do sem pessoa dentro do erro de Monte
Carlo da rodada v1, que usou o caminho oficial.

Regra 2: `momentos_com_pessoa_v2` lê as gramas só por `ocultar`, que apaga as células de M;
lixo nelas não altera média, covariância nem amostras.
"""

from dataclasses import dataclass

import numpy as np

from amaprob.imputation.base import ocultar
from amaprob.imputation.hierarquico import VARIANTES, Variancias, estimar_delta
from amaprob.imputation.momentos import momentos_ocultos

PISO_AUTOVALOR = 1e-8
BLOCO_PROJECAO = 2_048


@dataclass(frozen=True)
class Estatisticas:
    """Conferências da correção numa rodada (cenário, taxa, semente, variante)."""

    excesso_diag_max: float  # máx diag(Σ_novo) − diag(Σ_FA) sobre as ocultas (exigido ≤ 1e-12)
    projecoes_piso: int      # refeições com autovalores abaixo do piso (piso aplicado)
    projecoes_negativas: int  # refeições com autovalores negativas: Σ_novo não PSD


def estimar_delta_v2(
    residuo_historico: np.ndarray,
    observado_historico: np.ndarray,
    codigos: np.ndarray,
    n_pessoas: int,
    variancias: Variancias,
    variante: str = "gaveta",
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior de δ de `estimar_delta`, com o prior exato onde não há dado no histórico.

    No fator único, `estimar_delta` calcula v = 1/(1/τ² + n/σ²); com n = 0 isso é 1/(1/τ²), que
    pode diferir de τ² em 1 ulp. Aqui n = 0 grava o próprio τ², para que V = T seja exato na
    conferência do H2 na taxa 1,0 (Σ_novo = Σ_FA sem diferença). A variante por gaveta já grava
    τ_k² diretamente em `estimar_delta`.
    """
    delta, v = estimar_delta(
        residuo_historico, observado_historico, codigos, n_pessoas, variancias, variante
    )
    if variante == "fator":
        obs = np.asarray(observado_historico, dtype=bool) & ~np.isnan(residuo_historico)
        n_f = np.bincount(np.asarray(codigos, dtype=int)[obs.any(axis=1)], minlength=n_pessoas)
        v = np.where((n_f == 0)[:, None], np.float64(variancias.tau2_fator), v)
    return delta, v


def momentos_com_pessoa_v2(
    fa,
    G: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    delta: np.ndarray,
    v: np.ndarray,
    variancias: Variancias,
    variante: str,
) -> tuple[np.ndarray, np.ndarray, Estatisticas]:
    """Média (N, K), covariância (N, K, K) e conferências de Σ_novo = Σ_FA − T + V, no log.

    Condiciona a FA em X_O − δ̂, devolve δ̂ às ocultas e aplica a correção de variância. As
    células fora de M ficam com média NaN e covariância 0, como em `momentos_ocultos`.
    """
    if variante not in VARIANTES:
        raise ValueError(f"variante desconhecida: {variante!r}")
    M = np.asarray(M, dtype=bool)
    X = ocultar(G, M)
    media, covariancia = momentos_ocultos(fa, X - delta, P, M, C)
    linhas, colunas = np.nonzero(M)
    media[linhas, colunas] += delta[linhas, colunas]

    diag_antes = np.einsum("nii->ni", covariancia)
    if variante == "gaveta":
        correcao = (v - variancias.tau2[None, :])[linhas, colunas]
        covariancia[linhas, colunas, colunas] += correcao
    elif variante == "fator":
        # c_n = v_p − τ² constante por refeição: entra em todo o par de células ocultas
        correcao = np.where(M, v - variancias.tau2_fator, 0.0)
        covariancia += correcao[:, :, None] * M[:, None, :]
    # variante "nenhum": T = V = 0, Σ_novo = Σ_FA

    diag_depois = np.einsum("nii->ni", covariancia)
    excesso = float(np.max((diag_depois - diag_antes)[M])) if M.any() else 0.0
    piso, negativas = _projetar(covariancia, M)
    return media, covariancia, Estatisticas(excesso, piso, negativas)


def _projetar(covariancia: np.ndarray, M: np.ndarray) -> tuple[int, int]:
    """Eleva a PISO_AUTOVALOR os autovalores abaixo do piso no bloco oculto de cada refeição.

    Devolve (refeições com piso aplicado, refeições com autovalores negativas). A checagem usa a
    matriz com a diagonal fora de M preenchida com 1 (bloco oculto ⊕ identidade), cujos
    autovalores são os do bloco oculto e uns; o piso em 1 nunca é atingido pelas direções fora
    de M. A projeção é simétrica (autovalores + autovetores) e reescreve só o bloco oculto.
    """
    N, K = M.shape
    idx = np.arange(K)
    piso_cont = negativas_cont = 0
    for inicio in range(0, N, BLOCO_PROJECAO):
        b = slice(inicio, inicio + BLOCO_PROJECAO)
        A = covariancia[b].copy()
        A[:, idx, idx] = np.where(M[b], np.einsum("nii->ni", A), 1.0)
        minimo = np.linalg.eigvalsh(A).min(axis=1)
        for local in np.flatnonzero(minimo < PISO_AUTOVALOR):
            n = inicio + local
            ocultas = np.flatnonzero(M[n])
            if len(ocultas) == 0:
                continue  # sem ocultas o mínimo é 1, acima do piso
            bloco = covariancia[n][np.ix_(ocultas, ocultas)]
            autovalores, autovetores = np.linalg.eigh((bloco + bloco.T) / 2)
            negativas_cont += int((autovalores < 0).any())
            autovalores = np.maximum(autovalores, PISO_AUTOVALOR)
            covariancia[n][np.ix_(ocultas, ocultas)] = (autovetores * autovalores) @ autovetores.T
        piso_cont += int((minimo < PISO_AUTOVALOR).sum())
    return piso_cont, negativas_cont


def amostrar_bloco(
    media_log: np.ndarray,
    covariancia_log: np.ndarray,
    X: np.ndarray,
    M: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Amostras (S, n, K) em log1p(gramas): N(μ_novo, Σ_novo) por refeição, truncada em 0.

    A diagonal fora de M vira 1 antes do Cholesky, o que deixa a matriz bloco-diagonal sem
    alterar a marginal das ocultas. As células observadas saem com o valor verdadeiro de X (o
    deslocamento δ̂ é só para o condicionamento), e o corte em 0 é o mesmo de `FA.sample`.
    """
    n, K = M.shape
    idx = np.arange(K)
    A = covariancia_log.copy()
    A[:, idx, idx] = np.where(M, np.einsum("nii->ni", A), 1.0)
    L = np.linalg.cholesky(A)
    z = rng.standard_normal((n_samples, n, K))
    ocultas = np.where(M, media_log, 0.0)
    return np.where(
        M[None],
        np.maximum(np.einsum("nij,snj->sni", L, z) + ocultas[None], 0.0),
        X[None],
    )
