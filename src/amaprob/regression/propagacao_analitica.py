"""Propagação analítica da incerteza da imputação na regressão linear bayesiana.

Os braços existentes levam a incerteza da imputação por amostragem (S sorteios de `Imputer.sample`).
Este trata a entrada como variável aleatória e resolve a preditiva em forma fechada, numa passada e
sem gerador aleatório.

Regressão linear bayesiana (derivação em `blr.py`), com φ = (Z − media_z) / desvio_z e
w ~ N(m, S), condicionada na entrada aleatória Z:

    E[y | Z] = mᵀ φ(Z) + media_y
    Var[y | Z] = 1/β + φ(Z)ᵀ S φ(Z)

Pela lei da variância total, com média φ̄ = E[φ] e covariância Cφ = Cov(φ),

    E[y] = mᵀ φ̄ + media_y
    Var[y] = 1/β + φ̄ᵀ S φ̄ + mᵀ Cφ m + tr(S Cφ)

Os dois primeiros termos são a preditiva do `predict`; os dois últimos são a incerteza da entrada.

Só as K colunas de gramas são aleatórias: P e C são sempre observados e entram com variância 0.
Por isso os blocos usados são m_K e S_KK, e Cφ = Cg / (s sᵀ), com s = desvio_z das gramas.

Momentos em gramas. A condicional da imputação é gaussiana em log1p(gramas), com média μ e
covariância Σ (`imputation/momentos.py`). Se g = exp(x) − 1 com x ~ N(μ, Σ), a log-normal dá

    E[g_k] = exp(μ_k + Σ_kk / 2) − 1
    Cov(g_i, g_j) = exp(μ_i + μ_j + (Σ_ii + Σ_jj)/2) · (exp(Σ_ij) − 1)

Como 1 + E[g_k] = exp(μ_k + Σ_kk/2), então a covariância sai como
a_i a_j (exp(Σ_ij) − 1), com a = 1 + E[g].

Aproximação declarada: `Imputer.sample` corta as amostras em `log1p(g) >= 0`, ou seja em `g >= 0`, e
esta forma fechada não corta. Sem o corte, `g = expm1(x)` cairia em `(-1, 0)`, então o efeito é de no
máximo 1 g por célula oculta, desprezível contra a escala das porções.

Regra 2: `momentos_em_gramas` lê G só para copiar as células observadas (P & ~M); o resto é 0.
"""

import numpy as np

from amaprob.regression.base import montar_Z

LIMITE_G = 1e6  # 1 t em gramas, o teto da salvaguarda de plausibilidade
BLOCO = 4_096  # refeições por bloco: a covariância em gramas também é N × K × K


def momentos_em_gramas(
    G: np.ndarray, P: np.ndarray, M: np.ndarray, media_log: np.ndarray, cov_log: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Média (N, K) e covariância (N, K, K) em gramas: observadas copiadas, ocultas pela log-normal."""
    G = np.asarray(G, dtype=np.float64)
    P = np.asarray(P, dtype=bool)
    M = np.asarray(M, dtype=bool)
    O = P & ~M

    N, K = G.shape
    Eg = np.where(O, G, 0.0)
    Cg = np.zeros((N, K, K))
    for inicio in range(0, N, BLOCO):
        b = slice(inicio, inicio + BLOCO)
        M_b = M[b]
        linhas, colunas = np.nonzero(M_b)
        media = np.asarray(media_log[b], dtype=np.float64)[linhas, colunas]
        variancia = np.asarray(cov_log[b], dtype=np.float64)[linhas, colunas, colunas]
        Eg[b][linhas, colunas] = np.expm1(media + 0.5 * variancia)
        escala = np.zeros(M_b.shape)  # a = 1 + E[g] nas ocultas, 0 fora de M
        escala[linhas, colunas] = Eg[b][linhas, colunas] + 1.0
        Cg[b] = escala[:, :, None] * np.expm1(np.asarray(cov_log[b], dtype=np.float64)) * escala[:, None, :]

    _conferir(Eg, Cg)
    return Eg, Cg


def _conferir(Eg: np.ndarray, Cg: np.ndarray) -> None:
    """Só detecta: valor não finito ou acima de 1 t levanta erro com a gaveta, nunca corta.

    O teto vale para quantidades em gramas, então a covariância entra pela raiz da diagonal. A
    variância em g² não tem a mesma dimensão do teto, e as gavetas de porção grande (bebidas chegam
    a 9 kg observados) têm desvio imputado na casa do quilo sem nada de degenerado.
    """
    desvio = np.sqrt(np.einsum("nii->ni", Cg))
    for nome, valores in (("média", Eg), ("desvio-padrão", desvio)):
        problema = ~np.isfinite(valores) | (valores > LIMITE_G)
        if problema.any():
            gaveta = int(np.flatnonzero(problema.any(axis=0))[0])
            raise ValueError(f"{nome} em gramas não finito ou acima de 1 t na gaveta {gaveta}")


def prever_com_entrada_incerta(
    blr, Eg: np.ndarray, Cg: np.ndarray, P: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Média e variância da preditiva com a entrada de média Eg e covariância Cg (equações no topo)."""
    K = Eg.shape[1]
    phi = (montar_Z(Eg, P, C) - blr.media_z_) / blr.desvio_z_
    desvio_gramas = blr.desvio_z_[:K]
    C_phi = Cg / (desvio_gramas[:, None] * desvio_gramas[None, :])
    variancia = (
        1.0 / blr.beta_
        + np.einsum("ni,ij,nj->n", phi, blr.S_, phi)
        + np.einsum("i,nij,j->n", blr.m_[:K], C_phi, blr.m_[:K])
        + np.einsum("ij,nji->n", blr.S_[:K, :K], C_phi)
    )
    return phi @ blr.m_ + blr.media_y_, variancia
