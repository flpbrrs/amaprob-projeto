"""Efeito da pessoa sobre a FA condicional: x_nk = b_k(u_n) + δ_pk + W_k z_n + ε_nk.

δ_pk ~ N(0, τ_k²) é a porção da pessoa na gaveta k, e b_k(u) é a média condicional da FA
(`fa.py`, `media_condicional.py`), que permanece como está. O modelo estende a FA sem alterá-la:
o deslocamento δ entra no condicionamento e volta na saída, e a incerteza de δ entra na variância
das quantidades ocultas. Três peças:

1. `estimar_variancias`, no treino (pessoas com 2 dias): τ_k² é a covariância, entre pessoas, da
   média do resíduo no dia 1 com a do dia 2 (piso em 0), e σ_k² é a variância do resíduo por
   refeição menos τ_k² (piso positivo). Na variante de fator único um só δ_p cobre todas as
   gavetas, com τ² e σ² estimados do mesmo jeito sobre o fator de porção (a média dos resíduos
   das células presentes da refeição).

2. `estimar_delta`: a posterior conjunta de δ só a partir do histórico (Regra 2: o histórico é de
   outro dia, e as células ocultas do dia avaliado não entram). Com n_pk células da pessoa na
   gaveta k no histórico (uma por refeição) e média de resíduo r̄_pk,

       v_pk = (1/τ_k² + n_pk/σ_k²)⁻¹,   δ̂_pk = v_pk · n_pk · r̄_pk / σ_k².

   Sem dado da gaveta no histórico, δ̂_pk = 0 e v_pk = τ_k²; na variante por gaveta, o fator
   único do histórico é o valor de partida nesse caso (δ̂_pk = δ̂_p, o posterior do fator), com
   v_pk = τ_k². No fator único, as observações são os fatores por refeição (média dos resíduos
   das células observadas da refeição), n_p conta refeições com ao menos uma célula observada, e
   o mesmo par de fórmulas vale com τ² e σ² do fator.

3. `amostrar_com_pessoa` e `momentos_com_pessoa`: reaproveitam `FA.sample` e `momentos_ocultos`
   sem mudá-los. As células observadas do dia avaliado entram deslocadas (X_O − δ̂) no
   condicionamento e voltam com o valor verdadeiro na saída; nas ocultas, δ̂_p,k é somado de
   volta e v_pk entra na variância; na amostragem, um sorteio N(0, v_pk) por célula e amostra.

Aproximações declaradas: (a) as observações do próprio dia avaliado não atualizam δ, só o
histórico atualiza; (b) o corte da FA em log1p(g) ≥ 0 acontece no espaço deslocado, e a volta ao
espaço original recorta em ≥ 0, do mesmo jeito que a FA já corta.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from amaprob.imputation import media_condicional as mc
from amaprob.imputation.base import ocultar
from amaprob.imputation.momentos import momentos_ocultos

PISO_VARIANCIA = 1e-6
VARIANTES = ("nenhum", "fator", "gaveta")


@dataclass(frozen=True)
class Variancias:
    """τ² e σ² do efeito da pessoa, por gaveta e do fator de porção único."""

    tau2: np.ndarray  # (K,) entre-pessoas por gaveta
    sigma2: np.ndarray  # (K,) por refeição, já subtraído tau2
    tau2_fator: float
    sigma2_fator: float


def media_condicional(fa, P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """b_k(u) (N, K): média da FA condicional dada só a presença e o contexto."""
    return mc.montar_u(P, C, fa.padronizacao_) @ fa.coeficientes_media_.T


def residuos(fa, G: np.ndarray, P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """r = log1p(gramas) − b_k(u) nas células presentes; NaN fora delas."""
    r = np.log1p(np.asarray(G, dtype=np.float64)) - media_condicional(fa, P, C)
    return np.where(P, r, np.nan)


def _media_dias(valores: pd.Series, pessoa: np.ndarray, dia: np.ndarray) -> pd.DataFrame:
    """Média por pessoa e dia, com as colunas 1 e 2; só quem tem as duas dias sai na linha."""
    tabela = pd.DataFrame({"pessoa": pessoa, "dia": dia, "v": valores})
    medias = tabela.groupby(["pessoa", "dia"], sort=False).v.mean().unstack("dia")
    return medias.reindex(columns=[1, 2]).dropna()


def estimar_variancias(residuo: np.ndarray, pessoa: np.ndarray, dia: np.ndarray) -> Variancias:
    """τ_k² e σ_k² por gaveta e do fator único, com as pessoas do treino que têm 2 dias.

    τ_k² = covariância, entre pessoas, da média do resíduo no dia 1 com a do dia 2 (piso em 0);
    σ_k² = variância do resíduo por refeição menos τ_k² (piso em PISO_VARIANCIA). O fator de
    porção de uma refeição é a média dos resíduos das células presentes dela, e o mesmo par de
    contas vale sobre ele.
    """
    residuo = np.asarray(residuo, dtype=np.float64)
    N, K = residuo.shape
    pessoa, dia = np.asarray(pessoa), np.asarray(dia)

    tau2, sigma2 = np.zeros(K), np.full(K, PISO_VARIANCIA)
    for k in range(K):
        r_k = residuo[:, k]
        presente = ~np.isnan(r_k)
        if presente.sum() < 2:
            continue
        medias = _media_dias(pd.Series(r_k[presente]), pessoa[presente], dia[presente])
        if len(medias) >= 2:
            tau2[k] = max(float(np.cov(medias[1].to_numpy(), medias[2].to_numpy())[0, 1]), 0.0)
        variancia = float(np.var(r_k[presente], ddof=1))
        sigma2[k] = max(variancia - tau2[k], PISO_VARIANCIA)

    conta = (~np.isnan(residuo)).sum(axis=1)
    fator = np.where(conta > 0, np.nansum(residuo, axis=1) / np.maximum(conta, 1), np.nan)
    tem = ~np.isnan(fator)
    medias_f = _media_dias(pd.Series(fator[tem]), pessoa[tem], dia[tem])
    tau2_f = 0.0
    if len(medias_f) >= 2:
        tau2_f = max(float(np.cov(medias_f[1].to_numpy(), medias_f[2].to_numpy())[0, 1]), 0.0)
    sigma2_f = max(float(np.var(fator[tem], ddof=1)) - tau2_f, PISO_VARIANCIA)
    return Variancias(tau2=tau2, sigma2=sigma2, tau2_fator=tau2_f, sigma2_fator=sigma2_f)


def estimar_delta(
    residuo_historico: np.ndarray,
    observado_historico: np.ndarray,
    codigos: np.ndarray,
    n_pessoas: int,
    variancias: Variancias,
    variante: str = "gaveta",
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior de δ por pessoa e gaveta a partir do histórico: δ̂ (n_pessoas, K) e v (n_pessoas, K).

    `observado_historico` diz o que o cenário permite ler do histórico (no cenário completo, as
    células presentes; no mascarado, P & ~M da mesma taxa e semente). `codigos` associa cada linha
    do histórico a uma pessoa do conjunto avaliado.
    """
    if variante not in VARIANTES:
        raise ValueError(f"variante desconhecida: {variante!r}")
    K = np.asarray(residuo_historico).shape[1]
    if variante == "nenhum":
        return np.zeros((n_pessoas, K)), np.zeros((n_pessoas, K))

    obs = np.asarray(observado_historico, dtype=bool) & ~np.isnan(residuo_historico)
    codigos = np.asarray(codigos, dtype=int)

    # fator único: uma observação por refeição com ao menos uma célula observada
    conta_f = obs.sum(axis=1)
    tem_f = conta_f > 0
    soma_f = np.where(obs, residuo_historico, 0.0).sum(axis=1)
    media_f = np.where(tem_f, soma_f / np.maximum(conta_f, 1), 0.0)
    n_f = np.bincount(codigos[tem_f], minlength=n_pessoas).astype(np.float64)
    soma_f_p = np.bincount(codigos[tem_f], weights=media_f[tem_f], minlength=n_pessoas)
    with np.errstate(divide="ignore"):  # tau2 = 0: prior de precisão infinita, posterior exata em 0
        precisao_f = np.float64(1.0) / np.float64(variancias.tau2_fator)
        v_f = 1.0 / (precisao_f + n_f / variancias.sigma2_fator)
    delta_f = v_f * soma_f_p / variancias.sigma2_fator  # n_f = 0 → 0
    if variante == "fator":
        return (
            np.repeat(delta_f[:, None], K, axis=1),
            np.repeat(v_f[:, None], K, axis=1),
        )

    # por gaveta: uma observação por célula do histórico (uma por refeição)
    linhas, colunas = np.nonzero(obs)
    indice = codigos[linhas] * K + colunas
    total = n_pessoas * K
    n = np.bincount(indice, minlength=total).astype(np.float64).reshape(n_pessoas, K)
    soma = np.bincount(
        indice, weights=np.asarray(residuo_historico, dtype=np.float64)[linhas, colunas], minlength=total
    ).reshape(n_pessoas, K)
    with np.errstate(divide="ignore"):
        v = 1.0 / (1.0 / variancias.tau2[None, :] + n / variancias.sigma2[None, :])
    delta = v * soma / variancias.sigma2[None, :]
    sem_dado = n == 0  # fator único como valor de partida, v_pk = tau2 (a fórmula já dá tau2)
    delta = np.where(sem_dado, delta_f[:, None], delta)
    v = np.where(sem_dado, variancias.tau2[None, :], v)
    return delta, v


def amostrar_com_pessoa(
    fa,
    G: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    delta: np.ndarray,
    v: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    rng_ruido: np.random.Generator,
) -> np.ndarray:
    """Amostras (S, N, K) em log1p(gramas): condiciona a FA em X_O − δ̂ e devolve o espaço original.

    Nas células ocultas, δ̂_p,k é somado de volta e um sorteio N(0, v_pk) por célula e amostra
    acrescenta a incerteza da posterior de δ. Nas observadas, sai o valor verdadeiro: o deslocamento
    é só para o condicionamento. `rng_ruido` é separado de `rng` para que o sorteio da FA seja o
    mesmo com e sem efeito da pessoa.
    """
    X = ocultar(G, M)
    amostras = fa.sample(X - delta, P, M, C, n_samples, rng)

    linhas, colunas = np.nonzero(M)
    ruido = rng_ruido.normal(0.0, 1.0, size=(n_samples, len(linhas))) * np.sqrt(v[linhas, colunas])
    amostras[:, linhas, colunas] = np.maximum(
        amostras[:, linhas, colunas] + delta[linhas, colunas] + ruido, 0.0
    )
    obs_linhas, obs_colunas = np.nonzero(P & ~M)
    amostras[:, obs_linhas, obs_colunas] = X[obs_linhas, obs_colunas]
    return amostras


def momentos_com_pessoa(
    fa,
    G: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    delta: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Média (N, K) e covariância (N, K, K) da condicional no log, com δ̂ e v do efeito da pessoa.

    Com δ̂ = 0 e v = 0 devolve exatamente `momentos_ocultos`; com δ̂ = 0 e v = τ², a média não muda
    e só a diagonal das ocultas cresce v.
    """
    X = ocultar(G, M)
    media, covariancia = momentos_ocultos(fa, X - delta, P, M, C)
    linhas, colunas = np.nonzero(M)
    media[linhas, colunas] += delta[linhas, colunas]
    covariancia[linhas, colunas, colunas] += v[linhas, colunas]
    return media, covariancia
