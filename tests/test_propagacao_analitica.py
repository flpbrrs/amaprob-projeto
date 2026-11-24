"""Testes do braço de propagação analítica, em dados sintéticos.

Cobrem as três peças novas: `momentos_ocultos` (condicional em log), `momentos_em_gramas` (log-normal)
e `prever_com_entrada_incerta` (lei da variância total com a regressão linear bayesiana).

O teste de referência de `momentos_ocultos` usa células de porção média alta: as amostras de
`Imputer.sample` são truncadas em 0 e, com média perto de 5 no log e desvio pequeno, o truncamento
quase não pesa, então a média e a covariância empíricas comparam com a condicional gaussiana.
"""

import subprocess
import sys

import numpy as np
import pytest

from amaprob.imputation.base import ocultar
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.imputation.fa import FA
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.imputation.ppca import PPCA
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagacao_analitica import momentos_em_gramas, prever_com_entrada_incerta

N_AMOSTRAS = 20_000
SORTEIOS_MONTE_CARLO = 5_000

FABRICAS = {
    "ppca": lambda: PPCA(q=2),
    "ppca_condicional": lambda: PPCA(q=2, media_condicional=True),
    "fa": lambda: FA(q=2),
    "fa_condicional": lambda: FA(q=2, media_condicional=True),
    "porcao_media": lambda: PorcaoMedia(),
    "porcao_condicional": lambda: PorcaoMedia(media_condicional=True),
}


def _dados(n: int = 60, k: int = 4, q: int = 2, semente: int = 0):
    """Refeições sintéticas com porção média alta, para o truncamento em 0 não pesar."""
    rng = np.random.default_rng(semente)
    P = rng.random((n, k)) > 0.1
    W = rng.normal(0.0, 0.2, (k, q))
    media = rng.normal(5.0, 0.3, k)
    z = rng.standard_normal((n, q))
    G = np.expm1(media + z @ W.T + rng.normal(0.0, 0.15, (n, k)))
    G = np.where(P, np.maximum(G, 0.0), 0.0)
    C = rng.standard_normal((n, 13))
    M = P & (rng.random((n, k)) < 0.6)
    return G, P, M, C


def _blr(G: np.ndarray, P: np.ndarray, C: np.ndarray, semente: int = 0) -> RegressaoLinearBayesiana:
    y = G.sum(axis=1) + np.random.default_rng(semente).normal(0.0, 5.0, len(G))
    return RegressaoLinearBayesiana().fit(montar_Z(G, P, C), y)


def test_sem_ocultas_coincide_com_predict():
    G, P, _, C = _dados(n=120, k=5)
    blr = _blr(G, P, C)
    M = np.zeros_like(P)
    media_log = np.full(G.shape, np.nan)
    cov_log = np.zeros((len(G), G.shape[1], G.shape[1]))

    Eg, Cg = momentos_em_gramas(G, P, M, media_log, cov_log)
    np.testing.assert_allclose(Eg, G)
    assert not Cg.any()

    media, variancia = prever_com_entrada_incerta(blr, Eg, Cg, P, C)
    media_blr, variancia_blr = blr.predict(montar_Z(G, P, C))
    np.testing.assert_allclose(media, media_blr, atol=1e-10)
    np.testing.assert_allclose(variancia, variancia_blr, atol=1e-10)


@pytest.mark.parametrize("nome", list(FABRICAS))
def test_momentos_batem_com_as_amostras(nome: str):
    G, P, M, C = _dados()
    vazio = np.zeros_like(P)
    imputador = FABRICAS[nome]().fit(ocultar(G, vazio), P, vazio, C)
    X = ocultar(G, M)

    media, cov = momentos_ocultos(imputador, X, P, M, C)
    assert np.isnan(media[~M]).all()
    assert not cov[~M].any()  # linha de célula não oculta inteira em 0

    amostras = imputador.sample(X, P, M, C, N_AMOSTRAS, np.random.default_rng(7))
    emp_media = amostras.mean(axis=0)
    centrado = amostras - emp_media
    emp_cov = np.einsum("sni,snj->nij", centrado, centrado) / (N_AMOSTRAS - 1)

    np.testing.assert_allclose(media[M], emp_media[M], rtol=0.03)

    # A covariância empírica de 20.000 sorteios carrega erro de Monte Carlo de ~1% do desvio no
    # caso da variância (diagonal) e ~0,7% na covariância entre gavetas. O limite soma 3% da escala
    # natural a 3 erros-padrão de Monte Carlo, para o teste medir o modelo e não o sorteio. No
    # agregado, onde o erro cai com o número de células, vale o nominal de 3%.
    par = M[:, :, None] & M[:, None, :]
    escala = np.sqrt(np.einsum("nii->ni", cov))
    erro_padrao = np.sqrt(
        (np.einsum("nii->ni", cov)[:, :, None] * np.einsum("nii->ni", cov)[:, None, :] + cov**2)
        / (N_AMOSTRAS - 1)
    )
    limite = 0.03 * escala[:, :, None] * escala[:, None, :] + 3 * erro_padrao
    assert np.all(np.abs(emp_cov - cov)[par] <= limite[par])
    np.testing.assert_allclose(emp_cov[par].mean(), cov[par].mean(), rtol=0.03)


def test_analitica_bate_com_monte_carlo():
    rng = np.random.default_rng(1)
    n, k = 25, 3
    P = np.ones((n, k), dtype=bool)
    C = rng.standard_normal((n, 13))
    G = rng.gamma(2.0, 30.0, (n, k))
    blr = _blr(G, P, C)

    M = rng.random((n, k)) < 0.5
    media_log = np.where(M, rng.normal(4.5, 0.3, (n, k)), np.nan)
    A = rng.normal(0.0, 0.25, (n, k, k))
    cov_log = np.einsum("nij,nkj->nik", A, A)
    diagonal = np.arange(k)
    cov_log[:, diagonal, diagonal] += rng.uniform(0.10, 0.20, (n, k))
    cov_log *= M[:, :, None] & M[:, None, :]

    Eg, Cg = momentos_em_gramas(G, P, M, media_log, cov_log)
    media, variancia = prever_com_entrada_incerta(blr, Eg, Cg, P, C)

    # Monte Carlo exato, sem truncamento: sorteia o log, converte para gramas e usa a lei da
    # variância total sobre as preditivas de cada sorteio.
    amostras_log = np.repeat(np.log1p(G)[None], SORTEIOS_MONTE_CARLO, axis=0)
    for i in range(n):
        ocultas = np.flatnonzero(M[i])
        if not len(ocultas):
            continue
        bloco = cov_log[i][np.ix_(ocultas, ocultas)] + 1e-12 * np.eye(len(ocultas))
        L = np.linalg.cholesky(bloco)
        sorteio = rng.standard_normal((SORTEIOS_MONTE_CARLO, len(ocultas)))
        amostras_log[:, i, ocultas] = media_log[i, ocultas] + sorteio @ L.T

    Z = montar_Z(
        np.expm1(amostras_log).reshape(-1, k),
        np.tile(P, (SORTEIOS_MONTE_CARLO, 1)),
        np.tile(C, (SORTEIOS_MONTE_CARLO, 1)),
    )
    media_s, variancia_s = blr.predict(Z)
    media_mc = media_s.reshape(SORTEIOS_MONTE_CARLO, n).mean(axis=0)
    variancia_mc = (variancia_s.reshape(SORTEIOS_MONTE_CARLO, n) + media_s.reshape(SORTEIOS_MONTE_CARLO, n) ** 2).mean(axis=0)
    variancia_mc -= media_mc**2

    erro_padrao_media = np.sqrt(variancia_mc) / np.sqrt(SORTEIOS_MONTE_CARLO)
    assert np.all(np.abs(media - media_mc) <= 0.02 * np.sqrt(variancia) + 3 * erro_padrao_media)
    # A variância estimada por Monte Carlo tem erro relativo de ~sqrt(2/S) = 2%; o limite soma o
    # nominal de 2% a 3 erros-padrão. As médias sobre as refeições, sem o ruído do sorteio, fecham
    # no nominal.
    erro_padrao_variancia = variancia_mc * np.sqrt(2.0 / SORTEIOS_MONTE_CARLO)
    assert np.all(np.abs(variancia - variancia_mc) <= 0.02 * variancia + 3 * erro_padrao_variancia)
    np.testing.assert_allclose(media.mean(), media_mc.mean(), rtol=0.02)
    np.testing.assert_allclose(variancia.mean(), variancia_mc.mean(), rtol=0.02)


def test_covariancia_em_gramas_propriedades():
    G, P, M, C = _dados(n=40)
    vazio = np.zeros_like(P)
    imputador = FA(q=2).fit(ocultar(G, vazio), P, vazio, C)

    media_log, cov_log = momentos_ocultos(imputador, ocultar(G, M), P, M, C)
    Eg, Cg = momentos_em_gramas(G, P, M, media_log, cov_log)

    np.testing.assert_allclose(Cg, np.swapaxes(Cg, 1, 2))
    par = M[:, :, None] & M[:, None, :]
    assert np.all(Cg[~par] == 0)
    assert np.all(np.linalg.eigvalsh(Cg) >= -1e-8)
    assert np.all(Eg[P & ~M] == G[P & ~M])
    assert np.all(Eg[~P] == 0)


def test_lixo_nas_ocultas_nao_muda_o_resultado():
    G, P, M, C = _dados()
    vazio = np.zeros_like(P)
    imputador = PPCA(q=2).fit(ocultar(G, vazio), P, vazio, C)
    X = ocultar(G, M)
    X_com_lixo = X.copy()
    X_com_lixo[M] = np.random.default_rng(3).normal(0.0, 1e3, int(M.sum()))

    media, cov = momentos_ocultos(imputador, X, P, M, C)
    media_lixo, cov_lixo = momentos_ocultos(imputador, X_com_lixo, P, M, C)
    np.testing.assert_allclose(media_lixo, media, equal_nan=True)
    np.testing.assert_allclose(cov_lixo, cov)

    Eg, Cg = momentos_em_gramas(G, P, M, media, cov)
    Eg_lixo, Cg_lixo = momentos_em_gramas(G, P, M, media_lixo, cov_lixo)
    np.testing.assert_allclose(Eg_lixo, Eg)
    np.testing.assert_allclose(Cg_lixo, Cg)


def test_salvaguarda_de_plausibilidade():
    G = np.zeros((3, 2))
    P = np.ones((3, 2), dtype=bool)
    M = P.copy()
    media_log = np.full((3, 2), 20.0)  # expm1(20) ≈ 5 × 10⁸ g
    cov_log = np.zeros((3, 2, 2))
    with pytest.raises(ValueError, match="média"):
        momentos_em_gramas(G, P, M, media_log, cov_log)

    # Média moderada e variância grande no log: o teto de 1 t entra pelo desvio-padrão em gramas.
    media_log = np.full((3, 2), -2.0)
    cov_log = np.zeros((3, 2, 2))
    diagonal = np.arange(2)
    cov_log[:, diagonal, diagonal] += 28.0
    with pytest.raises(ValueError, match="desvio-padrão"):
        momentos_em_gramas(G, P, M, media_log, cov_log)


@pytest.mark.parametrize("modulo", [
    "amaprob.experiments.propagacao_analitica",
    "amaprob.experiments.juntar_propagacao_analitica",
    "amaprob.experiments.pergunta",
])
def test_modulo_do_experimento_importa_e_responde_help(modulo: str):
    """O `--help` força o import e a montagem do argparse.

    Os testes de cálculo não passam por aqui, e foi assim que uma anotação `-> Path` sem o import de
    `pathlib` derrubou a primeira rodada em paralelo. O subprocesso reproduz a chamada real.
    """
    resultado = subprocess.run(
        [sys.executable, "-m", modulo, "--help"], capture_output=True, text=True, timeout=600
    )
    assert resultado.returncode == 0, resultado.stderr[-2_000:]
    assert "usage" in resultado.stdout

