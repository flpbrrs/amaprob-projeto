"""Testes da escolha de gaveta para perguntar.

Cobrem as duas garantias da escolha: a redução prevista em forma fechada bate com a queda de
variância medida ao revelar a gaveta, e a escolha não olha as gramas das outras células ocultas.
"""

import numpy as np
from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from amaprob.experiments.pergunta import ESTRATEGIAS, escolher, prever
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.pergunta import reducao_por_gaveta
from amaprob.regression.propagacao_analitica import prever_com_entrada_incerta


def _regressao(n: int = 4_000, k: int = 4, semente: int = 0):
    """Regressão com posterior apertado (n grande), para o termo desprezado ser de segunda ordem."""
    rng = np.random.default_rng(semente)
    P = np.ones((n, k), dtype=bool)
    C = rng.standard_normal((n, 13))
    G = rng.gamma(2.0, 30.0, (n, k))
    y = G.sum(axis=1) + rng.normal(0.0, 5.0, n)
    return RegressaoLinearBayesiana().fit(montar_Z(G, P, C), y), P, C


def _entrada_gaussiana(n: int, k: int, M: np.ndarray, semente: int = 1):
    """Eg e Cg em gramas, gaussianos e simétricos, com covariância zero fora de M."""
    rng = np.random.default_rng(semente)
    Eg = rng.uniform(50.0, 400.0, (n, k))
    A = rng.normal(0.0, 30.0, (n, k, k))
    Cg = np.einsum("nij,nkj->nik", A, A)
    diagonal = np.arange(k)
    Cg[:, diagonal, diagonal] += rng.uniform(50.0, 200.0, (n, k))
    M_par = M[:, :, None] & M[:, None, :]
    return Eg, Cg * M_par


def test_reducao_bate_com_a_queda_de_variancia():
    n, k = 300, 4
    blr, P_reg, C_reg = _regressao()
    P, C = P_reg[:n], C_reg[:n]  # as mesmas 300 refeições da entrada
    rng = np.random.default_rng(2)
    M = rng.random((n, k)) < 0.6
    M[:, 0] = True  # toda refeição tem ao menos uma oculta
    Eg, Cg = _entrada_gaussiana(n, k, M)

    _, var_antes = prever_com_entrada_incerta(blr, Eg, Cg, P, C)
    reducao = reducao_por_gaveta(blr, Eg, Cg)
    assert np.all(reducao[~M] == 0)  # fora de M não há o que perguntar

    k_escolhida = np.argmax(np.where(M, reducao, -np.inf), axis=1)
    linhas = np.arange(n)
    # Covariância condicional dada a gaveta revelada: some a linha e a coluna, no resto não muda.
    condicionada = Cg - Cg[linhas, :, k_escolhida][:, :, None] * Cg[linhas, k_escolhida, :][:, None, :] / Cg[linhas, k_escolhida, k_escolhida][:, None, None]
    _, var_depois = prever_com_entrada_incerta(blr, Eg, condicionada, P, C)

    realizada = var_antes - var_depois
    prevista = reducao[linhas, k_escolhida]
    assert np.all(realizada > 0)
    np.testing.assert_allclose(prevista, realizada, rtol=0.01)

    # O termo desprezado (incerteza dos pesos vezes variância da entrada) é de segunda ordem.
    fracao = np.abs(realizada - prevista) / realizada
    assert fracao.mean() < 0.005


def _sinteticos(n: int = 400, k: int = 5, semente: int = 0):
    rng = np.random.default_rng(semente)
    P = rng.random((n, k)) > 0.05
    W = rng.normal(0.0, 0.2, (k, 2))
    media = rng.normal(5.0, 0.3, k)
    z = rng.standard_normal((n, 2))
    G = np.expm1(media + z @ W.T + rng.normal(0.0, 0.15, (n, k)))
    G = np.where(P, np.maximum(G, 0.0), 0.0)
    C = rng.standard_normal((n, 13))
    M = P & (rng.random((n, k)) < 0.6)
    M[:, 0] = True
    return G, P, M, C


def test_escolha_nao_le_as_outras_ocultas():
    G, P, M, C = _sinteticos()  # 5 gavetas
    vazio = np.zeros_like(P)
    imputador = FA(q=2).fit(ocultar(G, vazio), P, vazio, C)

    # Gramas absurdas nas ocultas: se alguma delas fosse lida, a escolha mudaria.
    G_lixo = G.copy()
    G_lixo[M] = np.random.default_rng(9).uniform(1e5, 1e6, int(M.sum()))

    blr = _regressao(k=G.shape[1])[0]
    referencia = prever(imputador, blr, G, P, M, C)
    lixo = prever(imputador, blr, G_lixo, P, M, C)
    for a, b in zip(referencia, lixo):
        np.testing.assert_allclose(b, a)

    for estrategia in ESTRATEGIAS:
        escolha = escolher(estrategia, referencia[3], referencia[2], blr, M, 0, 0, 0)
        escolha_lixo = escolher(estrategia, lixo[3], lixo[2], blr, M, 0, 0, 0)
        assert np.array_equal(escolha, escolha_lixo), estrategia
        assert np.all(M[np.arange(len(M)), escolha]), estrategia  # só escolhe oculta


def test_estrategias_de_nota_ordinam_como_o_nome_diz():
    """Maior porção escolhe a maior Eg; maior carboidrato pesa Eg pelo coeficiente da média."""
    blr, P, C = _regressao()
    M = np.array([[True, True, False, False]])
    Eg = np.array([[100.0, 300.0, 500.0, 700.0]])
    Cg = np.zeros((1, 4, 4))
    Cg[0, 0, 0] = 100.0
    Cg[0, 1, 1] = 900.0  # a segunda oculta é a mais incerta, mas não a maior

    assert escolher("maior_porcao", np.zeros((1, 4)), Eg, blr, M, 0, 0, 0)[0] == 1

    coeficiente = np.abs(blr.m_[:4]) / blr.desvio_z_[:4]
    esperado = int(np.argmax(np.where(M[0], coeficiente * Eg[0], -np.inf)))
    assert escolher("maior_carboidrato", np.zeros((1, 4)), Eg, blr, M, 0, 0, 0)[0] == esperado
