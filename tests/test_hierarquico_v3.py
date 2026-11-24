"""Testes do hierárquico v3 (variante por gaveta, Ψ* por refeição)."""

import numpy as np
import pytest

from amaprob.imputation import media_condicional as mc
from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from amaprob.imputation.hierarquico_v3 import (
    PISO_PSI, condicionar_com_psi, gavetas_no_piso, menor_autovalor_das_ocultas, momentos_com_pessoa_v3,
    psi_estrela,
)
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.imputation.ppca import observados
from test_pergunta import _sinteticos


@pytest.fixture(scope="module", params=[False, True], ids=["media_fixa", "media_condicional"])
def modelo(request):
    G, P, M, C = _sinteticos(n=600)
    vazio = np.zeros_like(P)
    fa = FA(q=2, media_condicional=request.param).fit(ocultar(G, vazio), P, vazio, C)
    return fa, G, P, M, C


def test_tau_zero_reproduz_a_fa(modelo):
    fa, G, P, M, C = modelo
    zero = np.zeros(G.shape)
    media, cov = momentos_com_pessoa_v3(fa, G, P, M, C, zero, zero, np.zeros(G.shape[1]))
    media_fa, cov_fa = momentos_ocultos(fa, ocultar(G, M), P, M, C)
    np.testing.assert_allclose(media, media_fa, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov, cov_fa, rtol=0, atol=1e-12)


def test_psi_igual_para_todas_coincide_com_o_condicionamento_padrao(modelo):
    fa, G, P, M, C = modelo
    O = observados(P, M)
    X_O = np.where(O, ocultar(G, M), 0.0)
    U = mc.montar_u(P, C, fa.padronizacao_) if fa.media_condicional else None
    media, cov = condicionar_com_psi(fa, X_O, O, M, U, np.broadcast_to(fa.psi_, G.shape))
    media_fa, cov_fa = momentos_ocultos(fa, ocultar(G, M), P, M, C)
    np.testing.assert_allclose(media, media_fa, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cov, cov_fa, rtol=0, atol=1e-12)


def test_sem_historico_e_a_fa_exata(modelo):
    """v = τ², δ̂ = 0 e τ² < ψ: ψ* = ψ bit a bit, e média e covariância saem idênticas à FA."""
    fa, G, P, M, C = modelo
    tau2 = 0.5 * fa.psi_
    v = np.broadcast_to(tau2, G.shape).copy()
    np.testing.assert_array_equal(psi_estrela(fa, v, tau2), np.broadcast_to(fa.psi_, G.shape))
    assert len(gavetas_no_piso(fa, tau2)) == 0


def test_com_historico_encolhe_e_continua_psd(modelo):
    fa, G, P, M, C = modelo
    tau2 = 2.0 * fa.psi_  # τ² > ψ em todas: piso acionado
    assert set(gavetas_no_piso(fa, tau2)) == set(range(G.shape[1]))
    rng = np.random.default_rng(3)
    v = tau2[None, :] * rng.uniform(0.1, 1.0, G.shape)
    delta = rng.normal(0.0, 0.2, G.shape)
    _, cov = momentos_com_pessoa_v3(fa, G, P, M, C, delta, v, tau2)
    assert menor_autovalor_das_ocultas(cov, M) >= PISO_PSI * 0.999
    # A variância de cada oculta é ao menos ψ* da célula.
    psi = psi_estrela(fa, v, tau2)
    diagonal = np.einsum("nkk->nk", cov)
    assert np.all(diagonal[M] >= psi[M] - 1e-12)


def test_media_soma_delta_de_volta_e_nao_le_as_ocultas(modelo):
    fa, G, P, M, C = modelo
    tau2 = 0.3 * fa.psi_
    v = 0.5 * np.broadcast_to(tau2, G.shape)
    delta = np.full(G.shape, 0.4)
    media, cov = momentos_com_pessoa_v3(fa, G, P, M, C, delta, v, tau2)
    G_lixo = G.copy()
    G_lixo[M] = 1e6
    media_lixo, cov_lixo = momentos_com_pessoa_v3(fa, G_lixo, P, M, C, delta, v, tau2)
    np.testing.assert_array_equal(media, media_lixo)
    np.testing.assert_array_equal(cov, cov_lixo)

    # Sem observadas (taxa 1), z segue a priori: a média é b(u) + δ̂ e a covariância W Wᵀ + Ψ*.
    M_tudo = P.copy()
    media1, cov1 = momentos_com_pessoa_v3(fa, G, P, M_tudo, C, delta, v, tau2)
    media_fa, _ = momentos_ocultos(fa, ocultar(G, M_tudo), P, M_tudo, C)
    np.testing.assert_allclose(media1[M_tudo], media_fa[M_tudo] + 0.4, atol=1e-12)
    n = int(np.flatnonzero(M_tudo.sum(axis=1) >= 2)[0])
    h = np.flatnonzero(M_tudo[n])
    esperado = fa.W_[h] @ fa.W_[h].T + np.diag(psi_estrela(fa, v, tau2)[n, h])
    np.testing.assert_allclose(cov1[n][np.ix_(h, h)], esperado, atol=1e-12)
