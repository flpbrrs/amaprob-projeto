"""Testes do hierárquico v4: Σ_novo = Σ_FA − c·D, c = min(1; 0,9/λ_max)."""

import numpy as np
import pytest

from amaprob.imputation.base import ocultar
from amaprob.imputation.hierarquico import Variancias
from amaprob.imputation.hierarquico_v2 import momentos_com_pessoa_v2
from amaprob.imputation.hierarquico_v4 import (
    RHO, VARIANTES_V4, fator_c, lambda_max, menor_autovalor_folga, momentos_com_pessoa_v4, reducao,
)
from amaprob.imputation.momentos import momentos_ocultos
from test_hierarquico_v2 import _diferenca_max, _fa


def _variancias(k: int, tau2: float, tau2_fator: float | None = None) -> Variancias:
    return Variancias(tau2=np.full(k, tau2), sigma2=np.full(k, 0.3),
                      tau2_fator=tau2 if tau2_fator is None else tau2_fator, sigma2_fator=0.3)


def _caso(semente: int):
    fa, G, P, C = _fa(semente=semente)
    M = P & (np.random.default_rng(semente + 10).random(P.shape) < 0.6)
    return fa, G, P, C, M


@pytest.mark.parametrize("variante", VARIANTES_V4)
def test_sem_historico_e_a_fa_exata(variante):
    """v = τ²: D = 0, c = 1 e Σ_novo = Σ_FA bit a bit (conferência 1, em todas as gavetas)."""
    fa, G, P, C, M = _caso(0)
    variancias = _variancias(G.shape[1], tau2=5.0)  # τ² grande: na v3 acionaria o piso
    v = (np.full_like(G, variancias.tau2_fator) if variante == "fator"
         else np.broadcast_to(variancias.tau2, G.shape).copy())
    zeros = np.zeros_like(G)
    media, cov, c, _ = momentos_com_pessoa_v4(fa, G, P, M, C, zeros, v, variancias, variante)
    media_fa, cov_fa = momentos_ocultos(fa, ocultar(G, M), P, M, C)
    assert _diferenca_max(media, media_fa) == 0.0
    assert np.array_equal(cov, cov_fa)
    assert np.all(c == 1.0)


@pytest.mark.parametrize("variante", VARIANTES_V4)
def test_reducao_pequena_reproduz_a_v2(variante):
    """Conferência 3: com D pequeno, c = 1 em toda refeição e a v4 é a v2 em 1e-12."""
    fa, G, P, C, M = _caso(1)
    variancias = _variancias(G.shape[1], tau2=0.02)
    rng = np.random.default_rng(2)
    v = 0.02 * rng.uniform(0.5, 1.0, G.shape)
    if variante == "fator":
        v = np.repeat(v[:, :1], G.shape[1], axis=1)  # constante por refeição
    delta = rng.normal(0.0, 0.1, G.shape)
    media4, cov4, c, _ = momentos_com_pessoa_v4(fa, G, P, M, C, delta, v, variancias, variante)
    media2, cov2, est = momentos_com_pessoa_v2(fa, G, P, M, C, delta, v, variancias, variante)
    assert np.all(c == 1.0)
    assert est.projecoes_piso == 0  # a v2 não projetou: a comparação é da fórmula
    assert _diferenca_max(media4, media2) <= 1e-12
    np.testing.assert_allclose(cov4, cov2, rtol=0, atol=1e-12)


@pytest.mark.parametrize("variante", VARIANTES_V4)
def test_reducao_grande_fica_acima_de_10_por_cento_da_fa(variante):
    """Conferência 2: Σ_novo − 0,1·Σ_FA ⪰ 0 e diag(Σ_novo) ≤ diag(Σ_FA), com c < 1 de fato."""
    fa, G, P, C, M = _caso(2)
    variancias = _variancias(G.shape[1], tau2=50.0)  # na v2, diagonal negativa
    v = np.zeros_like(G)  # histórico completo: V = 0
    _, cov, c, cov_fa = momentos_com_pessoa_v4(fa, G, P, M, C, np.zeros_like(G), v, variancias, variante)
    com_oculta = M.any(axis=1)
    assert np.all(c[com_oculta] < 1.0)
    assert menor_autovalor_folga(cov, cov_fa, M) >= -1e-10
    diag, diag_fa = np.einsum("nii->ni", cov), np.einsum("nii->ni", cov_fa)
    assert np.all(diag[M] <= diag_fa[M])
    # c·λ_max = ρ: a restrição fica ativa (menor autovalor da folga ≈ 0 em cada refeição)
    d = reducao(v, M, variancias, variante)
    np.testing.assert_allclose((c * lambda_max(cov_fa, d, M, variante))[com_oculta], RHO, rtol=1e-12)


@pytest.mark.parametrize("variante", VARIANTES_V4)
def test_lambda_max_bate_com_o_autovalor_generalizado(variante):
    fa, G, P, C, M = _caso(3)
    variancias = _variancias(G.shape[1], tau2=0.4)
    v = np.random.default_rng(4).uniform(0.0, 0.4, G.shape)
    if variante == "fator":
        v = np.repeat(v[:, :1], G.shape[1], axis=1)
    _, cov_fa = momentos_ocultos(fa, ocultar(G, M), P, M, C)
    d = reducao(v, M, variancias, variante)
    lam = lambda_max(cov_fa, d, M, variante)
    for n in np.flatnonzero(M.any(axis=1))[:30]:
        h = np.flatnonzero(M[n])
        S = cov_fa[n][np.ix_(h, h)]
        D = np.diag(d[n, h]) if variante == "gaveta" else d[n, h[0]] * np.ones((len(h), len(h)))
        autovalores = np.linalg.eigvals(np.linalg.solve(S, D)).real
        np.testing.assert_allclose(lam[n], autovalores.max(), rtol=1e-10, atol=1e-14)
    assert np.all(lam[~M.any(axis=1)] == 0.0)


def test_fator_c():
    np.testing.assert_array_equal(fator_c(np.array([0.0, 0.5, 0.9, 1.8, 9.0])), [1.0, 1.0, 1.0, 0.5, 0.1])


def test_lixo_nas_ocultas_nao_muda_nada():
    fa, G, P, C, M = _caso(5)
    variancias = _variancias(G.shape[1], tau2=3.0)
    v = np.full_like(G, 0.5)
    delta = np.random.default_rng(6).normal(0.0, 0.1, G.shape)
    G_lixo = G.copy()
    G_lixo[M] = 1e6
    for variante in VARIANTES_V4:
        a = momentos_com_pessoa_v4(fa, G, P, M, C, delta, v, variancias, variante)
        b = momentos_com_pessoa_v4(fa, G_lixo, P, M, C, delta, v, variancias, variante)
        assert _diferenca_max(a[0], b[0]) == 0.0
        for x, y in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(x, y)
