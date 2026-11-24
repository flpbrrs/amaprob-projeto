"""Testes da correção de variância Σ_novo = Σ_FA − T + V (`hierarquico_v2`).

As garantias da v2: sem pessoa (T = V = 0) a FA é reproduzida exatamente; sem histórico
(v = τ²) a covariância volta a ser exatamente a da FA, o que sustenta a conferência do H2 na
taxa 1,0; com histórico a diagonal só encolhe e a correção por gaveta mexe só na diagonal, enquanto
a de fator único mexe no bloco inteiro; Σ_novo não PSD é projetado e contado; a amostragem de
N(μ_novo, Σ_novo) bate com os momentos e copia as observadas; e lixo nas células ocultas do dia
avaliado não altera nada (Regra 2).
"""

import numpy as np

from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from amaprob.imputation.hierarquico import Variancias
from amaprob.imputation.hierarquico_v2 import (
    PISO_AUTOVALOR,
    amostrar_bloco,
    estimar_delta_v2,
    momentos_com_pessoa_v2,
)
from amaprob.imputation.momentos import momentos_ocultos

# 1/(1/x) ≠ x neste valor: é o caso em que a fórmula do fator único sem histórico
# (v = 1/(1/τ²)) divergiria de τ² em 1 ulp e quebraria a igualdade exata da conferência.
TAU2_FATOR_CONTRAEXEMPLO = 0.028382263201310803


def _fa(n: int = 400, k: int = 4, semente: int = 0):
    """FA condicional ajustada em dados sintéticos, sem máscara no ajuste."""
    rng = np.random.default_rng(semente)
    P = rng.random((n, k)) < 0.6
    C = rng.normal(size=(n, 13))
    G = np.where(P, rng.gamma(2.0, 25.0, (n, k)), 0.0)
    M = np.zeros_like(P)
    fa = FA(q=3, semente=0, media_condicional=True, max_iteracoes=60).fit(ocultar(G, M), P, M, C)
    return fa, G, P, C


def _variancias(k: int, tau2: float = 0.05, tau2_fator: float = 0.05) -> Variancias:
    return Variancias(
        tau2=np.full(k, tau2), sigma2=np.full(k, 0.3),
        tau2_fator=tau2_fator, sigma2_fator=0.3,
    )


def _diferenca_max(a: np.ndarray, b: np.ndarray) -> float:
    """Máximo |a − b| com NaN alinhado (a média tem NaN fora de M nas duas)."""
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    assert np.array_equal(nan_a, nan_b)
    dif = np.abs(a[~nan_a] - b[~nan_b])
    return float(dif.max()) if dif.size else 0.0


def test_sem_pessoa_reproduz_a_fa_exatamente():
    fa, G, P, C = _fa()
    k = G.shape[1]
    M = P & (np.random.default_rng(1).random(P.shape) < 0.5)
    variancias = _variancias(k)
    zeros = np.zeros_like(G)

    media, cov, est = momentos_com_pessoa_v2(fa, G, P, M, C, zeros, zeros, variancias, "nenhum")
    base_media, base_cov = momentos_ocultos(fa, ocultar(G, M), P, M, C)
    assert _diferenca_max(media, base_media) == 0.0
    assert _diferenca_max(cov, base_cov) == 0.0
    assert est.excesso_diag_max == 0.0


def test_sem_historico_reproduz_a_fa_exatamente_nas_duas_variantes():
    """v = τ² (posterior sem dado): −T + V = 0 e Σ_novo = Σ_FA bit a bit."""
    fa, G, P, C = _fa(semente=1)
    k = G.shape[1]
    M = P & (np.random.default_rng(2).random(P.shape) < 0.5)
    variancias = _variancias(k)
    zeros = np.zeros_like(G)

    media_ref, cov_ref, _ = momentos_com_pessoa_v2(fa, G, P, M, C, zeros, zeros, variancias, "nenhum")
    v_fator = np.full_like(G, variancias.tau2_fator)
    v_gaveta = np.broadcast_to(variancias.tau2, G.shape).copy()
    for variante, v in (("fator", v_fator), ("gaveta", v_gaveta)):
        media, cov, est = momentos_com_pessoa_v2(fa, G, P, M, C, zeros, v, variancias, variante)
        assert _diferenca_max(media, media_ref) == 0.0, variante
        assert _diferenca_max(cov, cov_ref) == 0.0, variante
        assert est.excesso_diag_max <= 1e-12, variante


def test_com_historico_a_diagonal_so_encolhe():
    fa, G, P, C = _fa(semente=2)
    k = G.shape[1]
    M = P & (np.random.default_rng(3).random(P.shape) < 0.5)
    variancias = _variancias(k)
    zeros = np.zeros_like(G)

    media_ref, cov_ref, _ = momentos_com_pessoa_v2(fa, G, P, M, C, zeros, zeros, variancias, "nenhum")
    diag_ref = np.einsum("nii->ni", cov_ref)
    casos = (
        ("gaveta", np.full_like(G, 0.5 * 0.05)),
        ("fator", np.full_like(G, 0.5 * 0.05)),
    )
    for variante, v in casos:
        media, cov, est = momentos_com_pessoa_v2(fa, G, P, M, C, zeros, v, variancias, variante)
        diag = np.einsum("nii->ni", cov)
        assert np.all((diag - diag_ref)[M] < 0.0), variante  # v < τ²: só encolhe
        assert est.excesso_diag_max <= 1e-12, variante
        # δ̂ = 0: a média é a mesma da FA (NaN fora de M nas duas)
        assert _diferenca_max(media, media_ref) == 0.0, variante


def test_correcao_por_gaveta_na_diagonal_e_de_fator_no_bloco_inteiro():
    fa, G, P, C = _fa(semente=3)
    k = G.shape[1]
    M = P & (np.random.default_rng(4).random(P.shape) < 0.5)
    variancias = _variancias(k, tau2=0.05, tau2_fator=0.05)
    zeros = np.zeros_like(G)

    _, cov_ref, _ = momentos_com_pessoa_v2(fa, G, P, M, C, zeros, zeros, variancias, "nenhum")

    _, cov_gaveta, _ = momentos_com_pessoa_v2(
        fa, G, P, M, C, zeros, np.full_like(G, 0.025), variancias, "gaveta"
    )
    dif_g = cov_gaveta - cov_ref
    linhas, colunas = np.nonzero(M)
    np.testing.assert_allclose(
        dif_g[linhas, colunas, colunas], -0.025, rtol=0, atol=1e-15
    )  # diferença de subtração em 1 ulp
    fora_diagonal = ~np.eye(k, dtype=bool)
    np.testing.assert_array_equal(dif_g[:, fora_diagonal], 0.0)

    _, cov_fator, _ = momentos_com_pessoa_v2(
        fa, G, P, M, C, zeros, np.full_like(G, 0.025), variancias, "fator"
    )
    dif_f = cov_fator - cov_ref
    bloco_oculto = M[:, :, None] & M[:, None, :]
    np.testing.assert_allclose(dif_f[bloco_oculto], -0.025, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(dif_f[~bloco_oculto], 0.0)


def test_estimar_delta_v2_grava_o_prior_exato_sem_historico():
    variancias = Variancias(
        tau2=np.array([0.04, 0.09]), sigma2=np.array([0.25, 0.16]),
        tau2_fator=TAU2_FATOR_CONTRAEXEMPLO, sigma2_fator=0.30,
    )
    residuo = np.zeros((3, 2))
    observado = np.zeros((3, 2), dtype=bool)  # histórico vazio
    codigos = np.array([0, 1, 0])

    delta, v = estimar_delta_v2(residuo, observado, codigos, 2, variancias, "fator")
    assert 1.0 / (1.0 / TAU2_FATOR_CONTRAEXEMPLO) != TAU2_FATOR_CONTRAEXEMPLO  # o contraexemplo vale
    np.testing.assert_array_equal(delta, np.zeros((2, 2)))
    np.testing.assert_array_equal(v, np.full((2, 2), TAU2_FATOR_CONTRAEXEMPLO))

    _, v_gaveta = estimar_delta_v2(residuo, observado, codigos, 2, variancias, "gaveta")
    np.testing.assert_array_equal(v_gaveta, np.broadcast_to(variancias.tau2, (2, 2)))


def test_sigma_novo_nao_psd_e_projetado_e_contado():
    fa, G, P, C = _fa(semente=4)
    k = G.shape[1]
    M = P & (np.random.default_rng(5).random(P.shape) < 0.5)
    # τ² enorme e v = 0 (histórico completo): a subtração de T deixa a diagonal negativa
    variancias = Variancias(
        tau2=np.full(k, 50.0), sigma2=np.full(k, 1.0),
        tau2_fator=50.0, sigma2_fator=1.0,
    )
    zeros = np.zeros_like(G)

    _, cov, est = momentos_com_pessoa_v2(fa, G, P, M, C, zeros, zeros, variancias, "gaveta")
    assert est.projecoes_negativas > 0
    assert est.projecoes_piso >= est.projecoes_negativas
    assert est.excesso_diag_max <= 1e-12  # a correção só encolhe; a projeção é numérica

    idx = np.arange(k)
    A = cov.copy()
    A[:, idx, idx] = np.where(M, np.einsum("nii->ni", A), 1.0)
    minimo = np.linalg.eigvalsh(A).min(axis=1)
    # a reconstrução S V Sᵀ de eigh devolve o piso com erro de arredondamento de ~1e-15 relativo
    assert np.all(minimo >= PISO_AUTOVALOR * (1 - 1e-6))


def test_amostragem_bate_com_os_momentos_e_copia_as_observadas():
    fa, G, P, C = _fa(n=60, k=4, semente=5)
    k = G.shape[1]
    M = P & (np.random.default_rng(6).random(P.shape) < 0.5)
    variancias = _variancias(k)
    rng = np.random.default_rng(7)
    delta = rng.normal(0.0, 0.1, G.shape)
    v = np.full_like(G, 0.025)

    media, cov, _ = momentos_com_pessoa_v2(fa, G, P, M, C, delta, v, variancias, "gaveta")
    X = ocultar(G, M)
    amostras = amostrar_bloco(media, cov, X, M, 5_000, np.random.default_rng(8))

    observadas = ~M
    np.testing.assert_array_equal(
        amostras[:, observadas], np.broadcast_to(X[observadas], (5_000,) + X[observadas].shape)
    )
    assert np.all(amostras[:, M] >= 0.0)  # truncamento em 0, como FA.sample

    ocultas = amostras[:, M]  # (S, n_ocultas)
    media_true = media[M]
    var_true = np.einsum("nii->ni", cov)[M]
    se_media = np.sqrt(var_true / len(amostras))  # erro padrão da média amostral por célula
    assert np.all(np.abs(ocultas.mean(axis=0) - media_true) <= 5 * se_media + 1e-3)
    assert np.all(np.abs(ocultas.var(axis=0) / var_true - 1) < 0.15)


def test_lixo_nas_ocultas_do_avaliado_nao_muda_nada():
    fa, G, P, C = _fa(n=80, k=4, semente=6)
    k = G.shape[1]
    M = P & (np.random.default_rng(9).random(P.shape) < 0.5)
    variancias = _variancias(k)
    delta = np.random.default_rng(10).normal(0.0, 0.1, G.shape)
    v = np.full_like(G, 0.025)

    media0, cov0, _ = momentos_com_pessoa_v2(fa, G, P, M, C, delta, v, variancias, "gaveta")
    G_lixo = G.copy()
    G_lixo[M] = 1e6
    media1, cov1, _ = momentos_com_pessoa_v2(fa, G_lixo, P, M, C, delta, v, variancias, "gaveta")
    assert _diferenca_max(media0, media1) == 0.0
    assert _diferenca_max(cov0, cov1) == 0.0

    a0 = amostrar_bloco(media0, cov0, ocultar(G, M), M, 5, np.random.default_rng(11))
    a1 = amostrar_bloco(media1, cov1, ocultar(G_lixo, M), M, 5, np.random.default_rng(11))
    np.testing.assert_array_equal(a0, a1)
