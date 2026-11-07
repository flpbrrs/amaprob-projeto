import numpy as np
import pytest

from amaprob.data.dataset import ARQUIVO_MASCARAS, ARQUIVO_REFEICOES, carregar_refeicoes
from amaprob.data.mascaras import gerar_mascaras
from amaprob.data.splits import TESTE
from amaprob.imputation.base import ocultar
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagation import (
    N_DOBRAS,
    PLUGIN,
    PLUGIN_RECALIBRADO,
    PROPAGACAO,
    dobras_por_pessoa,
    imputar_treino_cruzado,
    momentos_da_mistura,
    prever_bracos,
)

N, K, S = 900, 5, 30


@pytest.fixture(scope="module")
def dados():
    rng = np.random.default_rng(11)
    P = rng.random((N, K)) < 0.6
    G = np.where(P, np.exp(rng.normal(4.0, 0.6, size=(N, K))), 0.0)
    C = rng.random((N, 2))
    y = G @ rng.uniform(0.1, 0.6, size=K) + 5 * C[:, 0] + rng.normal(0, 5, size=N)
    pessoas = np.repeat(np.arange(N // 3), 3).astype(str)
    M = gerar_mascaras(P, taxas=(0.5,), sementes=(0,))[0, 0]
    return G, P, M, C, y, pessoas


def _ajustados(G, P, C, y):
    sem_mascara = np.zeros_like(P)
    imputador = PorcaoMedia().fit(ocultar(G, sem_mascara), P, sem_mascara, C)
    return imputador, RegressaoLinearBayesiana().fit(montar_Z(G, P, C), y)


def _geradores(semente=0):
    return [np.random.default_rng((semente, 1 + f)) for f in range(N_DOBRAS)]


def test_mesma_media_com_regressor_linear(dados):
    G, P, M, C, y, _ = dados
    imputador, regressor = _ajustados(G, P, C, y)
    bracos = prever_bracos(imputador, regressor, G, P, M, C, S, np.random.default_rng(0))
    np.testing.assert_allclose(bracos[PROPAGACAO][0].mean(axis=0), bracos[PLUGIN][0][0], rtol=1e-9)


def test_variancia_total():
    rng = np.random.default_rng(0)
    medias, variancias = rng.normal(0, 3, size=(4, 1)), rng.uniform(0.5, 2, size=(4, 1))
    x = np.linspace(-40, 40, 400_001)
    densidade = np.mean(np.exp(-((x - medias) ** 2) / (2 * variancias)) / np.sqrt(2 * np.pi * variancias), axis=0)
    media_numerica = np.trapezoid(x * densidade, x)
    var_numerica = np.trapezoid((x - media_numerica) ** 2 * densidade, x)
    media, var = momentos_da_mistura(medias, variancias)
    assert media[0] == pytest.approx(media_numerica, abs=1e-8)
    assert var[0] == pytest.approx(var_numerica, rel=1e-8)


def test_bracos_iguais_sem_ocultos(dados):
    G, P, _, C, y, pessoas = dados
    sem_mascara = np.zeros_like(P)
    imputador, regressor = _ajustados(G, P, C, y)
    G_recalibrado = imputar_treino_cruzado(
        PorcaoMedia, G, P, sem_mascara, C, dobras_por_pessoa(pessoas, np.random.default_rng(0)), S, _geradores()
    )
    recalibrado = RegressaoLinearBayesiana().fit(montar_Z(G_recalibrado, P, C), y)
    bracos = prever_bracos(imputador, regressor, G, P, sem_mascara, C, S, np.random.default_rng(0), recalibrado)

    media_direta, var_direta = regressor.predict(montar_Z(G, P, C))
    for braco in (PLUGIN, PLUGIN_RECALIBRADO, PROPAGACAO):
        medias, variancias = bracos[braco]
        np.testing.assert_allclose(medias, np.broadcast_to(media_direta, medias.shape), rtol=1e-8)
        np.testing.assert_allclose(variancias, np.broadcast_to(var_direta, variancias.shape), rtol=1e-8)


def test_dobras_por_pessoa(dados):
    *_, pessoas = dados
    dobras = dobras_por_pessoa(pessoas, np.random.default_rng(0))
    assert set(dobras) == set(range(N_DOBRAS))
    for pessoa in np.unique(pessoas):
        assert len(set(dobras[pessoas == pessoa])) == 1


def test_nenhum_imputador_ve_o_que_imputa(dados):
    G, P, M, C, _, pessoas = dados
    dobras = dobras_por_pessoa(pessoas, np.random.default_rng(0))
    G_alterado = G.copy()
    ocultos_da_dobra = M & (dobras == 0)[:, None]
    G_alterado[ocultos_da_dobra] = np.random.default_rng(9).uniform(1, 2_000, size=ocultos_da_dobra.sum())

    original = imputar_treino_cruzado(PorcaoMedia, G, P, M, C, dobras, S, _geradores())
    alterado = imputar_treino_cruzado(PorcaoMedia, G_alterado, P, M, C, dobras, S, _geradores())
    np.testing.assert_array_equal(original[dobras == 0], alterado[dobras == 0])
    assert not np.array_equal(original[dobras != 0], alterado[dobras != 0])


@pytest.mark.skipif(not (ARQUIVO_REFEICOES.exists() and ARQUIVO_MASCARAS.exists()), reason="dados processados ausentes")
def test_conjuntos_usados():
    from amaprob.experiments.propagacao import treino_e_validacao

    tr, va, *_ = treino_e_validacao()
    refeicoes = carregar_refeicoes()
    ids_de_teste = refeicoes.meal_id[refeicoes.conjunto == TESTE]
    usados = np.concatenate([tr.meal_id, va.meal_id])
    assert not np.isin(usados, ids_de_teste).any()
    assert len(usados) + len(ids_de_teste) == len(refeicoes.meal_id)


class _ImputadorFixo:
    def __init__(self, amostras_log):
        self.amostras_log = amostras_log

    def sample(self, X, P, M, C, n_samples, rng):
        return self.amostras_log


class _RegressorSomaDasGramas:
    def predict(self, Z):
        return Z[:, :K].sum(axis=1), np.ones(len(Z))


def test_plugin_usa_media_em_gramas():
    amostras_log = np.zeros((S, 1, K))
    amostras_log[:, 0, 0] = np.linspace(2.0, 5.0, S)
    P = np.ones((1, K), dtype=bool)
    M = np.zeros((1, K), dtype=bool)
    M[0, 0] = True
    bracos = prever_bracos(
        _ImputadorFixo(amostras_log), _RegressorSomaDasGramas(), np.zeros((1, K)), P, M, np.zeros((1, 2)), S,
        np.random.default_rng(0),
    )
    media_em_gramas = np.expm1(amostras_log[:, 0, 0]).mean()
    assert bracos[PLUGIN][0][0, 0] == pytest.approx(media_em_gramas)
    assert bracos[PLUGIN][0][0, 0] > np.expm1(amostras_log[:, 0, 0].mean())
