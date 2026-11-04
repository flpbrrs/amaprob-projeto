"""O experimento da taxa 0 nunca usa o conjunto de teste (Regra 6)."""

import numpy as np
import pytest

from amaprob.data.dataset import ARQUIVO_REFEICOES, Refeicoes, carregar_refeicoes
from amaprob.data.splits import TESTE, TREINO, VALIDACAO
from amaprob.experiments.taxa_zero import separar


def test_separar_ignora_teste_sintetico():
    conjunto = np.array([TREINO, TESTE, VALIDACAO, TESTE, TREINO])
    vazio = np.zeros((5, 1))
    r = Refeicoes(G=vazio, P=vazio.astype(bool), C=vazio, C_nomes=np.array(["c"]), y=np.zeros(5),
                  meal_id=np.arange(5), person_id=np.array(list("abcde")), conjunto=conjunto, gavetas=np.array(["g"]))
    tr, va = separar(r)
    np.testing.assert_array_equal(tr, [0, 4])
    np.testing.assert_array_equal(va, [2])


@pytest.mark.skipif(not ARQUIVO_REFEICOES.exists(), reason="dados processados ausentes")
def test_separar_ignora_teste_real():
    r = carregar_refeicoes()
    tr, va = separar(r)
    usados = np.concatenate([tr, va])
    assert not (r.conjunto[usados] == TESTE).any()
    assert len(usados) + (r.conjunto == TESTE).sum() == len(r.conjunto)
