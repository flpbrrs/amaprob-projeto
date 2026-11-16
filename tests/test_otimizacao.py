import numpy as np
import pytest

from amaprob.tuning.busca import (
    Avaliador, busca_aleatoria, melhor_ate_aqui, menor_dentro_da_tolerancia, otimizacao_bayesiana,
)

N_AVALIACOES = 12
N_INICIAIS = 4
PASSO = 0.1
GRADE = np.array([(a, b) for a in np.arange(0, 1 + PASSO / 2, PASSO) for b in np.arange(0, 1 + PASSO / 2, PASSO)])
MINIMO = np.array([0.3, 0.7])


def tigela(candidato: int) -> dict[str, float]:
    return {"nll": float(10 * np.sum((GRADE[candidato] - MINIMO) ** 2))}


def buscar(semente: int = 0, objetivo=tigela):
    avaliador = Avaliador(objetivo)
    aleatoria = busca_aleatoria(len(GRADE), avaliador, N_AVALIACOES, np.random.default_rng(semente))
    iniciais = [a.candidato for a in aleatoria[:N_INICIAIS]]
    bayesiana = otimizacao_bayesiana(GRADE, avaliador, N_AVALIACOES, iniciais, semente)
    return aleatoria, bayesiana


@pytest.fixture(scope="module")
def buscas():
    return buscar()


def test_aproxima_o_minimo_da_tigela(buscas):
    aleatoria, bayesiana = buscas
    melhor = min(bayesiana, key=lambda a: a.nll)
    assert np.all(np.abs(GRADE[melhor.candidato] - MINIMO) <= PASSO + 1e-9)
    assert melhor.nll <= min(a.nll for a in aleatoria)


def test_candidatos_distintos_e_da_grade(buscas):
    for avaliacoes in buscas:
        candidatos = [a.candidato for a in avaliacoes]
        assert len(candidatos) == N_AVALIACOES == len(set(candidatos))
        assert all(0 <= c < len(GRADE) for c in candidatos)


def test_inicio_comum(buscas):
    aleatoria, bayesiana = buscas
    for a, b in zip(aleatoria[:N_INICIAIS], bayesiana[:N_INICIAIS]):
        assert a.candidato == b.candidato and a.nll == b.nll


def test_reprodutivel_pela_semente(buscas):
    repetidas = buscar()
    for avaliacoes, de_novo in zip(buscas, repetidas):
        assert [a.candidato for a in avaliacoes] == [a.candidato for a in de_novo]


def test_reaproveita_entre_metodos():
    chamadas = []

    def contando(candidato):
        chamadas.append(candidato)
        return tigela(candidato)

    aleatoria, bayesiana = buscar(objetivo=contando)
    unicos = {a.candidato for a in aleatoria + bayesiana}
    assert sorted(chamadas) == sorted(unicos)
    assert all(a.reaproveitada for a in bayesiana[:N_INICIAIS])
    assert not any(a.reaproveitada for a in aleatoria)


def test_melhor_ate_aqui_nunca_aumenta(buscas):
    for avaliacoes in buscas:
        assert np.all(np.diff(melhor_ate_aqui(avaliacoes)) <= 0)


def test_menor_m_dentro_da_tolerancia():
    M = np.array([512, 320, 128, 448])
    nlls = np.array([3.900, 3.906, 3.950, 3.902])
    assert M[menor_dentro_da_tolerancia(M, nlls, 0.01)] == 320
    assert M[menor_dentro_da_tolerancia(M, nlls, 0.0)] == 512


def _contexto_sintetico(n: int, rng: np.random.Generator) -> np.ndarray:
    ocasiao = rng.choice(5, size=n, p=[0.3, 0.35, 0.15, 0.15, 0.05])
    return np.c_[np.eye(5)[ocasiao], rng.normal(size=(n, 8))]


def test_amostra_estratificada_pela_ocasiao():
    from amaprob.experiments.otimizacao import amostra_estratificada

    C = _contexto_sintetico(20_000, np.random.default_rng(1))
    indices = amostra_estratificada(C, 0.25, np.random.default_rng(0))
    assert len(indices) == len(set(indices)) and indices.min() >= 0 and indices.max() < len(C)
    proporcao = lambda linhas: np.bincount(linhas[:, :5].argmax(axis=1), minlength=5) / len(linhas)
    assert np.all(np.abs(proporcao(C[indices]) - proporcao(C)) < 0.01)
    assert np.array_equal(indices, amostra_estratificada(C, 0.25, np.random.default_rng(0)))


def test_grades_contem_a_partida_e_ficam_em_0_1():
    from amaprob.experiments.otimizacao import MODELOS

    for modelo in MODELOS.values():
        assert modelo.params(modelo.indice(modelo.partida)) == modelo.partida
        assert modelo.pontos.min() >= 0 and modelo.pontos.max() <= 1
        assert len(np.unique(modelo.pontos, axis=0)) == len(modelo.candidatos)


def _confirmacao(nll_partida, nll_escolhida, modelo="fa_condicional", escolhida=(6, 1000.0)):
    linhas = []
    for configuracao, (q, alfa), nlls in (("partida", (5, 1.0), nll_partida), ("escolhida", escolhida, nll_escolhida)):
        linhas += [{"modelo": modelo, "configuracao": configuracao, "q": q, "alfa": alfa, "semente": s, "nll": n}
                   for s, n in enumerate(nlls)]
    import pandas as pd
    return pd.DataFrame(linhas)


def test_adota_a_escolhida_so_com_melhora_nas_sementes_novas():
    from amaprob.experiments.otimizacao import MODELOS, adotada

    fa = MODELOS["fa_condicional"]
    # melhora grande só na semente 0, usada na escolha: fica a de partida
    origem, params = adotada(fa, _confirmacao([4.4736, 4.4692, 4.4841], [4.4594, 4.4755, 4.4989]))
    assert origem == "partida" and params == {"q": 5, "alfa": 1.0}
    # melhora de 0,02 nas sementes 1 e 2: adota a escolhida
    origem, params = adotada(fa, _confirmacao([4.47, 4.47, 4.48], [4.47, 4.45, 4.46]))
    assert origem == "escolhida" and params == {"q": 6, "alfa": 1000.0}
    # média acima do limite, mas piora numa semente nova: fica a de partida
    origem, _ = adotada(fa, _confirmacao([4.47, 4.47, 4.48], [4.47, 4.43, 4.485]))
    assert origem == "partida"


def test_configuracao_gravada_herda_o_q(tmp_path, monkeypatch):
    import json

    import amaprob.experiments.otimizacao as otimizacao

    monkeypatch.setattr(otimizacao, "CONFIGURACAO", tmp_path / "hiperparametros.json")
    otimizacao.gravar_configuracao(_confirmacao([4.47, 4.47, 4.48], [4.47, 4.45, 4.46]))
    gravada = json.loads((tmp_path / "hiperparametros.json").read_text(encoding="utf-8"))
    assert gravada["fa_condicional"] == {"q": 6, "alfa": 1000.0} and gravada["fa"] == {"q": 6}
