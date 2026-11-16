"""Busca de hiperparâmetros numa grade discreta de candidatos: aleatória e por otimização bayesiana.

Otimização bayesiana (Snoek et al., 2012; Frazier, 2018). Um processo gaussiano g ~ GP é ajustado às
avaliações feitas, com g(x) = −NLL(x) sobre as entradas levadas a [0, 1]; a posterior dele dá, em cada
candidato não avaliado, uma média μ(x) e um desvio σ(x). A aquisição Expected Improvement,

    EI(x) = E[max(g(x) − g*, 0)] = σ(x) [z Φ(z) + φ(z)],     z = (μ(x) − g*) / σ(x),

com g* a melhor avaliação até ali, pesa candidatos de média alta (explotação) e de incerteza alta
(exploração). Ela é calculada em log (LogEI; Ament et al., 2023), que não se anula numericamente
longe do melhor ponto. A cada passo, avalia-se o candidato de maior aquisição.

A grade é discreta (q e M são inteiros), então a aquisição é calculada em todos os candidatos não
avaliados, sem otimização contínua: a escolha é exata e reprodutível.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
import time

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

ALVO = "nll"


@dataclass(frozen=True)
class Avaliacao:
    candidato: int
    metricas: dict[str, float]
    segundos: float
    reaproveitada: bool = False

    @property
    def nll(self) -> float:
        return self.metricas[ALVO]


class Avaliador:
    """Calcula o objetivo uma vez por candidato; uma repetição devolve o resultado guardado, marcado."""

    def __init__(self, objetivo: Callable[[int], dict[str, float]]):
        self._objetivo = objetivo
        self._feitas: dict[int, Avaliacao] = {}

    def __call__(self, candidato: int) -> Avaliacao:
        if candidato in self._feitas:
            return replace(self._feitas[candidato], reaproveitada=True)
        inicio = time.perf_counter()
        metricas = self._objetivo(candidato)
        avaliacao = Avaliacao(candidato, metricas, time.perf_counter() - inicio)
        self._feitas[candidato] = avaliacao
        return avaliacao


def busca_aleatoria(n_candidatos: int, avaliador: Avaliador, n: int, rng: np.random.Generator) -> list[Avaliacao]:
    return [avaliador(int(i)) for i in rng.permutation(n_candidatos)[:n]]


def otimizacao_bayesiana(
    pontos: np.ndarray, avaliador: Avaliador, n: int, iniciais: list[int], semente: int,
) -> list[Avaliacao]:
    """`pontos`: (candidatos, dimensões) em [0, 1]. Começa pelos `iniciais` e segue pelo máximo do LogEI."""
    avaliacoes = [avaliador(i) for i in iniciais]
    while len(avaliacoes) < n:
        feitos = [a.candidato for a in avaliacoes]
        X = torch.tensor(pontos[feitos], dtype=torch.float64)
        g = -torch.tensor([[a.nll] for a in avaliacoes], dtype=torch.float64)
        torch.manual_seed(semente + len(avaliacoes))
        surrogate = SingleTaskGP(X, g)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(surrogate.likelihood, surrogate))
        restantes = np.setdiff1d(np.arange(len(pontos)), feitos)
        aquisicao = LogExpectedImprovement(surrogate, best_f=g.max())
        with torch.no_grad():
            valores = aquisicao(torch.tensor(pontos[restantes], dtype=torch.float64).unsqueeze(1))
        avaliacoes.append(avaliador(int(restantes[int(torch.argmax(valores))])))
    return avaliacoes


def melhor_ate_aqui(avaliacoes: list[Avaliacao]) -> np.ndarray:
    return np.minimum.accumulate([a.nll for a in avaliacoes])


def menor_dentro_da_tolerancia(valores: np.ndarray, nlls: np.ndarray, tolerancia: float) -> int:
    """Índice do menor valor (por exemplo, M) cuja NLL fica a até `tolerancia` da menor NLL."""
    valores, nlls = np.asarray(valores), np.asarray(nlls)
    aceitos = np.flatnonzero(nlls <= nlls.min() + tolerancia)
    return int(aceitos[np.argmin(valores[aceitos])])
