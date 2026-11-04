"""Teste de sanidade na taxa 0: pipeline de ponta a ponta sem nenhuma quantidade oculta.

1. Monta Z = [gramas | P | C] com todas as gramas observadas.
2. Ajusta no treino a referência de mínimos quadrados (o piso de erro, com resíduo gaussiano)
   e a regressão linear bayesiana.
3. Mede na validação: MAE, NLL, CRPS, PICP, MPIW, % de meia dose e cobertura por faixa da previsão.
4. Mede o tempo de ajuste de cada modelo. O baseline de imputação não faz nada na taxa 0, então
   seu tempo (ajuste + 100 amostras) é medido sobre a máscara de taxa 0,5 e semente 0, sem métricas.

O conjunto de teste não é lido (Regra 6). Resultados em results/taxa_zero_*.csv e tempos_ajuste.csv.

Uso: uv run python -m amaprob.experiments.taxa_zero
"""

import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import RAIZ, Refeicoes, carregar_mascaras, carregar_refeicoes
from amaprob.data.splits import TREINO, VALIDACAO
from amaprob.evaluation.metricas import avaliar, cobertura_por_faixa
from amaprob.imputation.base import ocultar
from amaprob.imputation.baseline import PorcaoMedia
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana

SAIDA = RAIZ / "results"
N_AMOSTRAS = 100
BLOCO = 5_000  # refeições por bloco na amostragem: 100 amostras de todo o conjunto não cabem na memória


class MinimosQuadrados:
    """Referência: a regra linear do piso de erro, com o resíduo do treino como variância única."""

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "MinimosQuadrados":
        A = np.c_[Z, np.ones(len(Z))]
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.var_ = np.var(y - A @ self.coef_)
        return self

    def predict(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        media = np.c_[Z, np.ones(len(Z))] @ self.coef_
        return media, np.full(len(Z), self.var_)


def separar(refeicoes: Refeicoes) -> tuple[np.ndarray, np.ndarray]:
    """Índices de treino e validação. As linhas de teste nunca são selecionadas."""
    return np.flatnonzero(refeicoes.conjunto == TREINO), np.flatnonzero(refeicoes.conjunto == VALIDACAO)


def _cronometrar(funcao):
    inicio = time.perf_counter()
    resultado = funcao()
    return resultado, time.perf_counter() - inicio


def tempo_baseline(refeicoes: Refeicoes, M: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[float, float]:
    """Segundos para ajustar o baseline no treino e gerar N_AMOSTRAS para a validação, em blocos."""
    r = refeicoes
    X_tr = ocultar(r.G[tr], M[tr])
    modelo, t_ajuste = _cronometrar(lambda: PorcaoMedia().fit(X_tr, r.P[tr], M[tr], r.C[tr]))
    rng = np.random.default_rng(0)
    t_amostras = 0.0
    for inicio in range(0, len(va), BLOCO):
        b = va[inicio : inicio + BLOCO]
        X_b = ocultar(r.G[b], M[b])
        _, t = _cronometrar(lambda: modelo.sample(X_b, r.P[b], M[b], r.C[b], N_AMOSTRAS, rng))
        t_amostras += t
    return t_ajuste, t_amostras


def main() -> None:
    refeicoes = carregar_refeicoes()
    mascaras = carregar_mascaras(refeicoes)  # confere o alinhamento
    tr, va = separar(refeicoes)
    Z = montar_Z(refeicoes.G, refeicoes.P, refeicoes.C)  # taxa 0: todas as gramas observadas
    y = refeicoes.y

    metricas, coberturas, tempos = [], [], []
    for nome, modelo in [("minimos_quadrados", MinimosQuadrados()), ("blr", RegressaoLinearBayesiana())]:
        _, t = _cronometrar(lambda: modelo.fit(Z[tr], y[tr]))
        media, var = modelo.predict(Z[va])
        metricas.append({"modelo": nome, **avaliar(y[va], media, var)})
        coberturas.append(cobertura_por_faixa(y[va], media, var).assign(modelo=nome))
        tempos.append({"modelo": nome, "etapa": "ajuste no treino", "segundos": t})
        if nome == "blr":
            print(f"BLR: alpha = {modelo.alpha_:.4g} | 1/sqrt(beta) = {1 / np.sqrt(modelo.beta_):.2f} g | "
                  f"gamma = {modelo.gamma_:.1f} de {Z.shape[1]} parâmetros | iterações = {modelo.n_iter_}")

    j = int(np.flatnonzero(mascaras.taxas == 0.5)[0])
    t_ajuste, t_amostras = tempo_baseline(refeicoes, mascaras.M[0, j], tr, va)
    tempos += [
        {"modelo": "porcao_media", "etapa": "ajuste no treino (taxa 0,5, semente 0)", "segundos": t_ajuste},
        {"modelo": "porcao_media", "etapa": f"{N_AMOSTRAS} amostras para a validação", "segundos": t_amostras},
    ]

    metricas = pd.DataFrame(metricas)
    coberturas = pd.concat(coberturas)[["modelo", "faixa", "media_min_g", "media_max_g", "n", "picp"]]
    tempos = pd.DataFrame(tempos)
    SAIDA.mkdir(exist_ok=True)
    metricas.to_csv(SAIDA / "taxa_zero_metricas.csv", index=False)
    coberturas.to_csv(SAIDA / "taxa_zero_cobertura.csv", index=False)
    tempos.to_csv(SAIDA / "tempos_ajuste.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\ntreino: {len(tr):,} refeições | validação: {len(va):,} refeições\n")
    print(metricas.round(3).to_string(index=False), "\n")
    print(coberturas.round(3).to_string(index=False), "\n")
    print(tempos.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
