"""Máscaras de quantidades ocultas sobre as gavetas presentes (Regra 4).

Mecanismo: MCAR (ausência completamente ao acaso), por gaveta. Para uma taxa r, cada célula
presente (P = 1) fica oculta com probabilidade r, de forma independente das demais células,
das gramas, do carboidrato e do conjunto do split. A presença em si nunca é ocultada: M ⊆ P.

Construção aninhada. Para cada semente, sorteia-se uma vez U ~ Uniforme[0, 1) com a forma de P e
define-se, para cada taxa r,

    M_r = P ∧ (U < r).

- Marginal correta: P(U < r) = r, então cada M_r, isolada, é o sorteio de Bernoulli acima.
- Aninhamento: r < r' implica (U < r) ⊆ (U < r'), logo M_r ⊆ M_r'. A curva "métrica × taxa"
  compara cada refeição com ela mesma, sem somar a variação de um novo sorteio por taxa.
- Extremos exatos: U < 0 nunca ocorre (M_0 vazia) e U < 1 sempre ocorre (M_1 = P).

A variação entre repetições vem das sementes. A justificativa completa está em docs/mascaramento.md.
"""

from collections.abc import Sequence

import numpy as np

TAXAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEMENTES = (0, 1, 2)


def gerar_mascaras(
    P: np.ndarray, taxas: Sequence[float] = TAXAS, sementes: Sequence[int] = SEMENTES
) -> np.ndarray:
    """Máscaras booleanas de forma (len(sementes), len(taxas), N, K); M[s, t] ⊆ P."""
    P = np.asarray(P, dtype=bool)
    M = np.empty((len(sementes), len(taxas), *P.shape), dtype=bool)
    for i, semente in enumerate(sementes):
        U = np.random.default_rng(semente).random(P.shape)
        for j, taxa in enumerate(taxas):
            M[i, j] = P & (U < taxa)
    return M
