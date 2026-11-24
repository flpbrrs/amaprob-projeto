"""Qual gaveta perguntar: redução esperada da variância preditiva ao revelar cada oculta.

O braço analítico (`regression/propagacao_analitica.py`) trata a entrada como variável aleatória e
devolve, por refeição,

    Var[y] = 1/β + φ̄ᵀ S φ̄ + mᵀ Cφ m + tr(S Cφ)

com φ̄ a média de φ, Cφ = Cg / (s sᵀ) a covariância da entrada padronizada, m e S o posterior dos
pesos. Os dois últimos termos são a incerteza da entrada; os dois primeiros são a preditiva que
sobraria se a entrada fosse conhecida.

Revelar a gaveta k substitui a marginal de x por x | x_k. A covariância condicional é

    Cφ⁽ᵏ⁾ = Cφ − (Cφ e_k)(Cφ e_k)ᵀ / Cφ_kk

e a média condicional é afim em x_k, com coeficiente Cφ[:, k] / Cφ_kk. Tomando a esperança sobre o
valor revelado (que é o que se pode calcular antes de perguntar, sem ler a resposta), os termos
cruzados da média se anulam e sobra

    mᵀ Cφ⁽ᵏ⁾ m = mᵀ Cφ m − (Cφ m)_k² / Cφ_kk

Por isso, com v = Cφ m (um vetor por refeição, em gramas), a redução esperada da variância é

    redução_k = v_k² / Cφ_kk

para o termo da média da entrada, e tr(S Cφ) cai para tr(S Cφ⁽ᵏ⁾), uma quantidade da ordem de
Cφ_kk · S_kk, produto da variância da entrada pela incerteza dos pesos. O termo é de segunda ordem e
fica fora da conta. A diferença entre a redução prevista e a realizada medida no
dado inclui esse termo, e é o que a tabela de conferência mostra.

Unidades: Cφ é adimensional (gramas divididos por desvio de gramas), m está em gramas, então v está
em gramas e v² / Cφ_kk em gramas², a mesma unidade da variância preditiva.

Gaveta fora de M tem Cg = 0, logo Cφ_kk = 0 e a redução é 0. A escolha nunca olha a grama verdadeira
de uma oculta: só a média e a covariância condicionais do modelo, que já vêm sem elas.
"""

import numpy as np


def reducao_por_gaveta(blr, Eg: np.ndarray, Cg: np.ndarray) -> np.ndarray:
    """Redução esperada da variância preditiva (g²) ao revelar cada gaveta, com forma (N, K).

    `Eg` e `Cg` são a média e a covariância da entrada em gramas (`momentos_em_gramas`). Fora de M a
    covariância é 0 e a redução sai 0.
    """
    K = Eg.shape[1]
    desvio = blr.desvio_z_[:K]
    C_phi = Cg / (desvio[:, None] * desvio[None, :])
    v = np.einsum("nij,j->ni", C_phi, blr.m_[:K])
    diagonal = np.einsum("nii->ni", C_phi)
    reducao = np.zeros_like(v)
    np.divide(v * v, diagonal, out=reducao, where=diagonal > 0)
    return reducao
