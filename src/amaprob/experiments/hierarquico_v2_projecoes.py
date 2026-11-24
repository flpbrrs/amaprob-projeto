"""Tamanho da projeção para semidefinida positiva no hierárquico v2 (variante por gaveta).

Em `imputation/hierarquico_v2.py`, Σ_novo = Σ_FA − T + V pode deixar de ser semidefinida positiva
no bloco oculto, e `_projetar` eleva os autovalores abaixo de 1e-8 a esse piso. Aqui se mede
quanto isso acrescenta: nas refeições projetadas, o traço adicionado
tr(Σ_projetada) − tr(Σ_novo), dividido pelo traço total de Σ_novo (bloco oculto) de todas as
refeições avaliadas. Também se registram o maior acréscimo relativo numa refeição
(acréscimo / tr(Σ_novo) da própria refeição) e as gavetas em que os autovalores negativos
aparecem, por duas leituras: a massa |λ| v_k² dos autovetores de autovalor negativo e a contagem de
diagonais negativas.

Mesmo cenário de `experiments/hierarquico_v2.py`: validação, pessoas com os dois dias, dia 1 como
histórico completo (H1), dia 2 avaliado, semente 0, taxas 0,5 e 1,0. A covariância projetada daqui
é conferida contra a de `momentos_com_pessoa_v2` (diferença máxima registrada).

Saída: results/hierarquico_v2_projecoes.csv (resumo por taxa) e
results/hierarquico_v2_projecoes_gavetas.csv (por gaveta).

Uso: uv run python -m amaprob.experiments.hierarquico_v2_projecoes
"""

import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import RAIZ, carregar_mascaras, carregar_refeicoes
from amaprob.data.splits import TREINO, VALIDACAO
from amaprob.experiments.diagnostico_dois_dias import carregar_dias
from amaprob.experiments.propagacao import ler_configuracao
from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from amaprob.imputation.hierarquico import estimar_variancias, residuos
from amaprob.imputation.hierarquico_v2 import _projetar, estimar_delta_v2, momentos_com_pessoa_v2
from amaprob.imputation.momentos import momentos_ocultos

SAIDA = RAIZ / "results"
TAXAS = (0.5, 1.0)
SEMENTE = 0
VARIANTE = "gaveta"


def sigma_novo_sem_projecao(fa, G, P, M, C, delta, v, variancias) -> np.ndarray:
    """Σ_FA − T + V no bloco oculto, antes da projeção (os passos de `momentos_com_pessoa_v2`)."""
    M = np.asarray(M, dtype=bool)
    _, covariancia = momentos_ocultos(fa, ocultar(G, M) - delta, P, M, C)
    linhas, colunas = np.nonzero(M)
    covariancia[linhas, colunas, colunas] += (v - variancias.tau2[None, :])[linhas, colunas]
    return covariancia


def medir(sigma: np.ndarray, projetada: np.ndarray, M: np.ndarray, gavetas) -> tuple[dict, pd.DataFrame]:
    traco = np.einsum("nkk,nk->n", sigma, M.astype(float))
    traco_proj = np.einsum("nkk,nk->n", projetada, M.astype(float))
    acrescimo = traco_proj - traco
    projetadas = np.abs(projetada - sigma).reshape(len(M), -1).max(axis=1) > 0
    massa_negativa = np.zeros(M.shape[1])
    refeicoes_com_negativo = np.zeros(M.shape[1])
    for n in np.flatnonzero(projetadas):
        ocultas = np.flatnonzero(M[n])
        bloco = sigma[n][np.ix_(ocultas, ocultas)]
        autovalores, autovetores = np.linalg.eigh((bloco + bloco.T) / 2)
        negativos = autovalores < 0
        if negativos.any():
            massa = (autovetores[:, negativos] ** 2 * -autovalores[negativos]).sum(axis=1)
            massa_negativa[ocultas] += massa
            refeicoes_com_negativo[ocultas] += massa > 1e-3 * massa.sum()
    diag = np.einsum("nkk->nk", sigma)
    diag_negativa = ((diag < 0) & M).sum(axis=0)
    relativo = np.where(traco > 0, acrescimo / np.where(traco > 0, traco, 1.0), np.nan)
    i = int(np.nanargmax(np.where(projetadas, relativo, -np.inf))) if projetadas.any() else None
    resumo = {
        "refeicoes": int(len(M)), "refeicoes_com_ocultas": int(M.any(axis=1).sum()),
        "refeicoes_projetadas": int(projetadas.sum()),
        "traco_total": float(traco.sum()), "acrescimo_total": float(acrescimo.sum()),
        "acrescimo_relativo_total": float(acrescimo.sum() / traco.sum()),
        "maior_acrescimo_relativo_refeicao": float(relativo[i]) if i is not None else 0.0,
        "maior_acrescimo_absoluto_refeicao": float(acrescimo.max()),
        "traco_da_refeicao_do_maior_relativo": float(traco[i]) if i is not None else np.nan,
        "ocultas_da_refeicao_do_maior_relativo": int(M[i].sum()) if i is not None else 0,
        "mediana_acrescimo_relativo_projetadas": float(np.nanmedian(relativo[projetadas])) if projetadas.any() else 0.0,
        "diagonais_negativas": int(diag_negativa.sum()),
    }
    por_gaveta = pd.DataFrame({
        "gaveta": gavetas, "ocultas": M.sum(axis=0), "massa_autovalor_negativo": massa_negativa,
        "fracao_massa_negativa": massa_negativa / max(massa_negativa.sum(), 1e-300),
        "refeicoes_com_negativo": refeicoes_com_negativo.astype(int), "diagonais_negativas": diag_negativa,
    }).sort_values("massa_autovalor_negativo", ascending=False)
    return resumo, por_gaveta


def main() -> None:
    inicio = time.perf_counter()
    refeicoes = carregar_refeicoes()
    mascaras = carregar_mascaras(refeicoes)
    dias = carregar_dias(refeicoes)

    validacao = np.flatnonzero(refeicoes.conjunto == VALIDACAO)
    dv = dias.iloc[validacao]
    dia1, dia2 = (dv.dia == 1).to_numpy(), (dv.dia == 2).to_numpy()
    duas = set(dv.loc[dia1, "person_id"]) & set(dv.loc[dia2, "person_id"])
    tem_duas = dv.person_id.isin(duas).to_numpy()
    hist, avaliar = validacao[dia1 & tem_duas], validacao[dia2 & tem_duas]
    ordem = np.array(sorted(duas))
    indice_de = pd.Index(ordem)
    cod_hist = indice_de.get_indexer(dv.person_id.to_numpy()[dia1 & tem_duas])
    cod_avaliar = indice_de.get_indexer(dv.person_id.to_numpy()[dia2 & tem_duas])

    configuracao = ler_configuracao()
    treino = np.flatnonzero(refeicoes.conjunto == TREINO)
    sem_mascara = np.zeros_like(refeicoes.P[treino])
    fa = FA(semente=0, media_condicional=True, **configuracao.get("fa_condicional", {})).fit(
        ocultar(refeicoes.G[treino], sem_mascara), refeicoes.P[treino], sem_mascara, refeicoes.C[treino]
    )
    variancias = estimar_variancias(
        residuos(fa, refeicoes.G[treino], refeicoes.P[treino], refeicoes.C[treino]),
        refeicoes.person_id[treino], dias.iloc[treino].dia.to_numpy(),
    )
    print(f"FA e τ²: {time.perf_counter() - inicio:.1f} s", flush=True)

    G_h, P_h, C_h = (a[hist] for a in (refeicoes.G, refeicoes.P, refeicoes.C))
    residuo_h = residuos(fa, G_h, P_h, C_h)
    G_a, P_a, C_a = (a[avaliar] for a in (refeicoes.G, refeicoes.P, refeicoes.C))
    si = int(np.flatnonzero(mascaras.sementes == SEMENTE)[0])

    resumos, gavetas = [], []
    for taxa in TAXAS:
        t = int(np.flatnonzero(mascaras.taxas == taxa)[0])
        M_a = mascaras.M[si, t][avaliar]
        delta_p, v_p = estimar_delta_v2(residuo_h, P_h, cod_hist, len(ordem), variancias, VARIANTE)  # H1
        delta, v = delta_p[cod_avaliar], v_p[cod_avaliar]

        sigma = sigma_novo_sem_projecao(fa, G_a, P_a, M_a, C_a, delta, v, variancias)
        projetada = sigma.copy()
        piso, negativas = _projetar(projetada, M_a)
        _, oficial, est = momentos_com_pessoa_v2(fa, G_a, P_a, M_a, C_a, delta, v, variancias, VARIANTE)

        resumo, por_gaveta = medir(sigma, projetada, M_a, refeicoes.gavetas)
        chave = {"cenario": "H1", "variante": VARIANTE, "taxa": taxa, "semente": SEMENTE}
        resumos.append({**chave, **resumo, "projecoes_piso": piso, "projecoes_negativas": negativas,
                        "projecoes_piso_oficial": est.projecoes_piso,
                        "diferenca_max_contra_oficial": float(np.max(np.abs(projetada - oficial)))})
        gavetas.append(por_gaveta.assign(**chave))
        print(pd.Series(resumos[-1]).to_string(), flush=True)

    pd.DataFrame(resumos).to_csv(SAIDA / "hierarquico_v2_projecoes.csv", index=False)
    pd.concat(gavetas, ignore_index=True).to_csv(SAIDA / "hierarquico_v2_projecoes_gavetas.csv", index=False)
    print(f"tempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
