"""Modelo hierárquico v3, variante por gaveta: Ψ* por refeição, sem projeção.

Mesmo cenário de `experiments/hierarquico_v2.py` (validação, pessoas com os dois dias, dia 1 como
histórico, dia 2 avaliado; H1 completo e H2 mascarado; taxas 0,25 a 1,0; 3 sementes; imputação,
três braços da regressão linear bayesiana e métricas por ocasião na taxa 1,0), com os momentos de
`imputation/hierarquico_v3.py`. A amostragem é o caminho gaussiano da v2 (`amostrar_bloco`:
Cholesky por refeição, truncamento em 0), com o mesmo gerador. O fator único fica na v2.

Conferências, que abortam a rodada se falharem:
1. H2 na taxa 1,0 (sem histórico): média e covariância em log idênticas às da FA sem pessoa
   (`momentos_ocultos`), exceto nas gavetas em que o piso de ψ − τ² é acionado, listadas;
2. nenhuma projeção: o menor autovalor do bloco oculto é positivo em todas as refeições.

Saídas novas em results/hierarquico_v3_*.csv, regravadas ao fim de cada (cenário, taxa).

Uso: uv run python -m amaprob.experiments.hierarquico_v3
"""

import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import OCASIOES, RAIZ, carregar_mascaras, carregar_refeicoes
from amaprob.data.splits import TREINO, VALIDACAO
from amaprob.evaluation.metricas import ORCAMENTO_BLOCO_BYTES, avaliar_imputacao, avaliar_mistura, celulas_de_imputacao
from amaprob.experiments.diagnostico_dois_dias import carregar_dias
from amaprob.experiments.hierarquico_v2 import (
    BRACOS, CENARIOS, CHAVES, FLUXO_AMOSTRAGEM, N_AMOSTRAS, OCASIOES_AVALIADAS, TAXAS, gerador,
)
from amaprob.experiments.propagacao import ler_configuracao
from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from amaprob.imputation.hierarquico import estimar_variancias, residuos
from amaprob.imputation.hierarquico_v2 import amostrar_bloco, estimar_delta_v2
from amaprob.imputation.hierarquico_v3 import (
    gavetas_no_piso, menor_autovalor_das_ocultas, momentos_com_pessoa_v3,
)
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagacao_analitica import momentos_em_gramas, prever_com_entrada_incerta
from amaprob.regression.propagation import COPIAS_POR_PREVISAO, PLUGIN, PROPAGACAO

SAIDA = RAIZ / "results"
PREFIXO = "hierarquico_v3"
VARIANTE = "gaveta"
IMPUTADOR = "fa_por_gaveta_v3"
ANALITICA = "analitica"


def medir(fa, blr, G, P, M, C, delta, v, tau2, rng):
    """Momentos v3, amostragem em blocos e os três braços (como `medir_variante_v2`)."""
    N, K = G.shape
    media_log, cov_log = momentos_com_pessoa_v3(fa, G, P, M, C, delta, v, tau2)
    X = ocultar(G, M)
    celulas = []
    medias = {PLUGIN: np.empty((1, N)), PROPAGACAO: np.empty((N_AMOSTRAS, N))}
    variancias = {PLUGIN: np.empty((1, N)), PROPAGACAO: np.empty((N_AMOSTRAS, N))}
    tamanho = max(1, ORCAMENTO_BLOCO_BYTES // (N_AMOSTRAS * (2 * K + C.shape[1]) * 8 * COPIAS_POR_PREVISAO))
    for inicio in range(0, N, tamanho):
        b = slice(inicio, inicio + tamanho)
        amostras_log = amostrar_bloco(media_log[b], cov_log[b], X[b], M[b], N_AMOSTRAS, rng)
        celulas.append(celulas_de_imputacao(G[b], amostras_log, M[b]))
        G_s = np.expm1(amostras_log)
        s, n_b, _ = G_s.shape
        Z = {PLUGIN: montar_Z(G_s.mean(axis=0), P[b], C[b]),
             PROPAGACAO: montar_Z(G_s.reshape(s * n_b, K), np.tile(P[b], (s, 1)), np.tile(C[b], (s, 1)))}
        for braco in (PLUGIN, PROPAGACAO):
            media, var = blr.predict(Z[braco])
            medias[braco][:, b] = media.reshape(-1, n_b)
            variancias[braco][:, b] = var.reshape(-1, n_b)
    Eg, Cg = momentos_em_gramas(G, P, M, media_log, cov_log)
    medias[ANALITICA], variancias[ANALITICA] = prever_com_entrada_incerta(blr, Eg, Cg, P, C)
    return pd.concat(celulas, ignore_index=True), medias, variancias, media_log, cov_log


def gravar(tabelas: dict[str, list[dict]]) -> None:
    for nome, linhas in tabelas.items():
        if not linhas:
            continue
        tabela = pd.DataFrame(linhas)
        tabela.to_csv(SAIDA / f"{PREFIXO}_{nome}_metricas.csv", index=False)
        if nome not in CHAVES:
            continue
        resumo = tabela.drop(columns=["semente"]).groupby(CHAVES[nome], sort=False).agg(["mean", "std"])
        resumo.columns = [f"{m}_{'media' if e == 'mean' else 'desvio'}" for m, e in resumo.columns]
        resumo.reset_index().to_csv(SAIDA / f"{PREFIXO}_{nome}_resumo.csv", index=False)


def main() -> None:
    inicio_total = time.perf_counter()
    tempos: list[dict] = []

    def marcar(etapa: str, comeco: float, **chave) -> None:
        segundos = time.perf_counter() - comeco
        tempos.append({"etapa": etapa, **chave, "segundos": segundos})
        print(f"  {etapa}: {segundos:.1f} s", flush=True)

    comeco = time.perf_counter()
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
    marcar("leitura", comeco)

    comeco = time.perf_counter()
    treino = np.flatnonzero(refeicoes.conjunto == TREINO)
    sem_mascara = np.zeros_like(refeicoes.P[treino])
    fa = FA(semente=0, media_condicional=True, **ler_configuracao().get("fa_condicional", {})).fit(
        ocultar(refeicoes.G[treino], sem_mascara), refeicoes.P[treino], sem_mascara, refeicoes.C[treino]
    )
    variancias = estimar_variancias(
        residuos(fa, refeicoes.G[treino], refeicoes.P[treino], refeicoes.C[treino]),
        refeicoes.person_id[treino], dias.iloc[treino].dia.to_numpy(),
    )
    blr = RegressaoLinearBayesiana().fit(
        montar_Z(refeicoes.G[treino], refeicoes.P[treino], refeicoes.C[treino]), refeicoes.y[treino]
    )
    marcar("FA condicional, τ², σ² e regressão no treino", comeco)

    tau2 = variancias.tau2
    no_piso = gavetas_no_piso(fa, tau2)
    piso = pd.DataFrame({
        "gaveta": refeicoes.gavetas, "psi": fa.psi_, "tau2": tau2, "psi_menos_tau2": fa.psi_ - tau2,
        "piso_acionado": np.isin(np.arange(len(tau2)), no_piso),
        "excesso_sobre_psi_sem_historico": np.where(np.isin(np.arange(len(tau2)), no_piso),
                                                     1e-6 + tau2 - fa.psi_, 0.0),
    })
    piso.to_csv(SAIDA / f"{PREFIXO}_piso.csv", index=False)
    print(f"gavetas no piso: {list(refeicoes.gavetas[no_piso])}", flush=True)

    G_h, P_h, C_h = (a[hist] for a in (refeicoes.G, refeicoes.P, refeicoes.C))
    residuo_h = residuos(fa, G_h, P_h, C_h)
    G_a, P_a, C_a, y_a = (a[avaliar] for a in (refeicoes.G, refeicoes.P, refeicoes.C, refeicoes.y))
    ocasiao_a = dias.ocasiao.to_numpy()[avaliar]
    fora_do_piso = ~np.isin(np.arange(len(tau2)), no_piso)

    tabelas: dict[str, list[dict]] = {"imputacao": [], "regressao": [], "ocasiao": [], "conferencia": []}
    for cenario in CENARIOS:
        for taxa in TAXAS:
            t = int(np.flatnonzero(mascaras.taxas == taxa)[0])
            comeco = time.perf_counter()
            for si, semente in enumerate(mascaras.sementes):
                M_h = mascaras.M[si, t][hist]
                M_a = mascaras.M[si, t][avaliar]
                observado_h = P_h if cenario == "H1" else P_h & ~M_h
                delta_p, v_p = estimar_delta_v2(residuo_h, observado_h, cod_hist, len(ordem), variancias, VARIANTE)
                delta, v = delta_p[cod_avaliar], v_p[cod_avaliar]
                celulas, medias, var_braco, media_log, cov_log = medir(
                    fa, blr, G_a, P_a, M_a, C_a, delta, v, tau2, gerador(semente, t, FLUXO_AMOSTRAGEM),
                )
                chave = {"cenario": cenario, "taxa": taxa, "semente": int(semente), "imputador": IMPUTADOR}
                tabelas["imputacao"].append({**chave, **avaliar_imputacao(celulas)})
                for braco in BRACOS:
                    resultado = avaliar_mistura(y_a, medias[braco], var_braco[braco])
                    tabelas["regressao"].append({**chave, "braco": braco, **resultado})
                    if taxa == 1.0:
                        for codigo in OCASIOES_AVALIADAS:
                            m = ocasiao_a == codigo
                            tabelas["ocasiao"].append({
                                **chave, "braco": braco, "ocasiao": OCASIOES[codigo],
                                **avaliar_mistura(y_a[m], np.atleast_2d(medias[braco])[:, m],
                                                  np.atleast_2d(var_braco[braco])[:, m]),
                            })

                minimo = menor_autovalor_das_ocultas(cov_log, M_a)
                conferencia = {**chave, "menor_autovalor": minimo, "media_log_max_diff_fa": np.nan,
                               "cov_log_max_diff_fa": np.nan}
                if cenario == "H2" and taxa == 1.0:
                    media_fa, cov_fa = momentos_ocultos(fa, ocultar(G_a, M_a), P_a, M_a, C_a)
                    celula = M_a & fora_do_piso[None, :]
                    par = celula[:, :, None] & celula[:, None, :]
                    conferencia["media_log_max_diff_fa"] = float(np.max(np.abs(media_log - media_fa)[celula]))
                    conferencia["cov_log_max_diff_fa"] = float(np.max(np.abs(cov_log - cov_fa)[par]))
                tabelas["conferencia"].append(conferencia)
                gravar(tabelas)
                if not minimo > 0:
                    raise ValueError(f"conferência 2 ({cenario}, taxa {taxa}, semente {semente}): menor autovalor {minimo}")
                if cenario == "H2" and taxa == 1.0 and (
                    conferencia["media_log_max_diff_fa"] != 0.0 or conferencia["cov_log_max_diff_fa"] != 0.0
                ):
                    raise ValueError(f"conferência 1 (H2 taxa 1,0, semente {semente}): {conferencia}")
            marcar(f"{cenario} taxa {taxa}", comeco, cenario=cenario, taxa=taxa)
            pd.DataFrame(tempos).to_csv(SAIDA / f"{PREFIXO}_tempos.csv", index=False)

    print(f"tempo total: {(time.perf_counter() - inicio_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
