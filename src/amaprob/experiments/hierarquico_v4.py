"""Modelo hierárquico v4: Σ_novo = Σ_FA − c·D, por gaveta e fator único, mesmo cenário da v2.

Mesmo cenário de `experiments/hierarquico_v2.py` (validação, pessoas com os dois dias, dia 1 como
histórico, dia 2 avaliado; H1 completo e H2 mascarado; taxas 0,25 a 1,0; 3 sementes; imputação,
três braços da regressão linear bayesiana e métricas por ocasião na taxa 1,0), com os momentos de
`imputation/hierarquico_v4.py` no lugar da projeção da v2. A amostragem é a da v2
(`amostrar_bloco`), com o mesmo gerador por variante; a variante sem pessoa é a da v2 (D = 0, a
mesma conta) e não é refeita.

Conferências, que abortam a rodada se falharem:
1. H2 na taxa 1,0 (sem histórico): média e covariância em log idênticas às da FA sem pessoa
   (`momentos_ocultos`), diferença 0, nas duas variantes e em todas as gavetas;
2. em toda refeição: menor autovalor de Σ_novo − 0,1·Σ_FA ≥ −1e-10 e diag(Σ_novo) ≤ diag(Σ_FA).
(A conferência 3, c = 1 reproduz a v2, é o teste sintético em `tests/test_hierarquico_v4.py`.)

Saídas novas em results/hierarquico_v4_*.csv, regravadas a cada (cenário, taxa, semente), mais
`hierarquico_v4_c.csv` (distribuição de c) e `hierarquico_v4_comparacao.csv` (sem pessoa, v2, v3
e v4 por gaveta, H1, taxas 0,5 e 1,0).

Uso: uv run python -m amaprob.experiments.hierarquico_v4
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
from amaprob.imputation.hierarquico_v4 import VARIANTES_V4, menor_autovalor_folga, momentos_com_pessoa_v4
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagacao_analitica import momentos_em_gramas, prever_com_entrada_incerta
from amaprob.regression.propagation import COPIAS_POR_PREVISAO, PLUGIN, PROPAGACAO

SAIDA = RAIZ / "results"
PREFIXO = "hierarquico_v4"
IMPUTADORES = {"gaveta": "fa_por_gaveta_v4", "fator": "fa_fator_unico_v4"}
ANALITICA = "analitica"
TOLERANCIA_FOLGA = -1e-10


def medir(fa, blr, G, P, M, C, delta, v, variancias, variante, rng):
    """Momentos v4, amostragem em blocos e os três braços (como `medir_variante_v2`)."""
    N, K = G.shape
    media_log, cov_log, c, cov_fa = momentos_com_pessoa_v4(fa, G, P, M, C, delta, v, variancias, variante)
    X = ocultar(G, M)
    celulas = []
    medias = {PLUGIN: np.empty((1, N)), PROPAGACAO: np.empty((N_AMOSTRAS, N))}
    var_braco = {PLUGIN: np.empty((1, N)), PROPAGACAO: np.empty((N_AMOSTRAS, N))}
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
            var_braco[braco][:, b] = var.reshape(-1, n_b)
    Eg, Cg = momentos_em_gramas(G, P, M, media_log, cov_log)
    medias[ANALITICA], var_braco[ANALITICA] = prever_com_entrada_incerta(blr, Eg, Cg, P, C)
    return pd.concat(celulas, ignore_index=True), medias, var_braco, media_log, cov_log, c, cov_fa


def resumo_c(c: np.ndarray, M: np.ndarray) -> dict:
    """Distribuição de c nas refeições com ao menos uma oculta."""
    c = c[M.any(axis=1)]
    abaixo = c[c < 1.0]
    return {
        "n_refeicoes_com_oculta": len(c),
        "fracao_c_igual_1": float(np.mean(c == 1.0)) if len(c) else np.nan,
        "n_c_menor_1": len(abaixo),
        "mediana_c_menor_1": float(np.median(abaixo)) if len(abaixo) else np.nan,
        "p10_c_menor_1": float(np.percentile(abaixo, 10)) if len(abaixo) else np.nan,
        "minimo_c": float(c.min()) if len(c) else np.nan,
    }


def gravar(tabelas: dict[str, list[dict]]) -> None:
    for nome, linhas in tabelas.items():
        if not linhas:
            continue
        tabela = pd.DataFrame(linhas)
        tabela.to_csv(SAIDA / f"{PREFIXO}_{nome}_metricas.csv", index=False)
        if nome not in CHAVES:
            continue
        valores = tabela.drop(columns=["semente"]).select_dtypes(exclude="bool")
        resumo = valores.groupby(CHAVES[nome], sort=False).agg(["mean", "std"])
        resumo.columns = [f"{m}_{'media' if e == 'mean' else 'desvio'}" for m, e in resumo.columns]
        resumo.reset_index().to_csv(SAIDA / f"{PREFIXO}_{nome}_resumo.csv", index=False)


def comparar() -> pd.DataFrame:
    """Sem pessoa | v2 | v3 | v4, por gaveta, H1, taxas 0,5 e 1,0 (médias das 3 sementes)."""
    fontes = [("sem pessoa", "hierarquico_v2", "fa_sem_pessoa"), ("v2", "hierarquico_v2", "fa_por_gaveta"),
              ("v3", "hierarquico_v3", "fa_por_gaveta_v3"), ("v4", PREFIXO, IMPUTADORES["gaveta"])]
    linhas = []
    for versao, prefixo, imputador in fontes:
        reg = pd.read_csv(SAIDA / f"{prefixo}_regressao_resumo.csv")
        imp = pd.read_csv(SAIDA / f"{prefixo}_imputacao_resumo.csv")
        for taxa in (0.5, 1.0):
            r = reg[(reg.cenario == "H1") & np.isclose(reg.taxa, taxa) & (reg.imputador == imputador)
                    & (reg.braco == PROPAGACAO)].iloc[0]
            i = imp[(imp.cenario == "H1") & np.isclose(imp.taxa, taxa) & (imp.imputador == imputador)].iloc[0]
            linhas.append({
                "versao": versao, "taxa": taxa,
                **{f"prop_{m}": r[f"{m}_media"] for m in ("mae_g", "nll", "crps_g", "picp", "mpiw_g")},
                **{f"imp_{m}": i[f"{m}_media"] for m in ("mae_log", "nll_log", "picp")},
            })
    return pd.DataFrame(linhas)


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

    G_h, P_h, C_h = (a[hist] for a in (refeicoes.G, refeicoes.P, refeicoes.C))
    residuo_h = residuos(fa, G_h, P_h, C_h)
    G_a, P_a, C_a, y_a = (a[avaliar] for a in (refeicoes.G, refeicoes.P, refeicoes.C, refeicoes.y))
    ocasiao_a = dias.ocasiao.to_numpy()[avaliar]

    tabelas: dict[str, list[dict]] = {"imputacao": [], "regressao": [], "ocasiao": [], "conferencia": []}
    linhas_c: list[dict] = []
    for cenario in CENARIOS:
        for taxa in TAXAS:
            t = int(np.flatnonzero(mascaras.taxas == taxa)[0])
            comeco = time.perf_counter()
            for si, semente in enumerate(mascaras.sementes):
                M_h = mascaras.M[si, t][hist]
                M_a = mascaras.M[si, t][avaliar]
                observado_h = P_h if cenario == "H1" else P_h & ~M_h
                referencia = None
                if cenario == "H2" and taxa == 1.0:
                    referencia = momentos_ocultos(fa, ocultar(G_a, M_a), P_a, M_a, C_a)
                for variante in VARIANTES_V4:
                    delta_p, v_p = estimar_delta_v2(residuo_h, observado_h, cod_hist, len(ordem), variancias, variante)
                    delta, v = delta_p[cod_avaliar], v_p[cod_avaliar]
                    celulas, medias, var_braco, media_log, cov_log, c, cov_fa = medir(
                        fa, blr, G_a, P_a, M_a, C_a, delta, v, variancias, variante,
                        gerador(semente, t, FLUXO_AMOSTRAGEM),
                    )
                    chave = {"cenario": cenario, "taxa": taxa, "semente": int(semente),
                             "imputador": IMPUTADORES[variante]}
                    tabelas["imputacao"].append({**chave, **avaliar_imputacao(celulas)})
                    for braco in BRACOS:
                        tabelas["regressao"].append({**chave, "braco": braco,
                                                     **avaliar_mistura(y_a, medias[braco], var_braco[braco])})
                        if taxa == 1.0:
                            for codigo in OCASIOES_AVALIADAS:
                                m = ocasiao_a == codigo
                                tabelas["ocasiao"].append({
                                    **chave, "braco": braco, "ocasiao": OCASIOES[codigo],
                                    **avaliar_mistura(y_a[m], np.atleast_2d(medias[braco])[:, m],
                                                      np.atleast_2d(var_braco[braco])[:, m]),
                                })
                    linhas_c.append({"variante": variante, **chave, **resumo_c(c, M_a), "_c": c[M_a.any(axis=1)]})

                    folga = menor_autovalor_folga(cov_log, cov_fa, M_a)
                    diag, diag_fa = np.einsum("nii->ni", cov_log), np.einsum("nii->ni", cov_fa)
                    excesso = float(np.max((diag - diag_fa)[M_a])) if M_a.any() else 0.0
                    conferencia = {**chave, "menor_autovalor_folga": folga, "diag_excesso_max": excesso,
                                   "media_log_max_diff_fa": np.nan, "cov_log_max_diff_fa": np.nan}
                    if referencia is not None:
                        media_fa, cov_ref = referencia
                        assert np.array_equal(np.isnan(media_log), np.isnan(media_fa))
                        conferencia["media_log_max_diff_fa"] = float(np.max(np.abs(media_log - media_fa)[M_a]))
                        conferencia["cov_log_max_diff_fa"] = float(np.max(np.abs(cov_log - cov_ref)))
                    tabelas["conferencia"].append(conferencia)
                    gravar(tabelas)
                    pd.DataFrame([{k: x for k, x in l.items() if k != "_c"} for l in linhas_c]).to_csv(
                        SAIDA / f"{PREFIXO}_c_por_semente.csv", index=False)
                    if not (folga >= TOLERANCIA_FOLGA and excesso <= 0.0):
                        raise ValueError(f"conferência 2 ({chave}): folga {folga}, excesso {excesso}")
                    if referencia is not None and (conferencia["media_log_max_diff_fa"] != 0.0
                                                   or conferencia["cov_log_max_diff_fa"] != 0.0):
                        raise ValueError(f"conferência 1 ({chave}): {conferencia}")
            marcar(f"{cenario} taxa {taxa}", comeco, cenario=cenario, taxa=taxa)
            pd.DataFrame(tempos).to_csv(SAIDA / f"{PREFIXO}_tempos.csv", index=False)

    # distribuição de c juntando as 3 sementes
    juntas = []
    for (variante, cenario, taxa), grupo in pd.DataFrame(linhas_c).groupby(["variante", "cenario", "taxa"], sort=False):
        c = np.concatenate(grupo["_c"].to_list())
        juntas.append({"variante": variante, "cenario": cenario, "taxa": taxa,
                       **resumo_c(c, np.ones((len(c), 1), dtype=bool))})
    pd.DataFrame(juntas).to_csv(SAIDA / f"{PREFIXO}_c.csv", index=False)
    comparacao = comparar()
    comparacao.to_csv(SAIDA / f"{PREFIXO}_comparacao.csv", index=False)

    pd.set_option("display.width", 240)
    print("\ndistribuição de c (3 sementes juntas):")
    print(pd.DataFrame(juntas).round(4).to_string(index=False))
    print("\nconferências:")
    print(pd.DataFrame(tabelas["conferencia"]).to_string(index=False))
    print("\ncomparação (por gaveta, H1, propagação e imputação):")
    print(comparacao.round(3).to_string(index=False))
    print(f"\ntempo total: {(time.perf_counter() - inicio_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
