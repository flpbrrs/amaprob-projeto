"""Modelo hierárquico v2: correção de variância Σ_novo = Σ_FA − T + V, mesmo cenário da v1.

Repete a rodada da v1 (`experiments/hierarquico.py`) com a contabilidade de variância corrigida
(`imputation/hierarquico_v2.py`): a FA condicional foi ajustada em todas as refeições, então a
covariância dela já inclui a variação entre pessoas τ², e somar a posterior v por cima a
duplicava. Mesmo cenário: validação, pessoas com os dois dias, dia 1 como histórico (H1 completo,
H2 mascarado), avaliação só no dia 2 com as máscaras existentes (taxas 0,25 a 1,0; 3 sementes),
imputação + os três braços da regressão linear bayesiana + métricas por ocasião na taxa 1,0.

A amostragem das três variantes, inclusive a sem pessoa (T = V = 0), sai de
`amostrar_bloco`: gaussiana N(μ_novo, Σ_novo) por refeição com Cholesky, truncada em 0, caminho
que difere do `FA.sample` oficial da v1. A comparação entre os caminhos é a conferência
`hierarquico_v2_conferencia_oficial.csv`: NLL e PICP do sem pessoa contra a rodada v1 (que usou
o caminho oficial), dentro do erro de Monte Carlo estimado pelo desvio das 3 sementes.

Conferências:
1. H2 na taxa 1,0 (sem histórico): δ̂ = 0 e v = τ², então média e covariância em log de todas as
   variantes são idênticas às da FA sem pessoa (diferença máxima 0) e as métricas de regressão
   das três variantes batem (diferença registrada, exigida 0);
2. em toda refeição, diag(Σ_novo) ≤ diag(Σ_FA) + 1e-12 (a correção nunca aumenta a diagonal);
3. contagem das refeições que precisaram de projeção (autovalores negativos e piso em 1e-8).

Saídas novas em results/hierarquico_v2_*.csv, regravadas ao fim de cada (cenário, taxa); a v1
não é tocada.

Uso: uv run python -m amaprob.experiments.hierarquico_v2
"""

import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import OCASIOES, RAIZ, carregar_mascaras, carregar_refeicoes
from amaprob.data.splits import TREINO, VALIDACAO
from amaprob.evaluation.metricas import (
    ORCAMENTO_BLOCO_BYTES,
    avaliar_imputacao,
    avaliar_mistura,
    celulas_de_imputacao,
)
from amaprob.experiments.diagnostico_dois_dias import carregar_dias
from amaprob.experiments.propagacao import ler_configuracao
from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from amaprob.imputation.hierarquico import VARIANTES, estimar_variancias, residuos
from amaprob.imputation.hierarquico_v2 import (
    amostrar_bloco,
    estimar_delta_v2,
    momentos_com_pessoa_v2,
)
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagacao_analitica import momentos_em_gramas, prever_com_entrada_incerta
from amaprob.regression.propagation import COPIAS_POR_PREVISAO, PLUGIN, PROPAGACAO

SAIDA = RAIZ / "results"
PREFIXO = "hierarquico_v2"
N_AMOSTRAS = 100  # amostras da propagação, o S do experimento oficial
TAXAS = (0.25, 0.5, 0.75, 1.0)
CENARIOS = ("H1", "H2")
ANALITICA = "analitica"
BRACOS = (PLUGIN, PROPAGACAO, ANALITICA)
IMPUTADORES = {"nenhum": "fa_sem_pessoa", "fator": "fa_fator_unico", "gaveta": "fa_por_gaveta"}
OCASIOES_AVALIADAS = (1, 2, 3, 4)  # café, almoço, lanche, jantar (V9017)
FLUXO_AMOSTRAGEM = 1
TOLERANCIA_CONFERENCIA = 1e-12
CHAVES = {
    "imputacao": ["cenario", "taxa", "imputador"],
    "regressao": ["cenario", "taxa", "imputador", "braco"],
    "ocasiao": ["cenario", "taxa", "imputador", "braco", "ocasiao"],
    "conferencia": ["cenario", "taxa", "imputador"],
}


def gerador(semente: int, indice_taxa: int, fluxo: int) -> np.random.Generator:
    return np.random.default_rng((int(semente), indice_taxa, fluxo))


def medir_variante_v2(
    fa: FA,
    blr: RegressaoLinearBayesiana,
    G: np.ndarray,
    P: np.ndarray,
    M: np.ndarray,
    C: np.ndarray,
    delta: np.ndarray,
    v: np.ndarray,
    variancias,
    variante: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, object]:
    """Momentos de Σ_novo, amostragem em blocos e os três braços; devolve também as conferências.

    Retorna as células ocultas, (médias, variâncias) por braço, a média e a covariância em log
    (para a conferência do H2 na taxa 1,0) e as estatísticas da correção de variância.
    """
    N, K = G.shape
    media_log, cov_log, estatisticas = momentos_com_pessoa_v2(
        fa, G, P, M, C, delta, v, variancias, variante
    )
    X = ocultar(G, M)
    celulas = []
    medias = {PLUGIN: np.empty((1, N)), PROPAGACAO: np.empty((N_AMOSTRAS, N))}
    variancias_braco = {PLUGIN: np.empty((1, N)), PROPAGACAO: np.empty((N_AMOSTRAS, N))}
    largura_Z = 2 * K + C.shape[1]
    tamanho = max(1, ORCAMENTO_BLOCO_BYTES // (N_AMOSTRAS * largura_Z * 8 * COPIAS_POR_PREVISAO))

    for inicio in range(0, N, tamanho):
        b = slice(inicio, inicio + tamanho)
        amostras_log = amostrar_bloco(media_log[b], cov_log[b], X[b], M[b], N_AMOSTRAS, rng)
        celulas.append(celulas_de_imputacao(G[b], amostras_log, M[b]))
        G_s = np.expm1(amostras_log)
        s, n_b, _ = G_s.shape
        Z = {
            PLUGIN: montar_Z(G_s.mean(axis=0), P[b], C[b]),
            PROPAGACAO: montar_Z(
                G_s.reshape(s * n_b, K), np.tile(P[b], (s, 1)), np.tile(C[b], (s, 1))
            ),
        }
        for braco in (PLUGIN, PROPAGACAO):
            media, var = blr.predict(Z[braco])
            medias[braco][:, b] = media.reshape(-1, n_b)
            variancias_braco[braco][:, b] = var.reshape(-1, n_b)

    Eg, Cg = momentos_em_gramas(G, P, M, media_log, cov_log)
    medias[ANALITICA], variancias_braco[ANALITICA] = prever_com_entrada_incerta(blr, Eg, Cg, P, C)
    return (
        pd.concat(celulas, ignore_index=True),
        (medias, variancias_braco),
        media_log,
        cov_log,
        estatisticas,
    )


def gravar(tabelas: dict[str, list[dict]]) -> None:
    """CSVs com sufixo _v2 e o resumo média ± desvio das sementes; a cada (cenário, taxa)."""
    for nome, linhas in tabelas.items():
        if not linhas:
            continue
        tabela = pd.DataFrame(linhas)
        tabela.to_csv(SAIDA / f"{PREFIXO}_{nome}_metricas.csv", index=False)
        valores = tabela.drop(columns=["semente"])
        resumo = valores.groupby(CHAVES[nome], sort=False).agg(["mean", "std"])
        resumo.columns = [
            f"{metrica}_{'media' if estatistica == 'mean' else 'desvio'}"
            for metrica, estatistica in resumo.columns
        ]
        resumo.reset_index().to_csv(SAIDA / f"{PREFIXO}_{nome}_resumo.csv", index=False)


def comparar_com_o_caminho_oficial(regressao: list[dict]) -> pd.DataFrame:
    """NLL e PICP do sem pessoa (plug-in e propagação) contra a rodada v1, do FA.sample oficial.

    As duas rodadas são estimativas independentes do mesmo número (mesmas máscaras, mesmo S, só
    o sorteio difere). O erro de Monte Carlo é 3 × o erro padrão da diferença das médias, com o
    desvio das 3 sementes de cada rodada; `dentro_erro_mc` marca a comparação.
    """
    arquivo = SAIDA / "hierarquico_regressao_metricas.csv"
    if not arquivo.exists():
        print("\nAVISO: rodada v1 não encontrada; conferência do caminho oficial pulada.")
        return pd.DataFrame()
    v1 = pd.read_csv(arquivo)
    v2 = pd.DataFrame(regressao)
    seletor = lambda t: (t.imputador == "fa_sem_pessoa") & (t.braco.isin([PLUGIN, PROPAGACAO]))
    v1, v2 = v1[seletor(v1)], v2[seletor(v2)]
    linhas = []
    for (cenario, taxa, braco) in v1.groupby(["cenario", "taxa", "braco"], sort=False).groups:
        a = v1[(v1.cenario == cenario) & (v1.taxa == taxa) & (v1.braco == braco)]
        b = v2[(v2.cenario == cenario) & (v2.taxa == taxa) & (v2.braco == braco)]
        n = len(a)
        for metrica in ("nll", "picp"):
            m1, s1 = float(a[metrica].mean()), float(a[metrica].std())
            m2, s2 = float(b[metrica].mean()), float(b[metrica].std())
            diff = m2 - m1
            tolerancia = 3.0 * np.sqrt((s1**2 + s2**2) / n)
            linhas.append({
                "cenario": cenario, "taxa": taxa, "braco": braco, "metrica": metrica,
                "v1_oficial_media": m1, "v1_oficial_desvio": s1,
                "v2_novo_media": m2, "v2_novo_desvio": s2,
                "diff": diff, "tolerancia_3se": tolerancia,
                "dentro_erro_mc": bool(abs(diff) <= tolerancia),
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
    marcar("leitura do dataset e da POF", comeco)

    # validação: histórico = dia 1, avaliação = dia 2, só quem tem os dois dias
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
    if min(cod_hist.min(), cod_avaliar.min()) < 0:
        raise ValueError("pessoa do histórico ou da avaliação fora do conjunto de pessoas com 2 dias.")
    print(
        f"validação com 2 dias: {len(ordem)} pessoas, {len(hist)} refeições de histórico, "
        f"{len(avaliar)} do dia 2 avaliadas",
        flush=True,
    )

    # FA condicional e resíduos no treino completo (τ² e σ² só saem do treino)
    comeco = time.perf_counter()
    configuracao = ler_configuracao()
    treino = np.flatnonzero(refeicoes.conjunto == TREINO)
    sem_mascara = np.zeros_like(refeicoes.P[treino])
    fa = FA(semente=0, media_condicional=True, **configuracao.get("fa_condicional", {})).fit(
        ocultar(refeicoes.G[treino], sem_mascara), refeicoes.P[treino], sem_mascara, refeicoes.C[treino]
    )
    residuo_treino = residuos(fa, refeicoes.G[treino], refeicoes.P[treino], refeicoes.C[treino])
    variancias = estimar_variancias(
        residuo_treino, refeicoes.person_id[treino], dias.iloc[treino].dia.to_numpy()
    )
    marcar(f"FA condicional no treino ({fa.n_iteracoes_} iterações) e τ², σ²", comeco)

    tau = pd.DataFrame({
        "escopo": ["fator_unico"] + ["gaveta"] * len(refeicoes.gavetas),
        "chave": ["fator_de_porcao", *refeicoes.gavetas],
        "tau2": np.r_[variancias.tau2_fator, variancias.tau2],
        "sigma2": np.r_[variancias.sigma2_fator, variancias.sigma2],
    })
    tau.to_csv(SAIDA / f"{PREFIXO}_tau.csv", index=False)

    comeco = time.perf_counter()
    blr = RegressaoLinearBayesiana().fit(
        montar_Z(refeicoes.G[treino], refeicoes.P[treino], refeicoes.C[treino]), refeicoes.y[treino]
    )
    marcar("regressão linear bayesiana no treino", comeco)

    G_h, P_h, C_h = (a[hist] for a in (refeicoes.G, refeicoes.P, refeicoes.C))
    residuo_h = residuos(fa, G_h, P_h, C_h)
    G_a, P_a, C_a, y_a = (
        a[avaliar] for a in (refeicoes.G, refeicoes.P, refeicoes.C, refeicoes.y)
    )
    ocasiao_a = dias.ocasiao.to_numpy()[avaliar]

    tabelas: dict[str, list[dict]] = {"imputacao": [], "regressao": [], "ocasiao": [], "conferencia": []}
    for cenario in CENARIOS:
        for taxa in TAXAS:
            indice_taxa = int(np.flatnonzero(mascaras.taxas == taxa)[0])
            comeco = time.perf_counter()
            violacao_momentos = 0.0
            violacao_excesso = 0.0
            for si, semente in enumerate(mascaras.sementes):
                M_h = mascaras.M[si, indice_taxa][hist]
                M_a = mascaras.M[si, indice_taxa][avaliar]
                observado_h = P_h if cenario == "H1" else P_h & ~M_h
                base_media = base_cov = None
                base_metricas: dict[str, dict] = {}
                for variante in VARIANTES:
                    delta_p, v_p = estimar_delta_v2(
                        residuo_h, observado_h, cod_hist, len(ordem), variancias, variante
                    )
                    delta, v = delta_p[cod_avaliar], v_p[cod_avaliar]
                    celulas, (medias, var_braco), media_log, cov_log, est = medir_variante_v2(
                        fa, blr, G_a, P_a, M_a, C_a, delta, v, variancias, variante,
                        gerador(semente, indice_taxa, FLUXO_AMOSTRAGEM),
                    )
                    nome = IMPUTADORES[variante]
                    chave = {"cenario": cenario, "taxa": taxa, "semente": int(semente), "imputador": nome}
                    tabelas["imputacao"].append({**chave, **avaliar_imputacao(celulas)})
                    metricas_braco = {}
                    for braco in BRACOS:
                        media, var = medias[braco], var_braco[braco]
                        resultado = avaliar_mistura(y_a, media, var)
                        metricas_braco[braco] = resultado
                        tabelas["regressao"].append({**chave, "braco": braco, **resultado})
                        if taxa == 1.0:
                            for codigo in OCASIOES_AVALIADAS:
                                m = ocasiao_a == codigo
                                tabelas["ocasiao"].append({
                                    **chave, "braco": braco, "ocasiao": OCASIOES[codigo],
                                    **avaliar_mistura(
                                        y_a[m], np.atleast_2d(media)[:, m], np.atleast_2d(var)[:, m]
                                    ),
                                })

                    # conferências contra a variante sem pessoa da mesma semente
                    if base_media is None:
                        base_media, base_cov = media_log, cov_log
                        base_metricas = metricas_braco
                        diff_media = diff_cov = 0.0
                        difs = {"mae_g": 0.0, "nll": 0.0, "picp": 0.0}
                    else:
                        diff_media = float(np.nanmax(np.abs(media_log - base_media)))
                        diff_cov = float(np.max(np.abs(cov_log - base_cov)))
                        difs = {
                            metrica: max(
                                abs(metricas_braco[b][metrica] - base_metricas[b][metrica])
                                for b in BRACOS
                            )
                            for metrica in ("mae_g", "nll", "picp")
                        }
                    violacao_excesso = max(violacao_excesso, est.excesso_diag_max - TOLERANCIA_CONFERENCIA)
                    if cenario == "H2" and taxa == 1.0:
                        violacao_momentos = max(
                            violacao_momentos, diff_media, diff_cov, *difs.values()
                        )
                    tabelas["conferencia"].append({
                        **chave,
                        "media_log_max_diff_sem_pessoa": diff_media,
                        "cov_log_max_diff_sem_pessoa": diff_cov,
                        "diag_excesso_max": est.excesso_diag_max,
                        "projecoes_piso": est.projecoes_piso,
                        "projecoes_negativas": est.projecoes_negativas,
                        "mae_g_max_diff_sem_pessoa": difs["mae_g"],
                        "nll_max_diff_sem_pessoa": difs["nll"],
                        "picp_max_diff_sem_pessoa": difs["picp"],
                    })

            gravar(tabelas)
            if violacao_excesso > 0.0:
                raise ValueError(
                    f"conferência 2 ({cenario} taxa {taxa}): diag(Σ_novo) − diag(Σ_FA) = "
                    f"{violacao_excesso + TOLERANCIA_CONFERENCIA} > 1e-12."
                )
            if violacao_momentos > 0.0:
                raise ValueError(
                    f"conferência 1 (H2 taxa 1,0): variante sem histórico não reproduziu a FA "
                    f"(diferença máxima {violacao_momentos})."
                )
            marcar(f"{cenario} taxa {taxa}", comeco, cenario=cenario, taxa=taxa)

    pd.DataFrame(tempos).to_csv(SAIDA / f"{PREFIXO}_tempos.csv", index=False)

    oficial = comparar_com_o_caminho_oficial(tabelas["regressao"])
    if len(oficial):
        oficial.to_csv(SAIDA / f"{PREFIXO}_conferencia_oficial.csv", index=False)

    pd.set_option("display.width", 240)
    resumos = {
        nome: pd.read_csv(SAIDA / f"{PREFIXO}_{nome}_resumo.csv")
        for nome in ("imputacao", "regressao", "ocasiao", "conferencia")
    }
    print("\nimputação (validação, dia 2):")
    print(resumos["imputacao"].round(3).to_string(index=False))
    print("\nregressão linear bayesiana:")
    print(resumos["regressao"].round(3).to_string(index=False))
    print("\nconferências (momentos e projeções):")
    print(resumos["conferencia"].round(6).to_string(index=False))
    if len(oficial):
        print("\nconferência do caminho oficial (NLL e PICP do sem pessoa vs. rodada v1):")
        print(oficial.round(5).to_string(index=False))
        falhas = oficial[~oficial.dentro_erro_mc]
        print(
            "\ncaminho oficial: "
            + ("todas as comparações dentro do erro de Monte Carlo." if falhas.empty
               else f"{len(falhas)} comparação(ões) fora do erro de Monte Carlo (ver CSV).")
        )
    print(f"\ntempo total: {(time.perf_counter() - inicio_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
