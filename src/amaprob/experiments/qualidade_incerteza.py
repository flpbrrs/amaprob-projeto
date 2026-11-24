"""Qualidade da incerteza: calibração, se o modelo sabe quando erra, gavetas e o caso dos doces.

Base: FA (análise fatorial) condicional + regressão linear bayesiana, como no experimento central;
braços plug-in, propagação S = 100 e analítico; taxas 0, 0,5 e 1,0; 3 sementes de máscara (na taxa
0 a máscara é vazia nas três, e só a semente 0 é medida). Validação apenas; o teste não é lido.

As previsões são feitas uma vez e guardadas por refeição em
`data/interim/qualidade_previsoes.npz` (fora do git): y, média, desvio, PIT (transformada
integral de probabilidade), limites do intervalo de 90% e máscara, por (taxa, semente, braço), e
para o braço analítico também a contribuição de cada gaveta à variância e os momentos da
imputação. As quatro partes leem só esse arquivo.

1. Calibração em vários níveis. A cobertura do intervalo central de nível α sai da PIT: com a CDF
   preditiva F contínua e crescente, y ∈ [F⁻¹((1−α)/2), F⁻¹((1+α)/2)] equivale a
   F(y) ∈ [(1−α)/2, (1+α)/2]. Na mistura, F é a média das CDFs das componentes. A cobertura de
   90% assim obtida é conferida contra o intervalo por bissecção de `metricas.intervalo`.
2. Se o modelo sabe quando erra: decis do desvio previsto, com a razão RMSE / desvio, em que o
   desvio é a raiz da variância média do decil (para uma preditiva calibrada, E[(y − μ)²] = σ²,
   então a razão é 1); correlação de Spearman entre desvio e |erro|; curva risco × cobertura.
3. Por gaveta: (a) carboidrato nas refeições com a gaveta presente e com ela oculta; (b) a
   imputação da célula oculta pelos momentos em forma fechada da FA condicional, em log1p(gramas)
   (gaussiana) e em gramas (log-normal; a média e o desvio são os da log-normal, e o intervalo de
   90% é a imagem do intervalo em log, com a mesma cobertura).
4. Doces, Açúcar, Cereais e Carnes por taxa, e a fração da variância preditiva atribuída a cada
   uma pela decomposição do braço analítico (`segmentos.decompor_variancia`).

Regra 2: as gramas de M só são lidas na avaliação (verdade da célula oculta e y).

Uso: uv run python -m amaprob.experiments.qualidade_incerteza [--reusar]
"""

import argparse
import time

import numpy as np
import pandas as pd
from plotnine import (
    aes, facet_wrap, geom_abline, geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual,
    scale_linetype_manual, scale_x_log10, scale_y_log10,
)
from scipy.special import ndtr
from scipy.stats import spearmanr

from amaprob.data.dataset import RAIZ
from amaprob.data.splits import VALIDACAO
from amaprob.evaluation.metricas import (
    ORCAMENTO_BLOCO_BYTES, TOLERANCIA_MEIA_DOSE_G, Z_INTERVALO, avaliar_mistura, intervalo,
)
from amaprob.evaluation.tema import CORES, TEMA
from amaprob.experiments.propagacao import FLUXO_VALIDACAO, IMPUTADORES, S, gerador, treino_e_medida
from amaprob.experiments.segmentos import decompor_variancia
from amaprob.imputation.base import ocultar
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagacao_analitica import momentos_em_gramas, prever_com_entrada_incerta
from amaprob.regression.propagation import PLUGIN, PROPAGACAO, momentos_da_mistura, prever_bracos

SAIDA = RAIZ / "results"
CACHE = RAIZ / "data" / "interim" / "qualidade_previsoes.npz"
IMPUTADOR = "fa_condicional"
TAXAS = (0.0, 0.5, 1.0)
ANALITICA = "analitica"
BRACOS = {PLUGIN: "plug-in", f"propagacao_S{S}": "propagação S = 100", ANALITICA: "analítica"}
NIVEIS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
CLASSES_PIT = 10
DECIS = 10
FRACOES = tuple(np.round(np.arange(0.1, 1.01, 0.1), 1))
PERMUTACOES = 50
GAVETAS_DO_CASO = ("Doces", "Açúcar", "Cereais", "Carnes")
BLOCO = 4_096


def pit_mistura(y: np.ndarray, medias: np.ndarray, variancias: np.ndarray) -> np.ndarray:
    """F(y) da mistura de pesos iguais (1/S) Σ N(μ_s, σ²_s); com S = 1, Φ((y − μ)/σ)."""
    medias = np.atleast_2d(medias)
    desvios = np.sqrt(np.atleast_2d(variancias))
    pit = np.empty(medias.shape[1])
    passo = max(1, ORCAMENTO_BLOCO_BYTES // (8 * medias.shape[0] * 2))
    for inicio in range(0, len(pit), passo):
        b = slice(inicio, inicio + passo)
        pit[b] = ndtr((y[b] - medias[:, b]) / desvios[:, b]).mean(axis=0)
    return pit


def cobertura_por_nivel(pit: np.ndarray, niveis=NIVEIS) -> pd.DataFrame:
    linhas = [{"nominal": a, "empirica": float(np.mean((pit >= (1 - a) / 2) & (pit <= (1 + a) / 2)))}
              for a in niveis]
    return pd.DataFrame(linhas)


def histograma_pit(pit: np.ndarray, classes: int = CLASSES_PIT) -> tuple[np.ndarray, float]:
    """Frequência relativa por classe de largura 1/classes e o desvio máximo de 1/classes."""
    frequencia = np.bincount(np.minimum((pit * classes).astype(int), classes - 1), minlength=classes) / len(pit)
    return frequencia, float(np.max(np.abs(frequencia - 1 / classes)))


def por_decil_do_desvio(y, media, desvio, dentro90, decis: int = DECIS) -> pd.DataFrame:
    """Por decil do desvio previsto: desvio médio, desvio RMS, RMSE, razão RMSE / desvio RMS e PICP90."""
    decil = pd.qcut(pd.Series(desvio).rank(method="first"), decis, labels=False) + 1
    tabela = pd.DataFrame({"decil": decil, "desvio": desvio, "var": desvio**2, "erro2": (y - media) ** 2,
                           "dentro": dentro90}).groupby("decil").agg(
        n=("desvio", "size"), desvio_medio=("desvio", "mean"), var_media=("var", "mean"),
        mse=("erro2", "mean"), picp90=("dentro", "mean"))
    tabela["desvio_rms"] = np.sqrt(tabela.pop("var_media"))
    tabela["rmse"] = np.sqrt(tabela.pop("mse"))
    tabela["razao_rmse_desvio"] = tabela.rmse / tabela.desvio_rms
    return tabela.reset_index()


def risco_cobertura(y, media, desvio, rng: np.random.Generator, fracoes=FRACOES,
                    permutacoes: int = PERMUTACOES) -> pd.DataFrame:
    """MAE e % de meia dose nas x% refeições de menor desvio, contra a ordem aleatória (média)."""
    erro = np.abs(y - media)
    n = len(y)
    ordem = np.argsort(desvio, kind="stable")
    aleatorias = [rng.permutation(n) for _ in range(permutacoes)]
    linhas = []
    for x in fracoes:
        m = int(round(x * n))
        sel, alea = erro[ordem[:m]], [erro[p[:m]] for p in aleatorias]
        linhas.append({
            "fracao": x, "mae_g": sel.mean(), "pct_meia_dose": 100 * np.mean(sel <= TOLERANCIA_MEIA_DOSE_G),
            "mae_g_aleatoria": float(np.mean([a.mean() for a in alea])),
            "pct_meia_dose_aleatoria": float(np.mean([100 * np.mean(a <= TOLERANCIA_MEIA_DOSE_G) for a in alea])),
        })
    return pd.DataFrame(linhas)


def resumo_de_grupo(y, media, desvio, li, ls) -> dict:
    """n, desvio médio, RMSE, razão RMSE / desvio RMS, PICP90 e MPIW de um grupo de refeições."""
    if len(y) == 0:
        return {"n": 0, "desvio_medio": np.nan, "rmse": np.nan, "razao_rmse_desvio": np.nan,
                "picp90": np.nan, "mpiw_g": np.nan}
    rmse = float(np.sqrt(np.mean((y - media) ** 2)))
    return {"n": len(y), "desvio_medio": float(desvio.mean()), "rmse": rmse,
            "razao_rmse_desvio": rmse / float(np.sqrt(np.mean(desvio**2))),
            "picp90": float(np.mean((li <= y) & (y <= ls))), "mpiw_g": float(np.mean(ls - li))}


def prever_tudo(tr, medida, taxas, sementes) -> dict[str, np.ndarray]:
    """Previsões dos três braços e momentos da imputação por (taxa, semente), num dicionário plano."""
    sem_mascara = np.zeros_like(tr.P)
    imputador = IMPUTADORES[IMPUTADOR]().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
    blr = RegressaoLinearBayesiana().fit(montar_Z(tr.G, tr.P, tr.C), tr.y)
    y, (N, K) = medida.y, medida.G.shape
    cache: dict[str, np.ndarray] = {"y": y, "G": medida.G, "P": medida.P}
    for taxa in TAXAS:
        t = int(np.flatnonzero(np.isclose(taxas, taxa))[0])
        for s, semente in enumerate(sementes if taxa > 0 else sementes[:1]):
            comeco = time.perf_counter()
            M = medida.M[s, t]
            chave = f"{taxa}_{int(semente)}"
            cache[f"M_{chave}"] = M
            bracos = prever_bracos(imputador, blr, medida.G, medida.P, M, medida.C, S,
                                   gerador(semente, t, FLUXO_VALIDACAO))

            media_a, var_a = np.empty(N), np.empty(N)
            contribuicao = np.zeros((N, K), dtype=np.float32)
            media_log, var_log = np.full((N, K), np.nan), np.full((N, K), np.nan)
            media_g, var_g = np.full((N, K), np.nan), np.full((N, K), np.nan)
            for inicio in range(0, N, BLOCO):
                b = slice(inicio, inicio + BLOCO)
                ml, cl = momentos_ocultos(imputador, ocultar(medida.G[b], M[b]), medida.P[b], M[b], medida.C[b])
                Eg, Cg = momentos_em_gramas(medida.G[b], medida.P[b], M[b], ml, cl)
                media_a[b], var_a[b] = prever_com_entrada_incerta(blr, Eg, Cg, medida.P[b], medida.C[b])
                contribuicao[b] = decompor_variancia(blr, medida.G[b], medida.P[b], M[b], medida.C[b], ml, cl)[0]
                oculta = M[b]
                media_log[b] = np.where(oculta, ml, np.nan)
                var_log[b] = np.where(oculta, np.einsum("nkk->nk", cl), np.nan)
                media_g[b] = np.where(oculta, Eg, np.nan)
                var_g[b] = np.where(oculta, np.einsum("nkk->nk", Cg), np.nan)

            previsoes = {PLUGIN: bracos[PLUGIN], f"propagacao_S{S}": bracos[PROPAGACAO],
                         ANALITICA: (media_a[None], var_a[None])}
            for braco, (medias, variancias) in previsoes.items():
                media, var = momentos_da_mistura(medias, variancias)
                li, ls = intervalo(medias, variancias)
                cache.update({f"{campo}_{chave}_{braco}": valor for campo, valor in (
                    ("media", media), ("desvio", np.sqrt(var)), ("pit", pit_mistura(y, medias, variancias)),
                    ("li", li), ("ls", ls))})
                cache[f"metricas_{chave}_{braco}"] = np.array(list(avaliar_mistura(y, medias, variancias).values()))
            cache.update({f"contribuicao_{chave}": contribuicao, f"var_total_{chave}": var_a,
                          f"media_log_{chave}": media_log, f"var_log_{chave}": var_log,
                          f"media_g_{chave}": media_g, f"var_g_{chave}": var_g,
                          f"segundos_{chave}": np.array(time.perf_counter() - comeco)})
            print(f"  taxa {taxa}, semente {semente}: {time.perf_counter() - comeco:.1f} s", flush=True)
    return cache


def combinacoes(cache) -> list[tuple[float, int]]:
    return sorted({(float(k.split("_")[1]), int(k.split("_")[2])) for k in cache if k.startswith("M_")})


def parte_1(cache) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    niveis, pit, conferencia = [], [], []
    y = cache["y"]
    for taxa, semente in combinacoes(cache):
        for braco in BRACOS:
            c = f"{taxa}_{semente}_{braco}"
            chave = {"taxa": taxa, "semente": semente, "braco": braco}
            niveis.append(cobertura_por_nivel(cache[f"pit_{c}"]).assign(**chave))
            frequencia, desvio_max = histograma_pit(cache[f"pit_{c}"])
            pit.append(pd.DataFrame({"classe": np.arange(1, CLASSES_PIT + 1), "frequencia": frequencia}).assign(
                desvio_max_uniforme=desvio_max, **chave))
            dentro_pit = np.mean(np.abs(cache[f"pit_{c}"] - 0.5) <= 0.45)
            dentro_bisseccao = np.mean((cache[f"li_{c}"] <= y) & (y <= cache[f"ls_{c}"]))
            conferencia.append({**chave, "picp90_pela_pit": dentro_pit, "picp90_pela_bisseccao": dentro_bisseccao,
                                "diferenca": dentro_pit - dentro_bisseccao})
    return pd.concat(niveis), pd.concat(pit), pd.DataFrame(conferencia)


def parte_2(cache) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decis, correlacoes, risco = [], [], []
    y = cache["y"]
    for taxa, semente in combinacoes(cache):
        for braco in BRACOS:
            c = f"{taxa}_{semente}_{braco}"
            media, desvio = cache[f"media_{c}"], cache[f"desvio_{c}"]
            dentro = (cache[f"li_{c}"] <= y) & (y <= cache[f"ls_{c}"])
            chave = {"taxa": taxa, "semente": semente, "braco": braco}
            decis.append(por_decil_do_desvio(y, media, desvio, dentro).assign(**chave))
            correlacoes.append({**chave, "spearman_desvio_erro": spearmanr(desvio, np.abs(y - media)).statistic})
            risco.append(risco_cobertura(y, media, desvio, np.random.default_rng(semente)).assign(**chave))
    return pd.concat(decis), pd.DataFrame(correlacoes), pd.concat(risco)


def parte_3(cache, gavetas) -> tuple[pd.DataFrame, pd.DataFrame]:
    carboidrato, imputacao = [], []
    y, G, P = cache["y"], cache["G"], cache["P"].astype(bool)
    for taxa, semente in combinacoes(cache):
        M = cache[f"M_{taxa}_{semente}"].astype(bool)
        for braco in BRACOS:
            c = f"{taxa}_{semente}_{braco}"
            media, desvio, li, ls = (cache[f"{campo}_{c}"] for campo in ("media", "desvio", "li", "ls"))
            for k, gaveta in enumerate(gavetas):
                for condicao, linhas in (("presente", P[:, k]), ("oculta", M[:, k])):
                    if condicao == "oculta" and taxa == 0:
                        continue
                    carboidrato.append({"taxa": taxa, "semente": semente, "braco": braco, "gaveta": gaveta,
                                        "condicao": condicao, **resumo_de_grupo(
                                            y[linhas], media[linhas], desvio[linhas], li[linhas], ls[linhas])})
        if taxa == 0:
            continue
        ml, vl, mg, vg = (cache[f"{campo}_{taxa}_{semente}"] for campo in ("media_log", "var_log", "media_g", "var_g"))
        for k, gaveta in enumerate(gavetas):
            linhas = M[:, k]
            if not linhas.any():
                continue
            verdade_g = G[linhas, k]
            verdade_log, mu, sd = np.log1p(verdade_g), ml[linhas, k], np.sqrt(vl[linhas, k])
            rmse_log = float(np.sqrt(np.mean((verdade_log - mu) ** 2)))
            rmse_g = float(np.sqrt(np.mean((verdade_g - mg[linhas, k]) ** 2)))
            imputacao.append({
                "taxa": taxa, "semente": semente, "gaveta": gaveta, "n": int(linhas.sum()),
                "desvio_medio_log": float(sd.mean()), "rmse_log": rmse_log,
                "razao_log": rmse_log / float(np.sqrt(np.mean(sd**2))),
                "desvio_medio_g": float(np.sqrt(vg[linhas, k]).mean()), "rmse_g": rmse_g,
                "razao_g": rmse_g / float(np.sqrt(np.mean(vg[linhas, k]))),
                "picp90": float(np.mean(np.abs(verdade_log - mu) <= Z_INTERVALO * sd)),
            })
    return pd.DataFrame(carboidrato), pd.DataFrame(imputacao)


def parte_4(carboidrato: pd.DataFrame, cache, gavetas) -> tuple[pd.DataFrame, pd.DataFrame]:
    caso = carboidrato[carboidrato.gaveta.isin(GAVETAS_DO_CASO)]
    fracoes = []
    for taxa, semente in combinacoes(cache):
        contribuicao = cache[f"contribuicao_{taxa}_{semente}"].astype(np.float64)
        total = cache[f"var_total_{taxa}_{semente}"]
        M = cache[f"M_{taxa}_{semente}"].astype(bool)
        for gaveta in GAVETAS_DO_CASO:
            k = int(np.flatnonzero(gavetas == gaveta)[0])
            oculta = M[:, k]
            fracoes.append({
                "taxa": taxa, "semente": semente, "gaveta": gaveta,
                "fracao_da_variancia_total": float(contribuicao[:, k].sum() / total.sum()),
                "fracao_da_variancia_da_entrada": float(contribuicao[:, k].sum() / max(contribuicao.sum(), 1e-300)),
                "fracao_media_nas_refeicoes_com_a_gaveta_oculta": (
                    float(np.mean(contribuicao[oculta, k] / total[oculta])) if oculta.any() else 0.0),
            })
    return caso, pd.DataFrame(fracoes)


def media_das_sementes(tabela: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    return tabela.drop(columns=["semente"]).groupby(chaves, sort=False).mean(numeric_only=True).reset_index()


def _rotular(tabela: pd.DataFrame) -> pd.DataFrame:
    tabela = tabela.copy()
    tabela["braco"] = pd.Categorical(tabela.braco.map(BRACOS), list(BRACOS.values()))
    tabela["taxa"] = pd.Categorical(tabela.taxa.map(lambda t: f"taxa {t:.1f}".replace(".", ",")),
                                    [f"taxa {t:.1f}".replace(".", ",") for t in TAXAS])
    return tabela


def figuras(niveis: pd.DataFrame, decis: pd.DataFrame, risco: pd.DataFrame) -> None:
    pasta = SAIDA / "figuras"
    pasta.mkdir(parents=True, exist_ok=True)
    cores = scale_color_manual(values=CORES[:3])
    legenda = "Validação, 45.639 refeições; FA condicional + regressão linear bayesiana; média de 3 sementes."

    dados = _rotular(media_das_sementes(niveis, ["taxa", "braco", "nominal"]))
    (ggplot(dados, aes("nominal", "empirica", color="braco"))
     + geom_abline(slope=1, intercept=0, color="#4D4D4D", linetype="dashed")
     + geom_line(size=1) + geom_point(size=2.4) + facet_wrap("~ taxa") + cores + TEMA
     + labs(title="Cobertura empírica por nível nominal", subtitle="Intervalos centrais de 50 a 99%; tracejada: calibração perfeita.",
            x="Cobertura nominal", y="Cobertura empírica", color="Braço", caption=legenda)
     ).save(pasta / "qualidade_calibracao.png", width=13, height=5.5, dpi=200, verbose=False)

    dados = _rotular(media_das_sementes(decis, ["taxa", "braco", "decil"]))
    (ggplot(dados, aes("desvio_rms", "rmse", color="braco"))
     + geom_abline(slope=1, intercept=0, color="#4D4D4D", linetype="dashed")
     + geom_line(size=1) + geom_point(size=2.4) + facet_wrap("~ taxa") + scale_x_log10() + scale_y_log10()
     + cores + TEMA
     + labs(title="Desvio previsto × erro realizado, por decil do desvio",
            subtitle="Tracejada: RMSE igual ao desvio previsto (o modelo sabe o tamanho do próprio erro).",
            x="Desvio previsto (g, raiz da variância média do decil)", y="RMSE realizado (g)", color="Braço",
            caption=legenda)
     ).save(pasta / "qualidade_desvio_rmse.png", width=13, height=5.5, dpi=200, verbose=False)

    dados = media_das_sementes(risco, ["taxa", "braco", "fracao"])
    longo = pd.concat([
        dados.assign(ordem="menor desvio primeiro", valor=dados.mae_g),
        dados.assign(ordem="ordem aleatória", valor=dados.mae_g_aleatoria),
    ])
    longo = _rotular(longo[longo.taxa > 0])
    (ggplot(longo, aes("fracao", "valor", color="braco", linetype="ordem"))
     + geom_line(size=1) + geom_point(size=1.8) + facet_wrap("~ taxa") + cores
     + scale_linetype_manual(values=["solid", "dashed"]) + TEMA
     + labs(title="Risco × cobertura", subtitle="MAE entre as x% refeições de menor desvio previsto.",
            x="Fração de refeições aceitas", y="MAE (g)", color="Braço", linetype="Ordem", caption=legenda)
     ).save(pasta / "qualidade_risco_cobertura.png", width=11, height=5.5, dpi=200, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reusar", action="store_true", help="lê as previsões de data/interim em vez de refazê-las")
    escolha = parser.parse_args()
    inicio = time.perf_counter()
    tr, medida, taxas, sementes, gavetas = treino_e_medida(VALIDACAO)
    if escolha.reusar and CACHE.exists():
        cache = dict(np.load(CACHE))
    else:
        cache = prever_tudo(tr, medida, taxas, sementes)
        np.savez_compressed(CACHE, **cache)
    tempo_previsao = time.perf_counter() - inicio
    gavetas = np.asarray(gavetas)

    niveis, pit, conferencia = parte_1(cache)
    decis, correlacoes, risco = parte_2(cache)
    carboidrato, imputacao = parte_3(cache, gavetas)
    caso, fracoes = parte_4(carboidrato, cache, gavetas)
    metricas = pd.DataFrame([
        {"taxa": taxa, "semente": semente, "braco": braco,
         **dict(zip(("mae_g", "nll", "crps_g", "picp", "mpiw_g", "pct_meia_dose"), cache[f"metricas_{taxa}_{semente}_{braco}"]))}
        for taxa, semente in combinacoes(cache) for braco in BRACOS
    ])
    tabelas = {
        "metricas": metricas, "niveis": niveis, "pit": pit, "conferencia_picp": conferencia, "decis": decis,
        "spearman": correlacoes, "risco_cobertura": risco, "gavetas_carboidrato": carboidrato,
        "gavetas_imputacao": imputacao, "doces": caso, "doces_fracao_variancia": fracoes,
    }
    for nome, tabela in tabelas.items():
        tabela.to_csv(SAIDA / f"qualidade_{nome}.csv", index=False)
    figuras(niveis, decis, risco)
    segundos = [float(cache[k]) for k in cache if k.startswith("segundos_")]
    pd.DataFrame({"etapa": ["previsões (todas as combinações)", "análises e figuras"],
                  "segundos": [sum(segundos) if segundos else tempo_previsao, time.perf_counter() - inicio - tempo_previsao]}
                 ).to_csv(SAIDA / "qualidade_tempos.csv", index=False)
    print(f"tempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
