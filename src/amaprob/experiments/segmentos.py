"""Análise segmentada e cenários difíceis (validação apenas, sem ler o teste).

Quatro etapas:

- A: na taxa 0 (gramas completas), regressão linear bayesiana e SVGP por segmento, com n,
  carboidrato médio, MAE, MAE relativo, NLL, CRPS, PICP90, MPIW e % de meia dose;
- B: dentro de cada segmento de local, ocasião e período, a cobertura por quintil do carboidrato
  previsto, para os dois regressores;
- C: com informação faltando (fa_condicional mais regressão linear), taxas 0,5 e 1,0, três
  sementes, braços plug-in, propagação S = 100 e analítica, com as mesmas métricas por segmento e a
  diferença de PICP contra o mesmo segmento na taxa 0;
- D: os segmentos ordenados por MAE relativo e por desvio da PICP em relação a 90%, a descrição dos
  três mais difíceis e dos três mais fáceis, a decomposição da variância preditiva por gaveta na
  taxa 1,0 e a variância do resíduo na taxa 0.

Segmentos por refeição, relendo a POF com as funções existentes (a camada de dados não muda):

- local (V9018): 1 = em casa ou levado de casa; 2 a 7 = fora de casa (merenda escolar,
  restaurante a quilo, cantina ou bar ou lanchonete, restaurante outros, vendedores ambulantes,
  fora de casa outros); 9 = não determinado. A refeição é "misto" quando o carboidrato se divide
  entre itens de dentro e de fora, e a fração de fora é gravada;
- ocasião (V9017): 1 café, 2 almoço, 3 lanche, 4 jantar, 5 ceia;
- período (V9015): madrugada 0 a 5 h, manhã 6 a 11 h, tarde 12 a 17 h, noite 18 a 23 h;
- dia: útil ou fim de semana (DIA_SEMANA) e atípico (DIA_ATIPICO: 1 = atípico, 2 = típico);
- região (5, pelo primeiro dígito da UF) e quintil de renda;
- tamanho (1, 2, 3, 4, 5+ gavetas presentes) e quintil do carboidrato verdadeiro.

Quintis por posto (`rank`), para não falhar com empates de renda. O MAE relativo é o MAE dividido
pelo carboidrato médio do segmento. A variância do resíduo é a média do erro quadrático na taxa 0.
O ranking e a escolha dos extremos usam segmentos com pelo menos 100 refeições na validação, para
que um segmento de 6 refeições não domine por ruído. Os extremos são escolhidos pelos dois
critérios, MAE relativo e desvio absoluto da PICP em relação a 90%.

O dicionário em `data/raw/pof2017_2018/Documentacao_20230713/` não está neste checkout. Os códigos
de V9018 e V9017 foram conferidos no instrumento POF 7 (Bloco de Consumo Alimentar Pessoal) e na
distribuição dos microdados; DIA_ATIPICO tem 1 = atípico e 2 = típico, coerente com 9,7% dos itens.

Uso: uv run python -m amaprob.experiments.segmentos [--etapas A B C D] [--recalcular-segmentos]
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
import scoringrules as sr
from plotnine import (
    aes, element_text, facet_wrap, geom_hline, geom_point, ggplot, labs, position_dodge,
    scale_color_manual, theme,
)

from amaprob.data.dataset import RAIZ
from amaprob.data.gavetas import DIVIDIR_K29, FUNDIR_K29, montar_gavetas
from amaprob.data.pof import read_consumo_alimentar
from amaprob.data.refeicoes import montar_itens_de_refeicao
from amaprob.data.splits import VALIDACAO
from amaprob.evaluation.metricas import (
    ORCAMENTO_BLOCO_BYTES, TOLERANCIA_MEIA_DOSE_G, Z_INTERVALO, intervalo,
)
from amaprob.evaluation.tema import CORES, TEMA
from amaprob.experiments.propagacao import (
    FLUXO_VALIDACAO, IMPUTADORES, S, gerador, ler_configuracao, treino_e_medida,
)
from amaprob.experiments.propagacao_analitica import prever_analitico
from amaprob.imputation.base import ocultar
from amaprob.imputation.momentos import momentos_ocultos
from amaprob.regression.base import montar_Z
from amaprob.regression.blr import RegressaoLinearBayesiana
from amaprob.regression.propagation import PLUGIN, PROPAGACAO, prever_bracos
from amaprob.regression.propagacao_analitica import momentos_em_gramas
from amaprob.regression.svgp import SVGP

SAIDA = RAIZ / "results"
CACHE_SEGMENTOS = RAIZ / "data/interim/segmentos_k29.csv"
ARQUIVO_CONSUMO = RAIZ / "data/raw/pof2017_2018/Dados_20230713/CONSUMO_ALIMENTAR.txt"
CLASSIFICACAO = RAIZ / "configs/classificacao_ibge.csv"
IMPUTADOR_ETAPA_C = "fa_condicional"
TAXAS_ETAPA_C = (0.5, 1.0)
MODELOS_ETAPA_A = {"blr": "Regressão linear bayesiana", "svgp": "SVGP"}

LOCAIS = {
    1: "Em casa ou levado de casa",
    2: "Merenda escolar",
    3: "Restaurante a quilo",
    4: "Cantina, bar, lanchonete, fast food",
    5: "Restaurante, outros",
    6: "Vendedores ambulantes ou de rua",
    7: "Fora de casa, outros",
    9: "Não determinado",
}
OCASIOES = {1: "Café da manhã", 2: "Almoço", 3: "Lanche", 4: "Jantar", 5: "Ceia", 6: "Outra ocasião", 9: "Não determinado"}
OCASIOES_INTERNAS = {1: "cafe", 2: "almoco", 3: "lanche", 4: "jantar", 5: "ceia"}
DIAS_ATIPICOS = {1: "atipico", 2: "tipico"}
DIAS_DA_SEMANA = {
    "Segunda-Feira": "dia_util", "Terca-Feira": "dia_util", "Quarta-Feira": "dia_util",
    "Quinta-Feira": "dia_util", "Sexta-Feira": "dia_util", "Sabado": "fim_de_semana", "Domingo": "fim_de_semana",
}
REGIOES = {0: "norte", 1: "nordeste", 2: "sudeste", 3: "sul", 4: "centro_oeste"}

# Ordem de exibição; o CSV guarda os rótulos, não os códigos.
NIVEIS = {
    "local": ["dentro", "fora", "misto", "nao_informado"],
    "ocasiao": ["cafe", "almoco", "lanche", "jantar", "ceia"],
    "periodo": ["madrugada", "manha", "tarde", "noite"],
    "dia_semana": ["dia_util", "fim_de_semana"],
    "dia_atipico": ["tipico", "atipico"],
    "regiao": ["norte", "nordeste", "sudeste", "sul", "centro_oeste"],
    "quintil_renda": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "tamanho": ["1", "2", "3", "4", "5+"],
    "quintil_carboidrato": ["Q1", "Q2", "Q3", "Q4", "Q5"],
}
DIMENSOES = list(NIVEIS)
DIMENSOES_DA_COBERTURA = ("local", "ocasiao", "periodo")
ROTULOS_DAS_DIMENSOES = {
    "local": "Local", "ocasiao": "Ocasião", "periodo": "Período", "dia_semana": "Dia da semana",
    "dia_atipico": "Dia atípico", "regiao": "Região", "quintil_renda": "Quintil de renda",
    "tamanho": "Gavetas presentes", "quintil_carboidrato": "Quintil de carboidrato",
}


def rotular_periodo(hora: pd.Series) -> pd.Series:
    """0 a 5 h madrugada, 6 a 11 h manhã, 12 a 17 h tarde, 18 a 23 h noite."""
    hora = pd.to_numeric(hora, errors="coerce")
    faixas = pd.cut(hora, bins=[-1, 5, 11, 17, 23], labels=NIVEIS["periodo"])
    return faixas.astype(object)


def rotular_local(fracao_dentro: pd.Series, fracao_fora: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Rótulo por refeição e fração de fora, calculada sobre o carboidrato conhecido.

    Refeição sem item de local conhecido fica "nao_informado". Se o carboidrato se divide entre
    itens de dentro e de fora, fica "misto" e a fração é gravada. Código 9 não entra na fração.
    """
    dentro = fracao_dentro.fillna(0.0)
    fora = fracao_fora.fillna(0.0)
    conhecido = dentro + fora
    fracao = np.where(conhecido > 0, fora / conhecido.replace(0, np.nan), np.nan)
    rotulo = np.select(
        [conhecido <= 0, fora <= 0, dentro <= 0],
        ["nao_informado", "dentro", "fora"],
        default="misto",
    )
    return pd.Series(rotulo, index=dentro.index), pd.Series(fracao, index=dentro.index)


def _ler_itens() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consumo bruto e itens das refeições válidas, com o mesmo meal_id do dataset."""
    consumo = read_consumo_alimentar(ARQUIVO_CONSUMO)
    gavetas = montar_gavetas(pd.read_csv(CLASSIFICACAO), DIVIDIR_K29, FUNDIR_K29)
    return consumo, montar_itens_de_refeicao(consumo, gavetas)


def construir_segmentos(itens: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uma linha por meal_id, com local, fração de fora, ocasião, período, dia e atípico."""
    itens = _ler_itens()[1] if itens is None else itens
    classe = itens.V9018.map({1: "dentro", **{k: "fora" for k in range(2, 8)}, 9: "desconhecido"})
    carboidrato = itens.assign(classe_local=classe).pivot_table(
        index="meal_id", columns="classe_local", values="CHOTOT", aggfunc="sum", fill_value=0.0,
    )
    for coluna in ("dentro", "fora", "desconhecido"):
        if coluna not in carboidrato:
            carboidrato[coluna] = 0.0
    local, fracao_fora = rotular_local(carboidrato["dentro"], carboidrato["fora"])

    primeiro = itens.groupby("meal_id")[["V9017", "V9015", "V9018", "DIA_SEMANA", "DIA_ATIPICO"]].first()
    hora = primeiro.V9015.str.rstrip("H").astype(int)
    tabela = pd.DataFrame({
        "local": local,
        "fracao_fora": fracao_fora,
        "ocasiao": primeiro.V9017.map(OCASIOES_INTERNAS),
        "periodo": rotular_periodo(hora),
        "dia_semana": primeiro.DIA_SEMANA.map(DIAS_DA_SEMANA),
        "dia_atipico": primeiro.DIA_ATIPICO.map(DIAS_ATIPICOS),
        "hora": hora,
    })
    tabela.index.name = "meal_id"
    return tabela


def carregar_segmentos(recalcular: bool = False) -> pd.DataFrame:
    """Cache em data/interim/: reler a POF custa 12 s e não muda entre execuções."""
    if CACHE_SEGMENTOS.exists() and not recalcular:
        return pd.read_csv(CACHE_SEGMENTOS).set_index("meal_id")
    tabela = construir_segmentos()
    CACHE_SEGMENTOS.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(CACHE_SEGMENTOS, index_label="meal_id")
    return tabela


def quintis_por_posto(valores: np.ndarray) -> pd.Series:
    """Cinco grupos de tamanho quase igual; o posto evita grupos vazios quando há empates."""
    posto = pd.Series(np.asarray(valores, dtype=np.float64)).rank(method="first")
    codigos = pd.qcut(posto, 5, labels=False, duplicates="drop")
    return codigos.map(lambda codigo: f"Q{int(codigo) + 1}")


def montar_segmentos_da_medida(medida, tabela: pd.DataFrame) -> pd.DataFrame:
    """Segmentos alinhados às linhas de `medida`, com região, renda, tamanho e quintis."""
    base = tabela.reindex(medida.meal_id)
    if base.local.isna().any():
        raise ValueError("segmentos ausentes para refeições da medida")
    regiao = pd.Series(medida.C[:, 7:12].argmax(axis=1)).map(REGIOES).to_numpy()
    renda = medida.C[:, 12]
    tamanho = np.clip(medida.P.sum(axis=1), 1, 5)
    tamanho_rotulo = np.where(tamanho < 5, tamanho.astype(str), "5+")
    return pd.DataFrame({
        "local": pd.Categorical(base.local.to_numpy(), NIVEIS["local"]),
        "fracao_fora": base.fracao_fora.to_numpy(),
        "ocasiao": pd.Categorical(base.ocasiao.to_numpy(), NIVEIS["ocasiao"]),
        "periodo": pd.Categorical(base.periodo.to_numpy(), NIVEIS["periodo"]),
        "dia_semana": pd.Categorical(base.dia_semana.to_numpy(), NIVEIS["dia_semana"]),
        "dia_atipico": pd.Categorical(base.dia_atipico.to_numpy(), NIVEIS["dia_atipico"]),
        "regiao": pd.Categorical(regiao, NIVEIS["regiao"]),
        "quintil_renda": pd.Categorical(quintis_por_posto(renda), NIVEIS["quintil_renda"]),
        "tamanho": pd.Categorical(tamanho_rotulo, NIVEIS["tamanho"]),
        "quintil_carboidrato": pd.Categorical(quintis_por_posto(medida.y), NIVEIS["quintil_carboidrato"]),
    })


def avaliar_linhas(y: np.ndarray, media: np.ndarray, var: np.ndarray) -> pd.DataFrame:
    """As métricas de `avaliar`, mas por refeição, para agregar por segmento."""
    y, media, var = (np.asarray(a, dtype=np.float64) for a in (y, media, var))
    erro = np.abs(y - media)
    meia_largura = Z_INTERVALO * np.sqrt(var)
    return pd.DataFrame({
        "y": y, "media": media, "mae_g": erro,
        "nll": 0.5 * np.log(2 * np.pi * var) + (y - media) ** 2 / (2 * var),
        "crps_g": sr.crps_normal(y, media, np.sqrt(var)),
        "picp": erro <= meia_largura,
        "mpiw_g": 2 * meia_largura,
        "pct_meia_dose": 100 * (erro <= TOLERANCIA_MEIA_DOSE_G),
    })


def avaliar_linhas_mistura(y: np.ndarray, medias: np.ndarray, variancias: np.ndarray) -> pd.DataFrame:
    """As métricas de `avaliar_mistura` por refeição; médias e variâncias (S, N)."""
    y = np.asarray(y, dtype=np.float64)
    medias = np.atleast_2d(np.asarray(medias, dtype=np.float64))
    variancias = np.atleast_2d(np.asarray(variancias, dtype=np.float64))
    S_, N = medias.shape
    media = medias.mean(axis=0)
    erro = np.abs(y - media)
    log_densidade = -0.5 * np.log(2 * np.pi * variancias) - (y - medias) ** 2 / (2 * variancias)
    nll = -(logsumexp(log_densidade, axis=0) - np.log(S_))
    crps = np.empty(N)
    for inicio in range(0, N, max(1, ORCAMENTO_BLOCO_BYTES // (8 * S_ * S_))):
        b = slice(inicio, inicio + max(1, ORCAMENTO_BLOCO_BYTES // (8 * S_ * S_)))
        crps[b] = sr.crps_mixnorm(y[b], medias[:, b], np.sqrt(variancias[:, b]), m_axis=0)
    li, ls = intervalo(medias, variancias)
    return pd.DataFrame({
        "y": y, "media": media, "mae_g": erro, "nll": nll, "crps_g": crps,
        "picp": (li <= y) & (y <= ls), "mpiw_g": ls - li,
        "pct_meia_dose": 100 * (erro <= TOLERANCIA_MEIA_DOSE_G),
    })


def agregar_por_segmento(
    linhas: pd.DataFrame, segmentos: pd.DataFrame, dimensoes=DIMENSOES,
) -> pd.DataFrame:
    """Uma linha por (segmento, valor), com n, carboidrato médio e as métricas."""
    tabelas = []
    for dimensao in dimensoes:
        agrupado = (
            linhas.assign(_valor=segmentos[dimensao].to_numpy())
            .groupby("_valor", observed=True)
            .agg(
                n=("y", "size"), carboidrato_medio_g=("y", "mean"), mae_g=("mae_g", "mean"),
                nll=("nll", "mean"), crps_g=("crps_g", "mean"), picp=("picp", "mean"),
                mpiw_g=("mpiw_g", "mean"), pct_meia_dose=("pct_meia_dose", "mean"),
            )
            .reset_index().rename(columns={"_valor": "valor"})
        )
        agrupado.insert(0, "segmento", dimensao)
        agrupado["valor"] = agrupado.valor.astype(str)
        agrupado["mae_relativo"] = agrupado.mae_g / agrupado.carboidrato_medio_g
        tabelas.append(agrupado)
    return pd.concat(tabelas, ignore_index=True)


def _cobertura_dentro(linhas: pd.DataFrame, segmentos: pd.DataFrame, modelo: str) -> pd.DataFrame:
    """Cobertura por quintil da média prevista dentro de cada segmento de local, ocasião e período."""
    tabelas = []
    for dimensao in DIMENSOES_DA_COBERTURA:
        for valor in NIVEIS[dimensao]:
            selecao = segmentos[dimensao].to_numpy() == valor
            if selecao.sum() < 5:
                continue
            media = linhas.loc[selecao, "media"].to_numpy()
            faixa = pd.qcut(media, 5, labels=False, duplicates="drop") + 1
            tabela = pd.DataFrame({
                "faixa": faixa, "media": media, "dentro": linhas.loc[selecao, "picp"].to_numpy(),
            }).groupby("faixa").agg(
                media_min_g=("media", "min"), media_max_g=("media", "max"),
                n=("dentro", "size"), picp=("dentro", "mean"),
            ).reset_index()
            tabelas.append(tabela.assign(modelo=modelo, segmento=dimensao, valor=valor))
    return pd.concat(tabelas, ignore_index=True)


def decompor_variancia(
    blr, G: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray, media_log: np.ndarray,
    cov_log: np.ndarray, bloco: int = 4_096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contribuição por gaveta (N, K), variância preditiva total (N,) e parte da entrada (N,)."""
    N, K = G.shape
    contribuicao = np.empty((N, K))
    total = np.empty(N)
    entrada = np.empty(N)
    m, S_ = blr.m_[:K], blr.S_[:K, :K]
    desvio = blr.desvio_z_[:K]
    for inicio in range(0, N, bloco):
        b = slice(inicio, inicio + bloco)
        Eg, Cg = momentos_em_gramas(G[b], P[b], M[b], media_log[b], cov_log[b])
        phi = (montar_Z(Eg, P[b], C[b]) - blr.media_z_) / blr.desvio_z_
        C_phi = Cg / (desvio[:, None] * desvio[None, :])
        base = 1.0 / blr.beta_ + np.einsum("ni,ij,nj->n", phi, blr.S_, phi)
        primeira = m[None, :] * np.einsum("nij,j->ni", C_phi, m)
        segunda = np.einsum("ij,nij->ni", S_, C_phi)
        contribuicao[b] = primeira + segunda
        total[b] = base + contribuicao[b].sum(axis=1)
        entrada[b] = contribuicao[b].sum(axis=1)
    return contribuicao, total, entrada


def _descricao_dos_segmentos(medida, segmentos: pd.DataFrame, medias: np.ndarray, gavetas: np.ndarray) -> pd.DataFrame:
    """Número médio de gavetas, carboidrato, fração com bebida e com açúcar, e resíduo na taxa 0."""
    bebidas = np.array([str(nome).startswith("Bebida") for nome in gavetas])
    acucar = np.array([str(nome) == "Açúcar" for nome in gavetas])
    com_bebida = (medida.P & bebidas[None, :]).any(axis=1)
    com_acucar = (medida.P & acucar[None, :]).any(axis=1)
    residuo = (medida.y - medias) ** 2
    tabelas = []
    for dimensao in DIMENSOES:
        tabela = pd.DataFrame({
            "valor": segmentos[dimensao].astype(str),
            "gavetas": medida.P.sum(axis=1),
            "carboidrato": medida.y,
            "bebida": com_bebida,
            "acucar": com_acucar,
            "residuo": residuo,
        }).groupby("valor").agg(
            n=("carboidrato", "size"), media_gavetas=("gavetas", "mean"),
            carboidrato_medio_g=("carboidrato", "mean"), fracao_com_bebida=("bebida", "mean"),
            fracao_com_acucar=("acucar", "mean"), variancia_residuo_g2=("residuo", "mean"),
        ).reset_index()
        tabela.insert(0, "segmento", dimensao)
        tabelas.append(tabela)
    return pd.concat(tabelas, ignore_index=True)


def _ranking(etapa_a: pd.DataFrame, n_minimo: int = 100) -> pd.DataFrame:
    """Ordena por MAE relativo e por desvio absoluto da PICP em relação a 90%.

    Segmentos com menos de `n_minimo` refeições ficam fora da ordenação: com n pequeno, o MAE
    relativo é ruído e dominaria os extremos (o local não informado tem 6 refeições na validação).
    """
    blr = etapa_a.query("modelo == 'blr' and n >= @n_minimo").copy()
    blr["desvio_picp"] = blr.picp - 0.90
    blr["posto_mae_relativo"] = blr.mae_relativo.rank(ascending=False, method="first").astype(int)
    blr["posto_desvio_picp"] = blr.desvio_picp.abs().rank(ascending=False, method="first").astype(int)
    return blr[[
        "segmento", "valor", "n", "carboidrato_medio_g", "mae_g", "mae_relativo", "picp",
        "desvio_picp", "posto_mae_relativo", "posto_desvio_picp",
    ]].sort_values(["posto_mae_relativo"]).reset_index(drop=True)


def _decomposicao_por_segmento(
    contribuicao: np.ndarray, total: np.ndarray, entrada: np.ndarray, segmentos: pd.DataFrame,
    gavetas: np.ndarray, n_topo: int = 5,
) -> pd.DataFrame:
    tabelas = []
    for dimensao in DIMENSOES:
        valores = segmentos[dimensao].astype(str).to_numpy()
        for valor in NIVEIS[dimensao]:
            selecao = valores == valor
            if selecao.sum() == 0:
                continue
            media_contrib = contribuicao[selecao].mean(axis=0)
            var_total = total[selecao].mean()
            var_entrada = entrada[selecao].mean()
            ordem = np.argsort(-media_contrib)[:n_topo]
            tabelas.append(pd.DataFrame({
                "segmento": dimensao, "valor": valor, "n": int(selecao.sum()),
                "posto": np.arange(1, len(ordem) + 1), "gaveta": gavetas[ordem],
                "contribuicao_media_g2": media_contrib[ordem],
                "fracao_variancia_preditiva": media_contrib[ordem] / var_total,
                "fracao_incerteza_entrada": media_contrib[ordem] / var_entrada,
                "variancia_preditiva_media_g2": var_total,
                "variancia_entrada_media_g2": var_entrada,
            }))
    return pd.concat(tabelas, ignore_index=True)


def _tabela_codigos(consumo: pd.DataFrame, itens: pd.DataFrame) -> pd.DataFrame:
    """Contagem de cada código, no consumo bruto e nos itens das refeições válidas.

    A diferença entre as duas contagens é a regra da camada de dados: só ocasiões 1 a 5 entram nas
    refeições, então V9017 = 6 e 9 aparecem no bruto e zeram nos itens válidos.
    """
    linhas = []
    for variavel, mapa in (
        ("V9018", LOCAIS), ("V9017", OCASIOES),
        ("DIA_ATIPICO", DIAS_ATIPICOS), ("DIA_SEMANA", DIAS_DA_SEMANA),
    ):
        bruto = consumo[variavel].value_counts()
        validos = itens[variavel].value_counts()
        for codigo, rotulo in mapa.items():
            linhas.append({
                "variavel": variavel, "codigo": codigo, "rotulo": rotulo,
                "n_itens_brutos": int(bruto.get(codigo, 0)),
                "n_itens_refeicoes_validas": int(validos.get(codigo, 0)),
            })
    return pd.DataFrame(linhas)


def _gravar(tabela: pd.DataFrame, nome: str) -> None:
    tabela.to_csv(SAIDA / f"segmentos_{nome}.csv", index=False)


def _figura_picp(etapa_a: pd.DataFrame, etapa_c: pd.DataFrame) -> None:
    """PICP por segmento na taxa 0 e nas taxas 0,5 e 1,0, braço plug-in e analítico."""
    serie = []
    blr = etapa_a.query("modelo == 'blr'")
    serie.append(blr.assign(etapa="Taxa 0 (BLR)")[["segmento", "valor", "picp", "etapa"]])
    if len(etapa_c):
        for taxa in TAXAS_ETAPA_C:
            for braco, rotulo in (("plugin", "plug-in"), ("analitica", "analítica")):
                recorte = etapa_c.query("taxa == @taxa and braco == @braco")
                serie.append(recorte.assign(etapa=f"Taxa {taxa:.1f} ({rotulo})")[["segmento", "valor", "picp", "etapa"]])
    dados = pd.concat(serie, ignore_index=True)
    dados = dados.groupby(["segmento", "valor", "etapa"], as_index=False).picp.mean()
    dados["segmento"] = pd.Categorical(
        dados.segmento.map(ROTULOS_DAS_DIMENSOES), list(ROTULOS_DAS_DIMENSOES.values())
    )
    ordem_das_etapas = (
        ["Taxa 0 (BLR)", "Taxa 0.5 (plug-in)", "Taxa 0.5 (analítica)", "Taxa 1.0 (plug-in)", "Taxa 1.0 (analítica)"]
    )
    dados["etapa"] = pd.Categorical(dados.etapa, ordem_das_etapas)
    grafico = (
        ggplot(dados, aes(x="valor", y="picp", color="etapa"))
        + geom_hline(yintercept=0.90, color="#4D4D4D", linetype="dashed", size=0.8)
        + geom_point(position=position_dodge(width=0.65), size=2.6)
        + facet_wrap("~ segmento", scales="free_x", ncol=3)
        + scale_color_manual(values=CORES[:4] + ("#4D4D4D",))
        + TEMA
        + labs(
            title="Cobertura do intervalo de 90% por segmento",
            subtitle="Tracejada: nominal de 90%. Taxa 0 com as gramas completas; taxas 0,5 e 1,0 com a FA condicional.",
            x="", y="PICP", color="Etapa",
            caption="Validação, 45.639 refeições. Média de 3 sementes de máscara nas taxas 0,5 e 1,0.",
        )
        + theme_rotacao()
    )
    pastas = SAIDA / "figuras"
    pastas.mkdir(parents=True, exist_ok=True)
    grafico.save(pastas / "segmentos_picp.png", width=18, height=13, dpi=200, verbose=False)


def theme_rotacao():
    return theme(axis_text_x=element_text(angle=45, hjust=1, size=9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--etapas", nargs="+", choices=list("ABCD"), default=list("ABCD"))
    parser.add_argument("--recalcular-segmentos", action="store_true", help="relê a POF e refaz o cache")
    escolha = parser.parse_args()

    inicio = time.perf_counter()
    tempos = []

    def marcar(etapa: str, comeco: float) -> None:
        segundos = time.perf_counter() - comeco
        tempos.append({"etapa": etapa, "segundos": segundos})
        print(f"  {etapa}: {segundos:.1f} s", flush=True)

    comeco = time.perf_counter()
    consumo, itens = _ler_itens()
    tabela = carregar_segmentos(escolha.recalcular_segmentos)
    _gravar(_tabela_codigos(consumo, itens), "codigos")
    marcar("segmentos", comeco)

    tr, medida, taxas, sementes, gavetas = treino_e_medida(VALIDACAO)
    segmentos = montar_segmentos_da_medida(medida, tabela)
    resumo = agregar_por_segmento(avaliar_linhas(medida.y, medida.y, np.ones_like(medida.y)), segmentos)
    _gravar(resumo[["segmento", "valor", "n", "carboidrato_medio_g"]], "resumo")
    fracao_fora = pd.DataFrame({
        "local": segmentos.local.astype(str), "fracao_fora": segmentos.fracao_fora,
    }).groupby("local").fracao_fora.mean().rename("fracao_fora_media").reset_index()
    print("refeições na validação:", f"{len(medida.y):,}", "| segmentos:", len(resumo), flush=True)

    # Etapa A: taxa 0, os dois regressores.
    comeco = time.perf_counter()
    Z_completo = montar_Z(medida.G, medida.P, medida.C)
    configuracao = ler_configuracao()
    regressores, medias_taxa_zero, variancias_taxa_zero = {}, {}, {}
    for nome in MODELOS_ETAPA_A:
        modelo = RegressaoLinearBayesiana() if nome == "blr" else SVGP(semente=0, **configuracao.get("svgp", {}))
        modelo.fit(montar_Z(tr.G, tr.P, tr.C), tr.y)
        regressores[nome] = modelo
        medias_taxa_zero[nome], variancias_taxa_zero[nome] = modelo.predict(Z_completo)
    marcar("ajuste dos regressores (taxa 0)", comeco)

    etapa_a = []
    for nome in MODELOS_ETAPA_A:
        comeco = time.perf_counter()
        linhas = avaliar_linhas(medida.y, medias_taxa_zero[nome], variancias_taxa_zero[nome])
        etapa_a.append(agregar_por_segmento(linhas, segmentos).assign(modelo=nome))
        marcar(f"etapa A, {nome}", comeco)
    etapa_a = pd.concat(etapa_a, ignore_index=True)
    _gravar(etapa_a, "etapa_a")

    if "B" in escolha.etapas:
        comeco = time.perf_counter()
        coberturas = []
        for nome in MODELOS_ETAPA_A:
            linhas = avaliar_linhas(medida.y, medias_taxa_zero[nome], variancias_taxa_zero[nome])
            coberturas.append(_cobertura_dentro(linhas, segmentos, nome))
        _gravar(pd.concat(coberturas, ignore_index=True), "etapa_b")
        marcar("etapa B", comeco)

    imputador = None
    if "C" in escolha.etapas or "D" in escolha.etapas:
        comeco = time.perf_counter()
        sem_mascara = np.zeros_like(tr.P)
        imputador = IMPUTADORES[IMPUTADOR_ETAPA_C]().fit(ocultar(tr.G, sem_mascara), tr.P, sem_mascara, tr.C)
        marcar(f"ajuste de {IMPUTADOR_ETAPA_C}", comeco)

    etapa_c = pd.DataFrame()
    if "C" in escolha.etapas:
        referencia = etapa_a.query("modelo == 'blr'").set_index(["segmento", "valor"]).picp

        linhas_c = []
        for s, semente in enumerate(sementes):
            for t, taxa in enumerate(taxas):
                if not any(np.isclose(taxa, alvo) for alvo in TAXAS_ETAPA_C):
                    continue
                comeco = time.perf_counter()
                M = medida.M[s, t]
                bracos = prever_bracos(
                    imputador, regressores["blr"], medida.G, medida.P, M, medida.C, S,
                    gerador(semente, t, FLUXO_VALIDACAO),
                )
                tempo_propagacao = time.perf_counter() - comeco
                comeco = time.perf_counter()
                media_ana, var_ana = prever_analitico(imputador, regressores["blr"], medida.G, medida.P, M, medida.C)
                tempo_analitica = time.perf_counter() - comeco
                tempos += [
                    {"etapa": f"etapa C, propagação S = {S}", "segundos": tempo_propagacao},
                    {"etapa": "etapa C, analítica", "segundos": tempo_analitica},
                ]
                bracos_medidos = {
                    "plugin": avaliar_linhas(medida.y, bracos[PLUGIN][0][0], bracos[PLUGIN][1][0]),
                    f"propagacao_S{S}": avaliar_linhas_mistura(medida.y, *bracos[PROPAGACAO]),
                    "analitica": avaliar_linhas(medida.y, media_ana, var_ana),
                }
                for braco, linhas in bracos_medidos.items():
                    tabela_braco = agregar_por_segmento(linhas, segmentos).assign(
                        imputador=IMPUTADOR_ETAPA_C, taxa=float(taxa), semente=int(semente), braco=braco,
                    )
                    tabela_braco["picp_taxa_zero"] = [
                        referencia.get((seg, val), np.nan) for seg, val in zip(tabela_braco.segmento, tabela_braco.valor)
                    ]
                    tabela_braco["picp_diferenca_taxa_zero"] = tabela_braco.picp - tabela_braco.picp_taxa_zero
                    linhas_c.append(tabela_braco)
                print(f"  etapa C, semente {semente}, taxa {taxa:.1f}: ok", flush=True)
        etapa_c = pd.concat(linhas_c, ignore_index=True)
        _gravar(etapa_c, "etapa_c")

    if "D" in escolha.etapas:
        assert imputador is not None
        comeco = time.perf_counter()
        ranking = _ranking(etapa_a)
        _gravar(ranking, "ranking")
        descricao = _descricao_dos_segmentos(medida, segmentos, medias_taxa_zero["blr"], gavetas)
        _gravar(descricao, "residuo")

        selecionados = []
        for criterio, coluna in (("mae_relativo", "posto_mae_relativo"), ("desvio_picp", "posto_desvio_picp")):
            ordenado = ranking.sort_values(coluna)
            selecionados.append(ordenado.head(3).assign(grupo="dificil", criterio=criterio))
            selecionados.append(ordenado.tail(3).assign(grupo="facil", criterio=criterio))
        selecionados = pd.concat(selecionados, ignore_index=True)
        descricao_para_merge = (
            descricao.rename(columns={"n": "n_descricao"}).drop(columns=["carboidrato_medio_g"])
        )
        dificuldade = selecionados.merge(descricao_para_merge, on=["segmento", "valor"], how="left")
        _gravar(dificuldade, "dificuldade")
        marcar("etapa D, ranking e descrição", comeco)

        comeco = time.perf_counter()
        t = int(np.flatnonzero(np.isclose(taxas, 1.0))[0])
        M = medida.M[0, t]
        media_log, cov_log = [], []
        for inicio_b in range(0, len(medida.y), 4_096):
            b = slice(inicio_b, inicio_b + 4_096)
            ml, cl = momentos_ocultos(
                imputador, ocultar(medida.G[b], M[b]), medida.P[b], M[b], medida.C[b],
            )
            media_log.append(ml)
            cov_log.append(cl)
        contribuicao, total, entrada = decompor_variancia(
            regressores["blr"], medida.G, medida.P, M, medida.C,
            np.concatenate(media_log), np.concatenate(cov_log),
        )
        _gravar(_decomposicao_por_segmento(contribuicao, total, entrada, segmentos, gavetas), "decomposicao")
        marcar("etapa D, decomposição na taxa 1,0", comeco)

    _gravar(pd.DataFrame(tempos), "tempos")
    _gravar(fracao_fora, "fracao_fora")

    try:
        _figura_picp(etapa_a, etapa_c)
    except Exception as erro:  # a figura não pode impedir a gravação das tabelas
        print(f"figura não gerada: {erro}", flush=True)

    pd.set_option("display.width", 220)
    print("\netapa A, BLR, primeiros segmentos:")
    print(etapa_a.query("modelo == 'blr'").head(12).round(3).to_string(index=False))
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
