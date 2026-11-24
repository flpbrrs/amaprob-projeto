"""Diagnóstico dia 1 → dia 2: contagens por conjunto e efeito da pessoa nos resíduos em log.

O dataset não guarda o QUADRO (72 = dia 1, 73 = dia 2): ele é recuperado relendo a POF com as
funções existentes (a camada de dados não muda) e associado a cada meal_id, com conferência do
alinhamento (mesmos meal_id e mesma pessoa do dataset). Análise de resíduo só no treino; contagens
no treino e na validação, porque a fase seguinte usaria o dia 1 da validação como histórico. O
teste não é lido.

1. Contagens por conjunto: pessoas com 1 e com 2 dias, refeições por pessoa e por dia.
2. Resíduo em log no treino: FA com média condicional (configs/hiperparametros.json) ajustada no
   treino completo; resíduo = log1p(gramas) − média condicional dada só P e C, nas células
   presentes. Só P e C entram na média: a comparação é com o que o modelo diria para qualquer
   pessoa no mesmo contexto, e o que sobra é a porção da pessoa.
3. Efeito da pessoa, entre as pessoas do treino com 2 dias: correlação entre a média do resíduo no
   dia 1 e no dia 2, e correlação intraclasse (ICC) com intervalo de 95% por bootstrap de pessoas,
   em três escopos: fator de porção único (média de todas as células presentes da pessoa no dia),
   por gaveta (as 10 mais frequentes) e por ocasião.

A ICC é a de via única, ICC(1), com as duas medidas (dia 1 e dia 2) por pessoa:

    ICC = (MS_entre − MS_dentro) / (MS_entre + MS_dentro)

negativa quando a variância entre pessoas é menor que a interna. O bootstrap reamostra pessoas
com substituição e usa os percentis 2,5 e 97,5.

Saídas: results/dois_dias_contagens.csv, results/dois_dias_icc.csv e results/dois_dias_tempos.csv.

Uso: uv run python -m amaprob.experiments.diagnostico_dois_dias
"""

import time

import numpy as np
import pandas as pd

from amaprob.data.dataset import RAIZ, carregar_refeicoes
from amaprob.data.gavetas import DIVIDIR_K29, FUNDIR_K29, montar_gavetas
from amaprob.data.pof import read_consumo_alimentar
from amaprob.data.refeicoes import montar_itens_de_refeicao
from amaprob.data.splits import TREINO
from amaprob.experiments.propagacao import ler_configuracao
from amaprob.imputation.base import ocultar
from amaprob.imputation.fa import FA
from amaprob.imputation.media_condicional import montar_u

SAIDA = RAIZ / "results"
ARQUIVO_CONSUMO = RAIZ / "data/raw/pof2017_2018/Dados_20230713/CONSUMO_ALIMENTAR.txt"
CLASSIFICACAO = RAIZ / "configs/classificacao_ibge.csv"
QUADROS = {72: 1, 73: 2}  # primeiro e segundo dia recordado (a verificar)
OCASIOES = {1: "cafe", 2: "almoco", 3: "lanche", 4: "jantar", 5: "ceia"}  # V9017
N_GAVETAS = 10
N_BOOTSTRAP = 2000
SEMENTE_BOOTSTRAP = 0
MINIMO_BOOTSTRAP = 10  # abaixo disso o intervalo por bootstrap é ruído


def carregar_dias(refeicoes) -> pd.DataFrame:
    """Uma linha por refeição do dataset: pessoa, dia (QUADRO) e ocasião, com alinhamento conferido."""
    consumo = read_consumo_alimentar(ARQUIVO_CONSUMO)
    gavetas = montar_gavetas(pd.read_csv(CLASSIFICACAO), DIVIDIR_K29, FUNDIR_K29)
    itens = montar_itens_de_refeicao(consumo, gavetas)
    primeiro = itens.groupby("meal_id")[["person_id", "QUADRO", "V9017"]].first().reindex(refeicoes.meal_id)
    if primeiro.QUADRO.isna().any():
        raise ValueError("QUADRO ausente para refeições do dataset: os meal_id não se alinham.")
    if not np.array_equal(primeiro.person_id.to_numpy(), refeicoes.person_id):
        raise ValueError("A pessoa do item difere da do dataset para o mesmo meal_id.")
    quadros = set(primeiro.QUADRO.unique())
    if not quadros <= set(QUADROS):
        raise ValueError(f"QUADRO fora de {sorted(QUADROS)}: {sorted(quadros)}.")
    return pd.DataFrame({
        "person_id": refeicoes.person_id,
        "dia": pd.Series(primeiro.QUADRO.to_numpy(dtype=int)).map(QUADROS).to_numpy(dtype=int),
        "ocasiao": primeiro.V9017.to_numpy(dtype=int),
        "conjunto": refeicoes.conjunto,
    })


def contar_pessoas_e_refeicoes(tabela: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por conjunto: pessoas (1 e 2 dias), refeições e refeições por pessoa e por dia."""
    linhas = []
    for conjunto, sub in tabela.groupby("conjunto", sort=False):
        dias_por_pessoa = sub.groupby("person_id").dia.nunique()
        linhas.append({
            "conjunto": conjunto,
            "n_pessoas": len(dias_por_pessoa),
            "pessoas_1_dia": int((dias_por_pessoa == 1).sum()),
            "pessoas_2_dias": int((dias_por_pessoa == 2).sum()),
            "n_refeicoes": len(sub),
            "refeicoes_dia_1": int((sub.dia == 1).sum()),
            "refeicoes_dia_2": int((sub.dia == 2).sum()),
            "refeicoes_por_pessoa": sub.groupby("person_id").size().mean(),
            "refeicoes_por_pessoa_dia": sub.groupby(["person_id", "dia"]).size().mean(),
        })
    return pd.DataFrame(linhas)


def residuos_do_treino(refeicoes, indices: np.ndarray, configuracao: dict) -> tuple[np.ndarray, int]:
    """Resíduo em log de cada célula presente do treino, contra a média condicional dada só P e C."""
    G, P, C = refeicoes.G[indices], refeicoes.P[indices], refeicoes.C[indices]
    sem_mascara = np.zeros_like(P)
    fa = FA(semente=0, media_condicional=True, **configuracao.get("fa_condicional", {})).fit(
        ocultar(G, sem_mascara), P, sem_mascara, C
    )
    media = montar_u(P, C, fa.padronizacao_) @ fa.coeficientes_media_.T
    residuo = np.log1p(G) - media
    residuo[~P] = np.nan
    return residuo, fa.n_iteracoes_


def matriz_dias(sub: pd.DataFrame) -> np.ndarray:
    """Média do resíduo por pessoa e dia, (n, 2); só quem tem células nos dois dias."""
    medias = sub.groupby(["person_id", "dia"]).residuo.mean().unstack("dia").reindex(columns=[1, 2])
    return medias.dropna().to_numpy(dtype=np.float64)


def icc_dois_dias(valores: np.ndarray) -> float:
    """ICC(1) de via única com duas medidas por pessoa; NaN com menos de duas pessoas."""
    x = np.asarray(valores, dtype=np.float64)
    n, k = x.shape
    if n < 2:
        return np.nan
    medias_pessoa = x.mean(axis=1)
    ms_entre = k * ((medias_pessoa - x.mean()) ** 2).sum() / (n - 1)
    ms_dentro = ((x - medias_pessoa[:, None]) ** 2).sum() / (n * (k - 1))
    denominador = ms_entre + ms_dentro
    return float((ms_entre - ms_dentro) / denominador) if denominador > 0 else np.nan


def correlacao_dias(valores: np.ndarray) -> float:
    """Correlação de Pearson entre a média do dia 1 e a do dia 2; NaN sem variância."""
    x = np.asarray(valores, dtype=np.float64)
    if len(x) < 2 or x[:, 0].std() == 0 or x[:, 1].std() == 0:
        return np.nan
    return float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])


def bootstrap_icc(valores: np.ndarray, n_boot: int = N_BOOTSTRAP, semente: int = SEMENTE_BOOTSTRAP):
    """Percentis 2,5 e 97,5 da ICC reamostrando pessoas; NaN com poucas pessoas."""
    x = np.asarray(valores, dtype=np.float64)
    if len(x) < MINIMO_BOOTSTRAP:
        return np.nan, np.nan
    rng = np.random.default_rng(semente)
    iccs = np.empty(n_boot)
    for b in range(n_boot):
        iccs[b] = icc_dois_dias(x[rng.integers(0, len(x), len(x))])
    return float(np.percentile(iccs, 2.5)), float(np.percentile(iccs, 97.5))


def resumo_pares(escopo: str, chave: str, sub: pd.DataFrame) -> dict:
    """n, correlação dia 1 × dia 2 e ICC com intervalo, para um recorte de células."""
    valores = matriz_dias(sub)
    icc = icc_dois_dias(valores)
    li, ls = bootstrap_icc(valores)
    return {
        "escopo": escopo, "chave": chave, "n_pessoas": len(valores),
        "correlacao_dia1_dia2": correlacao_dias(valores), "icc": icc, "icc_li": li, "icc_ls": ls,
    }


def analisar_efeito_da_pessoa(
    residuo: np.ndarray, pessoas: np.ndarray, dia: np.ndarray, ocasiao: np.ndarray,
    gavetas: np.ndarray, n_gavetas: int = N_GAVETAS,
) -> pd.DataFrame:
    """Tabela da ICC nos três escopos, sobre as células presentes do treino."""
    linhas, colunas = np.nonzero(~np.isnan(residuo))
    long = pd.DataFrame({
        "person_id": pessoas[linhas], "dia": dia[linhas], "ocasiao": ocasiao[linhas],
        "gaveta": gavetas[colunas], "residuo": residuo[linhas, colunas],
    })
    mais_frequentes = pd.Series(np.sum(~np.isnan(residuo), axis=0), index=gavetas).nlargest(n_gavetas).index
    resumos = [resumo_pares("fator_unico", "total", long)]
    resumos += [resumo_pares("gaveta", nome, long[long.gaveta == nome]) for nome in mais_frequentes]
    resumos += [
        resumo_pares("ocasiao", OCASIOES[codigo], long[long.ocasiao == codigo])
        for codigo in sorted(OCASIOES) if (long.ocasiao == codigo).any()
    ]
    return pd.DataFrame(resumos)


def main() -> None:
    inicio = time.perf_counter()
    tempos = []

    def marcar(etapa: str, comeco: float) -> None:
        tempos.append({"etapa": etapa, "segundos": time.perf_counter() - comeco})
        print(f"  {etapa}: {time.perf_counter() - comeco:.1f} s", flush=True)

    comeco = time.perf_counter()
    refeicoes = carregar_refeicoes()
    dias = carregar_dias(refeicoes)
    marcar("leitura da POF e QUADRO", comeco)

    contagem = contar_pessoas_e_refeicoes(dias[dias.conjunto != "teste"])
    contagem.to_csv(SAIDA / "dois_dias_contagens.csv", index=False)

    comeco = time.perf_counter()
    indices = np.flatnonzero(refeicoes.conjunto == TREINO)
    residuo, iteracoes = residuos_do_treino(refeicoes, indices, ler_configuracao())
    marcar(f"ajuste da FA condicional ({iteracoes} iterações)", comeco)

    comeco = time.perf_counter()
    sub = dias.iloc[indices]
    tabela_icc = analisar_efeito_da_pessoa(
        residuo, sub.person_id.to_numpy(), sub.dia.to_numpy(), sub.ocasiao.to_numpy(), refeicoes.gavetas,
    )
    tabela_icc.to_csv(SAIDA / "dois_dias_icc.csv", index=False)
    marcar("ICC e bootstrap", comeco)

    pd.DataFrame(tempos).to_csv(SAIDA / "dois_dias_tempos.csv", index=False)

    pd.set_option("display.width", 220)
    print("\ncontagens (treino e validação):")
    print(contagem.round(2).to_string(index=False))
    print("\nICC por escopo (treino, pessoas com 2 dias):")
    print(tabela_icc.round(3).to_string(index=False))

    fator = tabela_icc.query("escopo == 'fator_unico'").iloc[0]
    justifica = fator.icc > 0.1 and fator.icc_li > 0.05
    print(
        f"\nICC do fator de porção único = {fator.icc:.3f} (IC 95% {fator.icc_li:.3f} a "
        f"{fator.icc_ls:.3f}): o efeito da pessoa "
        + ("supera o critério (ICC > 0,1 e IC acima de 0,05)" if justifica else "não supera o critério (ICC > 0,1 e IC acima de 0,05)")
        + (", e um modelo hierárquico se justifica." if justifica else ", e um modelo hierárquico não se justifica.")
    )
    print(f"\ntempo total: {(time.perf_counter() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
