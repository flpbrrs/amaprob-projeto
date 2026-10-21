"""Piso de erro na taxa 0.0: quanto as gramas por gaveta explicam do carboidrato, sem nada escondido.

Método (detalhado em docs/piso-de-erro.md):
1. Ajusta no treino a regra linear  carbo ≈ b0 + Σ_k b_k · gramas_k  (mínimos quadrados).
2. Mede o erro absoluto médio (MAE) na validação. O teste não é usado.
3. Ranking: divide um grupo do IBGE por vez em seus subgrupos e mede o ganho.
4. Curva: acumula as divisões na ordem do ranking, de 21 gavetas até todos os subgrupos.

Uso: uv run python -m amaprob.experiments.piso_de_erro
"""

from pathlib import Path

import numpy as np
import pandas as pd

from amaprob.data.gavetas import montar_gavetas
from amaprob.data.pof import read_consumo_alimentar
from amaprob.data.refeicoes import gramas_por_gaveta, montar_itens_de_refeicao
from amaprob.data.splits import TREINO, VALIDACAO, dividir_por_pessoa

RAIZ = Path(__file__).resolve().parents[3]
SEED_SPLIT = 0
# Meia unidade de insulina com relação 1 U : 15 g, erro tratado como aceitável em estudo da
# Roche (fonte a verificar). A relação real varia de paciente para paciente.
TOLERANCIA_MEIA_DOSE_G = 7.5

# Proposta enxuta (~30 gavetas): divide os dois grupos de maior ganho no ranking, mas junta as
# bebidas por perfil nutricional em vez de usar os 14 subgrupos do IBGE.
PROPOSTA_DIVIDIR = {"Bebidas", "Farinhas e massas"}
PROPOSTA_FUNDIR = {
    **dict.fromkeys(
        ["Água", "Café", "Chá", "Refrigerantes diet/light", "Refrescos/sucos industrializados diet/light", "Destiladas"],
        "Bebidas / sem açúcar",
    ),
    **dict.fromkeys(
        ["Sucos", "Refrescos/sucos industrializados", "Refrigerantes", "Outras bebidas não-alcoólicas", "Bebidas à base de soja"],
        "Bebidas / açucaradas",
    ),
    "Bebidas lácteas": "Bebidas / lácteas",
    **dict.fromkeys(["Cerveja", "Vinho"], "Bebidas / cerveja e vinho"),
}


def avaliar_regra_linear(itens: pd.DataFrame, conjunto: pd.Series) -> dict:
    """Ajusta a regra linear no treino e mede na validação."""
    X = gramas_por_gaveta(itens)
    y = itens.groupby("meal_id").CHOTOT.sum().loc[X.index].to_numpy()
    conj = conjunto.loc[X.index].to_numpy()
    A = np.c_[X.to_numpy(), np.ones(len(X))]
    treino, validacao = conj == TREINO, conj == VALIDACAO
    beta, *_ = np.linalg.lstsq(A[treino], y[treino], rcond=None)
    residuo = y[validacao] - A[validacao] @ beta
    return {
        "K": X.shape[1],
        "mae_g": np.abs(residuo).mean(),
        "r2": 1 - residuo.var() / y[validacao].var(),
        "pct_ate_meia_dose": 100 * (np.abs(residuo) <= TOLERANCIA_MEIA_DOSE_G).mean(),
    }


def main() -> None:
    consumo = read_consumo_alimentar(RAIZ / "data/raw/pof2017_2018/Dados_20230713/CONSUMO_ALIMENTAR.txt")
    classificacao = pd.read_csv(RAIZ / "configs/classificacao_ibge.csv")
    itens = montar_itens_de_refeicao(consumo, montar_gavetas(classificacao, set()))
    conjunto = itens.groupby("meal_id").person_id.first().pipe(dividir_por_pessoa, seed=SEED_SPLIT)

    def com_gavetas(dividir, fundir=None) -> dict:
        gavetas = montar_gavetas(classificacao, dividir, fundir)
        return avaliar_regra_linear(itens.assign(gaveta=itens.V9001.map(gavetas)), conjunto)

    base = com_gavetas(set())
    y_val = itens.groupby("meal_id").CHOTOT.sum()[conjunto == VALIDACAO]
    y_tr = itens.groupby("meal_id").CHOTOT.sum()[conjunto == TREINO]
    print(f"refeições: {itens.meal_id.nunique():,}  |  MAE de prever sempre a média: {np.abs(y_val - y_tr.mean()).mean():.1f} g")

    ranking = []
    for grupo in sorted(classificacao.grupo.unique()):
        r = com_gavetas({grupo})
        ranking.append({"grupo": grupo, "K": r["K"], "ganho_mae_g": base["mae_g"] - r["mae_g"]})
    ranking = pd.DataFrame(ranking).sort_values("ganho_mae_g", ascending=False, ignore_index=True)

    curva = [{"grupos_divididos": "", **base}]
    for i in range(1, len(ranking) + 1):
        divididos = ranking.grupo[:i].tolist()
        curva.append({"grupos_divididos": " | ".join(divididos), **com_gavetas(set(divididos))})
    curva.append({"grupos_divididos": "PROPOSTA: Bebidas (4 perfis) | Farinhas e massas", **com_gavetas(PROPOSTA_DIVIDIR, PROPOSTA_FUNDIR)})
    curva = pd.DataFrame(curva)

    saida = RAIZ / "results"
    saida.mkdir(exist_ok=True)
    ranking.to_csv(saida / "piso_de_erro_ranking.csv", index=False)
    curva.to_csv(saida / "piso_de_erro_curva.csv", index=False)
    pd.set_option("display.width", 200)
    print(ranking.round(2).to_string())
    print(curva.assign(ultimo=curva.grupos_divididos.str.split(" \\| ").str[-1]).drop(columns="grupos_divididos").round(3).to_string())


if __name__ == "__main__":
    main()
