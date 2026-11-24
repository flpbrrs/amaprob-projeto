"""Figuras do artigo a partir dos CSVs já gravados em results/ (validação); não roda experimento.

Sem título dentro da figura: o título e a leitura das linhas de referência vão na legenda do artigo.
Cores dos braços como em `experiments/propagacao.py` (plug-in e propagação), com a analítica na quarta cor.

Saídas em results/figuras/: `artigo_cobertura_bracos.png`, `artigo_perguntas.png`, `artigo_historico.png` e
`artigo_desvio_rmse.png`.
"""

import argparse

import pandas as pd
from plotnine import (
    aes, coord_fixed, facet_wrap, geom_abline, geom_errorbar, geom_hline, geom_line, geom_point, ggplot, labs,
    scale_color_manual, scale_x_continuous, scale_x_log10, scale_y_continuous, scale_y_log10, theme,
)

from amaprob.data.dataset import RAIZ
from amaprob.evaluation.tema import CORES, TEMA

RESULTADOS = RAIZ / "results"
FIGURAS = RESULTADOS / "figuras"
DPI = 200

BRACOS = {"plugin": "Plug-in", "propagacao_S100": "Propagação (S = 100)", "analitica": "Analítica"}
CORES_DOS_BRACOS = [CORES[0], CORES[2], CORES[3]]
ESTRATEGIAS = {"otima": "Ótima", "maior_carboidrato": "Maior carboidrato", "maior_porcao": "Maior porção",
               "aleatoria": "Aleatória"}
VERSOES_HISTORICO = {"fa_sem_pessoa": "Sem histórico", "fa_por_gaveta_v4": "Com histórico do paciente"}
COBERTURA = "Cobertura do intervalo de 90%"
MARCAS_PICP = [0.6, 0.7, 0.8, 0.9]
EIXO_TAXA = scale_x_continuous(breaks=[0, 0.25, 0.5, 0.75, 1.0], labels=["0", "0,25", "0,5", "0,75", "1,0"])


def _virgula(valores) -> list[str]:
    return [f"{v:g}".replace(".", ",") for v in valores]


def media_e_desvio(tabela: pd.DataFrame, chaves: list[str], metrica: str) -> pd.DataFrame:
    """Média ± desvio entre as sementes, com as colunas `centro`, `inferior` e `superior`."""
    resumo = tabela.groupby(chaves, observed=True)[metrica].agg(["mean", "std"]).reset_index()
    desvio = resumo["std"].fillna(0.0)
    return resumo.assign(centro=resumo["mean"], inferior=resumo["mean"] - desvio, superior=resumo["mean"] + desvio)


def _categoria(serie: pd.Series, nomes: dict[str, str]) -> pd.Categorical:
    return pd.Categorical(serie.map(nomes), list(nomes.values()))


def _linhas_com_barras(dados: pd.DataFrame, x: str, cor: str, largura: float):
    return (ggplot(dados, aes(x=x, y="centro", color=cor))
            + geom_errorbar(aes(ymin="inferior", ymax="superior"), width=largura, size=0.8)
            + geom_line(size=1.2) + geom_point(size=3.5))


def cobertura_bracos() -> list[str]:
    """PICP90 × taxa por braço, FA condicional + regressão linear bayesiana.

    Na taxa 0 não há quantidade oculta e os três braços coincidem; o ponto vem da curva oficial.
    """
    fontes = ["propagacao_analitica_metricas.csv", "propagacao_metricas.csv"]
    analitica = pd.read_csv(RESULTADOS / fontes[0]).query("imputador == 'fa_condicional' and braco in @BRACOS")
    oficial = pd.read_csv(RESULTADOS / fontes[1]).query(
        "regressor == 'blr' and imputador == 'fa_condicional' and taxa == 0 and braco == 'plugin'")
    taxa_zero = pd.concat([oficial.assign(braco=braco) for braco in BRACOS])
    picp_taxa_zero = oficial.picp.mean()
    metricas = pd.concat([taxa_zero, analitica], ignore_index=True)
    dados = media_e_desvio(metricas, ["taxa", "braco"], "picp")
    dados["braco"] = _categoria(dados.braco, BRACOS)
    (_linhas_com_barras(dados, "taxa", "braco", 0.03)
     + geom_hline(yintercept=0.9, color="#4D4D4D", linetype="dashed", size=0.8)
     + geom_hline(yintercept=picp_taxa_zero, color="#4D4D4D", linetype="dotted", size=0.8)
     + EIXO_TAXA + scale_y_continuous(breaks=MARCAS_PICP, labels=_virgula(MARCAS_PICP))
     + scale_color_manual(values=CORES_DOS_BRACOS) + TEMA
     + labs(x="Taxa de quantidades ocultas", y=COBERTURA, color="Braço")
     ).save(FIGURAS / "artigo_cobertura_bracos.png", width=7, height=5.5, dpi=DPI, verbose=False)
    return fontes


def perguntas() -> list[str]:
    """% de refeições com erro até meia dose (7,5 g) × número de perguntas, por estratégia e taxa."""
    fonte = "pergunta_metricas.csv"
    metricas = pd.read_csv(RESULTADOS / fonte)
    dados = media_e_desvio(metricas, ["taxa", "estrategia", "n_perguntas"], "pct_meia_dose")
    dados["estrategia"] = _categoria(dados.estrategia, ESTRATEGIAS)
    dados["painel"] = pd.Categorical([f"taxa {t}".replace(".", ",") for t in dados.taxa],
                                     [f"taxa {t}".replace(".", ",") for t in sorted(dados.taxa.unique())])
    (_linhas_com_barras(dados, "n_perguntas", "estrategia", 0.12)
     + facet_wrap("~ painel") + scale_color_manual(values=list(CORES))
     + TEMA
     + labs(x="Número de perguntas (gavetas reveladas)", y="Refeições com erro até 7,5 g (%)",
            color="Estratégia")
     ).save(FIGURAS / "artigo_perguntas.png", width=10, height=5.5, dpi=DPI, verbose=False)
    return [fonte]


def historico() -> list[str]:
    """MAE, largura e cobertura × taxa, braço propagação, cenário H1 (histórico do dia 1).

    Sem histórico: FA condicional (variante sem pessoa da rodada v2); com histórico: hierárquico v4 por gaveta.
    """
    fontes = ["hierarquico_v2_regressao_metricas.csv", "hierarquico_v4_regressao_metricas.csv"]
    metricas = pd.concat([pd.read_csv(RESULTADOS / f) for f in fontes], ignore_index=True).query(
        "cenario == 'H1' and braco == 'propagacao' and imputador in @VERSOES_HISTORICO")
    paineis = {"mae_g": "MAE (g)", "mpiw_g": "Largura do intervalo de 90% (g)", "picp": COBERTURA}
    dados = pd.concat([media_e_desvio(metricas, ["taxa", "imputador"], m).assign(painel=nome)
                       for m, nome in paineis.items()], ignore_index=True)
    dados["imputador"] = _categoria(dados.imputador, VERSOES_HISTORICO)
    dados["painel"] = pd.Categorical(dados.painel, list(paineis.values()))
    (_linhas_com_barras(dados, "taxa", "imputador", 0.03)
     + geom_hline(pd.DataFrame({"painel": pd.Categorical([COBERTURA], list(paineis.values())), "valor": [0.9]}),
                  aes(yintercept="valor"), color="#4D4D4D", linetype="dashed", size=0.8)
     + facet_wrap("~ painel", scales="free_y")
     + scale_x_continuous(breaks=[0.25, 0.5, 0.75, 1.0], labels=["0,25", "0,5", "0,75", "1,0"])
     + scale_y_continuous(labels=_virgula)
     + scale_color_manual(values=[CORES[0], CORES[1]]) + TEMA + theme(panel_spacing_x=0.03)
     + labs(x="Taxa de quantidades ocultas", y="", color="Imputação")
     ).save(FIGURAS / "artigo_historico.png", width=13, height=5.5, dpi=DPI, verbose=False)
    return fontes


def desvio_rmse() -> list[str]:
    """Desvio previsto × RMSE por decil do desvio, eixos log de mesma escala e diagonal y = x.

    O plug-in tem desvio quase constante (só o ruído da regressão) e fica numa coluna vertical; ele é desenhado
    por último para não sumir sob as outras curvas. A taxa 0 fica de fora: nela os três braços coincidem.
    """
    fonte = "qualidade_decis.csv"
    decis = pd.read_csv(RESULTADOS / fonte).query("taxa > 0")
    dados = decis.groupby(["taxa", "braco", "decil"], as_index=False)[["desvio_rms", "rmse"]].mean()
    dados["braco"] = _categoria(dados.braco, BRACOS)
    dados["painel"] = pd.Categorical([f"taxa {t:.1f}".replace(".", ",") for t in dados.taxa])
    limites = (5.0, 100.0)
    marcas = [5, 10, 20, 50, 100]
    camadas = []
    for braco in ["Propagação (S = 100)", "Analítica", "Plug-in"]:
        parte = dados[dados.braco == braco]
        camadas += [geom_line(parte, size=1.1), geom_point(parte, size=2.6)]
    grafico = (ggplot(dados, aes("desvio_rms", "rmse", color="braco"))
               + geom_abline(slope=1, intercept=0, color="#4D4D4D", linetype="dashed", size=0.8))
    for camada in camadas:
        grafico += camada
    (grafico
     + facet_wrap("~ painel")
     + scale_x_log10(limits=limites, breaks=marcas, labels=_virgula(marcas))
     + scale_y_log10(limits=limites, breaks=marcas, labels=_virgula(marcas))
     + coord_fixed() + scale_color_manual(values=CORES_DOS_BRACOS, drop=False) + TEMA
     + labs(x="Desvio previsto (g, raiz da variância média do decil)", y="RMSE realizado (g)", color="Braço")
     ).save(FIGURAS / "artigo_desvio_rmse.png", width=8.2, height=5, dpi=DPI, verbose=False)
    return [fonte]


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    usados = []
    for gerar in (cobertura_bracos, perguntas, historico, desvio_rmse):
        usados += gerar()
    print("CSVs usados:", ", ".join(dict.fromkeys(usados)))


if __name__ == "__main__":
    main()
