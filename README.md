# amaprob-projeto

Projeto final de Aprendizagem de Máquina Probabilística (2026.2).

**Estimação probabilística do carboidrato de refeições brasileiras a partir de registros alimentares incompletos.**

No cuidado em diabetes tipo 1, o carboidrato da refeição orienta a dose de insulina, mas o registro do paciente raramente é completo. O projeto mascara quantidades de registros reais da POF 2017–2018 (Pesquisa de Orçamentos Familiares, IBGE), imputa o que falta com modelos probabilísticos e prevê o carboidrato com intervalo, verificando se a incerteza da imputação se propaga de forma calibrada.

O plano completo, as regras invioláveis e o protocolo experimental estão em [`AGENTS.md`](AGENTS.md), lido também pelos assistentes de código. Leia antes de contribuir.

## Ambiente

Gerenciado com [uv](https://docs.astral.sh/uv/) e Python 3.12.

```bash
uv sync          # cria .venv e instala as dependências travadas em uv.lock
uv run python    # executa dentro do ambiente
```

Novas dependências entram com `uv add <pacote>` e são combinadas com o time antes.

## Dados

Os microdados **não** são versionados. Cada pessoa baixa direto do IBGE para `data/raw/pof2017_2018/` e extrai cada `.zip` numa pasta de mesmo nome:

- Fonte: <https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados/>
- `Dados_20230713.zip` (~146 MB): contém `CONSUMO_ALIMENTAR.txt`
- `Documentacao_20230713.zip`: dicionário de variáveis e classificação dos grupos alimentares
- `Programas_de_Leitura_20230713.zip`: layout de referência (R e SAS)

Resultado esperado: `data/raw/pof2017_2018/Dados_20230713/CONSUMO_ALIMENTAR.txt`.

Com os microdados no lugar, prepare o dataset de refeições e as máscaras (cerca de 30 s):

```bash
uv run python -m amaprob.data.dataset
```

O comando grava `data/processed/refeicoes_k29.npz` (gramas, presença, carboidrato, identificadores e conjunto do split) e `data/processed/mascaras_k29.npz` (quantidades ocultas por taxa e semente). O mecanismo de mascaramento está descrito em [`docs/mascaramento.md`](docs/mascaramento.md).

```
data/
├── raw/         # microdados POF, como baixados do IBGE
├── interim/     # tabelas lidas
└── processed/   # dataset de refeições e máscaras (.npz)
```

## Experimentos

```bash
uv run python -m amaprob.experiments.taxa_zero   # teste de sanidade na taxa 0 (segundos)
```

Ajusta a regressão linear bayesiana no treino, mede na validação e grava em `results/` as métricas (`taxa_zero_metricas.csv`), a cobertura por faixa da previsão (`taxa_zero_cobertura.csv`) e os tempos de ajuste (`tempos_ajuste.csv`). O conjunto de teste não é lido.

```bash
uv run python -m amaprob.experiments.propagacao  # plug-in × propagação por taxa e imputador (cerca de 1 h)
```

Compara plug-in, plug-in recalibrado e propagação da incerteza da imputação nas taxas 0 a 1 e nas 3 sementes de máscara. Grava em `results/`:

| Arquivo | Conteúdo |
|---|---|
| `propagacao_resumo.csv` | média e desvio entre as sementes, por imputador, taxa e braço |
| `propagacao_metricas.csv` | uma linha por imputador, taxa, semente e braço |
| `propagacao_por_ocultas.csv` | métricas por número de gavetas ocultas na refeição, com a referência da taxa 0 |
| `propagacao_S.csv` | comparação de S = 100 e S = 500 amostras, com os tempos (registro; não é regravado) |
| `imputacao_metricas.csv`, `imputacao_por_gaveta.csv` | Parte 1: erro e calibração da imputação nas células ocultas, por imputador, taxa e semente |
| `imputadores_ajuste.csv` | tempo e iterações do ajuste de cada imputador |
| `figuras/propagacao_picp.png`, `figuras/propagacao_mpiw.png` | cobertura e largura do intervalo de 90% pela taxa, um painel por imputador |

A comparação de S ocupa cerca de 14 min; cada rodada da curva, cerca de 80 s.

## Testes

```bash
uv run pytest
```

Os testes de máscaras, split, gavetas e persistência usam dados sintéticos ou a classificação versionada e rodam sem os microdados. Os testes sobre os arquivos de `data/processed/` são pulados quando eles não existem.

## Estrutura

A estrutura de pastas segue a seção 8 do `AGENTS.md`. Os subpacotes de `src/amaprob/` são criados à medida que o código de cada etapa é escrito.
