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
uv run python -m amaprob.experiments.propagacao                   # plug-in × propagação por taxa, regressor e imputador
uv run python -m amaprob.experiments.propagacao --imputadores fa  # só a FA, juntando aos CSVs existentes
uv run python -m amaprob.experiments.propagacao --regressores svgp  # só o SVGP, com todos os imputadores
```

Compara plug-in, plug-in recalibrado e propagação da incerteza da imputação nas taxas 0 a 1 e nas 3 sementes de máscara, para cada par de regressor (regressão linear bayesiana e SVGP) e imputador (porção média, PPCA e FA, cada um com e sem média condicional), com os hiperparâmetros de `configs/hiperparametros.json`. O plug-in recalibrado existe só com a regressão linear. Com `--regressores` e `--imputadores`, as linhas dos outros pares e as referências da taxa 0 já gravadas ficam como estão; as métricas da Parte 1 só são calculadas quando a regressão linear roda, porque não dependem do regressor. Por padrão mede na validação e grava em `results/`:

| Arquivo | Conteúdo |
|---|---|
| `propagacao_resumo.csv` | média e desvio entre as sementes, por regressor, imputador, taxa e braço |
| `propagacao_metricas.csv` | uma linha por regressor, imputador, taxa, semente e braço |
| `propagacao_por_ocultas.csv` | métricas por número de gavetas ocultas na refeição, com a referência da taxa 0 de cada regressor |
| `regressores_ajuste.csv` | tempo, épocas e ELBO final do ajuste de cada regressor no treino completo |
| `propagacao_S.csv` | comparação de S = 100 e S = 500 amostras, com os tempos (registro; não é regravado) |
| `imputacao_metricas.csv`, `imputacao_por_gaveta.csv` | Parte 1: erro e calibração da imputação nas células ocultas, por imputador, taxa e semente |
| `imputadores_ajuste.csv` | tempo, iterações (com o teto do EM) e gavetas no piso de ruído (FA) do ajuste de cada imputador |
| `figuras/propagacao_picp.png`, `figuras/propagacao_mpiw.png` | cobertura e largura do intervalo de 90% pela taxa, um painel por regressor e imputador |

**Rodada no teste.** Com `--conjunto teste`, o ajuste continua só no treino e a medida passa para o teste; os mesmos arquivos são gravados em `results/teste/` (e `results/teste/figuras/`), e nada fora dessa pasta muda. A rodada é única (Regra 6) e roda em três chamadas, juntadas na pasta do teste, para que uma falha custe só uma parte:

```bash
uv run python -m amaprob.experiments.propagacao --conjunto teste --imputadores porcao_media porcao_condicional fa fa_condicional
uv run python -m amaprob.experiments.propagacao --conjunto teste --imputadores ppca
uv run python -m amaprob.experiments.propagacao --conjunto teste --imputadores ppca_condicional
```

Cada rodada da curva leva cerca de 80 s por imputador e por regressor. Os ajustes no treino completo e nas 5 dobras levam cerca de 16 min com o PPCA, 5 min com a FA e 9 min com o PPCA e com a FA condicionais; o do SVGP, cerca de 4 min. Tempos de parede: cerca de 1 h 30 com a regressão linear e os três imputadores sem média condicional, 27 min com `--imputadores fa`, 30 min com `--regressores blr --imputadores ppca_condicional` e 68 min com o SVGP e três imputadores; a execução completa, com os dois regressores e os seis imputadores, deve levar cerca de 4 h 45 (estimativa).

```bash
uv run python -m amaprob.experiments.otimizacao                        # Parte 3: os quatro modelos
uv run python -m amaprob.experiments.otimizacao --modelos svgp         # só o SVGP, juntando aos arquivos existentes
```

Otimização bayesiana (BoTorch) e busca aleatória, com 12 avaliações cada, dos botões de PPCA e FA com média condicional (q e α), da porção média condicional (α) e do SVGP (M), só na validação. A escolha é confirmada no treino completo contra a configuração de partida. Grava em `results/` e `configs/`:

| Arquivo | Conteúdo |
|---|---|
| `otimizacao_avaliacoes.csv` | uma linha por modelo, método e avaliação: hiperparâmetros, métricas, tempo, reaproveitada e melhor NLL até ali |
| `otimizacao_confirmacao.csv` | configuração de partida e escolhida no treino completo, por modelo e semente |
| `figuras/otimizacao_convergencia.png` | melhor NLL pelo número de avaliações, um painel por modelo |
| `configs/hiperparametros.json` | configuração escolhida de cada modelo, com o q herdado pelo PPCA e pela FA sem média condicional |

Os imputadores são avaliados num ajuste em 25% do treino mais a propagação na validação (de 1 a 6 min por avaliação, conforme o q); o SVGP, no treino completo (de 1,4 min com M = 32 a 12 min com M = 512). Tempos de parede: 3 h com os três imputadores e 1 h 40 com `--modelos svgp`; a execução completa leva cerca de 4 h 40. `--so-confirmacao` refaz só a variante de α convertido da confirmação (cerca de 25 min).

## Testes

```bash
uv run pytest
```

Os testes de máscaras, split, gavetas e persistência usam dados sintéticos ou a classificação versionada e rodam sem os microdados. Os testes sobre os arquivos de `data/processed/` são pulados quando eles não existem.

## Estrutura

A estrutura de pastas segue a seção 8 do `AGENTS.md`. Os subpacotes de `src/amaprob/` são criados à medida que o código de cada etapa é escrito.
