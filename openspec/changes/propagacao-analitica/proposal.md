# Proposal

## Why

Os braços de hoje levam a incerteza da imputação por amostragem: S sorteios de `Imputer.sample` passam pelo regressor e viram a mistura `(1/S) Σ N(μ_s, σ²_s)`. O resultado depende de S (100 no experimento oficial, com uma comparação em S = 500), e a mistura não tem quantil em forma fechada.

A regressão linear bayesiana admite um terceiro braço exato. Tratando a entrada como variável aleatória, com média e covariância, a lei da variância total dá a preditiva em forma fechada, numa passada e sem gerador aleatório. Se a forma fechada concordar com a propagação, ela serve de referência para separar erro de Monte Carlo de erro de modelo, e o custo cai de dezenas de segundos para cerca de um segundo por taxa.

O braço é um protótipo. A entrada no artigo está em aberto, e a medição no teste depende dessa decisão.

## What Changes

- **Momentos da condicional em log** (`imputation/momentos.py`): média e covariância de `x | x_O` para PPCA (análise de componentes principais probabilística), FA (análise fatorial) e porção média, com e sem média condicional. Substitui os sorteios pela condicional gaussiana, sem alterar os imputadores.
- **Momentos em gramas e previsão fechada** (`regression/propagacao_analitica.py`): conversão pela log-normal, com a covariância entre gavetas ocultas da mesma refeição, e a preditiva `E[y]`, `Var[y]` com `blr`.
- **Experimento próprio** (`experiments/propagacao_analitica.py`): plug-in, analítica e propagação com S = 10, 30 e 100, nas taxas 0,25 a 1,0, com as 3 sementes de máscara, na validação. Tabelas novas, sem tocar nos CSVs do experimento oficial.
- **Testes de concordância** (`tests/test_propagacao_analitica.py`): redução ao `predict` sem ocultas, momentos contra 20.000 amostras, forma fechada contra Monte Carlo sem truncamento e ausência de leitura de `X[M]`.

## Capabilities

### New Capabilities

- `propagacao-analitica`: previsão do carboidrato com a incerteza da imputação propagada em forma fechada, ao lado dos braços por amostragem.

### Modified Capabilities

Nenhuma. Os módulos e o experimento existentes não mudam.

## Impact

- Código novo: `src/amaprob/imputation/momentos.py`, `src/amaprob/regression/propagacao_analitica.py`, `src/amaprob/experiments/propagacao_analitica.py`, `src/amaprob/experiments/juntar_propagacao_analitica.py`, `tests/test_propagacao_analitica.py` e `tests/test_juntar_propagacao_analitica.py`.
- Resultados novos: `results/propagacao_analitica_metricas.csv`, `results/propagacao_analitica_por_ocultas.csv` e `results/propagacao_analitica_tempos.csv`.
- Sem dependência nova, sem alteração em `ppca.py`, `fa.py`, `baseline.py`, `blr.py`, `propagation.py` ou no experimento oficial.
- Aproximação declarada: o truncamento em 0 das amostras não entra na forma fechada.
