# Proposal

## Why

O dataset e as máscaras já estão fixados (change `mascaras-persistidas`), mas nenhum modelo foi ajustado ainda. O caminho mais simples de ponta a ponta tem três peças: contexto, regressão probabilística e métricas, avaliadas na taxa 0, em que nada está oculto. Ele funciona como teste de sanidade: o erro deve ficar perto do piso de ~8 g medido com K = 29. Sem esse circuito, as Partes 1 a 3 não têm onde se apoiar, e problemas de encanamento se confundiriam com resultados.

## What Changes

- **Contexto `C` no dataset:** ocasião (5 categorias), hora (codificação cíclica), macrorregião (5 categorias, a partir da UF) e `log1p` da renda, todos do arquivo de consumo alimentar. O dataset é regravado; as máscaras não mudam.
- **Contrato `ProbRegressor` com preditiva gaussiana:** `predict(Z)` devolve média e variância por refeição, no lugar de `predict_samples`. A NLL (log-verossimilhança negativa) passa a ser exata, e a propagação, na mudança seguinte, vira uma mistura de gaussianas. **Altera o contrato descrito no `AGENTS.md` (§6).**
- **Regressão linear bayesiana do zero:**
  - forma fechada conjugada, com entradas padronizadas pelo treino;
  - precisão do prior e do ruído estimadas pela maximização da evidência.
- **Contrato `Imputer` e baseline:**
  - o baseline é uma gaussiana por gaveta na escala `log1p`, ajustada só nas quantidades observadas;
  - testes de contrato e de não vazamento de `X[M]`;
  - o baseline não é avaliado em nenhuma taxa nesta mudança, porque na taxa 0 não há o que imputar.
- **Métricas para preditiva gaussiana:**
  - MAE (erro absoluto médio), NLL, CRPS (*continuous ranked probability score*), PICP (cobertura do intervalo de 90%) e MPIW (largura média do intervalo);
  - percentual de refeições com erro de até 7,5 g (meia dose; fonte a verificar);
  - cobertura do intervalo por faixa da previsão.
- **Experimento na taxa 0:** ajusta no treino, mede apenas na validação, grava as métricas em `results/` e registra o tempo de ajuste de cada modelo. O conjunto de teste não é lido; ele fica reservado às rodadas finais (Regra 6).
- Dependência nova: `scoringrules` (CRPS).
- `AGENTS.md` atualizado:
  - o contrato do §6;
  - o conteúdo de `C` (§2);
  - a métrica de meia dose adotada, que sai das decisões em aberto (§12);
  - o risco das caudas pesadas (§11).
- `backlog/decisions.md` registra o modelo de ruído com caudas pesadas ou com variância crescente como alternativa adiada.

Fora do escopo:
- plug-in × propagação e o baseline em taxas maiores que 0 (mudança seguinte);
- sexo e idade do arquivo `MORADOR`;
- SVGP, PPCA e FA.

## Capabilities

### New Capabilities

- `imputacao`: contrato dos imputadores probabilísticos (entrada com quantidades ocultas apagadas, saída em amostras) e o imputador de referência, a porção média por gaveta.
- `regressao-probabilistica`: contrato dos regressores com preditiva gaussiana e a regressão linear bayesiana com hiperparâmetros pela evidência.
- `metricas`: métricas de erro, calibração e qualidade probabilística para preditivas gaussianas.
- `sanidade-taxa-zero`: avaliação de ponta a ponta na taxa 0, na validação, com registro das métricas e do tempo de ajuste.

### Modified Capabilities

- `dataset-refeicoes`: acrescenta a matriz de contexto `C` e os nomes das suas colunas aos campos alinhados por refeição.

## Impact

- Código novo:
  - `src/amaprob/imputation/` (`base.py`, `baseline.py`);
  - `src/amaprob/regression/` (`base.py`, `blr.py`);
  - `src/amaprob/evaluation/metricas.py`;
  - `src/amaprob/experiments/taxa_zero.py`.
- Código alterado: `src/amaprob/data/dataset.py` (contexto `C`).
- Testes novos para imputação, regressão, métricas e contexto.
- Dependência: `scoringrules` (traz `scipy`).
- Arquivos regravados fora do git: `data/processed/refeicoes_k29.npz`. As máscaras continuam válidas, e a carga confere isso.
- Resultados novos em `results/` (CSV).
- Documentação: `AGENTS.md`, `backlog/decisions.md`, `README.md` (comando do experimento).
