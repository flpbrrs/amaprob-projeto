# Proposal

## Why

A Parte 2 exige mais de um regressor probabilístico, e hoje só existe a regressão linear bayesiana. O SVGP (Processo Gaussiano Variacional Esparso) é o segundo modelo obrigatório. Ele mostra se uma correção não linear sobre a reta melhora a previsão do carboidrato e se a propagação continua calibrada com um regressor cuja incerteza depende da entrada. É também o alvo da Parte 3 (número de pontos indutores).

## What Changes

- **SVGP como `ProbRegressor`**, em GPyTorch, com pontos indutores aprendidos e treino por minibatch na ELBO (limite inferior da evidência).
  - **Decisão tomada:** dependências PyTorch e GPyTorch. O PyTorch do PyPI no Windows já vem só com CPU.
  - **Decisão tomada:** média linear mais kernel RBF com um comprimento por coluna (ARD), sobre entradas padronizadas. O carboidrato é quase linear nas gramas (R² de 0,81), então o GP vira "regressão linear + correção não linear", e a comparação com a regressão linear bayesiana responde se a não linearidade ajuda.
  - **Decisão tomada:** M = 256 pontos indutores de partida (a Parte 3 otimiza M), iniciados numa amostra do treino; 60 épocas fixas (10 deixavam a ELBO ainda subindo; ver a medição de custo no design); semente no construtor; precisão `float64`.
- **Medição de custo antes da rodada:** tempo de ajuste no treino completo e tempo de previsão de 100 amostras da validação. **Ponto de decisão:** se a previsão de uma rodada passar de 2 minutos, a execução para e se decide entre subamostra estratificada da validação e M menor.
- **Regressor como dimensão do experimento de propagação**, ao lado do imputador.
  - **Decisão tomada:** os três imputadores com o SVGP. O plug-in recalibrado continua só com a regressão linear.
  - Execução parcial também por regressor (`--regressores svgp`), com junção aos CSVs existentes, como na FA. As métricas da Parte 1 não dependem do regressor e só são calculadas quando a regressão linear roda.
  - A referência da taxa 0 por número de ocultas passa a existir por regressor.
  - Figuras: uma grade regressor × imputador.

## Capabilities

### New Capabilities

Nenhuma.

### Modified Capabilities

- `regressao-probabilistica`: requisito do SVGP com média linear e pontos indutores.
- `propagacao`: regressor como dimensão da curva; execução parcial por regressor; referência da taxa 0 por regressor; recalibrado só com a regressão linear.

## Impact

- Dependências novas: `torch` e `gpytorch` (`uv add`).
- Código novo: `src/amaprob/regression/svgp.py`, `tests/test_svgp.py`.
- Código alterado: `src/amaprob/experiments/propagacao.py` (regressores, junção por regressor, figuras em grade).
- Resultados: coluna `regressor` nos CSVs de propagação (`blr` nas linhas existentes) e linhas do SVGP acrescentadas; resumo e figuras regravados. As métricas da Parte 1 não mudam.
- `AGENTS.md` §6 e §8 e README.
