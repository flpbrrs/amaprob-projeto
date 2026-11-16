# Proposal

## Why

A Parte 3 (ajuste de hiperparâmetros por otimização bayesiana) é obrigatória e é a última que falta. Hoje q = 5, M = 256 e α = 1 são valores de partida, escolhidos sem busca. A rodada única no teste (Regra 6) precisa usar botões escolhidos só na validação, de um jeito defensável. A regressão linear bayesiana já ajusta α e β do prior pela evidência, em forma fechada. Os botões de PPCA, FA e SVGP mudam a estrutura do modelo e custam um ajuste inteiro por tentativa, e é para esse caso que a otimização bayesiana serve.

## What Changes

- **Quatro buscas**, uma por modelo, minimizando uma NLL na validação:
  - **Decisão tomada:** PPCA condicional e FA condicional, cada um com q e α (força do ridge da média condicional).
  - **Decisão tomada:** porção média condicional (controle), com α, para a comparação com a FA condicional continuar justa.
  - SVGP, com M (número de pontos indutores). **Decisão tomada:** escolhe-se o menor M avaliado cuja NLL fica a até 0,01 da melhor, porque a NLL tende a melhorar sempre um pouco com M e o custo da previsão cresce com ele.
  - **Decisão tomada:** as versões sem média condicional não são buscadas e herdam o q da versão condicional. A porção média não tem botão, e a regressão linear se ajusta pela evidência.
- **Objetivo dos imputadores. Decisão tomada:** NLL do carboidrato no braço de propagação (regressão linear, S = 100) na taxa 0,5, semente 0, na validação.
- **Objetivo do SVGP:** NLL do carboidrato na taxa 0 (gramas completas) na validação, porque o regressor não depende do imputador.
- **Otimização bayesiana. Decisão tomada:** BoTorch, com surrogate de processo gaussiano e aquisição sobre a grade discreta de candidatos.
- **Busca aleatória** com o mesmo orçamento, para comparação. É o extra de `AGENTS.md` §7, que sai da suspensão.
- **Orçamento. Decisão tomada:** 12 avaliações por modelo e por método. As 4 primeiras são comuns aos dois métodos, e uma avaliação repetida não é recalculada.
- **α sem conversão pela amostra:** os imputadores são ajustados numa amostra estratificada do treino, com o α do candidato usado direto. α é a precisão de um prior, e no máximo a posteriori o α ótimo não depende do número de refeições: com menos dados o prior pesa mais, como deve. A confirmação mede também a escolhida com α convertido pela razão entre as refeições do treino e as da amostra (o mesmo freio relativo aos dados), como diagnóstico dessa transferência.
- **Confirmação no treino completo:** a configuração escolhida (e a variante com α convertido) e a de partida são reajustadas no treino completo e comparadas nas 3 sementes (taxa 0,5 para os imputadores, taxa 0 para o SVGP). O ganho da Parte 3 é medido aqui.
- **Medição de custo antes da rodada**, com ponto de parada.
- **Configuração escolhida gravada** para a rodada no teste usar, sem alterar o experimento de propagação nesta change.

## Capabilities

### New Capabilities

- `otimizacao-bayesiana`: buscas por otimização bayesiana e por busca aleatória, objetivos, registro das avaliações, confirmação no treino completo e configuração escolhida.

### Modified Capabilities

Nenhuma.

## Impact

- Dependência nova: `botorch` (`uv add`).
- Código novo: `src/amaprob/tuning/` (busca), `src/amaprob/experiments/otimizacao.py` (objetivos, execução, registro, figura) e `tests/test_otimizacao.py`.
- Resultados novos:
  - `results/otimizacao_avaliacoes.csv`
  - `results/otimizacao_confirmacao.csv`
  - `results/figuras/otimizacao_convergencia.png`
  - `configs/hiperparametros.json`
- O experimento de propagação e os resultados dele não mudam. Ler a configuração escolhida fica para a change da rodada no teste.
- `AGENTS.md` §4, §7 e §10 e README.
