# Tasks

## 1. Dependência e busca

- [x] 1.1 `uv add botorch`; verificar que `uv run python -c "import botorch"` funciona e que o `gpytorch` e o `torch` instalados não mudam de versão (ou registrar a mudança)
- [x] 1.2 Implementar `src/amaprob/tuning/busca.py` (design, Decisão 5): `Avaliador` com cache e marca de reaproveitada, `busca_aleatoria` sem reposição e `otimizacao_bayesiana` com início comum, `SingleTaskGP`, `LogExpectedImprovement` sobre os candidatos não avaliados e semente por passo; verificar com o teste 1.3
- [x] 1.3 Escrever `tests/test_otimizacao.py` com os cenários da spec `otimizacao-bayesiana` que não dependem dos dados: tigela quadrática numa grade pequena, com 12 avaliações (melhor a até um passo do mínimo e não pior que a aleatória); candidatos distintos e da grade; início comum aos dois métodos; mesma semente, mesma sequência; reaproveitamento entre métodos (objetivo chamado uma vez por candidato); melhor NLL até ali nunca aumenta; verificar com `uv run pytest tests/test_otimizacao.py`

## 2. Objetivos e amostra

- [x] 2.1 Implementar em `src/amaprob/experiments/otimizacao.py` (design, Decisões 1 a 4): grades de candidatos, amostra estratificada pela ocasião, objetivo dos imputadores (propagação com a regressão linear, taxa 0,5, semente 0) e do SVGP (taxa 0), com NLL, CRPS, PICP e MPIW; acrescentar ao teste os cenários da amostra (só treino, proporção por ocasião, mesma semente, mesma amostra); acrescentar o cenário da escolha do M pelo menor dentro da tolerância; verificar com `uv run pytest tests/test_otimizacao.py`

## 3. Medição de custo (ponto de decisão)

- [x] 3.1 Com os dados reais (design, Decisão 6): tempo de ajuste da FA e do PPCA condicionais na amostra de 25%, com q = 15; tempo de um objetivo completo de imputador; segundos por época do SVGP com M = 512, em 3 épocas, e a ELBO dessas épocas; estimar o tempo total. Se a estimativa passar de 4 h, ou se a ELBO com M = 512 indicar que 60 épocas não bastam, parar e levar a decisão para aprovação

## 4. Execução, registro e confirmação

- [x] 4.1 Completar `experiments/otimizacao.py` (design, Decisão 7): referência de partida, busca aleatória e otimização bayesiana por modelo, escolha, confirmação no treino completo (com a variante de α convertido nos condicionais), `results/otimizacao_avaliacoes.csv`, `results/otimizacao_confirmacao.csv`, figura de convergência, `configs/hiperparametros.json` com o q herdado pelas versões sem média condicional e `--modelos` com junção por modelo; verificar com `uv run pytest`
- [x] 4.2 Executar `uv run python -m amaprob.experiments.otimizacao`, se preciso em partes com `--modelos`; verificar que a confirmação de partida de cada imputador repete as métricas de `propagacao_metricas.csv` (regressão linear, propagação, taxa 0,5, 3 sementes), que não há candidato repetido por método e que as 4 primeiras avaliações dos dois métodos coincidem; registrar os tempos
- [x] 4.3 Conferir a leitura antes de interpretar: regra de adoção (design, Adoção) aplicada a cada modelo; `--so-confirmacao` para regravar a variante de α convertido, corrigida para α · N / n depois da primeira rodada dos imputadores; ganho da escolhida sobre a de partida nas sementes 1 e 2 (não usadas na escolha); escolhida com α direto contra α convertido; otimização bayesiana contra busca aleatória na curva de convergência; registrar a configuração adotada de cada modelo

## 5. Documentação

- [x] 5.1 Atualizar o `AGENTS.md` §4 (escala), §7 (buscas, objetivos, α convertido, confirmação, resultado) e §10 (busca aleatória fora dos extras suspensos) e o `README.md` (comando, arquivos gravados e tempos); verificar que os comandos rodam como escritos

## 6. Integração

- [x] 6.1 Rodar `uv run pytest` com todos os testes passando e `openspec validate otimizacao-bayesiana --strict` sem erros

## Workflow follow-up

- Mostrar a medição de custo antes da rodada, e depois as escolhas, a confirmação, a figura, os tempos e o diff para aprovação antes de commitar.
- Arquivar a change (`/opsx:archive`) depois da revisão.
- Propor a entrada do Diário do projeto no vault para aprovação.
