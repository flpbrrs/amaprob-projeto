# Tasks

## 1. Dependências e SVGP

- [x] 1.1 `uv add torch gpytorch`; verificar que `uv run python -c "import torch, gpytorch"` funciona e que `torch.cuda.is_available()` é falso (CPU)
- [x] 1.2 Implementar `regression/svgp.py` (design, Decisões 1 e 2): padronização de `Z` e `y`, média linear, RBF com ARD, pontos indutores iniciados no treino, ELBO por minibatch, semente, `float64`, `historico_elbo_`, `segundos_ajuste_` e previsão em blocos; verificar com o teste 1.3
- [x] 1.3 Escrever `tests/test_svgp.py` com os cenários do SVGP na spec `regressao-probabilistica`: reta recuperada (média e variância do ruído); NLL num conjunto separado menor que a da regressão linear em dados com parte não linear; variância maior fora dos dados; mesma semente, mesmas previsões; coluna constante sem divisão por zero. Dados pequenos, M e épocas pequenos, para o teste levar segundos; verificar com `uv run pytest tests/test_svgp.py tests/test_regressao.py`

## 2. Medição de custo (ponto de decisão)

- [x] 2.1 Medir com os dados reais um ajuste do SVGP (M = 256; medido com 10 épocas, que subajustaram, e fixado em 60) no treino completo (segundos por época, curva da ELBO, NLL na validação na taxa 0 contra a regressão linear) e a previsão de 100 amostras da validação na taxa 0,5, semente 0. Se a previsão passar de 2 minutos por rodada ou a ELBO ainda estiver subindo muito na última época, parar e levar a decisão para aprovação

## 3. Experimento com o regressor como dimensão

- [x] 3.1 Atualizar `experiments/propagacao.py` (design, Decisão 3): `REGRESSORES`, `--regressores`, recalibrado só com `blr`, referência da taxa 0 por regressor, Parte 1 só com `blr`, coluna `regressor` e junção por par, `results/regressores_ajuste.csv`, figuras em grade; verificar com `uv run pytest`
- [x] 3.2 Executar `uv run python -m amaprob.experiments.propagacao --regressores svgp`; verificar que as linhas `blr` de cada CSV são iguais às de `HEAD` (fora a coluna `regressor`), que os CSVs da Parte 1 e `propagacao_S.csv` não mudaram, que não há duplicatas por regressor, imputador, taxa, semente e braço e que, na taxa 0, os braços do SVGP coincidem com a referência dele; registrar o tempo total
- [x] 3.3 Conferir a leitura antes de interpretar: PICP da propagação do SVGP perto da taxa 0 dele; plug-in do SVGP caindo com a taxa; MAE, CRPS e MPIW do SVGP contra a regressão linear em cada taxa; cobertura por número de ocultas

## 4. Documentação

- [x] 4.1 Atualizar o `AGENTS.md` §6 (SVGP implementado, média linear, experimento com o regressor como dimensão) e §8, se a estrutura mudar, e o `README.md` (comando com `--regressores`, tempos); verificar que os comandos rodam como escritos

## 5. Integração

- [x] 5.1 Rodar `uv run pytest` com todos os testes passando e `openspec validate svgp --strict` sem erros

## Workflow follow-up

- Mostrar os resultados (resumo por regressor, imputador e braço; figuras), os tempos e o diff para aprovação antes de commitar.
- Arquivar a change (`/opsx:archive`) depois da revisão.
- Propor a entrada do Diário do projeto no vault para aprovação.
