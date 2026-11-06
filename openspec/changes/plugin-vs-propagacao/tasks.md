# Tasks

## 1. Dependências

- [ ] 1.1 Adicionar `matplotlib` e `scipy` como dependências diretas (`uv add matplotlib scipy`); verificar com `uv run python -c "import matplotlib, scipy.special; import scoringrules as sr; sr.crps_mixnorm"`

## 2. Métricas da mistura

- [ ] 2.1 Implementar em `evaluation/metricas.py`: `intervalo(medias, variancias)` (quantis exatos por bissecção, S = 1 pela fórmula fechada), `avaliar_mistura(y, medias, variancias)` (NLL por `logsumexp`, CRPS por `crps_mixnorm` em blocos) e `por_ocultas(y, media, li, ls, n_ocultas)`; verificar com o teste 2.2
- [ ] 2.2 Estender `tests/test_metricas.py` com os cenários da spec `metricas`: S = 1 igual a `avaliar`; componentes iguais; CDF nos quantis de uma mistura de dois componentes (tolerância 1e-6); NLL finita a 40 desvios; classes de `por_ocultas` (soma N, classe "4+", caso conhecido); verificar com `uv run pytest tests/test_metricas.py`

## 3. Propagação

- [ ] 3.1 Implementar `regression/propagation.py`: `amostras_em_gramas` (blocos pelo orçamento de memória), `prever_bracos` (plug-in e propagação das mesmas amostras, recalibrado pela mesma entrada do plug-in) e `imputar_treino_cruzado` (5 dobras por pessoa, geradores por dobra); verificar com o teste 3.2
- [ ] 3.2 Escrever `tests/test_propagacao.py` com dados sintéticos pequenos: média da mistura igual à do plug-in com a regressão linear; variância total; taxa 0 com os três braços iguais à previsão direta; dobras por pessoa (cada pessoa em uma dobra); alterar as gramas ocultas de uma dobra não muda a imputação dela; plug-in usa a média de `expm1`; verificar com `uv run pytest tests/test_propagacao.py`

## 4. Experimento

- [ ] 4.1 Implementar `experiments/propagacao.py` (ajuste no treino completo, escolha de S, laço por semente e taxa, CSVs e figuras) e executar `uv run python -m amaprob.experiments.propagacao`; verificar que, na taxa 0, as métricas dos três braços batem com `results/taxa_zero_metricas.csv` (linha `blr`) e que todos os arquivos da spec `propagacao` existem
- [ ] 4.2 Conferir a leitura dos resultados antes de interpretar: plug-in com PICP caindo com a taxa; plug-in e propagação com o mesmo MAE; tempo de cada S registrado. Se o CRPS com S = 500 passar de 10 minutos, aplicar a alternativa do design (`crps_ensemble` com estimador `fair` na comparação de S) e registrar no relatório
- [ ] 4.3 Estender `tests/test_taxa_zero.py` ou `tests/test_propagacao.py`, pulado sem os dados processados: o experimento nunca seleciona linhas de `teste`; verificar com `uv run pytest`

## 5. Documentação

- [ ] 5.1 Atualizar o `AGENTS.md`: §4 (imputador e regressor no treino completo; máscara do treino só para o braço recalibrado, com ajuste cruzado), §6 (plug-in pela média em gramas das amostras, terceiro braço, leitura por número de ocultas, pergunta "o modelo sabe quando não sabe?") e §11 se surgir risco novo; verificar relendo cada seção contra o `design.md`
- [ ] 5.2 Atualizar `docs/mascaramento.md` (papel da máscara no treino) e acrescentar ao `README.md` o comando do experimento; verificar que o comando roda como escrito

## 6. Integração

- [ ] 6.1 Rodar `uv run pytest` com todos os testes passando e `openspec validate plugin-vs-propagacao --strict` sem erros

## Workflow follow-up

- Mostrar os resultados (tabela de resumo, por ocultas, comparação de S, figuras), os tempos e o diff para aprovação antes de commitar.
- Arquivar a change (`/opsx:archive`) depois da revisão.
- Propor a entrada do Diário do projeto no vault para aprovação.
