# Tasks

## 1. Dependência

- [x] 1.1 Adicionar `scoringrules` (`uv add scoringrules`); verificar com `uv run python -c "import scoringrules"`

## 2. Contexto C no dataset

- [x] 2.1 Montar `C` (13 colunas) e `C_nomes` em `montar_refeicoes`, gravando e carregando os dois campos em `Refeicoes`; verificar estendendo `tests/test_dataset.py` (indicadoras do almoço e do Nordeste, hora cíclica 23H/0H × 11H/12H, renda negativa levada a 0, ida e volta do disco) com `uv run pytest tests/test_dataset.py`
- [x] 2.2 Regravar os arquivos com `uv run python -m amaprob.data.dataset` e estender `tests/test_dados_processados.py` (`C` com 13 colunas, sem ausentes; carga das máscaras sem erro); verificar com `uv run pytest`

## 3. Imputação

- [x] 3.1 Implementar `imputation/base.py` com o protocolo `Imputer` e `ocultar(G, M)`, e `imputation/baseline.py` com `PorcaoMedia`; verificar com o teste 3.2
- [x] 3.2 Escrever `tests/test_imputacao.py`: `NaN` exatamente em `M`; forma (S, N, K); observados copiados e ausentes em 0; ocultos finitos e ≥ 0; mesma semente gera amostras idênticas; não vazamento (gramas diferentes só em `M` → saídas idênticas); parâmetros só pelos observados; distribuição com μ ≫ σ; verificar com `uv run pytest tests/test_imputacao.py`

## 4. Regressão

- [x] 4.1 Implementar `regression/base.py` (protocolo `ProbRegressor`, `montar_Z`) e `regression/blr.py` (`RegressaoLinearBayesiana`: padronização, evidência, posterior e preditiva, com a derivação comentada); verificar com o teste 4.2
- [x] 4.2 Escrever `tests/test_regressao.py`: `Z` com 71 colunas; formas e variância positiva; reta recuperada e 1/√β ≈ 2 em dados sintéticos; variância maior longe dos dados; coluna constante; verificar com `uv run pytest tests/test_regressao.py`
- [x] 4.3 Conferir a implementação, num script avulso fora do repositório, contra o `BayesianRidge` do scikit-learn (`uv run --isolated --with scikit-learn`) nos dados reais da taxa 0: previsões e 1/√β devem coincidir até poucas casas decimais; registrar o resultado no relatório da sessão

## 5. Métricas

- [x] 5.1 Implementar `evaluation/metricas.py` (`avaliar`, `cobertura_por_faixa`); verificar com o teste 5.2
- [x] 5.2 Escrever `tests/test_metricas.py` com os casos conhecidos da spec `metricas` (MAE e meia dose, NLL e CRPS da normal padrão, escala do CRPS, PICP e MPIW, 5 faixas que somam N); verificar com `uv run pytest tests/test_metricas.py`

## 6. Experimento na taxa 0

- [x] 6.1 Implementar `experiments/taxa_zero.py` (referência de mínimos quadrados + regressão linear bayesiana no treino, métricas na validação, tempos com o baseline na taxa 0,5 e semente 0) e executar `uv run python -m amaprob.experiments.taxa_zero`; verificar que o MAE da regressão linear bayesiana fica a no máximo 0,5 g acima de 8,1 g e que os três CSVs existem em `results/`
- [x] 6.2 Escrever `tests/test_taxa_zero.py`, pulado sem os dados processados: o experimento nunca seleciona linhas de `teste` (verificado pela função que separa os conjuntos); verificar com `uv run pytest`

## 7. Documentação

- [x] 7.1 Atualizar o `AGENTS.md`: contrato `ProbRegressor` (§6), conteúdo de `C` (§2), métrica de meia dose adotada como secundária e retirada das decisões em aberto (§12), risco das caudas pesadas (§11); verificar relendo cada seção contra o `design.md`
- [x] 7.2 Registrar em `backlog/decisions.md` o modelo de ruído com caudas pesadas ou variância crescente (o que é, por que ficou de fora, quando voltar); acrescentar ao `README.md` o comando do experimento; verificar que o comando roda como escrito

## 8. Integração

- [x] 8.1 Rodar `uv run pytest` com todos os testes passando e `openspec validate regressao-taxa-zero --strict` sem erros

## Workflow follow-up

- Mostrar os resultados, os tempos e o diff para aprovação antes de commitar.
- Arquivar a change (`/opsx:archive`) depois da revisão.
- Propor a entrada do Diário do projeto no vault para aprovação.
