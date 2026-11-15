# Tasks

## 1. FA-EM

- [x] 1.1 Implementar `imputation/fa.py` com a derivação comentada (design, Decisões 1 a 6): passo E com Ψ dentro dos produtos, passo M por gaveta com ψ_k próprio, piso de 0,1% da variância observada (ver 2.1), ψ_k = 1 nas gavetas sem observados, `gavetas_no_piso_`, log-verossimilhança dos observados, parada por tolerância, `historico_log_verossimilhanca_` e amostragem pela posterior com truncamento em 0. `ppca.py` não muda; verificar com o teste 1.2
- [x] 1.2 Escrever `tests/test_fa.py` com os cenários da FA na spec `imputacao`: recuperação com ruído heterogêneo e metade ausente (10% em Frobenius de `WWᵀ + Ψ` e de μ; Ψ gaveta a gaveta); log-verossimilhança da FA maior que a do PPCA nos mesmos dados; monotonia; momentos da condicional contra o complemento de Schur de `WWᵀ + Ψ`; taxa 1 pela marginal; ausentes fora da verossimilhança; mesma semente, mesmos parâmetros; gaveta constante no piso; gaveta sem observados sem `NaN`. Na recuperação, usar tolerância menor e mais iterações que o padrão; verificar com `uv run pytest tests/test_fa.py`
- [x] 1.3 Acrescentar a FA (q = 2) ao `IMPUTADORES` de `tests/test_imputacao.py`; verificar com `uv run pytest tests/test_imputacao.py tests/test_ppca.py` (o PPCA continua passando sem mudança)

## 2. Medição de custo (ponto de decisão)

- [x] 2.1 Medir com os dados reais um ajuste da FA (q = 5) no treino completo: segundos por iteração, iterações, segundos totais, gavetas no piso e ψ por gaveta contra o σ² do PPCA (0,099). Se o ajuste passar de 5 minutos ou bater 500 iterações, parar e levar a decisão para aprovação
  - Registro (piso desligado): ajuste no treino completo em 61 s (152 iterações, 0,40 s por iteração, parada pela tolerância), log-verossimilhança por célula observada de −0,902 (PPCA: −0,947), sem queda entre iterações. Menores ψ_k / variância: Molhos e condimentos 0,48%, Açúcar 1,4%, Óleos e gorduras 2,9%. O piso de 1% prenderia Molhos e, no teste sintético, prendeu uma gaveta normal; decidido 0,1%, que não prende nenhuma.

## 3. Experimento com a FA

- [x] 3.1 Atualizar `experiments/propagacao.py` (design, Decisão 7): FA em `IMPUTADORES` e nos nomes; opção `--imputadores`; junção dos CSVs mantendo as linhas dos não executados e da referência `nenhum`; coluna `gavetas_no_piso` em `imputadores_ajuste.csv`; resumo e figuras sobre a tabela juntada, com largura 16; verificar com `uv run pytest`
- [x] 3.2 Executar `uv run python -m amaprob.experiments.propagacao --imputadores fa`; verificar que o `git diff` de `results/*.csv` só acrescenta linhas da FA (as do baseline, do PPCA e da referência iguais às de `HEAD`), que não há duplicatas por imputador, taxa, semente e braço, que na taxa 0 os braços da FA coincidem com os dos outros imputadores e que `propagacao_S.csv` não mudou; registrar o tempo total
  - Registro: execução de 26,6 min (ajustes da FA: 61 s no treino completo e 244 s nas 5 dobras, de 146 a 167 iterações, nenhuma gaveta no piso). Linhas antigas dos cinco CSVs idênticas às de `HEAD` (comparadas com leitura exata de floats), sem duplicatas, 60 linhas de referência preservadas; na taxa 0 os três imputadores coincidem; `propagacao_S.csv` inalterado. O diff de `imputadores_ajuste.csv` muda todas as linhas só pela coluna nova `gavetas_no_piso`.
- [x] 3.3 Conferir a leitura antes de interpretar: PICP da propagação da FA perto da taxa 0; MPIW e CRPS da FA contra o PPCA e o baseline, principalmente na taxa 1; métricas por gaveta nas gavetas em que o PPCA calibrava pior (preparações mistas, farofa, macarrão, cerveja/vinho); gavetas no piso

## 4. Documentação

- [x] 4.1 Atualizar o `AGENTS.md` §5 (FA implementada, piso no ruído) e §6 (experimento com três imputadores e execução parcial) e o `README.md` (comando com `--imputadores`); verificar que o comando roda como escrito

## 5. Integração

- [x] 5.1 Rodar `uv run pytest` com todos os testes passando e `openspec validate fa-em --strict` sem erros

## Workflow follow-up

- Mostrar os resultados (resumo por imputador e braço, Parte 1 global e por gaveta, figuras), os tempos e o diff para aprovação antes de commitar.
- Arquivar a change (`/opsx:archive`) depois da revisão.
- Propor a entrada do Diário do projeto no vault para aprovação.
