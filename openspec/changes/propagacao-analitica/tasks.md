# Tasks

## 1. Momentos da condicional em log

- [x] 1.1 `imputation/momentos.py` com `momentos_ocultos(imputador, X, P, M, C)`, lendo os internos dos imputadores (PPCA, FA e porção média, com e sem média condicional), em blocos de refeições. Verificar com um teste para cada um dos seis imputadores: média e covariância batem com 20.000 amostras de `Imputer.sample` em células de média alta, com tolerância de 3%
- [x] 1.2 Células fora de `M` com média NaN e covariância 0; covariância simétrica e semidefinida positiva. Verificar no mesmo teste e no teste de propriedades
- [x] 1.3 Regra 2: `X[M]` com lixo produz o mesmo resultado que NaN. Verificar com um teste que troca as ocultas por ruído de escala 1e3

## 2. Momentos em gramas e previsão em forma fechada

- [x] 2.1 `regression/propagacao_analitica.py` com `momentos_em_gramas` (log-normal) e `prever_com_entrada_incerta` (lei da variância total). Verificar com o teste de redução: sem ocultas, coincide com `blr.predict`
- [x] 2.2 Salvaguarda de plausibilidade (1 t, valor não finito) que só detecta, levantando `ValueError` com a gaveta. Verificar com um teste de média absurda
- [x] 2.3 Forma fechada contra Monte Carlo sem truncamento, com 5.000 sorteios e a lei da variância total. Verificar no teste de concordância, com 2% no agregado e tolerância informada pelo erro-padrão de Monte Carlo nas células

## 3. Experimento separado

- [x] 3.1 `experiments/propagacao_analitica.py` com plug-in, analítica e propagação em S = 10, 30 e 100, taxas 0,25 a 1,0, 3 sementes, na validação, e `--imputadores`. Verificar com `--help` e com a execução de uma taxa
- [x] 3.2 Tabelas novas (`propagacao_analitica_metricas.csv`, `propagacao_analitica_por_ocultas.csv`, `propagacao_analitica_tempos.csv`) com junção por imputador. Verificar que os CSVs do experimento oficial ficam idênticos pelo `git status`
- [x] 3.3 Registro de tempo por ajuste, por braço analítico e por propagação S = 100. Verificar nas colunas do CSV de tempos

## 4. Reprodutibilidade contra o experimento oficial

- [x] 4.1 Conferir as linhas de plug-in e propagação S = 100 com a regressão linear contra `results/propagacao_metricas.csv` do `main`, mostrando a maior diferença absoluta por métrica. Veredito: bateu (diferenças da ordem de 1e-11, ruído de ponto flutuante)

## 5. Proposta

- [x] 5.1 `proposal.md`, `design.md` e `tasks.md` no formato das changes arquivadas, com tom impessoal e sem nomes

## 6. Rodada na validação

- [ ] 6.1 Rodar `--imputadores fa_condicional porcao_media` nas taxas 0,25 a 1,0 e nas 3 sementes; gerar a tabela por imputador e taxa (MAE, NLL, CRPS, PICP90, MPIW) de plug-in, analítica e propagação S = 10, 30 e 100, com os tempos
- [ ] 6.2 Comparar a FA condicional na taxa 0,5 por número de quantidades ocultas (0, 1, 2, 3, 4+) entre os braços
- [ ] 6.3 Rodar os quatro imputadores restantes, para o protótipo completo, e juntar às tabelas

## 7. Rodadas em paralelo

- [x] 7.1 `--sufixo`, `--taxas` e `--sementes` no experimento, com sufixo vazio por padrão. Verificar com `--help` e com a rodada de uma combinação
- [x] 7.2 `experiments/juntar_propagacao_analitica.py`, com falha em coluna diferente, chave repetida e arquivo ausente. Verificar com os testes da junção
- [ ] 7.3 Medir o pico de memória de um `ppca` (q = 12) antes da rodada em paralelo e decidir entre 3 e 2 processos
- [ ] 7.4 Rodar `porcao_condicional`, `fa` e `ppca` em paralelo, com `ppca_condicional` na primeira vaga liberada, e juntar com o script de conferência
- [ ] 7.5 Conferir que o paralelismo não muda número: rodar uma combinação isolada (porção média condicional, taxa 0,5, semente 0) em processo único e comparar a linha de propagação S = 100 com a da rodada em paralelo, exigindo diferença zero

## Workflow follow-up

- Mostrar o que será commitado e fazer o commit só com aprovação.
- A medição no teste depende de aprovação (Regra 6) e fica fora desta change.
