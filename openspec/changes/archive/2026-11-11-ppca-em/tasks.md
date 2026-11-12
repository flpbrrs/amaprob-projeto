# Tasks

## 1. PPCA-EM

- [x] 1.1 Implementar `imputation/ppca.py` com a derivação comentada (design, Decisões 1 a 5): passo E vetorizado, passo M por gaveta com `[w_k; μ_k]` juntos, σ², log-verossimilhança dos observados, parada por tolerância e `historico_log_verossimilhanca_`, e amostragem pela posterior do latente com truncamento em 0; verificar com o teste 1.2
- [x] 1.2 Escrever `tests/test_ppca.py` com os cenários do PPCA na spec `imputacao`: forma fechada sem faltantes (tolerância 1e-3, autovalores bem separados); recuperação sintética com metade ausente (10% em Frobenius de `WWᵀ + σ²I` e de μ); log-verossimilhança monótona; momentos da condicional contra o complemento de Schur; taxa 1 pela marginal; ausentes fora da verossimilhança; mesma semente, mesmos parâmetros. Nos testes de forma fechada e de recuperação, usar tolerância bem menor e mais iterações que o padrão, para testar a solução convergida e não a regra de parada; verificar com `uv run pytest tests/test_ppca.py`
- [x] 1.3 Parametrizar `tests/test_imputacao.py` sobre `PorcaoMedia` e `PPCA` (q = 2, porque a fixture tem K = 6) nos testes do contrato (forma, observados, ocultos, gerador, vazamento, X[M] preenchido); verificar com `uv run pytest tests/test_imputacao.py`

## 2. Medição de custo (ponto de decisão)

- [x] 2.1 Medir com os dados reais: um ajuste do PPCA (q = 5) no treino completo (segundos por iteração, iterações, segundos totais) e 100 amostras da validação na taxa 0,5, semente 0; mostrar os números. Se o ajuste passar de 5 minutos, parar antes de seguir e diagnosticar pelo fator: iteração lenta → conferir se as contrações viram produtos de matrizes (BLAS), como `O @ WW.reshape(K, q*q)`; muitas iterações → inicialização e tolerância; só se os dois estiverem bem, levar subamostra ou agrupamento por padrão para aprovação
  - Registro: ajuste no treino completo em 184 s (486 iterações, 0,38 s por iteração, parada pela tolerância); σ² = 0,099; log-verossimilhança por célula observada de −1,109 para −0,947. 100 amostras da validação (taxa 0,5, semente 0) em 1,8 s. Abaixo dos 5 minutos: segue o desenho, com ~18 minutos de ajustes do PPCA por execução (completo + 5 dobras).

## 3. Métricas de imputação

- [x] 3.1 Implementar em `evaluation/metricas.py` `avaliar_imputacao` e `imputacao_por_gaveta` (design, Decisão 6); verificar com o teste 3.2
- [x] 3.2 Estender `tests/test_metricas.py` com os cenários da spec `metricas`: caso conhecido do MAE em gramas e em log; só células ocultas; amostras gaussianas (NLL, CRPS e intervalo); cobertura conhecida; soma das células por gaveta; verificar com `uv run pytest tests/test_metricas.py`

## 4. Propagação com o imputador como dimensão

- [x] 4.1 Em `regression/propagation.py`, criar `ajustar_dobras` e fazer `imputar_treino_cruzado` receber os imputadores já ajustados; atualizar `tests/test_propagacao.py` e acrescentar o cenário "Dobras ajustadas uma vez" (5 ajustes por imputador, contados com um imputador instrumentado). No teste "Nenhum imputador vê o que imputa", chamar `ajustar_dobras` de novo com `G_alterado` (reaproveitar as dobras ajustadas em `G` faria o teste passar sem testar nada); verificar com `uv run pytest tests/test_propagacao.py`
- [x] 4.2 Atualizar `experiments/propagacao.py` (design, Decisão 7): `IMPUTADORES` com porção média e PPCA; S = 100 fixo sem `escolher_S` e sem regravar `propagacao_S.csv`; dobras uma vez por imputador; referência da taxa 0 por número de ocultas; métricas da Parte 1 nas taxas > 0; CSVs com a coluna `imputador`; `results/imputadores_ajuste.csv`; figuras com um painel por imputador; verificar com `uv run pytest` (incluindo o teste de conjuntos usados)
- [x] 4.3 Executar `uv run python -m amaprob.experiments.propagacao`; verificar que existem todos os arquivos das specs `propagacao` e `metricas`, que na taxa 0 os braços dos dois imputadores coincidem com a referência e com `results/taxa_zero_metricas.csv` (linha `blr`), que o baseline reproduz os números da rodada anterior (comparar com os CSVs versionados em `HEAD`, até o erro de ponto flutuante) e que `propagacao_S.csv` não mudou
  - Registro: execução de 67,6 min. Baseline idêntico à rodada anterior (45 linhas de métricas e 180 por número de ocultas, diferença 0). Na taxa 0, braços, imputadores e referência coincidem com `taxa_zero_metricas.csv` (linha `blr`). `propagacao_S.csv` inalterado. Ajuste do PPCA: 194 s no treino completo (486 iterações) e 791 s nas 5 dobras; 2 das 5 dobras pararam no limite de 500 iterações.
- [x] 4.4 Conferir a leitura antes de interpretar: PICP do plug-in caindo com a taxa nos dois imputadores; MPIW da propagação do PPCA contra a do baseline nas taxas intermediárias; métricas da Parte 1 por gaveta (açúcar contra as outras; gavetas em que o PPCA calibra pior que o baseline, ver o risco do ruído isotrópico); tempo total da execução

## 5. Documentação

- [x] 5.1 Atualizar o `AGENTS.md`: §5 (decisão dos ausentes fechada como condicionantes; métricas da Parte 1 a partir das amostras; q = 5 de partida), §6 (experimento com o imputador como dimensão; S = 100 fixo) e §12 (sair a decisão dos ausentes); verificar relendo cada seção contra o `design.md`
- [x] 5.2 Atualizar `backlog/decisions.md`: encerrar "Ajustes do experimento de propagação para a próxima rodada" e registrar a extensão B+ (média de cada gaveta dependendo de `P` e `C`; quando voltar: taxa 1 ainda larga e tempo sobrando); atualizar o comando ou a descrição do experimento no `README.md` se mudar; verificar que o comando roda como escrito

## 6. Integração

- [x] 6.1 Rodar `uv run pytest` com todos os testes passando e `openspec validate ppca-em --strict` sem erros

## Workflow follow-up

- Mostrar os resultados (resumo por imputador e braço, por ocultas com a referência da taxa 0, métricas da Parte 1 globais e por gaveta, figuras), os tempos e o diff para aprovação antes de commitar.
- Arquivar a change (`/opsx:archive`) depois da revisão.
- Propor a entrada do Diário do projeto no vault para aprovação.
