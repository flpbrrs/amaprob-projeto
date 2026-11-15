# Proposal

## Why

Na validação, o PPCA (Análise de Componentes Principais Probabilística) ganha do baseline quando há quantidades informadas e perde na taxa 1: o intervalo fica cerca de 15 g mais largo, e as gavetas com pior cobertura são preparações mistas, farofa, macarrão e cerveja/vinho. A causa provável é o ruído isotrópico, um σ² único para gavetas de variabilidade muito diferente. A FA (Análise Fatorial) difere do PPCA só no ruído, que passa a ser diagonal, uma variância por gaveta. É o segundo modelo obrigatório da Parte 1 e testa diretamente essa causa.

## What Changes

- **FA-EM com dados faltantes**, do zero em NumPy e com derivação comentada, com a mesma estrutura do PPCA-EM: gavetas ausentes como condicionantes, regressão por gaveta no passo M e amostragem pela condicional. q = 5 de partida, a mesma do PPCA; a Parte 3 otimiza.
  - **Decisão tomada:** módulo novo `imputation/fa.py`, sem alterar o caminho numérico de `imputation/ppca.py`. Assim a junção aos resultados de 11/11 continua válida. O custo é repetir cerca de 15 linhas do passo M.
  - **Decisão tomada:** piso na variância de cada gaveta, ψ_k ≥ 0,1% da variância observada da gaveta, contra casos de Heywood (ψ_k → 0). As porções da POF vêm em degraus (o açúcar se concentra em 5, 20 e 30 g), o que favorece esse colapso. As gavetas que baterem no piso são registradas. O piso de 1% proposto antes foi descartado depois da medição (design, Decisão 3).
- **Medição de custo antes da rodada**, como no PPCA: um ajuste no treino completo, com o tempo e as iterações. O mesmo limite de 500 iterações; se a FA convergir mais devagar, o fato é registrado, e o limite não é aumentado às cegas.
- **Execução parcial do experimento:** a FA roda sozinha, e as linhas dela entram nos CSVs de 11/11.
  - **Decisão tomada:** rodar só a FA. O baseline reproduziu a rodada anterior sem diferença, e os geradores dependem da semente, da taxa e do fluxo, não do imputador. Repetir o baseline e o PPCA (~68 min) não traria informação nova.
  - A execução escolhe os imputadores. As linhas dos imputadores não executados e a referência da taxa 0 ficam como estavam; o resumo e as figuras são recalculados sobre a tabela juntada, com três painéis. A execução completa continua sendo o padrão, e é ela que entra na rodada final no teste.

## Capabilities

### New Capabilities

Nenhuma.

### Modified Capabilities

- `imputacao`: requisito da FA-EM com ausentes como condicionantes e piso no ruído; contrato, ausência de vazamento e reprodutibilidade valem também para a FA.
- `propagacao`: FA como terceiro imputador da curva; execução parcial com junção aos resultados existentes.

## Impact

- Código novo: `src/amaprob/imputation/fa.py`, `tests/test_fa.py`.
- Código alterado: `src/amaprob/experiments/propagacao.py` (FA, escolha de imputadores, junção, figuras com três painéis) e `tests/test_imputacao.py` (contrato parametrizado também na FA). `imputation/ppca.py` não muda.
- Sem dependências novas.
- Resultados: linhas da FA acrescentadas a `results/propagacao_*.csv`, `results/imputacao_*.csv` e `results/imputadores_ajuste.csv`; resumo e figuras regravados. As linhas do baseline e do PPCA não mudam, e o `git diff` confere isso.
- `AGENTS.md` §5 e §6 (FA implementada; experimento com três imputadores) e README.
