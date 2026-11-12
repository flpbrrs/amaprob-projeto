# Proposal

> Status: **pronta para implementar**. Escopo, decisões, design, specs e tarefas fechados.

## Why

O resultado principal (plug-in × propagação) existe só com o baseline, que ignora coocorrência e contexto; a largura do intervalo pode estar exagerando a incerteza da imputação. O PPCA (Análise de Componentes Principais Probabilística) é o primeiro imputador que explora a correlação entre gavetas: mostra se o intervalo estreita mantendo a calibração e abre a Parte 1, que ainda não tem métrica própria.

## What Changes

- **PPCA-EM com dados faltantes**, do zero em NumPy, com derivação comentada e dimensão latente q = 5.
  - **Decisão tomada:** gavetas ausentes como condicionantes (opção B). Células com `P = 0` ficam fora da verossimilhança, e o modelo aprende o tamanho das porções dado que a gaveta está presente, como o baseline. Motivos: 89% das células são zero; toda gaveta imputada está presente (`M ⊆ P`); com zeros, as porções imputadas sairiam puxadas para zero.
  - **Decisão tomada:** q = 5 como valor de partida; a Parte 3 otimiza q.
  - O passo E condiciona cada refeição só nas gavetas observadas; a amostragem sorteia o latente pela posterior e depois as gavetas ocultas.
  - A semente vai no construtor, porque o contrato `fit` não recebe `rng`.
- **Medição de custo antes do experimento:** tempo por iteração, iterações até convergir e tempo de 100 amostras da validação. **Ponto de decisão:** se um ajuste passar de 5 minutos, a execução para e se decide entre subamostra e agrupamento por padrão de presença.
- **Métricas da Parte 1** nas células ocultas da validação, por taxa > 0: MAE em gramas e em log, NLL e CRPS em log e PICP/MPIW do intervalo de 90%, globais e por gaveta.
  - **Decisão tomada:** tudo a partir das S amostras do imputador (NLL pela gaussiana com média e variância das amostras de cada célula; intervalo pelos quantis empíricos). Vale para qualquer imputador sem mudar o contrato `Imputer`; o custo é um pequeno erro de Monte Carlo. Alternativa descartada: um método de momentos condicionais exatos no contrato, que obrigaria todo imputador a implementá-lo.
  - Por gaveta, porque `backlog/decisions.md` pede para comparar o erro da gaveta de açúcar com o das outras.
- **Experimento de propagação com o imputador como dimensão:** baseline e PPCA na mesma execução, com as mesmas máscaras e geradores.
  - **Decisão tomada:** o baseline roda de novo junto, para que todos os CSVs saiam de uma só versão do código.
  - **Decisão tomada:** figuras com um painel por imputador, lado a lado, e as cores de braço atuais.
- **Ajuste cruzado sem retrabalho:** os 5 imputadores das dobras do braço recalibrado são ajustados uma vez por imputador e reaproveitados em todas as taxas e sementes. Eles usam o treino completo, sem máscara, e não dependem da taxa nem da semente; só a amostragem usa a máscara. Com o PPCA, isso evita 75 ajustes por execução (5 dobras × 5 taxas × 3 sementes).
- **Ajustes adiados da rodada anterior** (`backlog/decisions.md`):
  - S = 100 fixo, sem repetir a comparação com S = 500; `results/propagacao_S.csv` fica como registro.
  - Referência da taxa 0 por número de ocultas em `propagacao_por_ocultas.csv`, uma vez por taxa e semente, porque não depende do imputador.
- **Documentação:** `AGENTS.md` §5 e §12 (decisão dos ausentes fechada; métricas da Parte 1); `backlog/decisions.md` (entrada da próxima rodada encerrada; extensão B+ registrada).

## Capabilities

### New Capabilities

Nenhuma.

### Modified Capabilities

- `imputacao`: requisito do PPCA-EM com ausentes como condicionantes; contrato, ausência de vazamento e reprodutibilidade valem também para o PPCA.
- `propagacao`: S = 100 fixo no lugar da escolha de S; curva com o imputador como dimensão; dobras ajustadas uma vez; referência da taxa 0 por número de ocultas.
- `metricas`: métricas de imputação nas células ocultas, a partir das amostras.

## Impact

- Código novo: `src/amaprob/imputation/ppca.py`, `tests/test_ppca.py`.
- Código alterado: `src/amaprob/regression/propagation.py` (dobras pré-ajustadas), `src/amaprob/experiments/propagacao.py`, `src/amaprob/evaluation/metricas.py`, `tests/test_imputacao.py` (parametrizado nos dois imputadores), `tests/test_propagacao.py`, `tests/test_metricas.py`.
- Sem dependências novas.
- Resultados: CSVs e figuras de propagação regravados com os dois imputadores; novos `results/imputacao_metricas.csv`, `results/imputacao_por_gaveta.csv` e `results/imputadores_ajuste.csv`.
- `AGENTS.md` §5, §6 e §12; `backlog/decisions.md`.
