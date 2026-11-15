# Spec Delta

## ADDED Requirements

### Requirement: Execução parcial com junção
O experimento SHALL aceitar a escolha dos imputadores da execução; sem escolha, roda todos. Numa execução parcial, as linhas dos imputadores executados SHALL ser acrescentadas aos resultados existentes, e as linhas dos imputadores não executados e da referência da taxa 0 MUST NOT ser alteradas. O resumo e as figuras SHALL ser recalculados sobre a tabela juntada.

#### Scenario: Só um imputador novo
- **WHEN** o experimento roda só com a FA sobre resultados que já têm o baseline e o PPCA
- **THEN** os CSVs passam a ter as linhas da FA, as linhas do baseline, do PPCA e da referência da taxa 0 ficam iguais às de antes, e não há linha duplicada por imputador, taxa, semente e braço

#### Scenario: Mesmo resultado que a execução completa
- **WHEN** um imputador roda sozinho e numa execução completa, com as mesmas máscaras e sementes
- **THEN** as métricas dele são idênticas nos dois casos, porque os geradores dependem só da semente, da taxa e do fluxo

## MODIFIED Requirements

### Requirement: Curva por taxa e braço
O experimento SHALL rodar as taxas 0; 0,25; 0,5; 0,75 e 1 com as 3 sementes de máscara, para cada imputador (porção média, PPCA e FA), ajustar no treino e medir na validação. Ele MUST NOT ler as refeições de teste (Regra 6). SHALL gravar as métricas por imputador, taxa, semente e braço, a leitura por número de quantidades ocultas, o resumo com média e desvio entre as sementes e as figuras de PICP e MPIW pela taxa, com um painel por imputador.

#### Scenario: Conjuntos usados
- **WHEN** o experimento roda
- **THEN** nenhuma linha com conjunto `teste` entra em ajuste nem em métrica

#### Scenario: Arquivos gravados
- **WHEN** o experimento termina
- **THEN** existem `results/propagacao_metricas.csv` (uma linha por imputador, taxa, semente e braço), `results/propagacao_por_ocultas.csv`, `results/propagacao_resumo.csv` e as figuras `results/figuras/propagacao_picp.png` e `results/figuras/propagacao_mpiw.png`

#### Scenario: Referências nas figuras
- **WHEN** as figuras são geradas
- **THEN** cada uma tem um painel por imputador, com uma linha por braço e o desvio entre as sementes, e a de PICP tem linhas horizontais em 90% e na PICP da taxa 0

#### Scenario: Mesmas máscaras e geradores entre imputadores
- **WHEN** dois imputadores rodam na mesma taxa e semente
- **THEN** usam a mesma máscara e geradores com a mesma semente, e na taxa 0 as métricas dos braços são idênticas entre os imputadores
