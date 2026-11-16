# Spec Delta

## MODIFIED Requirements

### Requirement: Curva por taxa e braço
O experimento SHALL rodar as taxas 0; 0,25; 0,5; 0,75 e 1 com as 3 sementes de máscara, para cada par de regressor (regressão linear bayesiana e SVGP) e imputador (porção média, PPCA e FA), ajustar no treino e medir na validação. O braço plug-in recalibrado SHALL existir só com a regressão linear bayesiana. Ele MUST NOT ler as refeições de teste (Regra 6). SHALL gravar as métricas por regressor, imputador, taxa, semente e braço, a leitura por número de quantidades ocultas, o resumo com média e desvio entre as sementes e as figuras de PICP e MPIW pela taxa, com um painel por regressor e imputador.

#### Scenario: Conjuntos usados
- **WHEN** o experimento roda
- **THEN** nenhuma linha com conjunto `teste` entra em ajuste nem em métrica

#### Scenario: Arquivos gravados
- **WHEN** o experimento termina
- **THEN** existem `results/propagacao_metricas.csv` (uma linha por regressor, imputador, taxa, semente e braço), `results/propagacao_por_ocultas.csv`, `results/propagacao_resumo.csv` e as figuras `results/figuras/propagacao_picp.png` e `results/figuras/propagacao_mpiw.png`

#### Scenario: Referências nas figuras
- **WHEN** as figuras são geradas
- **THEN** cada uma tem um painel por regressor e imputador, com uma linha por braço e o desvio entre as sementes, e a de PICP tem linhas horizontais em 90% e na PICP da taxa 0 do regressor do painel

#### Scenario: Mesmas máscaras e geradores entre imputadores
- **WHEN** dois imputadores rodam na mesma taxa e semente, com o mesmo regressor
- **THEN** usam a mesma máscara e geradores com a mesma semente, e na taxa 0 as métricas dos braços são idênticas entre os imputadores

#### Scenario: Recalibrado só na regressão linear
- **WHEN** o experimento roda com o SVGP
- **THEN** as linhas do SVGP têm só os braços plug-in e propagação

### Requirement: Referência da taxa 0 por número de ocultas
Para cada regressor, taxa e semente, a leitura por número de ocultas SHALL incluir a previsão do regressor sobre as gramas completas, agrupada pelo número de gavetas ocultas da máscara daquela taxa e semente. A referência SHALL ser calculada uma vez por regressor, taxa e semente, porque não depende do imputador.

#### Scenario: Linha de referência
- **WHEN** `results/propagacao_por_ocultas.csv` é gravado
- **THEN** cada regressor, taxa e semente tem as linhas do braço de referência da taxa 0, com o imputador marcado como nenhum, e os números de refeições por classe iguais aos dos braços da mesma taxa e semente

#### Scenario: Referência na taxa 0
- **WHEN** a taxa é 0
- **THEN** a linha de referência de cada regressor coincide com as linhas dos braços de cada imputador com esse regressor

### Requirement: Execução parcial com junção
O experimento SHALL aceitar a escolha dos regressores e dos imputadores da execução; sem escolha, roda todos. Numa execução parcial, as linhas dos pares executados SHALL ser acrescentadas aos resultados existentes, e as linhas dos pares não executados e das referências da taxa 0 já gravadas MUST NOT ser alteradas. As métricas da Parte 1 SHALL ser calculadas só quando a regressão linear bayesiana roda, porque não dependem do regressor. O resumo e as figuras SHALL ser recalculados sobre a tabela juntada.

#### Scenario: Só um imputador novo
- **WHEN** o experimento roda só com a FA sobre resultados que já têm o baseline e o PPCA
- **THEN** os CSVs passam a ter as linhas da FA, as linhas do baseline, do PPCA e da referência da taxa 0 ficam iguais às de antes, e não há linha duplicada por regressor, imputador, taxa, semente e braço

#### Scenario: Só um regressor novo
- **WHEN** o experimento roda só com o SVGP sobre resultados que já têm a regressão linear com os três imputadores
- **THEN** os CSVs de propagação passam a ter as linhas do SVGP, as linhas da regressão linear ficam iguais às de antes (com `regressor` = `blr`), e os CSVs da Parte 1 não mudam

#### Scenario: Mesmo resultado que a execução completa
- **WHEN** um imputador roda sozinho e numa execução completa, com as mesmas máscaras e sementes
- **THEN** as métricas dele são idênticas nos dois casos, porque os geradores dependem só da semente, da taxa e do fluxo
