# Spec Delta

## MODIFIED Requirements

### Requirement: Curva por taxa e braço
O experimento SHALL rodar as taxas 0; 0,25; 0,5; 0,75 e 1 com as 3 sementes de máscara, para cada par de regressor (regressão linear bayesiana e SVGP) e imputador (porção média, PPCA, FA e as versões dos três com média condicional), ajustar no treino e medir na validação. O braço plug-in recalibrado SHALL existir só com a regressão linear bayesiana. Ele MUST NOT ler as refeições de teste (Regra 6). SHALL gravar as métricas por regressor, imputador, taxa, semente e braço, a leitura por número de quantidades ocultas, o resumo com média e desvio entre as sementes e as figuras de PICP e MPIW pela taxa, com um painel por regressor e imputador.

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
