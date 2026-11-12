# propagacao Specification

## Purpose
Levar a incerteza da imputação até a previsão de carboidrato e comparar três formas de fazê-lo (plug-in, plug-in recalibrado e propagação), sobre os contratos `Imputer` e `ProbRegressor`, em função da taxa de quantidades ocultas.

## Requirements

### Requirement: Ajuste no treino completo
O imputador e o regressor de referência SHALL ser ajustados uma vez, com as gramas completas do treino, e usados em todas as taxas e sementes. As máscaras SHALL definir as quantidades ocultas só nas refeições em que se mede; no treino, só para o braço recalibrado.

#### Scenario: Taxa 1 com imputador ajustável
- **WHEN** o experimento roda na taxa 1, em que todas as gavetas presentes da validação ficam ocultas
- **THEN** o imputador já está ajustado com quantidades observadas e as amostras são finitas

### Requirement: Plug-in pela média em gramas
O braço plug-in SHALL preencher cada quantidade oculta com a média, em gramas, das S amostras do imputador convertidas por `expm1`, e prever uma vez com o regressor de referência.

#### Scenario: Média em gramas, e não do log
- **WHEN** as amostras em `log1p` de uma célula oculta são conhecidas
- **THEN** o valor usado pelo plug-in é a média de `expm1` das amostras, e não `expm1` da média

### Requirement: Propagação como mistura
O braço de propagação SHALL prever com o regressor de referência para cada uma das S amostras, convertidas por `expm1`, e SHALL representar a preditiva como a mistura de pesos iguais `(1/S) Σ N(μ_s, σ²_s)`. Plug-in e propagação SHALL usar as mesmas amostras.

#### Scenario: Mesma média com regressor linear
- **WHEN** plug-in e propagação usam a regressão linear bayesiana e as mesmas amostras
- **THEN** a média da mistura é igual à média do plug-in em toda refeição, até o erro de ponto flutuante

#### Scenario: Variância total
- **WHEN** a preditiva da propagação é resumida em média e variância
- **THEN** a variância é a média das variâncias dos componentes mais a variância das médias dos componentes

### Requirement: Plug-in recalibrado com ajuste cruzado
O braço plug-in recalibrado SHALL usar um regressor ajustado com as gramas do treino mascarado na mesma taxa e semente, imputadas pela média em gramas, e SHALL receber na medida a mesma entrada do plug-in. A imputação do treino SHALL usar ajuste cruzado por pessoa em 5 dobras: o imputador que preenche as refeições de uma dobra MUST NOT ser ajustado com refeições das pessoas dessa dobra.

#### Scenario: Dobras por pessoa
- **WHEN** as dobras do treino são formadas
- **THEN** cada pessoa do treino está em exatamente uma dobra, e todas as refeições de uma pessoa ficam na mesma dobra

#### Scenario: Nenhum imputador vê o que imputa
- **WHEN** as gramas das células ocultas de uma dobra são alteradas e a imputação cruzada é refeita com os mesmos geradores
- **THEN** os valores imputados para essa dobra não mudam

### Requirement: Imputadores das dobras ajustados uma vez
Os imputadores das dobras SHALL ser ajustados com o treino completo das outras dobras, uma vez por imputador, e reaproveitados em todas as taxas e sementes, porque o ajuste sem máscara não depende delas.

#### Scenario: Dobras ajustadas uma vez
- **WHEN** o experimento roda todas as taxas e sementes para um imputador
- **THEN** são feitos exatamente 5 ajustes de imputador para as dobras, além do ajuste no treino completo

### Requirement: Sanidade na taxa 0
Na taxa 0, os três braços SHALL coincidir com a previsão direta do regressor de referência sobre as gramas completas.

#### Scenario: Braços iguais sem ocultos
- **WHEN** o experimento roda na taxa 0
- **THEN** média e variância dos três braços coincidem com as do regressor de referência em toda refeição da validação, até o erro de ponto flutuante

### Requirement: Curva por taxa e braço
O experimento SHALL rodar as taxas 0; 0,25; 0,5; 0,75 e 1 com as 3 sementes de máscara, para cada imputador (porção média e PPCA) na mesma execução, ajustar no treino e medir na validação. Ele MUST NOT ler as refeições de teste (Regra 6). SHALL gravar as métricas por imputador, taxa, semente e braço, a leitura por número de quantidades ocultas, o resumo com média e desvio entre as sementes e as figuras de PICP e MPIW pela taxa, com um painel por imputador.

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

### Requirement: Número fixo de amostras
A propagação SHALL usar S = 100 amostras do imputador em todas as taxas, sementes e imputadores, sem repetir a comparação com S = 500. O registro dessa comparação (`results/propagacao_S.csv`) MUST NOT ser regravado pelo experimento.

#### Scenario: S na tabela principal
- **WHEN** o experimento termina
- **THEN** toda linha de `results/propagacao_metricas.csv` tem S = 100 e `results/propagacao_S.csv` continua igual ao de antes da execução

### Requirement: Referência da taxa 0 por número de ocultas
Para cada taxa e semente, a leitura por número de ocultas SHALL incluir a previsão do regressor de referência sobre as gramas completas, agrupada pelo número de gavetas ocultas da máscara daquela taxa e semente. A referência SHALL ser calculada uma vez por taxa e semente, porque não depende do imputador.

#### Scenario: Linha de referência
- **WHEN** `results/propagacao_por_ocultas.csv` é gravado
- **THEN** cada taxa e semente tem as linhas do braço de referência da taxa 0, com o imputador marcado como nenhum, e os números de refeições por classe iguais aos dos braços da mesma taxa e semente

#### Scenario: Referência na taxa 0
- **WHEN** a taxa é 0
- **THEN** a linha de referência coincide com as linhas dos três braços de cada imputador
