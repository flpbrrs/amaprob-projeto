# Spec Delta

## MODIFIED Requirements

### Requirement: Curva por taxa e braço
O experimento SHALL rodar as taxas 0; 0,25; 0,5; 0,75 e 1 com as 3 sementes de máscara, para cada par de regressor (regressão linear bayesiana e SVGP) e imputador (porção média, PPCA, FA e as versões dos três com média condicional), ajustar no treino e medir no conjunto escolhido na execução: a validação (padrão) ou o teste. O braço plug-in recalibrado SHALL existir só com a regressão linear bayesiana. Na validação, o experimento MUST NOT ler as refeições de teste (Regra 6); no teste, as refeições de teste MUST NOT entrar em nenhum ajuste. SHALL gravar as métricas por regressor, imputador, taxa, semente e braço, a leitura por número de quantidades ocultas, o resumo com média e desvio entre as sementes e as figuras de PICP e MPIW pela taxa, com um painel por regressor e imputador.

#### Scenario: Conjuntos usados
- **WHEN** o experimento roda na validação
- **THEN** nenhuma linha com conjunto `teste` entra em ajuste nem em métrica

#### Scenario: Conjuntos usados no teste
- **WHEN** o experimento roda no teste
- **THEN** imputadores, imputadores das dobras e regressores são ajustados só com refeições do treino, as métricas são calculadas só com refeições do teste, e nenhuma refeição da validação entra em ajuste nem em métrica

#### Scenario: Arquivos gravados
- **WHEN** o experimento termina na validação
- **THEN** existem `results/propagacao_metricas.csv` (uma linha por regressor, imputador, taxa, semente e braço), `results/propagacao_por_ocultas.csv`, `results/propagacao_resumo.csv` e as figuras `results/figuras/propagacao_picp.png` e `results/figuras/propagacao_mpiw.png`

#### Scenario: Referências nas figuras
- **WHEN** as figuras são geradas
- **THEN** cada uma tem um painel por regressor e imputador, com uma linha por braço e o desvio entre as sementes, a de PICP tem linhas horizontais em 90% e na PICP da taxa 0 do regressor do painel, e a legenda nomeia o conjunto medido

#### Scenario: Mesmas máscaras e geradores entre imputadores
- **WHEN** dois imputadores rodam na mesma taxa e semente, com o mesmo regressor
- **THEN** usam a mesma máscara e geradores com a mesma semente, e na taxa 0 as métricas dos braços são idênticas entre os imputadores

#### Scenario: Recalibrado só na regressão linear
- **WHEN** o experimento roda com o SVGP
- **THEN** as linhas do SVGP têm só os braços plug-in e propagação

### Requirement: Execução parcial com junção
O experimento SHALL aceitar a escolha dos regressores e dos imputadores da execução; sem escolha, roda todos. Numa execução parcial, as linhas dos pares executados SHALL ser acrescentadas aos resultados existentes do mesmo conjunto, e as linhas dos pares não executados e das referências da taxa 0 já gravadas MUST NOT ser alteradas. As métricas da Parte 1 SHALL ser calculadas só quando a regressão linear bayesiana roda, porque não dependem do regressor. O resumo e as figuras SHALL ser recalculados sobre a tabela juntada.

#### Scenario: Só um imputador novo
- **WHEN** o experimento roda só com a FA sobre resultados que já têm o baseline e o PPCA
- **THEN** os CSVs passam a ter as linhas da FA, as linhas do baseline, do PPCA e da referência da taxa 0 ficam iguais às de antes, e não há linha duplicada por regressor, imputador, taxa, semente e braço

#### Scenario: Só um regressor novo
- **WHEN** o experimento roda só com o SVGP sobre resultados que já têm a regressão linear com os três imputadores
- **THEN** os CSVs de propagação passam a ter as linhas do SVGP, as linhas da regressão linear ficam iguais às de antes (com `regressor` = `blr`), e os CSVs da Parte 1 não mudam

#### Scenario: Mesmo resultado que a execução completa
- **WHEN** um imputador roda sozinho e numa execução completa, com as mesmas máscaras e sementes
- **THEN** as métricas dele são idênticas nos dois casos, porque os geradores dependem só da semente, da taxa e do fluxo

#### Scenario: Junção restrita ao conjunto
- **WHEN** uma execução parcial roda no teste
- **THEN** as linhas são juntadas só aos arquivos de `results/teste/`, e os arquivos da validação em `results/` ficam iguais aos de antes

## ADDED Requirements

### Requirement: Saída por conjunto
Na validação, o experimento SHALL gravar em `results/` e `results/figuras/`. No teste, SHALL gravar todos os arquivos (métricas, leitura por ocultas, resumo, Parte 1, tempos de ajuste e figuras) em `results/teste/` e `results/teste/figuras/`, com os mesmos nomes, e MUST NOT criar nem alterar arquivo fora dessa pasta.

#### Scenario: Arquivos do teste
- **WHEN** o experimento termina no teste
- **THEN** existem `results/teste/propagacao_metricas.csv`, `results/teste/propagacao_por_ocultas.csv`, `results/teste/propagacao_resumo.csv` e as figuras em `results/teste/figuras/`, e nenhum arquivo de `results/` fora de `results/teste/` mudou

### Requirement: Configuração adotada na validação
Os imputadores e os regressores SHALL ser construídos com os hiperparâmetros de `configs/hiperparametros.json` (q e α dos imputadores, M do SVGP), nos dois conjuntos. Um modelo sem entrada no arquivo SHALL usar o valor de partida. Os rótulos das figuras SHALL mostrar o valor usado por cada modelo.

#### Scenario: Valores lidos do arquivo
- **WHEN** o arquivo traz q = 12 e α = 17,8 para o PPCA condicional, q = 12 para o PPCA e M = 288 para o SVGP
- **THEN** o experimento ajusta esses modelos com esses valores, e os rótulos mostram q = 12 e M = 288

#### Scenario: Modelo sem entrada
- **WHEN** a porção média não tem entrada no arquivo
- **THEN** ela é construída sem botões, como antes

### Requirement: Iterações do EM no registro de ajuste
O registro do ajuste dos imputadores SHALL trazer o número de iterações do EM no treino completo e em cada dobra, e o teto de iterações, para que uma parada no teto seja identificável durante a execução e nos resultados.

#### Scenario: Parada no teto
- **WHEN** o ajuste de um imputador usa todas as iterações permitidas
- **THEN** o registro mostra o número de iterações igual ao teto para esse ajuste
