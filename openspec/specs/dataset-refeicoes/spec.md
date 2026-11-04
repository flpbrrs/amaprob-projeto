# dataset-refeicoes Specification

## Purpose

Montar, a partir dos microdados da POF, as matrizes por refeição usadas por todas as etapas de modelagem e avaliação, e persisti-las em disco numa ordem de linhas fixa, à qual as máscaras se prendem.

## Requirements

### Requirement: Uma linha por refeição, em ordem fixa
O dataset SHALL ter uma linha por refeição montada conforme a spec `refeicoes`, em ordem crescente de `meal_id`, com a configuração de 29 gavetas da spec `gavetas`. Todos os campos SHALL estar alinhados a essa mesma ordem de linhas.

#### Scenario: Dimensão na versão 20230713
- **WHEN** o dataset é montado a partir da versão 20230713
- **THEN** ele tem 304.670 linhas e 29 gavetas, com os nomes das gavetas na ordem das colunas da matriz de gramas

#### Scenario: Campos alinhados
- **WHEN** o dataset é carregado
- **THEN** gramas, presença, carboidrato, `meal_id`, `person_id` e conjunto têm o mesmo número de linhas, e a linha i de cada campo se refere à mesma refeição

### Requirement: Gramas e presença por gaveta
O dataset SHALL trazer as gramas por gaveta (`G`), iguais às da matriz de gramas da spec `refeicoes`, sem transformação. A presença (`P`) SHALL valer 1 exatamente nas células com pelo menos um item da refeição naquela gaveta e 0 nas demais.

#### Scenario: Gaveta presente
- **WHEN** uma refeição tem itens numa gaveta
- **THEN** `P` vale 1 nessa célula e `G` traz a soma das gramas desses itens

#### Scenario: Presença coincide com gramas positivas na versão 20230713
- **WHEN** o dataset é montado a partir da versão 20230713
- **THEN** `P` vale 1 exatamente onde `G` é maior que 0, em 941.320 células

### Requirement: Carboidrato da refeição como alvo
O dataset SHALL trazer `y`, a soma de `CHOTOT` dos itens de cada refeição, em gramas. O dataset MUST NOT trazer outro nutriente nem qualquer coluna da medida caseira.

#### Scenario: Alvo da refeição
- **WHEN** uma refeição tem itens com `CHOTOT` de 30 g e 12 g
- **THEN** o `y` dessa refeição vale 42

#### Scenario: Refeições acima do limiar
- **WHEN** o dataset é carregado
- **THEN** todo `y` é maior ou igual a 10

### Requirement: Pessoa e conjunto do split por linha
Cada linha SHALL trazer o `person_id` da refeição e o rótulo do conjunto (`treino`, `validacao` ou `teste`), sorteado conforme a spec `split-por-pessoa` com semente 0 e frações 70/15/15.

#### Scenario: Pessoas por conjunto na versão 20230713
- **WHEN** o dataset é montado a partir da versão 20230713
- **THEN** o treino tem 32.299 pessoas, a validação 6.921 e o teste 6.921, e nenhum `person_id` aparece em mais de um conjunto

### Requirement: Persistência reprodutível
O dataset SHALL ser gravado em `data/processed/`, fora do controle de versão. Montá-lo de novo a partir dos mesmos microdados e da mesma classificação SHALL produzir conteúdo idêntico. Carregar o dataset gravado SHALL devolver os mesmos valores montados.

#### Scenario: Ida e volta do disco
- **WHEN** o dataset é gravado e depois carregado
- **THEN** cada campo carregado é igual ao montado, valor a valor

#### Scenario: Nova montagem
- **WHEN** o dataset é montado duas vezes a partir dos mesmos arquivos
- **THEN** as duas montagens são idênticas
