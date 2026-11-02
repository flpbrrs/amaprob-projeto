# refeicoes Specification

## Purpose

Agrupar os itens da POF (Pesquisa de Orçamentos Familiares) em refeições que pedem insulina e expor, por refeição, as gramas consumidas em cada gaveta. A refeição é a unidade de análise de todas as etapas seguintes.

## Requirements

### Requirement: Pré-condições da entrada
A montagem de refeições SHALL receber itens com `V9015` preenchido e com todo `V9001` presente no mapeamento de gavetas. O código não trata a violação dessas condições. Ambas foram verificadas nos microdados da versão 20230713.

#### Scenario: Entrada da versão 20230713
- **WHEN** a montagem recebe a tabela lida da versão 20230713 e as gavetas montadas a partir da classificação versionada
- **THEN** nenhum item tem `V9015` ausente e todo item recebe uma gaveta

### Requirement: Identificador de pessoa
Cada item SHALL receber `person_id`: os valores de `COD_UPA`, `NUM_DOM`, `NUM_UC` e `COD_INFORMANTE`, nessa ordem, como texto e unidos por `-`.

#### Scenario: Composição do identificador
- **WHEN** um item tem `COD_UPA` = 110000016, `NUM_DOM` = 2, `NUM_UC` = 1 e `COD_INFORMANTE` = 1
- **THEN** o seu `person_id` é `110000016-2-1-1`

### Requirement: Ocasiões que pedem insulina
SHALL entrar na montagem apenas os itens com ocasião (`V9017`) igual a 1 (café da manhã), 2 (almoço), 3 (lanche), 4 (jantar) ou 5 (ceia).

#### Scenario: Ocasião excluída
- **WHEN** um item tem `V9017` igual a 6 (outra) ou 9 (não determinado)
- **THEN** o item não aparece em nenhuma refeição

### Requirement: Definição de refeição
Os itens com o mesmo `person_id`, `QUADRO`, `V9017` e `V9015` SHALL formar uma refeição, identificada por um `meal_id` inteiro. `V9015` registra só a hora cheia. O `meal_id` distingue refeições dentro de uma mesma montagem, sem outro significado.

#### Scenario: Mesma pessoa, ocasiões diferentes na mesma hora
- **WHEN** dois itens da mesma pessoa e do mesmo `QUADRO` têm o mesmo `V9015` e ocasiões diferentes
- **THEN** eles ficam em refeições diferentes

#### Scenario: Mesma ocasião em horas diferentes
- **WHEN** dois itens da mesma pessoa, do mesmo `QUADRO` e da mesma ocasião têm `V9015` diferente
- **THEN** eles ficam em refeições diferentes

#### Scenario: Mesma hora e ocasião em dias de registro diferentes
- **WHEN** dois itens da mesma pessoa, ocasião e `V9015` têm `QUADRO` diferente (72 e 73, dia 1 e dia 2 a verificar)
- **THEN** eles ficam em refeições diferentes

#### Scenario: Refeição de uma só pessoa
- **WHEN** as refeições são montadas
- **THEN** nenhum `meal_id` reúne itens de mais de um `person_id`

### Requirement: Mínimo de 10 g de carboidrato
SHALL permanecer apenas as refeições cuja soma de `CHOTOT` seja de pelo menos 10 g. Uma refeição permanece ou sai com todos os seus itens.

#### Scenario: Refeição no limiar
- **WHEN** a soma de `CHOTOT` de uma refeição é exatamente 10 g
- **THEN** a refeição permanece

#### Scenario: Refeição abaixo do limiar
- **WHEN** a soma de `CHOTOT` de uma refeição é menor que 10 g
- **THEN** nenhum item dessa refeição permanece

#### Scenario: Volume na versão 20230713
- **WHEN** a montagem recebe a tabela lida da versão 20230713
- **THEN** permanecem 968.908 itens em 304.670 refeições de 46.141 pessoas

### Requirement: Gaveta de cada item
Cada item que permanece SHALL receber a gaveta do seu `V9001` no mapeamento fornecido por quem chama. A escolha de gavetas não altera quais refeições permanecem.

#### Scenario: Troca da configuração de gavetas
- **WHEN** a mesma tabela de itens é montada com duas configurações de gavetas diferentes
- **THEN** o conjunto de refeições e de `meal_id` é o mesmo, e só a coluna de gaveta muda

### Requirement: Matriz de gramas por gaveta
A matriz SHALL ter uma linha por `meal_id`, em ordem crescente, e uma coluna por gaveta presente nos itens, ordenadas pelo ponto de código dos nomes (não pela ordem alfabética do português). Cada célula SHALL ser a soma de `QTD` dos itens da refeição naquela gaveta, em gramas, sem transformação, e 0 quando a gaveta não aparece na refeição.

#### Scenario: Itens da mesma gaveta
- **WHEN** uma refeição tem dois itens na mesma gaveta, com 100 g e 50 g
- **THEN** a célula dessa refeição e dessa gaveta vale 150

#### Scenario: Gaveta ausente na refeição
- **WHEN** uma gaveta não aparece em uma refeição
- **THEN** a célula correspondente vale 0

#### Scenario: Gaveta sem nenhum item
- **WHEN** uma gaveta da configuração não aparece em nenhum item
- **THEN** ela não vira coluna da matriz

#### Scenario: Dimensão com a configuração padrão
- **WHEN** a matriz é montada da versão 20230713, sem nenhum grupo dividido
- **THEN** ela tem 304.670 linhas e 21 colunas
