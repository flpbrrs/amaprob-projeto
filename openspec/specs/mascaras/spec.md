# mascaras Specification

## Purpose

Definir quais quantidades de gavetas presentes ficam ocultas em cada taxa e semente do protocolo experimental, e persistir essas máscaras antes de qualquer modelagem, para que todos os modelos sejam avaliados sobre as mesmas quantidades ocultas.

## Requirements

### Requirement: Máscara só sobre gavetas presentes
Uma máscara `M` SHALL ter a mesma forma da presença `P` e MUST NOT marcar como oculta uma célula em que a gaveta está ausente (`M ⊆ P`). A presença em si nunca é ocultada.

#### Scenario: Gaveta ausente
- **WHEN** `P` vale 0 numa célula
- **THEN** `M` vale 0 nessa célula, em qualquer taxa e semente

### Requirement: Sorteio independente por gaveta presente
Para uma taxa r, cada célula presente SHALL ficar oculta com probabilidade r, de forma independente das demais células, dos valores de gramas e carboidrato e do conjunto do split (mecanismo MCAR). O mascaramento é por gaveta: oculta todos os itens dela de uma vez.

#### Scenario: Taxa observada em dados sintéticos
- **WHEN** a máscara é gerada sobre uma presença sintética com n células presentes
- **THEN** a fração de células presentes ocultas difere de r por no máximo 4 desvios-padrão binomiais, √(r(1−r)/n)

#### Scenario: Taxa observada na versão 20230713
- **WHEN** as máscaras são geradas sobre o dataset da versão 20230713
- **THEN** em cada taxa e semente, a fração das 941.320 células presentes que fica oculta difere de r por menos de 0,002

#### Scenario: Refeição com uma só gaveta
- **WHEN** uma refeição tem uma única gaveta presente e a taxa é 0,25
- **THEN** essa gaveta fica oculta com probabilidade 0,25, como qualquer outra célula presente

#### Scenario: Mesmo protocolo em todos os conjuntos
- **WHEN** as máscaras são geradas
- **THEN** treino, validação e teste são mascarados pelo mesmo sorteio, sem regra própria por conjunto

### Requirement: Máscaras aninhadas entre taxas
Para uma mesma semente, toda célula oculta numa taxa r SHALL estar oculta em qualquer taxa maior que r.

#### Scenario: Taxas consecutivas
- **WHEN** as máscaras de uma semente são comparadas nas taxas 0,25 e 0,5
- **THEN** toda célula oculta em 0,25 também está oculta em 0,5

#### Scenario: Taxa zero
- **WHEN** a taxa é 0
- **THEN** nenhuma célula fica oculta

#### Scenario: Taxa um
- **WHEN** a taxa é 1
- **THEN** a máscara é igual a `P`

### Requirement: Grade do protocolo e reprodutibilidade
As máscaras SHALL ser geradas para as taxas 0, 0,25, 0,5, 0,75 e 1 e para as sementes 0, 1 e 2. A máscara SHALL depender só de `P`, da taxa e da semente.

#### Scenario: Mesma semente
- **WHEN** as máscaras são geradas duas vezes com a mesma `P` e a mesma semente
- **THEN** as duas são idênticas

#### Scenario: Sementes diferentes
- **WHEN** as máscaras da taxa 0,5 são geradas com sementes diferentes
- **THEN** elas diferem em alguma célula

### Requirement: Persistência alinhada ao dataset
As máscaras SHALL ser gravadas em `data/processed/`, em arquivo separado do dataset, junto com o `meal_id` de cada linha. Carregar máscaras cujo `meal_id` não coincide, na ordem, com o do dataset MUST falhar com erro explícito.

#### Scenario: Alinhamento conferido
- **WHEN** as máscaras gravadas são carregadas com o dataset que as gerou
- **THEN** a carga devolve as máscaras de cada taxa e semente, linha a linha alinhadas ao dataset

#### Scenario: Desalinhamento
- **WHEN** as máscaras são carregadas com um dataset de outra ordem ou outro conjunto de refeições
- **THEN** a carga falha com mensagem que indica o desalinhamento
