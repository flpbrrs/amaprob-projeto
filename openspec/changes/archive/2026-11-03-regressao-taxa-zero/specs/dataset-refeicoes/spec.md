# Spec Delta

## MODIFIED Requirements

### Requirement: Uma linha por refeição, em ordem fixa
O dataset SHALL ter uma linha por refeição montada conforme a spec `refeicoes`, em ordem crescente de `meal_id`, com a configuração de 29 gavetas da spec `gavetas`. Todos os campos SHALL estar alinhados a essa mesma ordem de linhas.

#### Scenario: Dimensão na versão 20230713
- **WHEN** o dataset é montado a partir da versão 20230713
- **THEN** ele tem 304.670 linhas e 29 gavetas, com os nomes das gavetas na ordem das colunas da matriz de gramas

#### Scenario: Campos alinhados
- **WHEN** o dataset é carregado
- **THEN** gramas, presença, contexto, carboidrato, `meal_id`, `person_id` e conjunto têm o mesmo número de linhas, e a linha i de cada campo se refere à mesma refeição

## ADDED Requirements

### Requirement: Contexto da refeição
O dataset SHALL trazer a matriz de contexto `C`, com os nomes das colunas, montada só com campos do arquivo de consumo alimentar. As colunas são: ocasião (`V9017`, 5 indicadoras), hora (`V9015` como seno e cosseno de 2π·h/24), macrorregião (primeiro dígito da `UF`, 5 indicadoras) e `log1p` da renda (`RENDA_TOTAL`, com valores negativos levados a 0).

#### Scenario: Colunas do contexto
- **WHEN** o dataset é carregado
- **THEN** `C` tem 13 colunas, sem valores ausentes, e os nomes das colunas acompanham a matriz

#### Scenario: Indicadoras
- **WHEN** uma refeição é um almoço (`V9017` = 2) de uma pessoa da região Nordeste (`UF` iniciada por 2)
- **THEN** a indicadora do almoço e a do Nordeste valem 1 e as demais indicadoras de ocasião e de região valem 0

#### Scenario: Hora cíclica
- **WHEN** duas refeições têm horários `23H` e `0H`
- **THEN** a distância entre os seus pares (seno, cosseno) é a mesma de duas refeições com `11H` e `12H`

#### Scenario: Máscaras continuam válidas
- **WHEN** o dataset é regravado com o contexto e as máscaras gravadas antes são carregadas com ele
- **THEN** a carga conclui sem erro de alinhamento
