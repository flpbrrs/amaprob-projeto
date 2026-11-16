# Spec Delta

## ADDED Requirements

### Requirement: Média condicional no PPCA e na FA
O PPCA e a FA SHALL aceitar, na construção, a opção de média condicional: a média de cada gaveta passa a ser uma função linear de `P` e `C` (padronizados pelo ajuste), mais um intercepto, com penalização ridge fixa nos coeficientes de `P` e `C`. As amostras de uma refeição SHALL usar a média condicional dela, inclusive quando todas as gavetas presentes estão ocultas.

#### Scenario: Redução ao modelo atual
- **WHEN** a média condicional é ajustada com `P` e `C` constantes em todas as refeições
- **THEN** os parâmetros e as amostras coincidem com os do mesmo modelo sem média condicional, a menos de 1e-6 relativo

#### Scenario: Recuperação em dados sintéticos
- **WHEN** o modelo com média condicional é ajustado com dados gerados com a média de cada gaveta dependendo de `P` e `C`, com metade das células ausentes ao acaso
- **THEN** os coeficientes da média se aproximam dos verdadeiros e, na taxa 1, a média das amostras de cada refeição fica mais perto da média verdadeira dela que a do modelo sem média condicional

#### Scenario: Colunas colineares
- **WHEN** `C` tem indicadoras que somam 1 e há gavetas que nunca aparecem juntas no conjunto de ajuste
- **THEN** o ajuste termina com parâmetros e amostras finitos

#### Scenario: Ausentes fora da verossimilhança
- **WHEN** os valores de `X` nas células com `P = 0` são trocados por quaisquer outros e o ajuste é refeito com a mesma semente
- **THEN** os parâmetros ajustados e as amostras são idênticos

#### Scenario: Reprodutível pela semente
- **WHEN** dois modelos com média condicional, com a mesma q e a mesma semente, são ajustados nos mesmos dados
- **THEN** os parâmetros ajustados são idênticos

### Requirement: Porção média condicional como controle
A porção média SHALL aceitar a mesma opção de média condicional: a média de cada gaveta como função linear de `P` e `C` pela mesma regressão ridge, sem fatores latentes, e o desvio pelo resíduo dessa regressão.

#### Scenario: Redução à porção média
- **WHEN** a porção média condicional é ajustada com `P` e `C` constantes em todas as refeições
- **THEN** a média, o desvio e as amostras coincidem com os da porção média

#### Scenario: Recupera a média condicional
- **WHEN** a porção média condicional é ajustada com dados gerados com a média de cada gaveta dependendo de `P` e `C`
- **THEN** a média ajustada se aproxima da verdadeira e, na taxa 1, as amostras ficam mais perto da média verdadeira que as da porção média

### Requirement: Modelo atual intacto sem a média condicional
Sem a opção de média condicional, o ajuste e as amostras da porção média, do PPCA e da FA MUST NOT mudar.

#### Scenario: Opção desligada
- **WHEN** o PPCA e a FA são construídos sem a opção de média condicional
- **THEN** os testes atuais do PPCA e da FA passam sem mudança, e os parâmetros e as amostras são os mesmos de antes da opção existir

### Requirement: Convergência do EM com média condicional
Com a média condicional, o EM SHALL seguir a mesma regra de parada do modelo sem ela, aplicada à log-verossimilhança dos observados menos a penalização ridge. Esse objetivo MUST NOT diminuir entre iterações, e o histórico dele SHALL ficar disponível após o ajuste.

#### Scenario: Objetivo penalizado monótono
- **WHEN** o PPCA e a FA com média condicional são ajustados em dados com células ausentes
- **THEN** o objetivo registrado não diminui de uma iteração para a seguinte, a menos de 1e-8 relativo

## MODIFIED Requirements

### Requirement: Contrato comum dos imputadores
Os requisitos de entrada sem os valores ocultos, de saída em amostras e de nenhum vazamento de `X[M]` SHALL valer para todos os imputadores do projeto, inclusive os imputadores com média condicional, e o mesmo conjunto de testes do contrato SHALL rodar sobre cada um deles.

#### Scenario: Contrato do PPCA
- **WHEN** os testes do contrato rodam com o PPCA
- **THEN** observados preservados, ausentes em 0, ocultos finitos e não negativos, reprodutibilidade pelo gerador e ausência de vazamento passam como no baseline

#### Scenario: Contrato da FA
- **WHEN** os testes do contrato rodam com a FA
- **THEN** observados preservados, ausentes em 0, ocultos finitos e não negativos, reprodutibilidade pelo gerador e ausência de vazamento passam como no baseline

#### Scenario: Contrato com média condicional
- **WHEN** os testes do contrato rodam com a porção média, o PPCA e a FA com média condicional
- **THEN** observados preservados, ausentes em 0, ocultos finitos e não negativos, reprodutibilidade pelo gerador e ausência de vazamento passam como no baseline
