# Spec Delta

## ADDED Requirements

### Requirement: SVGP com média linear
O SVGP SHALL modelar o alvo padronizado como uma função com prior de processo gaussiano, de média linear nas colunas padronizadas de `Z` e kernel RBF com um comprimento por coluna, mais ruído gaussiano homocedástico. SHALL usar M pontos indutores aprendidos e ser ajustado por minibatch na ELBO, só no conjunto de ajuste. M, o número de épocas e a semente SHALL ser fixados na construção.

#### Scenario: Recupera uma reta
- **WHEN** o SVGP é ajustado em dados sintéticos lineares com ruído gaussiano de desvio conhecido
- **THEN** as previsões se aproximam da reta verdadeira e a variância preditiva se aproxima da variância do ruído

#### Scenario: Captura o que a reta não captura
- **WHEN** SVGP e regressão linear bayesiana são ajustados em dados sintéticos com uma parte linear e uma parte não linear
- **THEN** a NLL do SVGP num conjunto separado é menor que a da regressão linear bayesiana

#### Scenario: Incerteza fora dos dados
- **WHEN** o SVGP prevê um ponto distante dos dados de ajuste
- **THEN** a variância preditiva nesse ponto é maior que nos pontos de ajuste

#### Scenario: Reprodutível pela semente
- **WHEN** dois SVGP com o mesmo M, as mesmas épocas e a mesma semente são ajustados nos mesmos dados
- **THEN** as médias e as variâncias previstas são idênticas

#### Scenario: Coluna constante
- **WHEN** uma coluna de `Z` é constante no conjunto de ajuste
- **THEN** o ajuste conclui sem divisão por zero e as previsões são finitas
