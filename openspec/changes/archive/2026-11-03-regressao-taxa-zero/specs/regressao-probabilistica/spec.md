# Spec Delta

## Purpose

Prever o carboidrato da refeição com uma distribuição preditiva gaussiana por refeição, a partir das quantidades por gaveta (observadas ou imputadas), da presença e do contexto, com a regressão linear bayesiana como primeiro modelo.

## ADDED Requirements

### Requirement: Entrada da regressão
A entrada `Z` SHALL ser a concatenação, nessa ordem, das gramas por gaveta (não do `log1p`), da presença `P` e do contexto `C`.

#### Scenario: Dimensão de Z
- **WHEN** `Z` é montada com 29 gavetas e 13 colunas de contexto
- **THEN** `Z` tem 29 + 29 + 13 = 71 colunas

### Requirement: Preditiva gaussiana por refeição
Todo regressor SHALL ser ajustado com `Z` e `y` e SHALL devolver, para cada linha de uma nova `Z`, a média e a variância de uma preditiva gaussiana, com variância estritamente positiva.

#### Scenario: Formas da saída
- **WHEN** um regressor ajustado recebe `Z` com N linhas
- **THEN** devolve dois vetores de N valores, médias e variâncias, com toda variância maior que 0

### Requirement: Regressão linear bayesiana conjugada
A regressão linear bayesiana SHALL usar pesos com prior gaussiano isotrópico de precisão α e ruído gaussiano homocedástico de precisão β, sobre colunas padronizadas pelas médias e desvios do conjunto de ajuste e com o alvo centrado. A preditiva SHALL ter variância 1/β + φᵀ S φ, em que S é a covariância posterior dos pesos.

#### Scenario: Recupera os coeficientes
- **WHEN** a regressão é ajustada em dados sintéticos lineares com ruído gaussiano de desvio conhecido e muitas linhas
- **THEN** as previsões se aproximam da reta verdadeira e a variância preditiva se aproxima da variância do ruído

#### Scenario: Incerteza fora dos dados
- **WHEN** a regressão é ajustada em poucas linhas e prevê um ponto distante dos dados de ajuste
- **THEN** a variância preditiva nesse ponto é maior que nos pontos de ajuste

#### Scenario: Coluna constante
- **WHEN** uma coluna de `Z` é constante no conjunto de ajuste
- **THEN** o ajuste conclui sem divisão por zero e a coluna não altera as previsões

### Requirement: Hiperparâmetros pela evidência
α e β SHALL ser estimados maximizando a evidência (verossimilhança marginal) no conjunto de ajuste, por iteração de ponto fixo até a convergência, sem usar validação nem teste.

#### Scenario: Ruído estimado
- **WHEN** a regressão é ajustada em dados sintéticos lineares com ruído de desvio 2 e muitas linhas
- **THEN** 1/√β fica próximo de 2
