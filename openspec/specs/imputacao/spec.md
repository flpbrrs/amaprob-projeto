# imputacao Specification

## Purpose

Definir como os imputadores recebem registros com quantidades ocultas e devolvem amostras das quantidades que faltam, sem nunca ler os valores ocultos, e fornecer o imputador de referência (porção média por gaveta).

## Requirements

### Requirement: Entrada sem os valores ocultos
Os imputadores SHALL receber `X` = `log1p(gramas)` com `NaN` em toda célula de `M`, junto com `P`, `M` e `C`. A construção dessa entrada SHALL apagar todas as células ocultas, de modo que nenhum imputador tenha acesso aos valores verdadeiros de `X[M]` (Regra 2).

#### Scenario: Células ocultas apagadas
- **WHEN** a entrada é construída a partir das gramas e de uma máscara
- **THEN** `X` é `NaN` exatamente nas células de `M` e igual a `log1p(gramas)` nas demais

### Requirement: Saída em amostras
O método de amostragem SHALL devolver um array de forma (`n_samples`, N, K). As células observadas SHALL ser cópias exatas da entrada, as células de gavetas ausentes SHALL valer 0, e apenas as células de `M` recebem valores sorteados, sem `NaN` e não negativos.

#### Scenario: Observados preservados
- **WHEN** um imputador ajustado gera amostras
- **THEN** em toda amostra, cada célula com `P` = 1 e `M` = 0 é igual à entrada, e cada célula com `P` = 0 vale 0

#### Scenario: Ocultos preenchidos
- **WHEN** um imputador ajustado gera amostras
- **THEN** toda célula de `M` tem valor finito e maior ou igual a 0 em toda amostra

#### Scenario: Reprodutível pelo gerador
- **WHEN** a amostragem é chamada duas vezes com geradores de mesma semente
- **THEN** as amostras são idênticas

### Requirement: Nenhum vazamento de X[M]
O ajuste e a amostragem MUST NOT depender dos valores verdadeiros das células ocultas.

#### Scenario: Valores ocultos alterados
- **WHEN** duas gramas diferem só em células de `M` e passam pelo mesmo ajuste e amostragem, com a mesma semente
- **THEN** as amostras resultantes são idênticas

#### Scenario: X[M] preenchido por engano
- **WHEN** o imputador recebe `X` com valores arbitrários (em vez de `NaN`) nas células de `M`
- **THEN** o ajuste e as amostras são idênticos aos obtidos com `NaN` nessas células

### Requirement: Porção média por gaveta
O imputador de referência SHALL modelar, para cada gaveta, `X` como gaussiano, com média e desvio-padrão estimados nas células observadas e presentes do conjunto de ajuste, sem usar coocorrências nem contexto. As amostras SHALL ser truncadas em 0, porque gramas não são negativas.

#### Scenario: Parâmetros pelos observados
- **WHEN** o imputador é ajustado
- **THEN** a média e o desvio-padrão de cada gaveta são os de `X` nas células com `P` = 1 e `M` = 0

#### Scenario: Distribuição das amostras
- **WHEN** muitas amostras são geradas para uma célula oculta de uma gaveta com média μ e desvio σ, com μ muito maior que σ
- **THEN** a média e o desvio das amostras se aproximam de μ e σ

### Requirement: PPCA com ausentes como condicionantes
O imputador PPCA SHALL modelar `X = Wz + μ + ε`, com `z ~ N(0, I_q)` e `ε ~ N(0, σ²I)`, estimado por EM com dados faltantes. A verossimilhança de cada refeição SHALL usar só as células com `P = 1` e `M = 0`. Células de gavetas ausentes MUST NOT entrar como zero observado. A dimensão latente q e a semente SHALL ser fixadas na construção.

#### Scenario: Ausentes fora da verossimilhança
- **WHEN** os valores de `X` nas células com `P = 0` são trocados por quaisquer outros e o ajuste é refeito com a mesma semente
- **THEN** os parâmetros ajustados e as amostras são idênticos

#### Scenario: Sem faltantes, solução fechada
- **WHEN** o PPCA é ajustado com todas as células observadas (`P = 1` em toda célula e `M` vazia)
- **THEN** σ² e `WWᵀ` coincidem com a solução de máxima verossimilhança em forma fechada (autovalores e autovetores da covariância amostral), com tolerância de 1e-3 relativa

#### Scenario: Recuperação em dados sintéticos
- **WHEN** o PPCA é ajustado com dados gerados por um PPCA conhecido, com metade das células ausentes ao acaso
- **THEN** `WWᵀ + σ²I` e `μ` se aproximam dos verdadeiros, com erro relativo de até 10% na norma de Frobenius

#### Scenario: Reprodutível pela semente
- **WHEN** dois PPCA com a mesma q e a mesma semente são ajustados nos mesmos dados
- **THEN** os parâmetros ajustados são idênticos

### Requirement: Convergência do EM
O EM SHALL parar quando a melhora relativa da log-verossimilhança dos observados ficar abaixo de uma tolerância fixa ou ao atingir um número máximo de iterações. A log-verossimilhança MUST NOT diminuir entre iterações, e o histórico dela SHALL ficar disponível após o ajuste.

#### Scenario: Log-verossimilhança monótona
- **WHEN** o PPCA é ajustado em dados com células ausentes
- **THEN** a log-verossimilhança registrada não diminui de uma iteração para a seguinte, a menos de 1e-8 relativo

### Requirement: Amostragem pela condicional
As amostras do PPCA SHALL vir da distribuição condicional das células de `M` dadas as células observadas da mesma refeição, sob os parâmetros ajustados, e SHALL ser truncadas em 0, como no baseline. Com todas as gavetas presentes ocultas, a condicional SHALL ser a marginal do modelo nas gavetas presentes.

#### Scenario: Momentos da condicional
- **WHEN** muitas amostras são geradas para uma refeição com parâmetros conhecidos, sem truncamento relevante (μ muito maior que o desvio)
- **THEN** a média e a covariância das amostras nas células ocultas se aproximam da média e da covariância condicionais gaussianas calculadas por complemento de Schur

#### Scenario: Taxa 1 pela marginal
- **WHEN** todas as gavetas presentes de uma refeição estão ocultas, sem truncamento relevante (μ muito maior que o desvio)
- **THEN** a média e a covariância das amostras se aproximam de `μ` e de `WWᵀ + σ²I` restritas às gavetas presentes

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

### Requirement: FA com ausentes como condicionantes
O imputador FA SHALL modelar `X = Wz + μ + ε`, com `z ~ N(0, I_q)` e `ε ~ N(0, Ψ)`, Ψ diagonal com uma variância por gaveta, estimado por EM com dados faltantes. A verossimilhança de cada refeição SHALL usar só as células com `P = 1` e `M = 0`. Células de gavetas ausentes MUST NOT entrar como zero observado. A dimensão latente q e a semente SHALL ser fixadas na construção.

#### Scenario: Ausentes fora da verossimilhança
- **WHEN** os valores de `X` nas células com `P = 0` são trocados por quaisquer outros e o ajuste é refeito com a mesma semente
- **THEN** os parâmetros ajustados e as amostras são idênticos

#### Scenario: Recuperação com ruído heterogêneo
- **WHEN** a FA é ajustada com dados gerados por uma FA conhecida, com variâncias de ruído diferentes entre as gavetas e metade das células ausentes ao acaso
- **THEN** `WWᵀ + Ψ` e `μ` se aproximam dos verdadeiros, com erro relativo de até 10% na norma de Frobenius, e Ψ se aproxima das variâncias verdadeiras gaveta a gaveta

#### Scenario: FA distingue ruídos que o PPCA iguala
- **WHEN** FA e PPCA são ajustados nos mesmos dados com ruído heterogêneo
- **THEN** a log-verossimilhança dos observados da FA é maior que a do PPCA

#### Scenario: Reprodutível pela semente
- **WHEN** duas FA com a mesma q e a mesma semente são ajustadas nos mesmos dados
- **THEN** os parâmetros ajustados são idênticos

### Requirement: Piso no ruído da FA
Cada variância de ruído da FA SHALL ficar acima de um piso proporcional à variância observada da gaveta, e SHALL ser finita e positiva em toda gaveta, inclusive nas que não têm célula observada no ajuste. O ajuste SHALL registrar quais gavetas terminaram no piso.

#### Scenario: Gaveta constante
- **WHEN** uma gaveta tem todos os valores observados iguais, e o resíduo dela vai a 0
- **THEN** a variância de ruído dela termina no piso, a gaveta aparece no registro, e o ajuste e as amostras são finitos

#### Scenario: Gaveta sem observados
- **WHEN** uma gaveta não tem nenhuma célula observada no conjunto de ajuste
- **THEN** o ajuste termina sem `NaN` nos parâmetros e as amostras das outras gavetas são finitas

### Requirement: Convergência do EM da FA
O EM da FA SHALL seguir a mesma regra de parada do PPCA (tolerância relativa fixa ou número máximo de iterações). A log-verossimilhança dos observados MUST NOT diminuir entre iterações, e o histórico dela SHALL ficar disponível após o ajuste.

#### Scenario: Log-verossimilhança monótona
- **WHEN** a FA é ajustada em dados com células ausentes
- **THEN** a log-verossimilhança registrada não diminui de uma iteração para a seguinte, a menos de 1e-8 relativo

### Requirement: Amostragem da FA pela condicional
As amostras da FA SHALL vir da distribuição condicional das células de `M` dadas as células observadas da mesma refeição, sob os parâmetros ajustados, e SHALL ser truncadas em 0. Com todas as gavetas presentes ocultas, a condicional SHALL ser a marginal do modelo nas gavetas presentes.

#### Scenario: Momentos da condicional
- **WHEN** muitas amostras são geradas para uma refeição com parâmetros conhecidos, sem truncamento relevante (μ muito maior que o desvio)
- **THEN** a média e a covariância das amostras nas células ocultas se aproximam da média e da covariância condicionais gaussianas de `WWᵀ + Ψ`, calculadas por complemento de Schur

#### Scenario: Taxa 1 pela marginal
- **WHEN** todas as gavetas presentes de uma refeição estão ocultas, sem truncamento relevante (μ muito maior que o desvio)
- **THEN** a média e a covariância das amostras se aproximam de `μ` e de `WWᵀ + Ψ` restritas às gavetas presentes

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
