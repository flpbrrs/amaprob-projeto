# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: Contrato comum dos imputadores
Os requisitos de entrada sem os valores ocultos, de saída em amostras e de nenhum vazamento de `X[M]` SHALL valer para todos os imputadores do projeto, e o mesmo conjunto de testes do contrato SHALL rodar sobre cada um deles.

#### Scenario: Contrato do PPCA
- **WHEN** os testes do contrato rodam com o PPCA
- **THEN** observados preservados, ausentes em 0, ocultos finitos e não negativos, reprodutibilidade pelo gerador e ausência de vazamento passam como no baseline

#### Scenario: Contrato da FA
- **WHEN** os testes do contrato rodam com a FA
- **THEN** observados preservados, ausentes em 0, ocultos finitos e não negativos, reprodutibilidade pelo gerador e ausência de vazamento passam como no baseline
