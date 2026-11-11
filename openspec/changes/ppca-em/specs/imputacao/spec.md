# Spec Delta

## ADDED Requirements

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
Os requisitos de entrada sem os valores ocultos, de saída em amostras e de nenhum vazamento de `X[M]` SHALL valer para todos os imputadores do projeto, e o mesmo conjunto de testes do contrato SHALL rodar sobre cada um deles.

#### Scenario: Contrato do PPCA
- **WHEN** os testes do contrato rodam com o PPCA
- **THEN** observados preservados, ausentes em 0, ocultos finitos e não negativos, reprodutibilidade pelo gerador e ausência de vazamento passam como no baseline
