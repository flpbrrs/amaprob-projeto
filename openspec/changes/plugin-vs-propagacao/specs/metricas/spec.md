# Spec Delta

## Purpose

Medir erro, calibração e qualidade probabilística das previsões de carboidrato, de forma idêntica para todos os modelos, a partir do alvo e da preditiva gaussiana (média e variância) de cada refeição.

## ADDED Requirements

### Requirement: Métricas da mistura de gaussianas
Para uma preditiva em mistura de pesos iguais `(1/S) Σ N(μ_s, σ²_s)`, as métricas SHALL ter as mesmas chaves das métricas gaussianas. O MAE e o percentual de meia dose SHALL usar a média da mistura. A NLL SHALL ser exata, via `logsumexp`. O CRPS SHALL ser o da mistura de normais, exato. O intervalo de 90% SHALL ser o central da mistura, pelos quantis exatos 0,05 e 0,95.

#### Scenario: Um componente
- **WHEN** a mistura tem S = 1
- **THEN** todas as métricas coincidem com as da preditiva gaussiana de mesma média e variância

#### Scenario: Componentes iguais
- **WHEN** a mistura tem S componentes idênticos N(μ, σ²)
- **THEN** todas as métricas coincidem com as de N(μ, σ²)

#### Scenario: Quantis da mistura
- **WHEN** o intervalo de 90% de uma mistura de dois componentes distantes é calculado
- **THEN** a CDF da mistura vale 0,05 no limite inferior e 0,95 no superior, com tolerância de 1e-6

#### Scenario: Observação longe da mistura
- **WHEN** y está a mais de 40 desvios de todos os componentes
- **THEN** a NLL é finita

### Requirement: Leitura por número de quantidades ocultas
As métricas SHALL poder ser reportadas por número de gavetas ocultas na refeição, nas classes 0, 1, 2, 3 e "4+", com o número de refeições, a PICP, a MPIW, o MAE e o percentual de meia dose de cada classe. A leitura SHALL receber os limites do intervalo, para servir tanto à preditiva gaussiana quanto à mistura.

#### Scenario: Classes
- **WHEN** a leitura é calculada para N refeições
- **THEN** cada refeição cai em exatamente uma classe, as classes sem refeições não aparecem e a soma das refeições é N

#### Scenario: Caso conhecido
- **WHEN** duas refeições têm 0 ocultas, uma com y dentro e outra fora do intervalo, e uma refeição tem 5 ocultas com y dentro
- **THEN** a classe 0 tem n = 2 e PICP de 0,5, e a classe "4+" tem n = 1 e PICP de 1
