# metricas Specification

## Purpose

Medir erro, calibração e qualidade probabilística das previsões de carboidrato, de forma idêntica para todos os modelos, a partir do alvo e da preditiva gaussiana (média e variância) de cada refeição.

## Requirements

### Requirement: Erro da média
O MAE (erro absoluto médio) SHALL ser a média de |y − média|, em gramas. O percentual de meia dose SHALL ser o percentual de refeições com |y − média| ≤ 7,5 g.

#### Scenario: Caso conhecido
- **WHEN** y = (10, 20, 30) e as médias são (12, 20, 20)
- **THEN** o MAE vale 4 g e o percentual de meia dose vale 66,7%

### Requirement: Qualidade probabilística
A NLL (log-verossimilhança negativa) SHALL ser a média de −log N(y | média, variância). O CRPS (*continuous ranked probability score*) SHALL ser a média do CRPS da normal com essa média e variância, em gramas.

#### Scenario: Normal padrão em zero
- **WHEN** y = 0, média 0 e variância 1
- **THEN** a NLL vale ½·log(2π) ≈ 0,9189 e o CRPS vale ≈ 0,2337

#### Scenario: Escala do CRPS
- **WHEN** y, a média e o desvio são multiplicados por 10
- **THEN** o CRPS também é multiplicado por 10

### Requirement: Intervalo de 90%
O intervalo SHALL ser o central de 90% da preditiva: média ± z·desvio, com z = Φ⁻¹(0,95). A PICP (cobertura) SHALL ser a fração de refeições com y dentro do intervalo, e a MPIW SHALL ser a largura média, em gramas.

#### Scenario: Cobertura e largura conhecidas
- **WHEN** y = (0, 1, 3), médias 0 e desvios 1
- **THEN** a PICP vale 2/3 e a MPIW vale 2·z ≈ 3,29

### Requirement: Cobertura por faixa da previsão
A cobertura SHALL ser também reportada por quintil da média prevista (faixas com o mesmo número de refeições), porque a cobertura global pode esconder falta de cobertura nas refeições grandes e sobra nas pequenas.

#### Scenario: Faixas
- **WHEN** a cobertura por faixa é calculada para N refeições
- **THEN** há 5 faixas, cada uma com os limites de média prevista, o número de refeições e a PICP da faixa, e a soma das refeições é N

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
