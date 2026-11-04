# Spec Delta

## Purpose

Medir erro, calibração e qualidade probabilística das previsões de carboidrato, de forma idêntica para todos os modelos, a partir do alvo e da preditiva gaussiana (média e variância) de cada refeição.

## ADDED Requirements

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
