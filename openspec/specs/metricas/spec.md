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

### Requirement: Erro da imputação nas células ocultas
A partir das S amostras do imputador em `log1p(gramas)` e das gramas verdadeiras, as métricas de imputação SHALL considerar só as células de `M`. O MAE em gramas SHALL usar a média de `expm1` das amostras, como o plug-in; o MAE em log SHALL usar a média das amostras em log. As gramas verdadeiras SHALL ser lidas só nesta avaliação (Regra 2).

#### Scenario: Caso conhecido
- **WHEN** duas células ocultas têm gramas verdadeiras 10 e 100 e as amostras de cada uma são constantes em `log1p(20)` e `log1p(100)`
- **THEN** o MAE em gramas vale 5 e o MAE em log vale `|log1p(20) − log1p(10)| / 2`

#### Scenario: Só células ocultas
- **WHEN** as gramas verdadeiras fora de `M` são trocadas por outras
- **THEN** as métricas de imputação não mudam

### Requirement: Qualidade probabilística da imputação
Por célula oculta, a NLL SHALL ser a da gaussiana com a média e a variância das S amostras em log, e o CRPS SHALL ser o do conjunto de amostras em log. O intervalo de 90% SHALL ir dos quantis empíricos 0,05 a 0,95 das amostras; a PICP é a fração de células com a verdade dentro e a MPIW é a largura média em gramas.

#### Scenario: Amostras gaussianas
- **WHEN** as amostras de uma célula são muitas e vêm de N(μ, σ²) em log
- **THEN** a NLL e o CRPS se aproximam dos da normal N(μ, σ²) avaliada na verdade, e o intervalo se aproxima de μ ± 1,645σ

#### Scenario: Cobertura conhecida
- **WHEN** a verdade está dentro do intervalo das amostras em duas de três células
- **THEN** a PICP vale 2/3

### Requirement: Métricas de imputação por gaveta
As métricas de imputação SHALL ser reportadas também por gaveta, com o número de células ocultas de cada uma, para comparar o erro entre gavetas (como a de açúcar adicionado).

#### Scenario: Soma das células
- **WHEN** as métricas por gaveta são calculadas
- **THEN** há uma linha por gaveta com células ocultas, e a soma das células é o total de células de `M`
