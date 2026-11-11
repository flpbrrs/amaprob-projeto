# Spec Delta

## ADDED Requirements

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
