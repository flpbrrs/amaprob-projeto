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
