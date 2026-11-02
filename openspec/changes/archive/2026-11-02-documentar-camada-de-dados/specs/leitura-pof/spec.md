# Spec Delta

## Purpose

Ler o arquivo de consumo alimentar da POF 2017–2018 (Pesquisa de Orçamentos Familiares, do IBGE, Instituto Brasileiro de Geografia e Estatística) em uma tabela de itens, expondo só as colunas que podem servir de entrada ou de alvo sem violar as regras de vazamento do projeto.

## ADDED Requirements

### Requirement: Uma linha por item consumido
A leitura SHALL interpretar `CONSUMO_ALIMENTAR.txt` como arquivo de largura fixa, sem cabeçalho, no layout do dicionário de variáveis da versão 20230713, e SHALL produzir uma linha da tabela para cada linha do arquivo.

#### Scenario: Arquivo da versão 20230713
- **WHEN** a leitura recebe o `CONSUMO_ALIMENTAR.txt` da versão 20230713
- **THEN** a tabela tem 1.175.390 linhas, uma por item consumido

### Requirement: Conjunto fixo de colunas
A tabela SHALL conter exatamente estas 22 colunas: `UF`, `ESTRATO_POF`, `TIPO_SITUACAO_REG`, `COD_UPA`, `NUM_DOM`, `NUM_UC`, `COD_INFORMANTE`, `QUADRO`, `SEQ`, `V9001`, `V9015`, `V9016`, `V9017`, `V9018`, `COD_PREPARACAO_FINAL`, `QTD`, `COD_TBCA`, `CHOTOT`, `PESO_FINAL`, `RENDA_TOTAL`, `DIA_SEMANA`, `DIA_ATIPICO`.

#### Scenario: Colunas da tabela
- **WHEN** a leitura termina
- **THEN** o conjunto de colunas é exatamente o listado no requisito, sem colunas a mais nem a menos

#### Scenario: Gramas e carboidrato do item
- **WHEN** um item é lido
- **THEN** `QTD` traz as gramas consumidas e `CHOTOT` o carboidrato total do item, ambos em gramas e como valores numéricos

### Requirement: Medida caseira nunca é lida
A leitura MUST NOT expor `V9005`, `V9007`, `GRAMATURA1` nem `COD_UNIDADE_MEDIDA_FINAL`. Como `QTD` = `V9005` × `GRAMATURA1`, essas colunas tornariam a quantidade determinística a partir da medida caseira.

#### Scenario: Colunas da medida caseira ausentes
- **WHEN** a leitura termina
- **THEN** nenhuma das colunas `V9005`, `V9007`, `GRAMATURA1` e `COD_UNIDADE_MEDIDA_FINAL` está na tabela

### Requirement: Carboidrato como único nutriente
A leitura SHALL expor `CHOTOT` como única coluna de composição nutricional. Os nutrientes da POF são `QTD` × composição: como entrada, vazariam quantidades mascaradas, por isso servem apenas como alvo.

#### Scenario: Nenhum outro nutriente
- **WHEN** a leitura termina
- **THEN** `CHOTOT` é a única coluna da tabela derivada da composição nutricional do item

### Requirement: Tipos e valores ausentes
`V9015` (horário), `COD_TBCA` e `DIA_SEMANA` SHALL vir como texto, sem espaços nas bordas. As demais colunas SHALL vir como numéricas, com `V9001` como código inteiro. Campo vazio no arquivo SHALL virar valor ausente.

#### Scenario: Texto sem espaços de preenchimento
- **WHEN** um campo de texto ocupa menos posições do que a largura reservada no layout
- **THEN** o valor lido não tem espaços no início nem no fim (por exemplo, o horário vem como `12H`)

#### Scenario: Código do alimento compatível com a classificação
- **WHEN** um item é lido
- **THEN** `V9001` é um inteiro comparável diretamente aos códigos de `configs/classificacao_ibge.csv`

#### Scenario: Campo opcional vazio
- **WHEN** `V9016` ou `COD_PREPARACAO_FINAL` está em branco no arquivo
- **THEN** a tabela traz valor ausente nessa célula, sem descartar a linha
