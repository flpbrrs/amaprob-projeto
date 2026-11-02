# gavetas Specification

## Purpose

Mapear cada código de alimento da POF (`V9001`) para uma gaveta (grupo alimentar), a partir da classificação hierárquica do IBGE (Instituto Brasileiro de Geografia e Estatística), com granularidade escolhida por quem chama.

## Requirements

### Requirement: Conversão da planilha hierárquica do IBGE
A conversão SHALL transformar a planilha "Classificação dos grupos de Consumo Alimentar" (descartando as duas primeiras linhas) em uma tabela com uma linha por item e colunas `V9001`, `grupo`, `subgrupo` e `descricao`. Os textos SHALL vir sem espaços nas bordas.

#### Scenario: Linha que abre um grupo
- **WHEN** uma linha tem o nome na primeira coluna
- **THEN** ela abre um novo grupo, zera o subgrupo corrente e não gera item

#### Scenario: Linha que abre um subgrupo
- **WHEN** uma linha não tem nome de grupo nem código, mas tem descrição
- **THEN** ela abre um novo subgrupo dentro do grupo corrente e não gera item

#### Scenario: Linha de item
- **WHEN** uma linha tem código
- **THEN** ela gera um item com esse código como inteiro, o grupo e o subgrupo correntes e a sua descrição

### Requirement: Classificação versionada no repositório
A classificação convertida SHALL ficar em `configs/classificacao_ibge.csv`, gerada a partir do arquivo `Classifica*grupos*.xlsx` da documentação dos microdados (`Documentacao_20230713`). A geração requer o pacote `openpyxl`.

#### Scenario: Conteúdo do arquivo versionado
- **WHEN** `configs/classificacao_ibge.csv` é lido
- **THEN** ele tem 1.593 itens com `V9001` único, 20 grupos e 116 subgrupos, e todo item tem subgrupo

### Requirement: Grupo do IBGE como gaveta padrão
Todo código cujo grupo não foi indicado para divisão SHALL receber como gaveta o nome do seu grupo do IBGE.

#### Scenario: Nenhum grupo dividido
- **WHEN** as gavetas são montadas sem nenhum grupo para dividir
- **THEN** há 21 gavetas distintas: os 20 grupos do IBGE e a gaveta `Açúcar`

### Requirement: Divisão de grupos em subgrupos
Para cada grupo indicado para divisão, a gaveta de um código SHALL ser `<grupo> / <subgrupo>`. O valor especial `"todos"` SHALL dividir todos os grupos.

#### Scenario: Um grupo dividido
- **WHEN** só o grupo `Bebidas` é indicado para divisão
- **THEN** cada código de `Bebidas` recebe a gaveta `Bebidas / <subgrupo>` e os demais grupos continuam com a gaveta do grupo

#### Scenario: Todos os grupos divididos
- **WHEN** a divisão recebe `"todos"`
- **THEN** há 117 gavetas distintas: os 116 subgrupos e a gaveta `Açúcar`

### Requirement: Fusão de subgrupos em gaveta comum
Um mapeamento opcional `{subgrupo: nome da gaveta}` SHALL dar a gaveta indicada aos códigos desses subgrupos, desde que o grupo deles esteja sendo dividido. Subgrupos fora do mapeamento mantêm `<grupo> / <subgrupo>`. O mapeamento identifica o subgrupo só pelo nome.

#### Scenario: Subgrupos fundidos em grupo dividido
- **WHEN** `Bebidas` é dividido e o mapeamento associa `Cerveja` e `Vinho` a `Bebidas / cerveja e vinho`
- **THEN** os códigos dos dois subgrupos recebem essa mesma gaveta

#### Scenario: Fusão em grupo não dividido
- **WHEN** o mapeamento cita um subgrupo de um grupo que não foi indicado para divisão
- **THEN** os códigos desse subgrupo continuam com a gaveta do grupo

#### Scenario: Nome de subgrupo único entre grupos
- **WHEN** a classificação versionada é usada
- **THEN** nenhum nome de subgrupo aparece em mais de um grupo, então a fusão por nome não é ambígua

### Requirement: Açúcar de mesa como gaveta própria
O código `6906602` (açúcar de mesa) SHALL receber sempre a gaveta `Açúcar`, independentemente da divisão e da fusão. Os demais códigos do grupo `Doces` seguem as regras gerais.

#### Scenario: Açúcar com divisão total
- **WHEN** a divisão recebe `"todos"`
- **THEN** o código `6906602` recebe `Açúcar`, e não `Doces / Mel/rapadura e açúcar de mesa`

### Requirement: Mapeamento por código
O resultado SHALL associar uma gaveta a cada código da classificação, indexado por `V9001`. Um código fora da classificação é violação de pré-condição da montagem de refeições (ver `refeicoes`), não um caso tratado aqui.

#### Scenario: Cobertura do mapeamento
- **WHEN** as gavetas são montadas a partir da classificação versionada
- **THEN** cada um dos 1.593 códigos tem exatamente uma gaveta
