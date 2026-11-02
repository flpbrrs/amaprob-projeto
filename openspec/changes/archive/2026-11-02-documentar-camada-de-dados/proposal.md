# Proposal

## Why

A camada de dados (`src/amaprob/data/`) já está implementada e alimenta todas as etapas seguintes. Mesmo assim, nenhuma spec nem teste descreve o que ela garante. Registrar como spec o comportamento atual dá às próximas mudanças (máscaras, configuração de gavetas, testes) uma base sobre a qual escrever deltas, e torna explícitas as suposições que hoje só estão no código.

## What Changes

- Documenta, como requisitos, o comportamento **já implementado** da leitura dos microdados, das gavetas, da montagem de refeições e da divisão treino/validação/teste.
- Declara como pré-condições, verificadas nos microdados da versão 20230713, duas suposições que o código não protege: horário sempre preenchido e todo código de alimento coberto pela classificação. A falta de proteção está registrada em `backlog/decisions.md`.
- Nenhum código muda. Nenhum comportamento é acrescentado.

Fica fora deste escopo tudo o que o `AGENTS.md` descreve e o código ainda não faz: escala `log1p`, as matrizes `P`, `M` e `C`, `y` como saída da camada de dados, máscaras, variáveis do arquivo `MORADOR` e a configuração de 29 gavetas. Essa configuração existe hoje apenas como constante do experimento de piso de erro, fora de `src/amaprob/data/`.

## Capabilities

### New Capabilities

- `leitura-pof`: leitura do arquivo de consumo alimentar da POF (Pesquisa de Orçamentos Familiares) em uma tabela de itens, com as colunas permitidas pelas regras de vazamento.
- `gavetas`: conversão da classificação hierárquica do IBGE (Instituto Brasileiro de Geografia e Estatística) e mapeamento de cada código de alimento para uma gaveta, com granularidade configurável.
- `refeicoes`: agrupamento dos itens em refeições que pedem insulina e matriz de gramas por gaveta.
- `split-por-pessoa`: atribuição de cada linha a treino, validação ou teste, sorteada por pessoa com semente fixa.

### Modified Capabilities

Nenhuma: não há specs anteriores.

## Impact

- Código: nenhum. As specs descrevem `src/amaprob/data/pof.py`, `gavetas.py`, `refeicoes.py` e `splits.py` como estão hoje.
- Dados versionados: `configs/classificacao_ibge.csv` passa a ter seu formato descrito em spec.
- Depois do archive, quatro specs novas em `openspec/specs/`. Mudanças futuras nesses módulos passam a exigir delta.
