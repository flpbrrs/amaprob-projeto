# Proposal

## Why

Todas as partes do projeto comparam modelos sobre as mesmas quantidades ocultas. Hoje não existe nem a máscara nem uma ordem fixa de refeições à qual ela possa se prender. Sem isso, as Regras 3 e 4 do `AGENTS.md` (split por indivíduo; máscaras geradas antes da modelagem, com semente fixa e persistidas) não têm como ser verificadas, e o pipeline de ponta a ponta da Parte 2 não tem entrada estável. Esta mudança cria essa fundação e os primeiros testes automatizados do repositório.

## What Changes

- Adiciona `pytest` como dependência de desenvolvimento e uma pasta `tests/`.
- Move a configuração de 29 gavetas (Bebidas em 4 perfis nutricionais + Farinhas e massas dividida em subgrupos) do experimento de piso de erro para a camada de dados. O experimento passa a importá-la de lá, sem mudar o resultado.
- Monta e persiste em `data/processed/` o **dataset de refeições** com K = 29: gramas por gaveta (`G`), presença (`P`), carboidrato (`y`), `meal_id`, `person_id`, conjunto do split e nomes das gavetas, todos alinhados à mesma ordem de linhas.
- Gera e persiste, em arquivo separado, as **máscaras** `M` para as taxas `{0.0, 0.25, 0.5, 0.75, 1.0}` e as sementes `{0, 1, 2}`:
  - mecanismo MCAR (ausência completamente ao acaso), com sorteio independente de Bernoulli por gaveta presente;
  - máscaras aninhadas entre taxas: um único número uniforme por célula, oculto quando menor que a taxa;
  - o `meal_id` acompanha as máscaras, para conferir o alinhamento ao carregar.
- Testes de máscara (`M ⊆ P`, taxa observada, aninhamento, extremos 0 e 1, reprodutibilidade pela semente) e de split (nenhum `person_id` em dois conjuntos, frações sobre pessoas).
- Documento `docs/mascaramento.md` com a justificativa do mecanismo e do aninhamento, para revisão do time e do professor.

Fica fora deste escopo, e vai para a mudança seguinte (baseline + regressão linear bayesiana):
- a matriz de contexto `C`;
- o teste de que a imputação não lê `X[M]` (depende do contrato `Imputer`);
- os testes das métricas.

Também fica fora a proteção contra entradas inválidas na montagem de refeições, registrada em `backlog/decisions.md`.

## Capabilities

### New Capabilities

- `dataset-refeicoes`: montagem e persistência das matrizes por refeição (gramas, presença, carboidrato, identificadores e conjunto do split) com a configuração de gavetas adotada, em ordem de linhas fixa.
- `mascaras`: geração das máscaras de quantidades ocultas sobre as gavetas presentes e a persistência delas, com o mecanismo, as taxas e as sementes do protocolo experimental.

### Modified Capabilities

- `gavetas`: acrescenta como requisito a configuração adotada de 29 gavetas, que passa a pertencer à camada de dados.

## Impact

- Código novo: `src/amaprob/data/mascaras.py` (geração das máscaras), `src/amaprob/data/dataset.py` (montagem, persistência e carga do dataset e das máscaras) e `tests/`.
- Código alterado: `src/amaprob/data/gavetas.py` recebe a configuração K = 29; `src/amaprob/experiments/piso_de_erro.py` passa a importá-la.
- Dependência nova: `pytest` (grupo de desenvolvimento).
- Arquivos gerados em `data/processed/`, fora do git: o dataset e as máscaras em `.npz`.
- Documentação: `docs/mascaramento.md`; `README.md` com o comando de preparação e o de testes.
