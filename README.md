# amaprob-projeto

Projeto final de Aprendizagem de Máquina Probabilística (2026.2).

**Estimação probabilística do carboidrato de refeições brasileiras a partir de registros alimentares incompletos.**

No cuidado em diabetes tipo 1, o carboidrato da refeição orienta a dose de insulina, mas o registro do paciente raramente é completo. O projeto mascara quantidades de registros reais da POF 2017–2018 (Pesquisa de Orçamentos Familiares, IBGE), imputa o que falta com modelos probabilísticos e prevê o carboidrato com intervalo, verificando se a incerteza da imputação se propaga de forma calibrada.

O plano completo, as regras invioláveis e o protocolo experimental estão em [`AGENTS.md`](AGENTS.md), lido também pelos assistentes de código. Leia antes de contribuir.

## Ambiente

Gerenciado com [uv](https://docs.astral.sh/uv/) e Python 3.12.

```bash
uv sync          # cria .venv e instala as dependências travadas em uv.lock
uv run python    # executa dentro do ambiente
```

Novas dependências entram com `uv add <pacote>` e são combinadas com o time antes.

## Dados

Os microdados **não** são versionados. Cada pessoa baixa direto do IBGE para `data/raw/pof2017_2018/` e extrai cada `.zip` numa pasta de mesmo nome:

- Fonte: <https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados/>
- `Dados_20230713.zip` (~146 MB): contém `CONSUMO_ALIMENTAR.txt`
- `Documentacao_20230713.zip`: dicionário de variáveis e classificação dos grupos alimentares
- `Programas_de_Leitura_20230713.zip`: layout de referência (R e SAS)

Resultado esperado: `data/raw/pof2017_2018/Dados_20230713/CONSUMO_ALIMENTAR.txt`.

Com os microdados no lugar, prepare o dataset de refeições e as máscaras (cerca de 30 s):

```bash
uv run python -m amaprob.data.dataset
```

O comando grava `data/processed/refeicoes_k29.npz` (gramas, presença, carboidrato, identificadores e conjunto do split) e `data/processed/mascaras_k29.npz` (quantidades ocultas por taxa e semente). O mecanismo de mascaramento está descrito em [`docs/mascaramento.md`](docs/mascaramento.md).

```
data/
├── raw/         # microdados POF, como baixados do IBGE
├── interim/     # tabelas lidas
└── processed/   # dataset de refeições e máscaras (.npz)
```

## Testes

```bash
uv run pytest
```

Os testes de máscaras, split, gavetas e persistência usam dados sintéticos ou a classificação versionada e rodam sem os microdados. Os testes sobre os arquivos de `data/processed/` são pulados quando eles não existem.

## Estrutura

A estrutura de pastas segue a seção 8 do `AGENTS.md`. Os subpacotes de `src/amaprob/` são criados à medida que o código de cada etapa é escrito.
