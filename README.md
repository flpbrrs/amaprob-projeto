# amaprob-projeto

Projeto final de Aprendizagem de Máquina Probabilística (2026.2).

**Estimação probabilística do carboidrato de refeições brasileiras a partir de registros alimentares incompletos.**

No cuidado em diabetes tipo 1, o carboidrato da refeição orienta a dose de insulina, mas o registro do paciente raramente é completo. O projeto mascara quantidades de registros reais da POF 2017–2018 (Pesquisa de Orçamentos Familiares, IBGE), imputa o que falta com modelos probabilísticos e prevê o carboidrato com intervalo, verificando se a incerteza da imputação se propaga de forma calibrada.

O plano completo, as regras invioláveis e o protocolo experimental estão em [`CLAUDE.md`](CLAUDE.md). Leia antes de contribuir.

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

```
data/
├── raw/         # microdados POF, como baixados do IBGE
├── interim/     # tabelas lidas
└── processed/   # X, P, C, y, person_id, máscaras, splits
```

## Estrutura

A estrutura de pastas segue a seção 8 do `CLAUDE.md`. Os subpacotes de `src/amaprob/` são criados à medida que o código de cada etapa é escrito.
