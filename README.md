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

Os microdados **não** são versionados. Cada pessoa baixa localmente para `data/raw/` (instruções de download a definir).

```
data/
├── raw/         # microdados POF, como baixados do IBGE
├── interim/     # tabelas lidas
└── processed/   # X, P, C, y, person_id, máscaras, splits
```

## Estrutura

A estrutura de pastas segue a seção 8 do `CLAUDE.md`. Os subpacotes de `src/amaprob/` são criados à medida que o código de cada etapa é escrito.
