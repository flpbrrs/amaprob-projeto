# Tasks

## 1. Ambiente de testes

- [x] 1.1 Adicionar `pytest` ao grupo de desenvolvimento (`uv add --dev pytest`), configurar `testpaths = ["tests"]` no `pyproject.toml` e criar `tests/`; verificar que `uv run pytest` executa (sem testes, código de saída 5)

## 2. Configuração de 29 gavetas na camada de dados

- [x] 2.1 Mover `PROPOSTA_DIVIDIR` e `PROPOSTA_FUNDIR` de `experiments/piso_de_erro.py` para `data/gavetas.py` como `DIVIDIR_K29` e `FUNDIR_K29`, com o `piso_de_erro` importando de lá; verificar que o módulo importa sem erro
- [x] 2.2 Escrever `tests/test_gavetas.py` com os cenários do delta `gavetas` (29 gavetas distintas, perfis de Bebidas, Farinhas e massas por subgrupo, Açúcar próprio) sobre `configs/classificacao_ibge.csv`; verificar com `uv run pytest tests/test_gavetas.py`

## 3. Split por pessoa sob teste

- [x] 3.1 Escrever `tests/test_splits.py` com `person_id` sintéticos: cada pessoa em um único conjunto, frações aplicadas sobre pessoas, mesmo rótulo com linhas embaralhadas; verificar com `uv run pytest tests/test_splits.py`

## 4. Geração das máscaras

- [x] 4.1 Implementar `data/mascaras.py` com `TAXAS`, `SEMENTES` e `gerar_mascaras(P, taxas, sementes)` (um uniforme por célula e semente, `M = P ∧ (U < r)`), com docstring da derivação; verificar com o teste 4.2
- [x] 4.2 Escrever `tests/test_mascaras.py` com `P` sintética: `M ⊆ P`, taxa dentro de 4 desvios-padrão binomiais, aninhamento, taxa 0 vazia, taxa 1 = `P`, mesma semente idêntica, sementes diferentes distintas, refeição de uma gaveta mascarável em 0,25; verificar com `uv run pytest tests/test_mascaras.py`
- [x] 4.3 Escrever `docs/mascaramento.md` (mecanismo Bernoulli por gaveta presente e por que não a contagem por refeição; aninhamento e comparação pareada; sementes; limitações MCAR e por gaveta; formato dos arquivos), em tom neutro, com siglas expandidas; verificar que cada afirmação numérica bate com o `design.md` ou com a execução da tarefa 5.3

## 5. Dataset e persistência

- [x] 5.1 Implementar `data/dataset.py`: dataclass `Refeicoes`, `montar_refeicoes(consumo, gavetas)`, gravação e carga do dataset (`.npz`, `allow_pickle=False`), gravação e carga das máscaras com as três conferências de alinhamento (`meal_id`, nomes das gavetas, taxa 1 = `P`); verificar com o teste 5.2
- [x] 5.2 Escrever `tests/test_dataset.py` com dados sintéticos: ida e volta do disco igual valor a valor, `P` = gramas > 0, `y` = soma de `CHOTOT`, erro explícito com `meal_id` em outra ordem e com `P` diferente; verificar com `uv run pytest tests/test_dataset.py`
- [x] 5.3 Adicionar o `__main__` que lê a POF, monta o dataset K = 29, gera as máscaras e grava `data/processed/refeicoes_k29.npz` e `data/processed/mascaras_k29.npz`; executar `uv run python -m amaprob.data.dataset` e registrar o tempo e o tamanho dos arquivos
- [x] 5.4 Escrever `tests/test_dados_processados.py` sobre os arquivos reais, pulado se não existirem: 304.670 × 29, 941.320 células presentes, 32.299/6.921/6.921 pessoas sem interseção, `y ≥ 10`, `M ⊆ P`, taxa observada a menos de 0,002 de r em cada taxa e semente, carga alinhada; verificar com `uv run pytest`
- [x] 5.5 Atualizar o `README.md` com o passo de preparação dos dados e o comando de testes; verificar que os comandos documentados rodam como escritos

## 6. Integração

- [x] 6.1 Rodar `uv run python -m amaprob.experiments.piso_de_erro` e verificar que a linha `PROPOSTA` da curva continua com K = 29 e MAE de 8,1 g na validação
- [x] 6.2 Rodar `uv run pytest` com todos os testes passando e `openspec validate mascaras-persistidas --strict` sem erros

## Workflow follow-up

- Mostrar o diff e a mensagem de commit para aprovação antes de commitar.
- Arquivar a change (`/opsx:archive`) depois da revisão.
