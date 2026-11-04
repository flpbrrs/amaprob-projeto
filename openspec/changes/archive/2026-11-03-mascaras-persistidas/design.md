# Design

## Context

A camada de dados já monta refeições, gavetas e o split por pessoa (specs `leitura-pof`, `gavetas`, `refeicoes`, `split-por-pessoa`), mas não grava nada em disco. Cada script relê a POF (≈ 15 s de leitura e ≈ 2 s de montagem). A configuração de 29 gavetas existe só como constante de `src/amaprob/experiments/piso_de_erro.py`. O repositório não tem testes.

Fatos verificados na versão 20230713 que orientam o desenho:
- Nenhum item tem `QTD` nulo, zero ou negativo. Por isso a presença por item coincide com `G > 0` nas 941.320 células presentes.
- A configuração de 29 gavetas gera exatamente 29 colunas.
- Gavetas presentes por refeição: média de 3,1. **15,6% das refeições têm uma gaveta só.**

## Goals / Non-Goals

**Goals:**
- Fixar a ordem de linhas e as máscaras antes de qualquer modelo (Regra 4).
- Tornar as Regras 3 e 4 verificáveis por teste automatizado.
- Manter a geração das máscaras como função pura de `(P, taxas, sementes)`, testável sem a POF.

**Non-Goals:**
- Contexto `C`, transformação `log1p` e a entrada dos imputadores (`X` com `NaN` em `M`): ficam para a mudança da Parte 2. A `log1p` é aplicada na entrada da imputação, não gravada.
- Mascaramento não aleatório (MAR/MNAR) e omissão de itens inteiros: limitações declaradas.

## Decisions

### 1. Mecanismo: Bernoulli independente por gaveta presente

Cada célula presente fica oculta com probabilidade r, sem depender das outras.

Alternativas consideradas:
- **Contagem exata por refeição** (ocultar `round(r × n_presentes)` gavetas em cada refeição). Rejeitada por viés: nas 15,6% de refeições com uma gaveta só, `round(0,25)` e `round(0,5)` valem 0 (o Python arredonda 0,5 para o par). Essas refeições nunca seriam mascaradas abaixo da taxa 0,75. Em geral, a taxa efetiva passaria a depender do tamanho da refeição, o que deixa de ser MCAR.
- **Contagem exata global** (ocultar exatamente r × 941.320 células sorteadas no conjunto todo). A taxa fica exata, mas o ganho é desprezível: o desvio-padrão da taxa no sorteio de Bernoulli é no máximo √(0,25/941.320) ≈ 0,0005. Além disso, as células deixam de ser independentes.

O Bernoulli independente é a definição direta de MCAR. Ele se descreve numa frase na Metodologia, e a taxa só é garantida em média, com desvio irrelevante nesta escala.

### 2. Aninhamento por um uniforme por célula

Para cada semente, sorteia-se uma vez `U ~ Uniforme[0, 1)` com a forma de `P`, e `M_r = P ∧ (U < r)`.

Consequências:
- **Marginal correta:** `P(U < r) = r`, então cada taxa, isolada, segue o mecanismo da Decisão 1.
- **Aninhamento:** se r < r', então `U < r` implica `U < r'`.
- **Extremos exatos:** como `U < 1` sempre e `U < 0` nunca, a taxa 1 é igual a `P` e a taxa 0 fica vazia.
- **Comparação pareada:** a curva "métrica × taxa" compara cada refeição com ela mesma. As diferenças entre taxas refletem só as quantidades a mais que foram ocultadas, não um novo sorteio. Isso reduz o ruído do resultado principal.

Alternativa: sortear uma máscara independente por taxa. Rejeitada porque soma às diferenças entre taxas a variação de sortear de novo.

Custo: dentro de uma semente, as máscaras de taxas diferentes não são independentes. A variação entre repetições vem das 3 sementes.

### 3. Sementes 0, 1 e 2, separadas da semente do split

O split continua fixo (semente 0). Só o sorteio das máscaras varia entre as repetições. Assim, o desvio reportado mede a sensibilidade ao sorteio das quantidades ocultas, sem misturar uma troca de pessoas entre os conjuntos.

### 4. Dois arquivos `.npz` em `data/processed/`

- `refeicoes_k29.npz`: `G` (float, N × K), `P` (bool, N × K), `y` (float, N), `meal_id` (int, N), `person_id` (texto, N), `conjunto` (texto, N), `gavetas` (texto, K).
- `mascaras_k29.npz`: `M` (bool, sementes × taxas × N × K), `taxas`, `sementes`, `meal_id`, `gavetas`.

Por que `.npz`: os campos são arrays NumPy, e o formato não pede dependência nova. Os textos ficam em dtype Unicode de largura fixa, carregados com `allow_pickle=False`, sem executar código ao ler. As alternativas Parquet e HDF5 pedem `pyarrow` ou `h5py`, e um array booleano de 4 dimensões não cabe naturalmente em tabela.

Por que dois arquivos: a mudança seguinte acrescenta `C` ao dataset. Com arquivos separados, regravar o dataset não toca nas máscaras.

Tamanho: `M` ocupa ≈ 130 MB em memória (3 × 5 × 304.670 × 29 bytes). Comprimido, fica bem menor em disco, porque a maior parte das células é 0.

### 5. Conferência de alinhamento na carga

A carga das máscaras recebe o dataset e confere três coisas:
1. `meal_id` igual, na ordem.
2. Nomes das gavetas iguais.
3. A máscara da taxa 1 igual a `P`, em todas as sementes.

O `meal_id` sozinho não basta: pela spec `refeicoes`, trocar a configuração de gavetas mantém os `meal_id` e muda `P`. A terceira conferência sai de graça, porque a taxa 1 é exatamente `P` (Decisão 2), e ela detecta qualquer mudança de presença. Qualquer falha levanta `ValueError` com a causa.

### 6. Organização do código

- `src/amaprob/data/gavetas.py` recebe `DIVIDIR_K29` e `FUNDIR_K29`. O `piso_de_erro` passa a importá-las, sem mudar o resultado.
- `src/amaprob/data/mascaras.py` contém `gerar_mascaras(P, taxas, sementes) -> M` (pura) e as constantes `TAXAS` e `SEMENTES`.
- `src/amaprob/data/dataset.py`:
  - a dataclass imutável `Refeicoes` com os campos do arquivo;
  - `montar_refeicoes(consumo, gavetas)`, que recebe o mapeamento `V9001 → gaveta` (o `__main__` passa a configuração K = 29, e os testes passam um mapeamento sintético);
  - gravação e carga do dataset e das máscaras;
  - um `__main__` que prepara os dois arquivos: `uv run python -m amaprob.data.dataset`.

A dataclass dá nome e tipo aos campos que a mudança seguinte vai passar adiante, sem criar hierarquia.

### 7. Testes com `pytest`

`pytest` entra no grupo de desenvolvimento (`uv add --dev pytest`).

- **Testes rápidos com dados sintéticos** (não precisam da POF): máscaras (todos os cenários da spec `mascaras`), split por pessoa, ida e volta do disco e erro de desalinhamento.
- **Gavetas K = 29:** usam o `configs/classificacao_ibge.csv` versionado, que está disponível sem os microdados.
- **Teste sobre os arquivos reais:** confere contagens, `M ⊆ P` e a taxa observada. É pulado (`pytest.skip`) se os `.npz` não existirem, para que quem não tem a POF rode a suíte.

## Risks / Trade-offs

- [O dataset é regravado com outra configuração e as máscaras antigas ficam no disco] → A conferência da Decisão 5 falha na carga, com mensagem explícita.
- [A taxa observada em uma combinação sai da tolerância de 0,002] → É improvável (≈ 4 desvios-padrão) e, como o sorteio é determinístico, aparece já na primeira execução. Nesse caso, a tolerância é revista com base no desvio-padrão, sem trocar a semente para "passar".
- [Mascaramento por gaveta oculta vários itens de uma vez] → Simplificação declarada no `AGENTS.md`; repetida em `docs/mascaramento.md`.
- [MCAR não reproduz a omissão real, que é sistemática] → Limitação declarada no artigo.
- [≈ 130 MB de máscaras em memória] → Cabe com folga. Se os laços pesados pedirem, a carga pode devolver uma semente por vez, sem mudar o arquivo.

## Migration Plan

Não há dados persistidos anteriores. Depois da implementação, cada membro do trio roda `uv sync` e depois `uv run python -m amaprob.data.dataset`. O `README.md` passa a trazer esse passo.
