# Mascaramento: quais quantidades ficam ocultas

> Código: `src/amaprob/data/mascaras.py` (geração) e `src/amaprob/data/dataset.py` (persistência). Testes: `tests/test_mascaras.py`, `tests/test_dados_processados.py`.

## A pergunta

O projeto simula registros alimentares incompletos: a pessoa informa **quais** alimentos comeu, mas não **quanto** de alguns deles. Para isso, partimos dos registros reais da POF (Pesquisa de Orçamentos Familiares), em que todas as quantidades são conhecidas, e **escondemos** parte delas. O registro completo funciona como gabarito: o modelo recebe o registro com quantidades ocultas, estima o que falta, e a estimativa é comparada com o valor real.

A **máscara** `M` diz quais quantidades ficam ocultas. Ela tem a forma da matriz de presença `P` (refeições × gavetas):

- `P[i, k] = 1` se a refeição `i` tem algum alimento da gaveta `k`. A presença é **sempre observada**.
- `M[i, k] = 1` se a quantidade (gramas) da gaveta `k` na refeição `i` está **oculta**. Só gavetas presentes podem ser ocultadas: `M ⊆ P`.

A **taxa** r é a fração das quantidades presentes que fica oculta. O protocolo usa r ∈ {0; 0,25; 0,5; 0,75; 1}. Em r = 1, nenhuma quantidade é informada: "sei o que comi, não sei quanto".

## Por que as máscaras são geradas antes e gravadas em disco

Todos os modelos precisam ser avaliados sobre **as mesmas quantidades ocultas**. Se cada modelo sorteasse a própria máscara, a diferença entre eles misturaria "modelo melhor" com "sorteio mais fácil". Por isso as máscaras são geradas uma vez, com semente fixa, antes de qualquer modelagem, e gravadas em `data/processed/mascaras_k29.npz` (Regra 4 do `AGENTS.md`). Treino, validação e teste são mascarados pelo mesmo sorteio.

## Decisão 1: cada gaveta presente é sorteada de forma independente

Para uma taxa r, **cada gaveta presente de cada refeição fica oculta com probabilidade r**, independentemente das outras gavetas, das quantidades, do carboidrato e do conjunto (treino, validação ou teste). É uma moeda viciada por célula.

Esse mecanismo é chamado de **MCAR** (*missing completely at random*, ausência completamente ao acaso). Duas alternativas foram consideradas:

| Alternativa | Como funcionaria | Por que não |
|---|---|---|
| Contagem exata por refeição | Em cada refeição, ocultar `round(r × n)` das `n` gavetas presentes | **Viés.** 15,6% das refeições têm uma gaveta só. Para elas, `round(0,25 × 1) = 0` e `round(0,5 × 1) = 0` (o Python arredonda 0,5 para o par). Essas refeições nunca seriam mascaradas abaixo de r = 0,75, e a taxa efetiva passaria a depender do tamanho da refeição |
| Contagem exata no conjunto todo | Ocultar exatamente r × 941.320 células, sorteadas entre todas | Taxa exata, mas ganho desprezível: com o sorteio independente, o desvio-padrão da taxa é no máximo √(0,25 / 941.320) ≈ 0,0005. Além disso, as células deixam de ser independentes |

Gavetas presentes por refeição (versão 20230713, 304.670 refeições):

| Gavetas presentes | 1 | 2 | 3 | 4 | 5 | 6 ou mais |
|---|---|---|---|---|---|---|
| Refeições | 47.498 (15,6%) | 58.711 | 85.011 | 69.231 | 27.681 | 16.538 |

Com o sorteio independente, a taxa é garantida em média. Nos dados reais, a taxa observada fica a menos de 0,001 do alvo (tabela abaixo).

**Por gaveta, não por item:** ocultar uma gaveta esconde de uma vez todos os alimentos dela naquela refeição. É uma simplificação: na prática, uma pessoa pode lembrar a quantidade do arroz e esquecer a do macarrão, mesmo que ambos estejam na mesma gaveta.

## Decisão 2: máscaras aninhadas entre as taxas

Para cada semente, cada célula recebe **uma única nota sorteada** U entre 0 e 1. Em cada taxa r, ficam ocultas as gavetas presentes com nota menor que r:

```
M_r = P ∧ (U < r)
```

Exemplo com uma refeição de quatro gavetas:

```
gaveta:        Arroz  Feijao  Carnes  Acucar
nota U:        0.12   0.61    0.33    0.89
taxa 0.25:      X      .       .       .
taxa 0.50:      X      .       X       .
taxa 0.75:      X      X       X       .
taxa 1.00:      X      X       X       X
```

Propriedades:

- **Cada taxa, isolada, segue a Decisão 1:** a chance de U ser menor que r é exatamente r.
- **Aninhamento:** o que está oculto em 0,25 continua oculto em 0,5, em 0,75 e em 1.
- **Extremos exatos:** em r = 0 nada fica oculto; em r = 1 a máscara é igual a `P`.

**Por que aninhar.** O resultado principal do projeto é uma curva: erro e calibração do intervalo em função da taxa. Com máscaras aninhadas, cada ponto da curva usa **as mesmas refeições**, e de um ponto para o seguinte só se acrescentam quantidades ocultas. É como um exame de vista que mostra letras cada vez menores ao mesmo paciente: a diferença entre as linhas reflete só a dificuldade. Com uma máscara independente por taxa, a diferença entre dois pontos somaria o efeito da taxa e a variação de um novo sorteio.

**Custo.** Dentro de uma semente, as máscaras de taxas diferentes não são independentes entre si. A variação entre repetições vem das sementes (Decisão 3).

## Decisão 3: três sementes, com o split fixo

As máscaras são geradas com as sementes 0, 1 e 2. A divisão treino/validação/teste fica fixa (semente 0, por pessoa). Assim, a média e o desvio-padrão reportados entre as três repetições medem a sensibilidade ao **sorteio das quantidades ocultas**, sem misturar uma troca das pessoas entre os conjuntos.

## Resultado nos dados reais

Fração das 941.320 células presentes que ficou oculta:

| Semente | 0 | 0,25 | 0,5 | 0,75 | 1 |
|---|---|---|---|---|---|
| 0 | 0 | 0,25073 | 0,50083 | 0,75073 | 1 |
| 1 | 0 | 0,25014 | 0,49993 | 0,74994 | 1 |
| 2 | 0 | 0,25021 | 0,50035 | 0,74991 | 1 |

## Limitações

- **MCAR não é como as pessoas omitem de verdade.** Na vida real, a omissão é sistemática: é mais provável esquecer a quantidade do açúcar no café do que a do prato principal. Os resultados valem para ausência ao acaso, e a extensão para mecanismos não aleatórios fica como trabalho futuro.
- **Mascaramento por gaveta** (ver Decisão 1).
- **A presença nunca é ocultada.** O cenário estudado é "sei o que comi, não sei quanto". Esquecer um alimento inteiro é outro problema e fica fora do escopo.

## Arquivos e uso

`uv run python -m amaprob.data.dataset` grava dois arquivos em `data/processed/` (≈ 11 MB no total, cerca de 30 s):

| Arquivo | Conteúdo |
|---|---|
| `refeicoes_k29.npz` | `G` (gramas por gaveta), `P`, `y` (carboidrato), `meal_id`, `person_id`, `conjunto`, `gavetas` |
| `mascaras_k29.npz` | `M` (sementes × taxas × refeições × gavetas), `taxas`, `sementes`, `meal_id`, `gavetas` |

```python
from amaprob.data.dataset import carregar_mascaras, carregar_refeicoes

refeicoes = carregar_refeicoes()
mascaras = carregar_mascaras(refeicoes)  # falha se as máscaras não forem destas refeições
M = mascaras.M[0, 2]  # semente 0, taxa 0,5
```

A carga das máscaras confere se elas pertencem ao dataset carregado: mesmo `meal_id` na mesma ordem, mesmas gavetas e máscara da taxa 1 igual a `P`. Se o dataset for regravado com outra configuração de gavetas, a carga falha com mensagem explícita, em vez de usar máscaras desalinhadas.
