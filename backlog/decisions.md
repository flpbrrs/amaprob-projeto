# Decisões adiadas

Complexidade que ficou de fora de propósito e que só deve voltar quando aparecer uma dor concreta.

## Montagem de refeições sem proteção contra entradas fora do esperado

`src/amaprob/data/refeicoes.py` supõe duas condições na entrada e não verifica nenhuma delas:

- **Horário (`V9015`) ausente:** `groupby(...).ngroup()` dá `meal_id = -1` a todas as linhas com chave nula, e elas se juntam numa única "refeição" artificial.
- **Código `V9001` fora da classificação:** o item recebe `gaveta = NaN`, sai da matriz de gramas no `pivot_table`, mas o `CHOTOT` dele continua somando no carboidrato da refeição.

As duas condições valem nos microdados da versão 20230713: `V9015` não tem nulos e os 1.593 códigos da classificação cobrem todos os itens. A spec `refeicoes` registra essas condições como pré-condições, não como comportamento garantido.

**Quando voltar:** em uma troca de versão dos microdados, de classificação ou de fonte de dados. Nesse caso, a montagem deve falhar com erro explícito em vez de produzir refeições inválidas sem aviso.

## Modelo de ruído da regressão com caudas pesadas ou variância crescente

A regressão linear bayesiana usa ruído gaussiano homocedástico (uma variância única para todas as refeições). Na taxa 0 (validação), o resíduo tem curtose em excesso de ~30 (0 numa gaussiana) e MAE/desvio de 0,58 (0,80 numa gaussiana). O intervalo de 90% cobre 93,1% no total, mas vai de 97,6% nas refeições com menor previsão a 84,1% nas maiores (`results/taxa_zero_cobertura.csv`).

Alternativas adiadas:

- **Verossimilhança t de Student:** caudas pesadas, mas sem forma fechada; pede inferência variacional ou MCMC (cadeias de Markov via Monte Carlo).
- **Variância que cresce com a refeição** (por exemplo, proporcional à previsão ou ao número de gavetas): também sem forma fechada.
- **Transformar o alvo** (raiz ou log do carboidrato): quebra a relação quase linear entre gramas e carboidrato.

**Por que ficou de fora:** plug-in e propagação dividem o mesmo modelo de ruído, então a comparação entre eles continua justa. A falha de calibração da taxa 0 é medida e reportada, e a curva é lida em relação a ela.

**Quando voltar:** se a curva plug-in × propagação ficar ambígua por causa da referência da taxa 0, ou se o SVGP mostrar cobertura por faixa muito diferente da regressão linear bayesiana.

## Regra do IBGE para os gramas de açúcar adicionado

Na POF, o informante só diz se adicionou açúcar (sim ou não); os gramas foram estimados depois pelo IBGE, com uma regra que a documentação dos microdados não descreve (ver `AGENTS.md` §2). Assim, a parte do açúcar no carboidrato da refeição (`y`) e na gaveta de açúcar é uma estimativa, não uma medida declarada.

**Por que ficou de fora:** declarar a limitação no artigo basta enquanto ela não afetar a conclusão. Reconstituir a regra (cruzando os gramas de açúcar com o alimento e a quantidade a que ele foi adicionado) não muda nenhum modelo.

**Quando voltar:** se o erro de imputação da gaveta de açúcar destoar muito das outras gavetas (uma regra fixa deixaria o açúcar fácil demais de prever), ou se a conclusão plug-in × propagação mudar quando a gaveta de açúcar sai da análise.
