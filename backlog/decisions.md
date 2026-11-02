# Decisões adiadas

Complexidade que ficou de fora de propósito e que só deve voltar quando aparecer uma dor concreta.

## Montagem de refeições sem proteção contra entradas fora do esperado

`src/amaprob/data/refeicoes.py` supõe duas condições na entrada e não verifica nenhuma delas:

- **Horário (`V9015`) ausente:** `groupby(...).ngroup()` dá `meal_id = -1` a todas as linhas com chave nula, e elas se juntam numa única "refeição" artificial.
- **Código `V9001` fora da classificação:** o item recebe `gaveta = NaN`, sai da matriz de gramas no `pivot_table`, mas o `CHOTOT` dele continua somando no carboidrato da refeição.

As duas condições valem nos microdados da versão 20230713: `V9015` não tem nulos e os 1.593 códigos da classificação cobrem todos os itens. A spec `refeicoes` registra essas condições como pré-condições, não como comportamento garantido.

**Quando voltar:** em uma troca de versão dos microdados, de classificação ou de fonte de dados. Nesse caso, a montagem deve falhar com erro explícito em vez de produzir refeições inválidas sem aviso.
