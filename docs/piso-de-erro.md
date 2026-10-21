# Piso de erro: quantas gavetas usar?

> Status: **proposta em validação**. Código: `src/amaprob/experiments/piso_de_erro.py`. Resultados: `results/piso_de_erro_*.csv`.

## A pergunta

O modelo não enxerga "arroz" ou "pão". Ele enxerga **gavetas**: grupos de alimentos parecidos (Cereais, Panificados…). Para cada refeição, ele sabe quantos gramas há em cada gaveta. Se uma gaveta mistura alimentos muito diferentes, o modelo erra mesmo sabendo tudo. Exemplo: "Bebidas" junta água (0 g de carboidrato) e refrigerante (~10 g por 100 ml).

**Quanto erro as gavetas sozinhas já impõem, antes de escondermos qualquer quantidade?** Esse erro é o "piso": o melhor que dá para fazer na taxa 0.0. Quando escondermos quantidades, o erro só sobe a partir dele.

## O método, em uma frase

Damos a uma regra simples **todas as gramas de cada gaveta** e pedimos para ela adivinhar o carboidrato da refeição. O erro que sobra mede quanta informação se perde ao juntar alimentos diferentes na mesma gaveta.

## Passo a passo

1. **Refeições.** Usamos as refeições que pedem insulina: mesma pessoa, dia, ocasião e horário; café, almoço, lanche, jantar ou ceia; pelo menos 10 g de carboidrato. São 304.670 refeições.
2. **Divisão por pessoa.** 70% das pessoas para treino, 15% para validação, 15% para teste, com semente fixa (`seed=0`). Os dois dias de uma pessoa ficam sempre no mesmo conjunto. **O teste não é usado aqui.**
3. **A regra.** Para cada refeição, `carbo ≈ b0 + b1·gramas_gaveta1 + b2·gramas_gaveta2 + …`. Cada `b` é o "carboidrato por grama" daquela gaveta (ex.: Cereais ≈ 0,29). Os `b` são aprendidos no **treino** por mínimos quadrados (a reta que melhor passa pelos pontos).
4. **O erro.** Aplicamos a regra nas refeições da **validação** e medimos:
   - **MAE** (erro absoluto médio): em média, quantos gramas a estimativa erra;
   - **R²**: fração da variação do carboidrato que a regra explica (1 = tudo);
   - **% até meia dose**: refeições com erro de até 7,5 g (ver abaixo).
5. **Ranking.** Partindo dos 20 grupos do IBGE (+ gaveta própria de Açúcar = 21), dividimos **um grupo por vez** nos seus subgrupos e medimos quanto o erro cai. Isso mostra onde dividir vale a pena.
6. **Curva.** Acumulamos as divisões na ordem do ranking, de 21 gavetas até todos os 116 subgrupos.

## Resultados (validação)

Referência: prever sempre a média dá MAE de 31,6 g.

| Gavetas (K) | O que foi dividido | MAE (g) | R² | % até meia dose |
|---|---|---|---|---|
| 21 | nada (grupos do IBGE + Açúcar) | 11,8 | 0,82 | 49% |
| **29** | **Proposta:** Bebidas em 4 perfis + Farinhas e massas | **8,1** | **0,90** | **67%** |
| 39 | Bebidas (14 subgrupos) + Farinhas e massas | 8,0 | 0,90 | 67% |
| 50 | + Frutas | 7,1 | 0,92 | 71% |
| 117 | todos os subgrupos | 5,2 | 0,95 | 79% |

Ganho ao dividir cada grupo sozinho (top 6): Bebidas −1,8 g · Farinhas e massas −1,7 g · Frutas −0,8 g · Laticínios −0,4 g · Panificados −0,3 g · Doces −0,3 g. Os demais ficam abaixo de 0,1 g.

## A proposta de 29 gavetas

- **Os 20 grupos do IBGE como base**, com o Açúcar de mesa em gaveta própria (o paciente costuma saber que adoçou, mas não quanto).
- **Farinhas e massas** dividida nos seus 6 subgrupos: a farinha de mandioca (muito densa em carboidrato) deixa de se misturar com o macarrão cozido.
- **Bebidas** em 4 perfis nutricionais, em vez dos 14 subgrupos do IBGE:
  - **sem açúcar:** água, café, chá, refrigerantes e refrescos diet/light, destiladas;
  - **açucaradas:** sucos, refrescos e sucos industrializados, refrigerantes, outras não alcoólicas, bebidas à base de soja;
  - **lácteas;**
  - **cerveja e vinho.**

Ela alcança praticamente o mesmo erro da versão com 39 gavetas, usando 10 a menos.

**Por que não 117 gavetas, se o erro é menor?** Porque o piso é só metade da história. Cada refeição usa 3 ou 4 gavetas. Com 117, quase todas ficam vazias em quase todas as refeições, e os modelos de imputação (Parte 1) têm muito pouco exemplo por gaveta para aprender tamanhos de porção e combinações. Esse custo não aparece nesta medição.

## Critério de meia dose

Com a relação típica de 1 unidade de insulina para 15 g de carboidrato, meia unidade equivale a 7,5 g. Um estudo da Roche trata a meia dose como erro aceitável (**fonte a verificar**). Observações:

- A relação insulina:carboidrato varia de paciente para paciente, então 7,5 g é uma referência, não um limite universal.
- Mesmo com todas as gramas conhecidas, a regra linear com 29 gavetas acerta dentro da meia dose em 67% das refeições. A meia dose funciona melhor como **métrica de avaliação** ("% de refeições dentro da meia dose") do que como meta de erro médio.

## Limitações desta medição

- A regra é **linear e não usa contexto**. Os modelos da Parte 2 (ex.: Processo Gaussiano) podem ficar um pouco abaixo deste piso.
- Uma única divisão treino/validação (`seed=0`).
- Outliers foram mantidos (há refeições com mais de 1 kg de carboidrato registrado).
- O ranking divide um grupo inteiro nos subgrupos do IBGE; outras divisões (como os perfis de Bebidas) são escolhas manuais, justificadas por nutrição.

## Como reproduzir

```bash
uv sync
# (uma vez) gerar a classificação do IBGE a partir da planilha original:
uv run --with openpyxl python -m amaprob.data.gavetas
uv run python -m amaprob.experiments.piso_de_erro
```

`configs/classificacao_ibge.csv` (versionado) traz cada alimento com seu grupo e subgrupo do IBGE, para revisão.
