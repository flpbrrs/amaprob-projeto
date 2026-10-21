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

## Como ler os CSVs

### `results/piso_de_erro_ranking.csv`: qual grupo vale a pena dividir?

Cada linha é um teste isolado: parte das 21 gavetas básicas e divide **apenas um grupo** nos seus subgrupos do IBGE.

| Coluna | Significado |
|---|---|
| `grupo` | Grupo dividido no teste |
| `K` | Total de gavetas após a divisão (ex.: Bebidas → 21 − 1 + 14 = 34) |
| `ganho_mae_g` | Quanto o erro médio caiu, em gramas, em relação às 21 gavetas |

As linhas estão ordenadas do maior para o menor ganho.

### `results/piso_de_erro_curva.csv`: e se as divisões se acumularem?

As divisões entram uma após a outra, na ordem do ranking (Bebidas; Bebidas + Farinhas e massas; …).

| Coluna | Significado |
|---|---|
| `grupos_divididos` | Grupos já divididos até a linha (vazio = nenhum) |
| `K` | Total de gavetas |
| `mae_g` | Erro absoluto médio por refeição, em gramas |
| `r2` | Fração da variação do carboidrato entre refeições explicada pela regra (0 = nada, 1 = tudo) |
| `pct_ate_meia_dose` | % das refeições com erro de até 7,5 g |

- Várias linhas finais se repetem (K = 116): grupos com um único subgrupo, como "Sopas e caldos", não mudam ao ser "divididos".
- A última linha (`PROPOSTA`) não segue o ranking: é a configuração de 29 gavetas, incluída para comparação.

## A proposta de 29 gavetas

- **Os 20 grupos do IBGE como base**, com o Açúcar de mesa em gaveta própria (o paciente costuma saber que adoçou, mas não quanto).
- **Farinhas e massas** dividida nos seus 6 subgrupos: a farinha de mandioca (muito densa em carboidrato) deixa de se misturar com o macarrão cozido.
- **Bebidas** em 4 perfis nutricionais, em vez dos 14 subgrupos do IBGE:
  - **sem açúcar:** água, café, chá, refrigerantes e refrescos diet/light, destiladas;
  - **açucaradas:** sucos, refrescos e sucos industrializados, refrigerantes, outras não alcoólicas, bebidas à base de soja;
  - **lácteas;**
  - **cerveja e vinho.**

Ela alcança praticamente o mesmo erro da versão com 39 gavetas, usando 10 a menos.

### Por que 29, e não mais?

29 não é um ótimo matemático: nesta medição, mais gavetas sempre reduzem o erro. É um ponto de equilíbrio, sustentado por quatro argumentos.

**1. Custo-benefício de cada gaveta extra.**

| Trecho | Queda do erro | Gavetas a mais | Ganho por gaveta |
|---|---|---|---|
| 21 → 29 | 3,7 g | 8 | ~0,47 g |
| 29 → 39 | 0,1 g | 10 | ~0,01 g |
| 29 → 117 | 2,9 g | 88 | ~0,03 g |

As 8 primeiras gavetas extras rendem, cada uma, cerca de 15 vezes mais que as 88 seguintes.

**2. Custo para a imputação (Parte 1), não medido aqui.** Cada refeição usa 3 ou 4 gavetas. Com 117, quase todas ficam vazias em quase todas as refeições, várias gavetas têm poucos exemplos para aprender o tamanho da porção, e os pares de gavetas cujas combinações precisam ser aprendidas passam de ~400 (K = 29) para ~6.800 (K = 117).

**3. Interpretabilidade.** Cada gaveta tem um sentido nutricional claro (ex.: "bebidas açucaradas", "farinha de mandioca").

**4. Coerência com o plano**, que previa K entre 15 e 30.

O argumento 2 é um raciocínio, ainda não uma medição. Ele pode ser verificado depois com uma análise de sensibilidade da Parte 1 (ex.: 29 vs. 50 gavetas).

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
