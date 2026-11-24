# Qualidade da incerteza

Validação (45.639 refeições); o conjunto de teste não é lido.

- **Modelo:** FA (análise fatorial) condicional com regressão linear bayesiana, como no
  experimento central.
- **Braços:** plug-in, propagação S = 100 e analítico.
- **Taxas e sementes:** taxas 0, 0,5 e 1,0, com 3 sementes de máscara. Na taxa 0, a máscara é
  vazia e os três braços coincidem.
- **Código e números:** `experiments/qualidade_incerteza.py` e `results/qualidade_*.csv`.
- **Previsões por refeição:** ficam em `data/interim/qualidade_previsoes.npz`, fora do git.
- **Conferência:** as métricas globais reproduzem `propagacao_metricas.csv` e
  `propagacao_analitica_metricas.csv` em até 3·10⁻¹⁰.

Siglas: MAE (erro absoluto médio), RMSE (raiz do erro quadrático médio), PIT (transformada integral
de probabilidade, F(y) com a CDF preditiva), PICP90 (cobertura do intervalo de 90%), MPIW
(largura média do intervalo de 90%).

Em todas as tabelas, a razão RMSE / desvio usa como desvio a raiz da variância prevista média do
grupo. Uma preditiva calibrada tem E[(y − μ)²] = σ², então a razão é 1. Abaixo de 1, o modelo
superestima o próprio erro; acima de 1, subestima.

## 1. O modelo quantifica bem? Calibração em vários níveis

Cobertura empírica dos intervalos centrais (média de 3 sementes). A cobertura de cada nível sai
da PIT, e a de 90% coincide com a do intervalo por bissecção (diferença 0).

| taxa | braço | 50% | 60% | 70% | 80% | 90% | 95% | 99% | desvio máx. da PIT uniforme |
|---|---|---|---|---|---|---|---|---|---|
| 0 | os três (iguais) | 0,740 | 0,798 | 0,848 | 0,889 | 0,931 | 0,950 | 0,973 | 0,117 |
| 0,5 | plug-in | 0,480 | 0,551 | 0,623 | 0,695 | 0,780 | 0,832 | 0,904 | 0,074 |
| 0,5 | propagação | 0,644 | 0,731 | 0,810 | 0,879 | 0,935 | 0,960 | 0,983 | 0,064 |
| 0,5 | analítica | 0,681 | 0,773 | 0,850 | 0,902 | 0,941 | 0,958 | 0,977 | 0,075 |
| 1,0 | plug-in | 0,251 | 0,319 | 0,394 | 0,479 | 0,590 | 0,671 | 0,794 | 0,219 |
| 1,0 | propagação | 0,573 | 0,678 | 0,777 | 0,866 | 0,939 | 0,968 | 0,990 | 0,052 |
| 1,0 | analítica | 0,634 | 0,748 | 0,849 | 0,912 | 0,946 | 0,962 | 0,978 | 0,090 |

O desvio máximo da PIT é medido em 10 classes (uniforme = 0,10 por classe).

- **Taxa 0:** sobram classes no centro e faltam nas pontas (0,118 contra 0,055 por classe). É a
  forma de um resíduo de caudas pesadas sob uma gaussiana.
- **Plug-in na taxa 1,0:** 32% das refeições caem na primeira classe da PIT e 20% na última.
- **Figura:** `results/figuras/qualidade_calibracao.png`.

**Resposta.** A cobertura de 90% está perto do nominal na propagação e na analítica (0,93 a
0,95), mas a calibração em vários níveis mostra que isso é um ponto de cruzamento. Os intervalos
estreitos cobrem demais (50% nominal: 0,57 a 0,74 empírico), e o de 99% cobre de menos na analítica
(0,973 a 0,978). Essa forma já aparece na taxa 0, com as gramas completas, e vem das caudas
pesadas do resíduo da regressão. O plug-in fica abaixo do nominal em todos os níveis com
quantidades ocultas.

## 2. O modelo sabe quando está errando?

**Por decil do desvio previsto, braço analítico** (média de 3 sementes):

| decil | taxa 0,5: desvio (g) | RMSE (g) | razão | PICP90 | taxa 1,0: desvio (g) | RMSE (g) | razão | PICP90 |
|---|---|---|---|---|---|---|---|---|
| 1 | 13,8 | 9,3 | 0,67 | 0,97 | 19,2 | 21,1 | 1,10 | 0,93 |
| 2 | 13,8 | 16,4 | 1,19 | 0,89 | 25,2 | 25,6 | 1,01 | 0,93 |
| 3 | 13,9 | 14,5 | 1,04 | 0,94 | 30,0 | 30,6 | 1,02 | 0,93 |
| 4 | 15,8 | 14,5 | 0,92 | 0,94 | 33,5 | 34,8 | 1,04 | 0,94 |
| 5 | 19,8 | 20,2 | 1,02 | 0,94 | 35,8 | 35,6 | 0,99 | 0,95 |
| 6 | 23,7 | 22,1 | 0,93 | 0,95 | 37,5 | 35,2 | 0,94 | 0,95 |
| 7 | 27,4 | 27,2 | 0,99 | 0,94 | 39,8 | 37,8 | 0,95 | 0,96 |
| 8 | 32,1 | 33,1 | 1,03 | 0,94 | 43,3 | 40,7 | 0,94 | 0,95 |
| 9 | 36,8 | 34,6 | 0,94 | 0,95 | 47,4 | 41,9 | 0,89 | 0,95 |
| 10 | 58,1 | 42,5 | 0,73 | 0,96 | 73,8 | 50,8 | 0,69 | 0,97 |

A propagação tem o mesmo perfil, com razões de 0,67 a 1,19 na taxa 0,5 e de 0,68 a 1,16 na
taxa 1,0.

- **Plug-in:** o desvio fica em 13,8 g em todos os decis e taxas, e a razão vai de 1,5 a 3,2.
- **Taxa 0:** o desvio é praticamente constante (13,8 g), porque o ruído da regressão é
  homocedástico, e o RMSE vai de 5,7 a 25,1 g entre os decis.
- **Figura:** `results/figuras/qualidade_desvio_rmse.png`.

**Correlação de Spearman entre o desvio previsto e |erro|:**

| taxa | plug-in | propagação | analítica |
|---|---|---|---|
| 0 | 0,294 | 0,294 | 0,294 |
| 0,5 | 0,144 | 0,494 | 0,495 |
| 1,0 | 0,087 | 0,308 | 0,315 |

**Risco × cobertura:** MAE (g) e % de meia dose entre as x% refeições de menor desvio. Pela ordem
aleatória, o MAE fica no valor global em todo x.

| taxa | braço | 10% | 20% | 50% | 80% | 100% (= aleatória) |
|---|---|---|---|---|---|---|
| 0,5 | plug-in | 14,1 / 42,5% | 13,4 / 46,2% | 13,4 / 46,8% | 14,5 / 43,7% | 15,7 / 41,6% |
| 0,5 | propagação | 6,0 / 73,2% | 8,0 / 67,1% | 9,2 / 60,6% | 12,8 / 47,9% | 15,7 / 41,6% |
| 0,5 | analítica | 6,1 / 73,2% | 7,9 / 67,1% | 9,1 / 60,5% | 12,7 / 47,7% | 15,6 / 41,5% |
| 1,0 | plug-in | 23,7 / 18,9% | 22,8 / 20,0% | 22,4 / 20,9% | 23,8 / 20,1% | 24,8 / 19,9% |
| 1,0 | propagação | 13,7 / 33,5% | 15,9 / 30,2% | 20,0 / 24,5% | 22,5 / 21,6% | 24,8 / 19,9% |
| 1,0 | analítica | 13,6 / 32,7% | 15,5 / 29,7% | 19,9 / 24,3% | 22,3 / 21,6% | 24,7 / 19,7% |

Figura: `results/figuras/qualidade_risco_cobertura.png`.

**Resposta.** Com a propagação e a analítica, sim: o RMSE acompanha o desvio previsto (razão
entre 0,89 e 1,19 do 2º ao 9º decil), a correlação de Spearman sobe de 0,29 para 0,49 na taxa
0,5, e aceitar só os 10% de menor desvio reduz o MAE de 15,6 para 6,0 g (taxa 0,5) e de 24,7 para
13,6 g (taxa 1,0). Há duas exceções:
- no decil de maior desvio, o modelo superestima o erro (razão 0,69 a 0,73);
- o plug-in não sabe quando erra, porque o desvio dele não muda com as ocultas.

## 3. Incerteza por gaveta

**(a) Carboidrato, refeições com a gaveta oculta, taxa 0,5, propagação.** O detalhe completo
(29 gavetas × presente e oculta × braço × taxa) está em `qualidade_gavetas_carboidrato.csv`.

| gaveta | n por semente | desvio (g) | RMSE (g) | razão | PICP90 | MPIW (g) |
|---|---|---|---|---|---|---|
| Doces | 1.488 | 49,2 | 37,3 | 0,50 | 0,93 | 124,0 |
| Laticínios | 1.691 | 35,8 | 30,2 | 0,71 | 0,95 | 106,3 |
| Óleos e gorduras | 3.663 | 27,2 | 23,1 | 0,76 | 0,96 | 84,8 |
| Açúcar | 6.747 | 27,2 | 26,9 | 0,85 | 0,94 | 85,0 |
| Bebidas / sem açúcar | 8.678 | 23,5 | 23,3 | 0,89 | 0,95 | 73,9 |
| Carnes | 8.290 | 25,6 | 27,9 | 0,98 | 0,94 | 81,1 |
| Cereais | 8.420 | 30,5 | 31,5 | 0,99 | 0,93 | 96,4 |
| Raízes e tubérculos | 1.172 | 42,1 | 48,5 | 1,11 | 0,91 | 128,9 |
| Frutas | 2.166 | 23,4 | 28,4 | 1,12 | 0,92 | 74,2 |
| Sopas e caldos | 556 | 20,5 | 25,4 | 1,16 | 0,91 | 65,9 |
| Preparações mistas | 303 | 31,5 | 41,2 | 0,81 | 0,86 | 95,9 |

**Razão e PICP90 por braço** (gaveta oculta, taxa 0,5):

| braço | razão RMSE / desvio | PICP90 |
|---|---|---|
| propagação | 0,50 a 1,16 | 0,86 a 0,96 |
| analítica | 0,51 a 1,15 | 0,88 a 0,97 |
| plug-in | 1,66 a 3,51 | 0,47 a 0,83 |

**Taxa 0, gaveta presente:** a razão vai de 0,38 (Macarrão instantâneo) a 1,65 (Doces), e a
PICP90 de 0,80 (Doces) a 0,99. Com as gramas completas, o desvio é o mesmo para toda refeição,
e as gavetas diferem só no erro.

**(b) Imputação da célula oculta, pela forma fechada da FA condicional** (taxa 0,5, média de 3
sementes; detalhe em `qualidade_gavetas_imputacao.csv`). O CSV por gaveta do main
(`imputacao_por_gaveta.csv`) tem PICP e MAE, mas não o desvio previsto nem o RMSE; por isso a
tabela é calculada aqui.

| gaveta | desvio em log | RMSE em log | razão em log | desvio (g) | RMSE (g) | razão em g | PICP90 |
|---|---|---|---|---|---|---|---|
| Doces | 1,24 | 1,26 | 1,01 | 129,4 | 90,0 | 0,42 | 0,90 |
| Laticínios | 0,83 | 0,85 | 1,02 | 174,2 | 106,0 | 0,52 | 0,89 |
| Bebidas / açucaradas | 0,43 | 0,36 | 0,78 | 144,5 | 132,8 | 0,84 | 0,97 |
| Açúcar | 0,43 | 0,47 | 1,03 | 8,3 | 9,0 | 0,90 | 0,96 |
| Cereais | 0,55 | 0,56 | 1,01 | 76,2 | 83,7 | 1,09 | 0,90 |
| Carnes | 0,63 | 0,63 | 0,99 | 86,3 | 98,8 | 1,12 | 0,90 |
| Molhos e condimentos | 0,48 | 0,62 | 1,19 | 8,7 | 13,9 | 1,37 | 0,88 |
| Suplementos (n = 18) | 0,34 | 0,55 | 1,62 | 12,5 | 17,9 | 1,35 | 0,80 |

Nas 29 gavetas, na taxa 1,0, a razão em log tem mediana 1,01 (de 0,74 a 1,57) e a PICP90
mediana é 0,90 (de 0,82 a 0,97).

**Resposta.** A incerteza da imputação está bem dimensionada em log em quase todas as gavetas
(razão de 0,99 a 1,03 nas maiores). Em gramas, ela se afasta em duas direções:
- nas gavetas de porção muito variável (Doces, Laticínios), a variância da log-normal fica
  grande demais (razão 0,42 e 0,52);
- nas de porção pequena e irregular (Molhos, Suplementos), fica pequena demais.

No carboidrato, a propagação e a analítica mantêm a PICP90 entre 0,86 e 0,97 por gaveta oculta,
e o plug-in fica entre 0,47 e 0,83.

## 4. A incerteza dos doces está crescendo?

**Braço analítico, média de 3 sementes** (a propagação está no CSV, com o mesmo padrão). Na taxa
1,0, toda gaveta presente está oculta, e as duas linhas coincidem.

| gaveta | condição | taxa | n por semente | desvio (g) | RMSE (g) | razão | PICP90 | MPIW (g) |
|---|---|---|---|---|---|---|---|---|
| Doces | presente | 0 | 2.991 | 13,8 | 22,7 | 1,65 | 0,80 | 45,4 |
| Doces | presente | 0,5 | 2.991 | 36,6 | 32,6 | 0,62 | 0,89 | 120,3 |
| Doces | oculta | 0,5 | 1.488 | 52,4 | 36,6 | 0,51 | 0,95 | 172,5 |
| Doces | oculta | 1,0 | 2.991 | 59,4 | 41,5 | 0,56 | 0,95 | 195,3 |
| Açúcar | presente | 0 | 13.522 | 13,8 | 12,0 | 0,87 | 0,94 | 45,4 |
| Açúcar | oculta | 0,5 | 6.747 | 27,5 | 26,8 | 0,85 | 0,95 | 90,5 |
| Açúcar | oculta | 1,0 | 13.522 | 41,7 | 37,2 | 0,84 | 0,95 | 137,3 |
| Cereais | presente | 0 | 16.795 | 13,8 | 11,8 | 0,86 | 0,95 | 45,4 |
| Cereais | oculta | 0,5 | 8.420 | 30,8 | 31,4 | 0,98 | 0,94 | 101,4 |
| Cereais | oculta | 1,0 | 16.795 | 38,6 | 38,2 | 0,95 | 0,95 | 127,0 |
| Carnes | presente | 0 | 16.544 | 13,8 | 12,7 | 0,92 | 0,94 | 45,4 |
| Carnes | oculta | 0,5 | 8.290 | 25,9 | 27,7 | 0,97 | 0,95 | 85,2 |
| Carnes | oculta | 1,0 | 16.544 | 39,1 | 39,4 | 0,95 | 0,95 | 128,5 |

**Fração da variância preditiva atribuída à gaveta** (decomposição do braço analítico; na taxa 0
não há variância de entrada):

| gaveta | taxa | da variância total | da variância de entrada | média nas refeições com a gaveta oculta |
|---|---|---|---|---|
| Doces | 0,5 | 17,9% | 23,2% | 45,4% |
| Doces | 1,0 | 17,1% | 19,3% | 39,3% |
| Açúcar | 0,5 | 2,9% | 3,8% | 18,2% |
| Açúcar | 1,0 | 5,4% | 6,1% | 16,4% |
| Cereais | 0,5 | 10,4% | 13,5% | 52,6% |
| Cereais | 1,0 | 12,0% | 13,5% | 41,2% |
| Carnes | 0,5 | 0,2% | 0,2% | 1,4% |
| Carnes | 1,0 | 0,3% | 0,4% | 1,0% |

**Resposta.** Sim, a incerteza dos doces cresce com a taxa, e mais rápido que o erro. O desvio
previsto vai de 13,8 a 59,4 g, enquanto o RMSE vai de 22,7 a 41,5 g, e a razão passa de 1,65
(subestimada com as gramas completas) para 0,51 a 0,62 (superestimada com os doces ocultos). Nas
outras três gavetas, a razão fica estável entre 0,84 e 0,98.

Os doces estão em 6,6% das refeições e respondem por 17 a 18% da variância preditiva total. A
causa está na imputação: em log, ela é calibrada (razão 1,01), mas o desvio de 1,24 em log faz a
variância da log-normal em gramas crescer mais que o erro (razão 0,42). A PICP90 continua entre
0,89 e 0,95, porque o intervalo é assimétrico e o excesso fica na cauda direita.

A imputação dos doces é calibrada em log (razão 1,01). Em gramas, a razão RMSE / desvio abaixo de 1
reflete a cauda direita da log-normal, em que a variância é dominada por valores raros. Para
gavetas assimétricas, a medida adequada é a cobertura por quantil (parte 3a).

## Tempos

| etapa | tempo |
|---|---|
| ajuste da FA condicional e da regressão | cerca de 1,5 min |
| previsões, 7 combinações de taxa e semente | 7,3 min (58 a 64 s cada) |
| análises e figuras | 4 s |
| total | 8,9 min |

## Suposições e limitações

1. **Taxa 0 com uma semente:** a máscara é vazia nas três sementes, então só a semente 0 é medida.
2. **Desvio do decil:** é a raiz da variância média do decil. Na mistura, a variância é a total
   (média das variâncias + variância das médias).
3. **Dois primeiros decis da taxa 0,5:** têm o mesmo desvio (13,8 g), o das refeições sem
   quantidade oculta. A divisão entre eles segue a ordem do pequeno termo de incerteza dos pesos,
   e por isso a razão oscila entre 0,67 e 1,19 ali.
4. **Imputação pela forma fechada:** a parte 3b usa a gaussiana em log sem o corte em 0 que
   `FA.sample` aplica. A PICP90 resultante bate com a do CSV do main só aproximadamente: aqui o
   intervalo é exato, lá vem dos quantis de 100 amostras.
5. **Ordem aleatória:** é a média de 50 permutações com semente fixa; o valor esperado é o MAE
   global em todo x.
6. **Conferência:** as métricas globais dos três braços reproduzem as rodadas oficiais.
