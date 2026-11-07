# Proposal

> Status: **pronta para implementar**. Escopo, decisões, design, specs e tarefas fechados.

## Why

A hipótese central do projeto é que propagar a incerteza da imputação até a regressão produz intervalos calibrados, enquanto o plug-in (imputar uma vez e prever) subestima a incerteza. O pipeline da taxa 0 já roda (change `regressao-taxa-zero`), mas nenhuma taxa maior que 0 foi avaliada. Esta mudança produz o resultado principal do artigo: a curva de erro e calibração em função da taxa de mascaramento.

O uso pretendido orienta o desenho: as refeições completas (prescrição, POF) existem antes do uso, e a falta de quantidade só aparece no relato do paciente, que sabe o que comeu e esquece quanto. A pergunta do artigo é **"o modelo sabe quando não sabe?"**: a incerteza por refeição separa as estimativas confiáveis das que pediriam mais informação, melhor que uma margem de erro fixa?

## What Changes

- **Ajuste no treino completo:** o imputador e o regressor são ajustados com as gramas completas do treino, um de cada para todas as taxas e sementes. As máscaras atuam na validação; no treino, só para o braço recalibrado.
  - **Decisão tomada:** com o treino mascarado na mesma taxa, a taxa 1 deixaria o imputador sem nenhuma quantidade observada, e ela é o cenário "sei o que comi, não sei quanto". Substitui o protocolo de máscaras no treino, cujo motivo era o VAE, hoje fora do escopo.
- **Plug-in:** cada quantidade oculta é preenchida com a média, em gramas, das amostras do imputador convertidas por `expm1`, e a regressão prevê uma vez.
  - **Decisão tomada:** converter a média do log (e^μ − 1) subestimaria as gramas. Para uma log-normal, a média em gramas é e^(μ+σ²/2) − 1, cerca de 16% acima de e^μ − 1 com σ ≈ 0,6. O plug-in perderia por um viés de conversão, e não pela incerteza que a hipótese mede.
  - Por que a média das amostras, e não a fórmula fechada: a fórmula vale só para o baseline gaussiano, enquanto a média das amostras vale para qualquer imputador (PPCA e FA incluídos) e usa só o contrato `Imputer.sample`.
- **Propagação:** para cada uma das S amostras da imputação, as gramas imputadas (`expm1`) entram em `Z`, e a regressão devolve `(μ_s, σ²_s)`. A preditiva é a mistura `(1/S) Σ N(μ_s, σ²_s)`.
  - Plug-in e propagação usam o mesmo regressor e as mesmas amostras. Com a regressão linear, os dois têm a mesma média preditiva (a média das previsões é a previsão da média das gramas); a diferença aparece só na incerteza (NLL, CRPS, PICP, MPIW).
- **Terceiro braço, plug-in recalibrado (margem fixa):** regressor treinado com as gramas imputadas (média) do treino mascarado, um por taxa e semente; na validação, a mesma entrada do plug-in. A variância de ruído absorve o erro de imputação como uma folga igual para toda refeição.
  - **Decisão tomada:** só com a regressão linear bayesiana; no SVGP fica de fora se o tempo apertar, e isso é declarado.
  - **Decisão tomada:** a imputação do treino usa ajuste cruzado por pessoa (5 dobras). O imputador de cada dobra é ajustado sem as pessoas que ele imputa, para que nenhum imputador veja as gramas que vai preencher (Regra 2).
  - Não se combina com a propagação, porque contaria a incerteza da imputação duas vezes.
- **Leitura por número de quantidades ocultas:** cobertura, largura, MAE e percentual de meia dose por número de gavetas ocultas na refeição (0, 1, 2, 3, 4 ou mais), para os três braços.
  - **Decisão tomada:** a largura do intervalo do plug-in e do plug-in recalibrado é praticamente constante na regressão linear, então faixas de largura não comparariam os braços. O número de quantidades ocultas sai da máscara, existe igual nos três braços e é conhecido pelo app no uso.
- **Número de amostras S:** comparar S = 100 e S = 500 na propagação, na taxa 0,5 e semente 0.
  - **Decisão tomada:** adota-se S = 100 se o CRPS variar menos de 1% (relativo) e a PICP menos de 0,5 ponto percentual entre os dois; senão, S = 500. O tempo de cada S fica registrado.
- **Métricas para mistura:** NLL exata (via `logsumexp`), CRPS exato da mistura de normais, intervalo central de 90% pelos quantis exatos da mistura e MAE pela média da mistura.
- **Imputador nesta mudança:** só o baseline (porção média). PPCA e FA entram depois como mais uma linha do mesmo experimento, sem mudar o código da propagação.
- **Experimento:** taxas 0 a 1 e as 3 sementes de máscara, ajuste no treino e medida na validação. Amostragem e previsão em blocos de refeições.
- **Formato do resultado:** tabela taxa × braço com média ± desvio das 3 sementes e figuras de PICP e MPIW pela taxa, com a PICP da taxa 0 como referência (além de 90%).

## Capabilities

### New Capabilities

- `propagacao`: plug-in, plug-in recalibrado e propagação da incerteza da imputação para a regressão, sobre os contratos `Imputer` e `ProbRegressor`, e o experimento por taxa.

### Modified Capabilities

- `metricas`: métricas para a preditiva em mistura de gaussianas e leitura por número de quantidades ocultas.

## Impact

- Código novo: `src/amaprob/regression/propagation.py`, `src/amaprob/experiments/propagacao.py`, `tests/test_propagacao.py`.
- Código alterado: `src/amaprob/evaluation/metricas.py`, `tests/test_metricas.py`.
- Novas dependências: `plotnine` (gramática de gráficos sobre o `matplotlib`), para as figuras, e `scipy`, para as métricas da mistura. Pacotes consolidados.
- Tema comum das figuras em `src/amaprob/evaluation/tema.py`, com as fontes Roboto Condensed versionadas em `assets/fonts/`, para que todas as figuras do artigo tenham o mesmo padrão em qualquer máquina.
- `AGENTS.md` §4 (treino completo; máscara no treino só para o braço recalibrado) e §6 (plug-in pela média em gramas das amostras, terceiro braço, pergunta "o modelo sabe quando não sabe?"); `docs/mascaramento.md` (papel da máscara no treino).
- Resultados novos em `results/` (CSV) e `results/figuras/` (PNG).
