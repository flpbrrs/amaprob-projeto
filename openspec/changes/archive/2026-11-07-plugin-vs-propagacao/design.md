# Design

## Context

O pipeline da taxa 0 roda (change `regressao-taxa-zero`): dataset K = 29 e máscaras em `data/processed/`, contratos `Imputer` e `ProbRegressor`, baseline `PorcaoMedia`, `RegressaoLinearBayesiana` e `metricas.avaliar`. Fatos medidos que orientam o desenho:

- Validação, taxa 0: MAE de 7,87 g, PICP de 93,1% e MPIW de 45,4 g. A incerteza dos pesos (`φᵀSφ`) é ≈ 0,03% da variância preditiva: o intervalo da regressão é, na prática, média ± z/√β, com largura quase constante.
- Tempos: ajuste da regressão linear bayesiana em 0,24 s; ajuste do baseline em 0,11 s; 100 amostras do baseline para a validação (~45 mil refeições) em 0,34 s, em blocos de 5.000 refeições.
- Memória: 100 amostras do treino inteiro ocupariam ≈ 5 GB (100 × 213 mil × 29 × 8 bytes).
- `M₁ = P` (spec `mascaras`): na taxa 1, nenhuma quantidade presente fica observada.
- `scoringrules` 0.11.0 instalado oferece `crps_normal`, `crps_mixnorm` e `crps_ensemble`. `scipy` vem como dependência dele.

## Goals / Non-Goals

**Goals:**
- A curva taxa × braço (plug-in, plug-in recalibrado, propagação) com o baseline e a regressão linear bayesiana, nas taxas 0 a 1 e nas 3 sementes.
- A leitura por número de quantidades ocultas, que responde "o modelo sabe quando não sabe?".
- Código da propagação que aceite PPCA, FA e SVGP depois sem alteração, só pelos contratos.

**Non-Goals:**
- PPCA, FA e SVGP (frentes paralelas).
- O braço recalibrado com o SVGP.
- Avaliação no conjunto de teste (fica para a rodada final).
- Métricas de imputação da Parte 1 (erro nas células ocultas): entram com o PPCA.

## Decisions

### 1. Imputador e regressor ajustados no treino completo

O imputador é ajustado com `ocultar(G_tr, M vazia)` e o regressor com `montar_Z(G_tr, P_tr, C_tr)`, uma vez para todo o experimento. As máscaras da validação definem o que é imputado na medida.

- Alternativa descartada: ajustar o imputador no treino mascarado na mesma taxa. Na taxa 1 (`M₁ = P`), o baseline teria média e desvio indefinidos e o PPCA não teria quantidade nenhuma para aprender.
- Correspondência com o uso: as refeições completas existem antes do uso; a falta só aparece no relato.
- O EM com dados faltantes continua necessário para o PPCA e a FA mesmo assim, porque cada refeição tem um conjunto diferente de gavetas presentes (proposta do `AGENTS.md` §12: gavetas ausentes como condicionantes, e não como zeros).

### 2. Os três braços

Para uma taxa e uma semente, com `M_va` a máscara da validação e as mesmas S amostras `X̃_s = imputador.sample(ocultar(G_va, M_va), …)`:

| Braço | Entrada na validação | Regressor | Preditiva |
|---|---|---|---|
| `plugin` | `Ḡ = (1/S) Σ expm1(X̃_s)` | completo | `N(μ(Ḡ), σ²(Ḡ))` |
| `plugin_recalibrado` | `Ḡ` (a mesma) | recalibrado (Decisão 3) | `N(μ_r(Ḡ), σ²_r(Ḡ))` |
| `propagacao` | `G_s = expm1(X̃_s)`, s = 1..S | completo | `(1/S) Σ N(μ(G_s), σ²(G_s))` |

- Nas células observadas, `expm1(log1p(G))` devolve `G` até o erro de ponto flutuante.
- Plug-in e propagação saem das mesmas amostras, numa só passada por bloco. Com a regressão linear, `mean_s μ(G_s) = μ(Ḡ)` por linearidade: as médias coincidem, e o teste confere isso.
- Na taxa 0 (M vazia), as amostras são cópias da entrada, e os três braços coincidem com a previsão direta do regressor completo. O regressor recalibrado também coincide, porque o treino imputado é o treino completo. É o teste de sanidade do experimento.

### 3. Braço recalibrado com ajuste cruzado por pessoa

1. As pessoas do treino são divididas em 5 dobras (permutação das pessoas únicas com `default_rng(0)`; dobra = posição mod 5).
2. Para cada dobra f, um imputador é ajustado com o treino completo das outras dobras, e as refeições de f são imputadas com `ocultar(G_f, M_f)` e S amostras; `Ḡ_f` é a média em gramas.
3. O regressor recalibrado é ajustado com `montar_Z(Ḡ_tr, P_tr, C_tr)` e `y_tr`, um por taxa e semente.

- Motivo: um imputador ajustado no treino completo já viu as gramas que vai preencher; com o ajuste cruzado, nenhum imputador vê os valores que imputa (Regra 2).
- Alternativa descartada: usar o imputador do treino completo e declarar a exceção. No baseline o efeito seria desprezível, mas com PPCA a exceção cresceria.
- Custo: 5 ajustes do baseline e S amostras do treino em blocos por taxa e semente, poucos segundos.

### 4. Métricas da mistura

`metricas.avaliar_mistura(y, medias, variancias)` recebe `(S, N)` e devolve as mesmas chaves de `avaliar`. Com S = 1, os valores coincidem com `avaliar`.

- **Média:** `μ̄ = mean_s μ_s`. MAE e percentual de meia dose usam `μ̄`.
- **NLL exata:** `−mean_n [logsumexp_s(log N(y | μ_s, σ²_s)) − log S]` (`scipy.special.logsumexp`).
- **CRPS exato:** `scoringrules.crps_mixnorm` com pesos iguais. O custo é O(S²) por refeição; o cálculo é feito em blocos de refeições com orçamento de memória fixo (Decisão 6). Alternativa: `crps_ensemble` com uma amostra por componente, mais rápido mas com ruído de Monte Carlo, que contaminaria a comparação de S.
- **Intervalo de 90%:** quantis exatos 0,05 e 0,95 da mistura, por bissecção na CDF `(1/S) Σ Φ((x − μ_s)/σ_s)` (`scipy.special.ndtr`), com o intervalo inicial `[min_s(μ_s − 8σ_s), max_s(μ_s + 8σ_s)]` e 60 iterações. PICP = fração com y dentro; MPIW = largura média. A aproximação gaussiana por momentos foi descartada: a mistura não é gaussiana quando a gaveta oculta domina a refeição.
- `scipy` passa a ser dependência direta, porque o código a importa.

### 5. Leitura por número de quantidades ocultas

`metricas.por_ocultas(y, media, li, ls, n_ocultas)` agrupa as refeições pelo número de gavetas ocultas (`M.sum(axis=1)`), nas classes 0, 1, 2, 3 e "4+", e devolve por classe: `n`, `picp`, `mpiw_g`, `mae_g` e `pct_meia_dose`.

- A função recebe os limites do intervalo (`li`, `ls`), e não a variância, para servir à gaussiana e à mistura. `metricas.intervalo(medias, variancias)` devolve `(li, ls)` para `(S, N)`; com S = 1, é a fórmula média ± z·desvio.
- Leitura esperada: a margem fixa cobre demais nas refeições com 0 ou 1 oculta e de menos nas com muitas; a propagação deve ficar estável entre as classes.
- Faixas de largura do intervalo foram descartadas: a largura do plug-in e da margem fixa é quase constante na regressão linear.

### 6. Blocos e geradores

- Um orçamento de memória (`ORCAMENTO_BYTES = 256 MB`) define o bloco de refeições: `B = ORCAMENTO / (S · K · 8)` para as amostras e `B = ORCAMENTO / (S² · 8)` para o CRPS da mistura.
- Geradores: `default_rng((semente_mascara, indice_taxa, fluxo))`, com fluxo 0 para a validação e fluxo 1 + f para a dobra f do ajuste cruzado. Reprodutível por taxa e semente e independente da ordem dos laços.

### 7. Escolha de S

Antes da curva, a propagação roda com S = 100 e S = 500 na taxa 0,5 e semente 0. Adota-se S = 100 se `|CRPS₁₀₀ − CRPS₅₀₀| / CRPS₅₀₀ < 1%` e `|PICP₁₀₀ − PICP₅₀₀| < 0,5` ponto percentual; senão, S = 500. O tempo de parede de cada S (amostragem, previsão e métricas) vai para `results/propagacao_S.csv`, e o S adotado é impresso e gravado na tabela principal.

### 8. Experimento e saídas

`uv run python -m amaprob.experiments.propagacao`:

1. Carrega dataset e máscaras; separa treino e validação com `taxa_zero.separar` (o teste nunca é lido).
2. Ajusta o imputador e o regressor completos no treino.
3. Escolhe S (Decisão 7).
4. Para cada semente e taxa: os três braços na validação e as métricas globais e por número de ocultas.
5. Grava:
   - `results/propagacao_metricas.csv`: `taxa, semente, braco, imputador, S` e as métricas de `avaliar`;
   - `results/propagacao_por_ocultas.csv`: `taxa, semente, braco, n_ocultas` e as colunas de `por_ocultas`;
   - `results/propagacao_resumo.csv`: média e desvio das métricas entre as sementes, por taxa e braço;
   - `results/propagacao_S.csv`;
   - `results/figuras/propagacao_picp.png` e `propagacao_mpiw.png`: uma linha por braço, barras de desvio entre as sementes, linhas horizontais em 90% e na PICP da taxa 0.

As figuras usam `plotnine` com o tema comum `evaluation/tema.TEMA` (fonte Roboto Condensed de `assets/fonts/`), o mesmo para todos os experimentos do projeto. Alternativa descartada: `matplotlib` direto, com estilo repetido em cada script.

Tempo esperado: dezenas de segundos por taxa e semente com S = 100; a comparação com S = 500 domina pelo CRPS O(S²).

### 9. Organização do código

```
src/amaprob/
  regression/  propagation.py  (amostras_em_gramas, prever_bracos, imputar_treino_cruzado)
  evaluation/  metricas.py     (+ avaliar_mistura, intervalo, por_ocultas)
               tema.py         (TEMA e CORES das figuras)
  experiments/ propagacao.py
```

`propagation.py` depende só dos contratos; o experimento combina as peças e escolhe os modelos.

## Risks / Trade-offs

- [CRPS da mistura O(S²) com S = 500 pode levar minutos] → Calculado em blocos com orçamento de memória; se passar de 10 minutos, a comparação de S usa `crps_ensemble` (estimador `fair`) nas duas versões de S, e a curva principal continua exata com o S adotado.
- [Bissecção vetorizada sobre S × N] → 60 iterações em blocos; o teste confere a CDF nos quantis com tolerância de 1e-6.
- [Treino completo muda o protocolo de 02/11 (máscaras no treino com o mesmo protocolo)] → `AGENTS.md` §4 e `docs/mascaramento.md` atualizados nesta mudança; as máscaras do treino continuam geradas e servem ao braço recalibrado.
- [Baseline ignora coocorrência e contexto, então a curva pode exagerar a incerteza da imputação] → É a referência; PPCA e FA entram como novas linhas do mesmo experimento.
- [Caudas pesadas: a PICP da taxa 0 já fica em 93,1%] → A curva é lida em relação à taxa 0, como decidido na change anterior.

## Migration Plan

1. `uv add matplotlib scipy plotnine` e `uv sync`.
2. Nenhum dado é regravado: o experimento lê `data/processed/` como está.
