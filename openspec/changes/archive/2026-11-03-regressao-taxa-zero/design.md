# Design

## Context

O dataset K = 29 e as máscaras já estão gravados em `data/processed/` (specs `dataset-refeicoes` e `mascaras`). Não há nenhum modelo, métrica ou contrato implementado. O `AGENTS.md` descreve os contratos `Imputer` (§5) e `ProbRegressor` (§6); este último devolve só amostras.

Fatos medidos na validação com a regra linear de mínimos quadrados sobre G e P (piso), que orientam o desenho:
- MAE de 7,85 g, desvio-padrão do resíduo de 13,6 g, curtose em excesso de 30. O resíduo tem caudas pesadas: MAE/desvio = 0,58, contra 0,80 para uma gaussiana.
- Um intervalo gaussiano de 90% com variância única cobre 93,1% no total. Por quintil da previsão, a cobertura vai de 97,5% (refeições pequenas) a 84,1% (refeições grandes).
- UF e `RENDA_TOTAL` são constantes por pessoa. `V9015` vai de `0H` a `23H`. Há 86 itens com renda menor ou igual a 0.
- `scoringrules` funciona com NumPy 2.5. O `properscoring` está sem manutenção e emite avisos de sintaxe no Python 3.12.

## Goals / Non-Goals

**Goals:**
- Circuito completo dados → `Z` → regressão → métricas, rodando em segundos, com o MAE perto do piso.
- Contratos estáveis para a mudança seguinte (plug-in × propagação) usar sem alterações.
- Testes obrigatórios restantes do `AGENTS.md`: métricas com casos conhecidos e não vazamento de `X[M]`.

**Non-Goals:**
- Modelos de ruído não gaussianos ou heterocedásticos (adiados; ver Riscos).
- Avaliar imputação em qualquer taxa.
- Métricas para misturas de gaussianas: entram com a propagação.

## Decisions

### 1. Ruído gaussiano homocedástico

A regressão usa o modelo conjugado padrão. Alternativas consideradas:
- **Verossimilhança t de Student ou variância crescente com a refeição:** perderia a forma fechada e exigiria inferência variacional ou MCMC.
- **Transformar o alvo (raiz ou log):** quebraria a relação quase linear entre gramas e carboidrato.

Plug-in e propagação vão dividir o mesmo modelo de ruído, então a comparação entre eles continua justa. Para não esconder o problema, a cobertura também é reportada por faixa da previsão (spec `metricas`). A curva de calibração será lida em relação à PICP da taxa 0, e não só em relação a 90%. A alternativa fica registrada em `backlog/decisions.md`.

### 2. Contrato `ProbRegressor` com preditiva gaussiana

```python
class ProbRegressor(Protocol):
    def fit(self, Z: np.ndarray, y: np.ndarray) -> "ProbRegressor": ...
    def predict(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Média e variância da preditiva gaussiana, cada uma de forma (N,)."""
```

O contrato anterior (`predict_samples`) obrigaria a estimar a densidade por kernel para obter a NLL. A regressão linear bayesiana e o SVGP já produzem gaussianas. Na propagação, cada amostra s da imputação gera `(μ_s, σ²_s)`, e a preditiva vira a mistura `(1/S) Σ N(μ_s, σ²_s)`:
- a NLL da mistura sai exata via `logsumexp`;
- o CRPS da mistura sai por amostras.

O parâmetro `groups` do contrato antigo sai, porque nenhum modelo do escopo usa grupos. O `AGENTS.md` §6 é atualizado.

### 3. Regressão linear bayesiana do zero (Bishop, *Pattern Recognition and Machine Learning*, §3.3 e §3.5)

- **Padronização:** cada coluna de `Z` é padronizada pela média e pelo desvio do conjunto de ajuste. Colunas com desvio 0 ficam com desvio 1 e, depois de centradas, valem 0. O alvo é centrado pela média `ȳ`. Isso equivale a um intercepto sem penalização, e o prior isotrópico passa a tratar as colunas na mesma escala.
- **Posterior:** `S⁻¹ = αI + βΦᵀΦ`, `m = βSΦᵀy`.
- **Evidência:** com `λᵢ` os autovalores de `βΦᵀΦ` e `γ = Σ λᵢ/(α + λᵢ)`, o ponto fixo é `α ← γ / mᵀm` e `1/β ← ‖y − Φm‖² / (N − γ)`, iterado até a variação relativa ficar abaixo de 1e-6 (no máximo 200 iterações). Como `ΦᵀΦ` tem apenas 71 × 71, cada iteração é barata.
- **Preditiva:** média `φᵀm + ȳ`, variância `1/β + φᵀSφ`.
- **Leitura honesta:** com ~213 mil refeições no treino, `φᵀSφ` é desprezível perto de `1/β`, e o intervalo na taxa 0 vira, na prática, média ± z/√β. A parte bayesiana pesa na estimação de α e β e, na propagação, quando a incerteza da imputação entra pela mistura.

### 4. Contexto `C` só do consumo alimentar

São 13 colunas:
- ocasião: 5 indicadoras;
- hora: seno e cosseno de 2π·h/24, para 23H ficar perto de 0H;
- macrorregião: 5 indicadoras, pelo primeiro dígito da UF;
- `log1p(max(RENDA_TOTAL, 0))`.

As indicadoras são completas (sem categoria de referência). Com o prior gaussiano, a colinearidade com o intercepto não impede o ajuste. `C` é montado em `montar_refeicoes` a partir dos itens (primeiro valor por refeição) e gravado com `C_nomes`.

Ao regravar o dataset, `meal_id`, gavetas e `P` não mudam, então a conferência de carga das máscaras continua passando.

### 5. Imputação: entrada sem os ocultos e baseline em `log1p`

- `ocultar(G, M) -> X` aplica `log1p` e põe `NaN` em `M`. É o único ponto que vê as gramas verdadeiras antes de um imputador; o imputador nunca recebe `G`.
- **Baseline:** para cada gaveta k, `μ_k` e `σ_k` de `X[:, k]` nas células com `P = 1` e `M = 0`. A amostra é `max(N(μ_k, σ_k), 0)` nas células de `M`. As células observadas são copiadas e as ausentes ficam em 0.
- **Teste de não vazamento:** duas matrizes de gramas que diferem só em `M` passam por `ocultar`, `fit` e `sample` com a mesma semente. As saídas devem ser idênticas.
- O baseline não é avaliado nesta mudança (taxa 0). Só os testes de contrato e o tempo de ajuste o exercitam.

### 6. Métricas com `scoringrules`

`metricas.avaliar(y, media, var) -> dict` devolve MAE, NLL, CRPS, PICP, MPIW e `pct_meia_dose`. `metricas.cobertura_por_faixa(y, media, var)` devolve a tabela por quintil.
- O CRPS usa `scoringrules.crps_normal`. Alternativas: `properscoring`, que está sem manutenção, ou a fórmula fechada à mão.
- `z = NormalDist().inv_cdf(0.95)`, da biblioteca padrão.
- A métrica de meia dose (7,5 g) é secundária, com a fonte marcada como "a verificar".

### 7. Experimento `taxa_zero`

`uv run python -m amaprob.experiments.taxa_zero` faz o seguinte:
1. Carrega o dataset e as máscaras; o carregamento das máscaras confere o alinhamento.
2. Monta `Z` com todas as gramas.
3. Ajusta no treino a referência de mínimos quadrados (variância do resíduo do treino) e a regressão linear bayesiana.
4. Mede na validação e grava três CSVs em `results/`.
5. Mede o tempo do baseline sobre a máscara de taxa 0,5 e semente 0: o ajuste no treino e 100 amostras para a validação, geradas em blocos de 5.000 refeições. Gerar 100 amostras do treino inteiro de uma vez ocuparia ≈ 5 GB (100 × 213 mil × 29 × 8 bytes); a propagação, na mudança seguinte, precisa do mesmo processamento em blocos.

Nenhuma linha de teste é selecionada.

### 8. Organização do código

```
src/amaprob/
  imputation/  base.py (Imputer, ocultar)   baseline.py (PorcaoMedia)
  regression/  base.py (ProbRegressor, montar_Z)   blr.py (RegressaoLinearBayesiana)
  evaluation/  metricas.py
  experiments/ taxa_zero.py
```

Os modelos dependem só dos contratos, e o experimento só combina as peças.

## Risks / Trade-offs

- [Caudas pesadas: a PICP da taxa 0 deve ficar acima de 90%, com falta de cobertura nas refeições grandes] → Cobertura por faixa reportada desde já, curva lida em relação à taxa 0, alternativa no backlog. O SVGP, na mudança seguinte, dá um segundo modelo de ruído para comparar.
- [A truncagem em 0 no baseline desloca a média das amostras quando μ_k é pequeno] → Afeta só gavetas de porção muito pequena na escala `log1p`. É aceitável para uma referência; o teste de distribuição usa μ ≫ σ.
- [Mudança de contrato em relação ao `AGENTS.md`] → O `AGENTS.md` é atualizado nesta mudança, antes de qualquer outro modelo existir.
- [`scoringrules` traz `scipy`, um pacote grande] → `scipy` já está na stack prevista do projeto.

## Migration Plan

1. Atualizar o código e regravar o dataset com `uv run python -m amaprob.data.dataset`. As máscaras são regeneradas iguais, porque o sorteio depende só de `P` e da semente; a carga confere.
2. O trio roda `uv sync` (nova dependência) e o mesmo comando de preparação.
