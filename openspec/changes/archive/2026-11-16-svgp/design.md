# Design

## Context

Motivação em `proposal.md`. Estado atual que orienta o desenho:

- Contrato `ProbRegressor` (`fit(Z, y)`, `predict(Z) → (média, variância)`); `Z` com 71 colunas (29 gramas, 29 presenças, 13 de contexto). A regressão linear bayesiana padroniza as colunas e centra o alvo, e uma coluna constante fica com φ = 0.
- `prever_bracos` chama `regressor.predict` com S × B linhas por bloco de refeições; a propagação prevê S × N = 100 × 45.639 ≈ 4,6 milhões de linhas por rodada da curva.
- O experimento tem o imputador como dimensão, `--imputadores` e junção aos CSVs; o regressor é fixo (`RegressaoLinearBayesiana`), e o recalibrado usa um segundo ajuste dela.
- Máquina: CPU de 6 núcleos e 16 GB de memória, sem GPU.

## Goals / Non-Goals

**Goals:**
- SVGP correto atrás do contrato, reprodutível pela semente e com custo medido antes da rodada.
- Regressor como dimensão do experimento, sem rodar de novo as linhas da regressão linear.

**Non-Goals:**
- Escolha de M e das épocas por otimização bayesiana (Parte 3).
- Plug-in recalibrado com o SVGP.
- Verossimilhança de caudas pesadas (o mesmo ruído gaussiano da regressão linear; ver `backlog/decisions.md`).

## Decisions

### 1. Modelo

Sobre `φ(z)` (colunas padronizadas pelo treino, com desvio 1 nas constantes) e `t = (y − ȳ) / s_y`:

- `f ~ GP(m, k)`, com `m(φ) = aᵀφ + b` (média linear, `LinearMean`) e `k = σ_f² · RBF` com um comprimento por coluna (ARD);
- `t | f ~ N(f(φ), σ²)`.

Aproximação variacional esparsa (Hensman et al., 2013): M pontos indutores `u = f(Z_u)` com `q(u) = N(m_u, S_u)` (Cholesky completo) e posições aprendidas. A ELBO por minibatch é a soma das esperanças de log-verossimilhança, reescalada por N / B, menos `KL(q(u) || p(u))`.

Preditiva: `q(f*)` de média `m(φ*) + k*ᵀ K_uu⁻¹ (m_u − m(Z_u))` e variância `k** − k*ᵀ K_uu⁻¹ (K_uu − S_u) K_uu⁻¹ k*`, mais σ², e depois a volta para a escala de y (`média · s_y + ȳ`, `variância · s_y²`).

### 2. Implementação

- `regression/svgp.py`, classe `SVGP(n_indutores=256, epocas=60, tamanho_lote=1024, taxa_aprendizado=0.01, semente=0)`, com GPyTorch (`ApproximateGP`, `CholeskyVariationalDistribution`, `VariationalStrategy` com `learn_inducing_locations=True`, `GaussianLikelihood`, `VariationalELBO`) e otimizador Adam sobre os parâmetros do modelo e da verossimilhança.
- Pontos indutores iniciados em M linhas do treino, sorteadas com `default_rng(semente)`.
- Reprodutibilidade: `torch.manual_seed(semente)` e um `torch.Generator` próprio para embaralhar os minibatches.
- `float64` em tudo: as variâncias pequenas e o Cholesky de `K_uu` ficam estáveis sem ajuste de jitter. Se a medição mostrar custo alto, `float32` é a primeira alternativa, levada para aprovação.
- `predict` em blocos de linhas com `torch.no_grad()`, devolvendo arrays NumPy.
- `historico_elbo_` (uma entrada por época) e `segundos_ajuste_` ficam disponíveis após o ajuste.
- Épocas fixas, e não parada por convergência: o custo fica previsível e o ajuste, determinístico. A curva da ELBO é conferida na medição.

### 3. Regressor como dimensão do experimento

- `REGRESSORES = {"blr": RegressaoLinearBayesiana, "svgp": lambda: SVGP(semente=0)}`; `--regressores` e `--imputadores`; os pares executados são o produto dos dois.
- Para cada regressor: ajuste uma vez no treino completo; referência da taxa 0 por número de ocultas por taxa e semente; para cada imputador, `prever_bracos` com os mesmos geradores (as amostras do imputador são as mesmas para os dois regressores). O recalibrado só existe com `blr`.
- Métricas da Parte 1 só quando `blr` está entre os regressores executados, porque não dependem do regressor.
- Coluna `regressor` em `propagacao_metricas.csv`, `propagacao_por_ocultas.csv` e `propagacao_resumo.csv`. Na junção, as linhas antigas sem a coluna recebem `blr`. São removidas as linhas existentes dos pares executados, e a referência `nenhum` de um regressor que já tem referência gravada fica como está.
- Novo `results/regressores_ajuste.csv` (regressor, segundos, épocas, ELBO final), que guarda os tempos para a junção com Desempenho.
- Figuras com `facet_grid(regressor ~ imputador)`; a linha pontilhada da PICP da taxa 0 é a do regressor de cada linha da grade.

### 4. Medição de custo e ponto de decisão

Antes da rodada: um ajuste do SVGP no treino completo (segundos por época, curva da ELBO) e a previsão de 100 amostras da validação (taxa 0,5, semente 0).

- Previsão de até 2 minutos por rodada: segue com a validação inteira (45 rodadas; estimativa de 30 a 60 minutos).
- Acima disso: a execução para e a decisão volta para aprovação (subamostra estratificada da validação, M menor ou `float32`).

### 5. Organização do código

```
src/amaprob/
  regression/  svgp.py          (SVGP)
  experiments/ propagacao.py    (+ REGRESSORES, --regressores, junção por regressor, grade nas figuras)
tests/
  test_svgp.py                  (reta, não linear contra a regressão linear, fora dos dados, semente, coluna constante)
```

## Risks / Trade-offs

- [Épocas fixas podem deixar o SVGP subajustado] → Curva da ELBO conferida na medição: com 10 épocas a ELBO ainda subia e o ruído ficava no dobro do da regressão linear; com 60, a ELBO assenta por volta da 25ª época. M e épocas entram na Parte 3.
- [Custo da previsão com S × N linhas] → Medido antes da rodada, com ponto de decisão; previsão em blocos.
- [Caudas pesadas: o ruído homocedástico cobre demais as refeições pequenas e de menos as grandes] → O mesmo da regressão linear; a leitura por faixa e por número de ocultas continua relativa à taxa 0 de cada regressor.
- [A variância do SVGP cresce longe dos pontos indutores: refeições raras podem ganhar intervalos largos] → Leitura esperada de um GP; aparece na cobertura por faixa.
- [Mudança de esquema dos CSVs (coluna `regressor`)] → Linhas antigas conferidas iguais às de `HEAD`, com a coluna nova à parte, como na FA.

## Migration Plan

1. `uv add torch gpytorch`.
2. CSVs de propagação com a coluna `regressor` e as linhas do SVGP; CSVs da Parte 1 inalterados.
