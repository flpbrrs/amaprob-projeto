# Design

## Context

Motivação em `proposal.md`. Estado atual que orienta o desenho:

- PPCA (`ppca.py`) e FA (`fa.py`) por EM com ausentes como condicionantes, em `x = log1p(gramas)`. O passo M já é uma regressão por gaveta de `x_k` em `z̃ = [z; 1]`, somada só nas refeições em que k foi observada; o intercepto dessa regressão é `μ_k`.
- `fit` e `sample` já recebem `P` e `C` pelo contrato `Imputer`, mas não os usam.
- Treino com 213.240 refeições, K = 29 gavetas e 13 colunas de contexto (5 indicadoras de ocasião, seno e cosseno da hora, 5 indicadoras de macrorregião e `log1p` da renda).
- Ajuste no treino completo: PPCA ~3 min e FA ~1 min; com as 5 dobras do recalibrado, ~16 min e ~5 min.

## Goals / Non-Goals

**Goals:**
- Média de cada gaveta dependendo de `P` e `C` no PPCA e na FA, com o caminho atual intacto quando a opção está desligada.
- EM estável com colunas colineares e monótono no objetivo penalizado.

**Non-Goals:**
- Interações entre colunas de `u` ou média não linear.
- Mudar a covariância (W, σ² ou Ψ) em função de `P` e `C`.
- Escolher q de novo (fica para a otimização bayesiana).

## Decisions

### 1. Modelo

Com `u_n = [1; φ(P_n); φ(C_n)]` (d = 1 + 29 + 13 = 43), em que φ padroniza cada coluna pela média e pelo desvio do conjunto de ajuste (desvio 0 vira 1, e a coluna fica em 0):

    x_nk = w_kᵀ z_n + b_kᵀ u_n + ε_nk

O único termo novo é `b_kᵀ u_n` no lugar de `μ_k`; o primeiro elemento de `b_k` é o intercepto. A marginal dos observados vira `x_O ~ N(B_O u_n, C_O)`, com a mesma `C_O` de antes. O passo E não muda, a menos de `r_n = x_O − B_O u_n`.

### 2. Penalização ridge e EM do máximo a posteriori

Na soma da gaveta k, as colunas de `u` são colineares: `P_k` vale 1 em toda refeição em que k é observada (repete o intercepto), as indicadoras de ocasião e as de região somam 1, e gavetas que nunca aparecem com k dão colunas nulas. Objetivo:

    L(θ) = log p(X_O | θ) − (α/2) Σ_k ‖b_k^{(sem intercepto)}‖²,   α = 1

Equivale a uma priori `N(0, 1/α)` em cada coeficiente padronizado: o efeito de um desvio de uma coluna sobre `log1p(gramas)`. Efeitos reais ficam bem abaixo de 1, então a priori quase não pesa onde os dados informam, e fixa em 0 as direções sem apoio.

Passo M em dois tempos (ECM, Meng e Rubin, 1993): primeiro `[w_k; b_k]` com o ruído da iteração anterior, depois o ruído com os coeficientes novos. Com `z̃_n = [z_n; u_n]` e `D` diagonal (1 nos coeficientes penalizados, 0 em z e no intercepto):

    [w_k; b_k] = (Σ_{n: k∈O_n} E[z̃_n z̃_nᵀ] + s_k α D)⁻¹ Σ_{n: k∈O_n} x_nk E[z̃_n]

com `s_k = σ²` no PPCA e `s_k = ψ_k` na FA. A atualização do ruído é a de hoje (a penalização não depende dele). Cada passo do ECM não diminui `L`, e o passo E é exato. Por isso o histórico, a parada e o teste de monotonia usam `L`.

`E[z̃ z̃ᵀ]` tem os blocos `E[zzᵀ]`, `E[z] u_nᵀ` e `u_n u_nᵀ`.

### 3. Custo e memória

`Σ O_nk u_n u_nᵀ` por `O.T @ (N × d²)` ocuparia ~3 GB. Os blocos `Σ u uᵀ` e `Σ x u` de cada gaveta não mudam entre iterações e são calculados uma vez, gaveta a gaveta, só nas linhas observadas. Por iteração mudam `Σ E[zzᵀ]` (como hoje) e `Σ E[z] uᵀ` (K produtos de `n_k × q` por `n_k × d`). Os sistemas são de dimensão q + d = 48.

Iniciação: `b_k` pela regressão ridge de `x_k` em `u` (sem z) nas linhas observadas; W como hoje; ruído pelo resíduo dessa regressão.

### 4. Implementação

- Opção `media_condicional: bool = False` no construtor de `PPCA` e de `FA`. Com `False`, o código executado é o atual, linha por linha; com `True`, `media_` deixa de ser um vetor e vira `B_` (K × d), e a média de cada refeição é calculada por `u_n`.
- Atributos novos: `coeficientes_media_` (`B_`), `media_u_`/`desvio_u_` (padronização) e `historico_objetivo_` (o `L` penalizado). Com a opção desligada, o histórico atual continua em `historico_log_verossimilhanca_`.
- A derivação acima entra comentada no topo de cada módulo (AGENTS §9).

### 5. Controle: porção média condicional

`PorcaoMedia(media_condicional=True)`: a média de cada gaveta pela regressão ridge de `x_k` em `u` nas células observadas (a mesma iniciação do PPCA e da FA, com a variância observada da gaveta como escala da penalização) e o desvio pelo resíduo. É a média condicional sem fatores. Na taxa 1, a FA com média condicional difere dele só pela covariância entre gavetas e pela estimação conjunta; a diferença entre os dois é a contribuição do modelo fatorial.

### 6. Experimento

- `IMPUTADORES` ganha `porcao_condicional`, `ppca_condicional` e `fa_condicional` (rótulos "Porção média cond.", "PPCA cond. (q = 5)" e "FA cond. (q = 5)"), rodados com `--imputadores`, com a junção já existente.
- Largura das figuras proporcional ao número de imputadores na grade.

### 7. Rodada e critério

1. `--regressores blr --imputadores porcao_condicional fa_condicional` (~35 min). Critério (proposta): CRPS da propagação na taxa 1 abaixo de 17,97, PICP ≥ 92%, sem piorar o CRPS nas taxas 0,25 a 0,75 em relação à FA.
2. Se passar: `--regressores blr --imputadores ppca_condicional`, depois `--regressores svgp --imputadores porcao_condicional ppca_condicional fa_condicional`. O controle entra na leitura: quanto do ganho da FA com média condicional sobre a porção média também aparece na porção média condicional.
3. Se não passar: a change fecha com o resultado negativo e a decisão volta para Felipe.

## Risks / Trade-offs

- [Extrapolação em combinações raras de `P`: média condicional alta e gramas imputadas extremas, como no caso da refeição de 7.717 g] → Ridge nos coeficientes; cauda das gramas imputadas medida na taxa 1 antes da rodada.
- [α fixo sem otimizar] → Escala justificada pela padronização; sensibilidade conferida na medição com α = 0,1 e α = 10 se a cauda ou o CRPS indicarem problema.
- [Mais iterações do EM com 43 colunas a mais] → Tempo e iterações medidos antes da rodada.
- [Rodada final no teste mais longa com cinco imputadores] → Decisão de Felipe em 17/11, junto com o resultado do critério.

## Migration Plan

1. Implementação com a opção desligada por padrão; testes atuais do PPCA e da FA sem mudança.
2. CSVs com as linhas novas acrescentadas pela junção; linhas existentes conferidas iguais às do `HEAD`.
