# Design

## Context

Motivação em `proposal.md`. Estado atual que orienta o desenho:

- `imputation/ppca.py`: PPCA-EM vetorizado, com σ² fatorado no passo E (`A = σ²I + WᵀW`, `Cov[z] = σ²A⁻¹`), regressão por gaveta no passo M e amostragem por Cholesky de `A`. Ajuste no treino completo em ~3 min (486 iterações); 2 das 5 dobras pararam no limite de 500 iterações.
- `experiments/propagacao.py`: `IMPUTADORES` como dicionário, laço por imputador com geradores `default_rng((semente, taxa, fluxo))` que não dependem do imputador; `main()` regrava todos os CSVs e as figuras a cada execução.
- Resultados de 11/11 versionados em `results/`: baseline e PPCA nas 5 taxas e 3 sementes. A execução completa levou 67,6 min.

## Goals / Non-Goals

**Goals:**
- FA-EM correto e vetorizado, com a mesma estrutura do PPCA-EM e derivação comentada.
- A FA como terceira linha do experimento, sem rodar de novo o baseline e o PPCA.

**Non-Goals:**
- Mudar o PPCA ou o baseline.
- Escolha de q (Parte 3); extensão B+; avaliação no teste.

## Decisions

### 1. Modelo e passo E

Por refeição n, com `O_n` = gavetas observadas, `r_n = x_O − μ_O` e Ψ = diag(ψ):

- `A_n = I_q + W_Oᵀ Ψ_O⁻¹ W_O` (q × q)
- `E[z_n] = A_n⁻¹ W_Oᵀ Ψ_O⁻¹ r_n`, `Cov[z_n] = A_n⁻¹` (sem o fator de ruído do PPCA)

Vetorização como no PPCA, com o ruído dentro dos produtos: `W_Oᵀ Ψ_O⁻¹ W_O = O @ (w_k w_kᵀ / ψ_k)` e `b_n = ((R / ψ) ⊙ O) @ W`.

Log-verossimilhança dos observados (lema do determinante e Woodbury):

- `log|C_O| = Σ_{k∈O_n} log ψ_k + log|A_n|`
- `r_nᵀ C_O⁻¹ r_n = r_nᵀ Ψ_O⁻¹ r_n − b_nᵀ A_n⁻¹ b_n`

Com Ψ = σ²I, as fórmulas recaem nas do PPCA (`A_FA = A_PPCA / σ²`); o teste de recuperação e a comparação de log-verossimilhança com o PPCA conferem isso indiretamente.

### 2. Passo M

A regressão de `[w_k; μ_k]` em `z̃ = [z; 1]` por gaveta é a mesma do PPCA: o peso 1/ψ_k é constante dentro da gaveta e se cancela. Muda só o ruído, que passa a ser por gaveta:

- `ψ_k = (1/n_k) Σ_{n: k∈O_n} (x_nk² − 2 x_nk w̃_kᵀ E[z̃_n] + w̃_kᵀ E[z̃_n z̃_nᵀ] w̃_k)`, com `n_k` = número de refeições em que k é observada.

### 3. Piso no ruído (casos de Heywood)

- `ψ_k ← max(ψ_k, 0,001 · v_k, 10⁻⁶)`, com `v_k` = variância observada da gaveta k no conjunto de ajuste; o mínimo absoluto cobre a gaveta constante (`v_k = 0`).
- Por que 0,1%, e não 1%: no teste sintético, 1% prendeu uma gaveta normal (ψ verdadeiro de 0,6% da variância) e superestimou o ψ dela em 54%. Nos dados reais, sem piso, a menor razão ψ_k / v_k é 0,48% (Molhos e condimentos), seguida de Açúcar (1,4%) e Óleos e gorduras (2,9%). Com 0,1%, nenhuma gaveta trava, e "no piso" passa a significar colapso de fato.
- Gavetas sem observados: `ψ_k = 1` fixo e `w_k = 0`. Assim o ruído fica finito, e `0 · ∞` não gera `NaN` que se espalhe para todas as refeições.
- `gavetas_no_piso_` guarda as gavetas que terminaram no piso; o experimento grava a contagem em `results/imputadores_ajuste.csv`.
- O piso pode quebrar a monotonia estrita do EM na iteração em que é ativado. O teste de monotonia usa dados sem gaveta no piso, e o histórico do ajuste real fica registrado.

### 4. Inicialização, convergência e construtor

Iguais às do PPCA: μ = média dos observados, W com entradas N(0; 0,1²) por `default_rng(semente)`, ψ_k = `v_k` (e não a média única); parada por melhora relativa abaixo de `1e-6` ou 500 iterações. `FA(q=5, semente=0, tolerancia=1e-6, max_iteracoes=500)`.

### 5. Amostragem

`z = E[z_n] + L⁻ᵀ u`, com `A_n = L Lᵀ` e `u ~ N(0, I_q)`; `x_H = μ_H + W_H z + √ψ_H ⊙ v`, com `v ~ N(0, I)` e truncamento em 0. Sem observados, a posterior é a priori, e as amostras seguem a marginal `N(μ_H, W_H W_Hᵀ + Ψ_H)`.

### 6. Código separado do PPCA

`imputation/fa.py` importa só `observados` de `ppca.py`. O passo M repete cerca de 15 linhas da regressão por gaveta. Alternativa descartada: extrair um núcleo comum e fazer o PPCA usá-lo. Isso mudaria a ordem das operações de ponto flutuante do PPCA e tiraria a garantia de que as linhas de 11/11 são as que o código atual produz.

### 7. Execução parcial do experimento

- `IMPUTADORES` ganha `"fa": lambda: FA(q=5, semente=0)`, e `NOMES_DOS_IMPUTADORES` ganha `"FA (q = 5)"`.
- `uv run python -m amaprob.experiments.propagacao --imputadores fa`: sem a opção, todos rodam. A leitura dos argumentos usa `argparse`, da biblioteca padrão.
- A referência da taxa 0 é calculada como hoje, mas na junção prevalece a do arquivo existente.
- Junção, por CSV: remover das linhas existentes as dos imputadores executados, acrescentar as novas no fim e manter as linhas `nenhum` do arquivo quando existirem. Assim o `git diff` mostra só linhas acrescentadas.
- O resumo é recalculado sobre `propagacao_metricas.csv` juntado, e as figuras saem com um painel por imputador. A largura passa de 12 para 16 polegadas, por causa dos três painéis.
- `imputadores_ajuste.csv` recebe as linhas da FA e a coluna `gavetas_no_piso`, vazia para os outros imputadores.

### 8. Medição de custo e ponto de decisão

Antes da rodada, um ajuste da FA no treino completo mede o tempo, as iterações e as gavetas no piso.

- Até 5 minutos: a rodada segue (6 ajustes mais 15 rodadas da curva; estimativa de ~35 min, metade da execução de 11/11).
- Acima disso, ou se o ajuste bater o limite de iterações no treino completo: a execução para e a decisão volta para aprovação.

### 9. Organização do código

```
src/amaprob/
  imputation/  fa.py           (FA)
  experiments/ propagacao.py   (+ FA, --imputadores, junção, figuras com três painéis)
tests/
  test_fa.py                   (recuperação, FA × PPCA, monotonia, piso, condicional, semente)
  test_imputacao.py            (contrato parametrizado também na FA)
```

## Risks / Trade-offs

- [Casos de Heywood nas gavetas em degraus ou raras] → Piso de 0,1% da variância observada e registro das gavetas no piso. As adições (molhos, açúcar, óleos), cujos gramas a POF estima, têm os menores ψ; as métricas por gaveta mostram se o intervalo delas fica estreito demais.
- [EM da FA mais lento que o do PPCA] → Mesmo limite de 500 iterações, histórico registrado e ponto de decisão antes da rodada.
- [A junção depende de o código do baseline e do PPCA não ter mudado desde 11/11] → `ppca.py`, `baseline.py`, `propagation.py` e as métricas não mudam nesta change; o `git diff` dos CSVs confere que só entram linhas novas.
- [A FA pode não superar o PPCA na taxa 1] → Leitura válida: ela testaria a causa e a descartaria. Na taxa 1 o ganho só pode vir do ruído por gaveta e da correlação entre as ocultas.

## Migration Plan

1. Nenhuma dependência nova; nenhum dado regravado em `data/processed/`.
2. CSVs de `results/` com linhas da FA acrescentadas; resumo e figuras regravados.
