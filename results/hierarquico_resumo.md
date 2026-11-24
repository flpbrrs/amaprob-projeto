# Modelo hierárquico: efeito da pessoa na imputação (versão final, v4)

Validação apenas; o teste não é lido. Código: `src/amaprob/imputation/hierarquico_v4.py`,
`src/amaprob/experiments/hierarquico_v4.py`, `tests/test_hierarquico_v4.py`. Números:
`results/hierarquico_v4_*.csv`; o braço sem pessoa vem de `results/hierarquico_v2_*.csv` (a mesma
conta, com D = 0, a mesma amostragem e o mesmo gerador).

## Cenário

- Pessoas da validação com os dois dias de registro: o dia 1 é o histórico, e o dia 2 é avaliado
  com as máscaras existentes (taxas 0,25 a 1,0; 3 sementes).
- H1: histórico completo (todas as gramas do dia 1). H2: histórico mascarado com a mesma taxa.
- Imputador: FA (análise fatorial) condicional ajustada no treino completo. As variâncias entre
  pessoas τ² e dentro da pessoa σ² saem dos resíduos do treino.
- Regressão: regressão linear bayesiana; tabela com o braço de propagação (S = 100).
- Métricas: MAE (erro absoluto médio), NLL (log-verossimilhança negativa), CRPS (continuous
  ranked probability score), PICP (cobertura do intervalo de 90%) e MPIW (largura média do
  intervalo de 90%).

## Modelo

O efeito da pessoa δ_p desloca o log1p(gramas) de cada gaveta. A posterior de δ_p a partir do
histórico, N(δ̂_p, V_p), vem da conjugação normal–normal com o prior N(0, T) e o ruído σ².

A FA condicional foi ajustada em todas as refeições, então a covariância condicional Σ_FA que ela
devolve já contém a variação entre pessoas. Saber o histórico troca o prior T pela posterior V, e
o bloco oculto de cada refeição fica

    μ_novo = μ_FA(x_O − δ̂) + δ̂
    Σ_novo = Σ_FA − c·D,   D = T − V ⪰ 0

- por gaveta: T = diag(τ_k²), V = diag(v_pk), D = diag(τ_k² − v_pk);
- fator único: T = τ²·11ᵀ, V = v_p·11ᵀ, D = (τ² − v_p)·11ᵀ (posto 1).

A redução é limitada por refeição:

    c = min(1; 0,9 / λ_max),   λ_max = maior autovalor de Σ_FA^(−1/2) D Σ_FA^(−1/2)

Com Σ_FA = L Lᵀ, Σ_novo − 0,1·Σ_FA = L (0,9·I − c·L⁻¹DL⁻ᵀ) Lᵀ ⪰ 0, porque c·λ_max ≤ 0,9.
Logo Σ_novo ⪰ 0,1·Σ_FA: nenhuma direção perde mais de 90% da variância da FA, e a matriz é
positiva definida sem projeção. Como diag(D) ≥ 0, diag(Σ_novo) ≤ diag(Σ_FA).

- Por gaveta, λ_max é o maior autovalor de SᵀS, com S = L⁻¹ diag(√d).
- No fator único, λ_max = d · 1_Hᵀ Σ_FA⁻¹ 1_H.
- Sem histórico, V = T bit a bit, D = 0, c = 1 e Σ_novo = Σ_FA exatamente.

As amostras saem de N(μ_novo, Σ_novo) por refeição (Cholesky), truncadas em 0 e convertidas para
gramas.

## Resultados

Médias de 3 sementes. Propagação: regressão linear bayesiana, carboidrato em g. Imputação: células
ocultas, log1p(gramas). Sem histórico útil, H1 e H2 do sem pessoa coincidem.

| cenário | taxa | modelo | MAE (g) | NLL | CRPS (g) | PICP | MPIW (g) | MAE log | NLL log | PICP imp. |
|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 0,25 | sem pessoa | 11,69 | 4,238 | 8,98 | 0,930 | 60,2 | 0,419 | 0,758 | 0,900 |
| H1 | 0,25 | v4 por gaveta | 11,27 | 4,214 | 8,71 | 0,927 | 58,4 | 0,382 | 2,041 | 0,832 |
| H1 | 0,25 | v4 fator único | 11,54 | 4,227 | 8,88 | 0,929 | 59,2 | 0,413 | 0,836 | 0,890 |
| H1 | 0,50 | sem pessoa | 15,81 | 4,480 | 11,58 | 0,933 | 76,9 | 0,449 | 0,832 | 0,898 |
| H1 | 0,50 | v4 por gaveta | 14,80 | 4,426 | 10,94 | 0,932 | 73,9 | 0,399 | 1,665 | 0,851 |
| H1 | 0,50 | v4 fator único | 15,42 | 4,458 | 11,33 | 0,932 | 74,8 | 0,440 | 0,915 | 0,890 |
| H1 | 0,75 | sem pessoa | 20,23 | 4,742 | 14,38 | 0,937 | 94,9 | 0,479 | 0,903 | 0,896 |
| H1 | 0,75 | v4 por gaveta | 18,53 | 4,637 | 13,30 | 0,938 | 91,5 | 0,414 | 1,248 | 0,874 |
| H1 | 0,75 | v4 fator único | 19,48 | 4,704 | 13,92 | 0,935 | 91,5 | 0,468 | 0,931 | 0,891 |
| H1 | 1,00 | sem pessoa | 24,92 | 5,001 | 17,42 | 0,940 | 114,6 | 0,508 | 0,968 | 0,893 |
| H1 | 1,00 | v4 por gaveta | 22,34 | 4,861 | 15,75 | 0,947 | 111,0 | 0,428 | 0,846 | 0,895 |
| H1 | 1,00 | v4 fator único | 23,63 | 4,939 | 16,61 | 0,939 | 109,8 | 0,494 | 0,939 | 0,891 |
| H2 | 0,25 | sem pessoa | 11,69 | 4,238 | 8,98 | 0,930 | 60,2 | 0,419 | 0,758 | 0,900 |
| H2 | 0,25 | v4 por gaveta | 11,37 | 4,217 | 8,77 | 0,927 | 58,7 | 0,396 | 1,973 | 0,829 |
| H2 | 0,25 | v4 fator único | 11,55 | 4,226 | 8,89 | 0,928 | 59,3 | 0,414 | 0,835 | 0,889 |
| H2 | 0,50 | sem pessoa | 15,81 | 4,480 | 11,58 | 0,933 | 76,9 | 0,449 | 0,832 | 0,898 |
| H2 | 0,50 | v4 por gaveta | 15,27 | 4,447 | 11,23 | 0,931 | 75,2 | 0,429 | 1,490 | 0,853 |
| H2 | 0,50 | v4 fator único | 15,53 | 4,461 | 11,40 | 0,931 | 75,0 | 0,444 | 0,914 | 0,889 |
| H2 | 0,75 | sem pessoa | 20,23 | 4,742 | 14,38 | 0,937 | 94,9 | 0,479 | 0,903 | 0,896 |
| H2 | 0,75 | v4 por gaveta | 19,75 | 4,706 | 14,07 | 0,935 | 94,0 | 0,463 | 1,073 | 0,878 |
| H2 | 0,75 | v4 fator único | 19,91 | 4,729 | 14,19 | 0,933 | 92,6 | 0,475 | 0,932 | 0,888 |
| H2 | 1,00 | sem pessoa | 24,92 | 5,001 | 17,42 | 0,940 | 114,6 | 0,508 | 0,968 | 0,893 |
| H2 | 1,00 | v4 por gaveta | 24,92 | 5,001 | 17,42 | 0,940 | 114,6 | 0,508 | 0,968 | 0,893 |
| H2 | 1,00 | v4 fator único | 24,92 | 5,001 | 17,42 | 0,940 | 114,6 | 0,508 | 0,968 | 0,893 |

Leitura:

- **Regressão:** o histórico melhora a regressão em todas as taxas. No H1 por gaveta, o MAE cai
  de 0,4 a 2,6 g e a NLL de 0,02 a 0,14, com a PICP mantida entre 0,93 e 0,95. O fator único
  ganha menos.
- **Histórico mascarado (H2):** o ganho encolhe com a taxa e some na taxa 1,0, em que o
  histórico não tem nenhuma quantidade observada.

## Conferências

As três passaram; as duas primeiras abortariam a rodada se falhassem.

1. **H2 na taxa 1,0, sem histórico:** a média e a covariância em log são idênticas às da FA sem
   pessoa (diferença máxima 0), nas duas variantes, nas 3 sementes e em todas as 29 gavetas.
2. **Em toda refeição de todas as rodadas:**
   - o menor autovalor de Σ_novo − 0,1·Σ_FA no bloco oculto é −5,6·10⁻¹⁶ no pior caso, dentro
     do limite de −10⁻¹⁰;
   - diag(Σ_novo) ≤ diag(Σ_FA), com excesso máximo 0.
3. **Teste sintético:** quando D é pequeno o bastante para que c = 1 em toda refeição, a v4
   reproduz a fórmula sem limite, Σ_FA − T + V, em 10⁻¹² (`tests/test_hierarquico_v4.py`).

## Distribuição de c

Refeições do dia 2 com ao menos uma oculta, 3 sementes juntas (34.182 refeições na taxa 0,25,
50.957 na 0,5, 58.432 na 0,75 e 61.848 na 1,0).

| variante | cenário | taxa | fração com c = 1 | mediana de c (c < 1) | percentil 10 (c < 1) | mínimo |
|---|---|---|---|---|---|---|
| por gaveta | H1 | 0,25 | 0,739 | 0,473 | 0,108 | 0,055 |
| por gaveta | H1 | 0,50 | 0,711 | 0,447 | 0,087 | 0,054 |
| por gaveta | H1 | 0,75 | 0,691 | 0,305 | 0,081 | 0,054 |
| por gaveta | H1 | 1,00 | 0,683 | 0,275 | 0,069 | 0,054 |
| por gaveta | H2 | 0,25 | 0,768 | 0,499 | 0,108 | 0,056 |
| por gaveta | H2 | 0,50 | 0,781 | 0,515 | 0,108 | 0,058 |
| por gaveta | H2 | 0,75 | 0,827 | 0,613 | 0,108 | 0,060 |
| por gaveta | H2 | 1,00 | 1,000 | sem c < 1 | sem c < 1 | 1,000 |
| fator único | H1 | 0,25 | 0,945 | 0,554 | 0,433 | 0,244 |
| fator único | H1 | 0,50 | 0,947 | 0,542 | 0,431 | 0,264 |
| fator único | H1 | 0,75 | 0,967 | 0,532 | 0,429 | 0,281 |
| fator único | H1 | 1,00 | 1,000 | sem c < 1 | sem c < 1 | 1,000 |
| fator único | H2 | 0,25 | 0,946 | 0,554 | 0,450 | 0,244 |
| fator único | H2 | 0,50 | 0,952 | 0,565 | 0,457 | 0,264 |
| fator único | H2 | 0,75 | 0,977 | 0,650 | 0,509 | 0,330 |
| fator único | H2 | 1,00 | 1,000 | sem c < 1 | sem c < 1 | 1,000 |

- **Por gaveta:** o limite age em 17 a 32% das refeições com histórico útil, e no percentil 10
  a redução fica em 7 a 11% do que a fórmula sem limite pediria.
- **Fator único:** o limite quase não age (c = 1 em 95% ou mais das refeições).
- **Fator único no H1, taxa 1,0:** c = 1 em todas as refeições. Sem gramas observadas no dia
  avaliado, Σ_FA é a marginal, que já comporta a redução inteira.

## Limitações

**Imputação subcoberta nas taxas intermediárias.** Na variante por gaveta, nas taxas 0,25 a 0,75,
a imputação cobre menos que o intervalo nominal: PICP de 0,83 a 0,88 e NLL em log de 1,07 a 2,04,
contra cerca de 0,90 e 0,76 a 0,90 sem pessoa. A regressão continua calibrada.

A causa é onde a FA guarda a variação entre pessoas: nos fatores W z, e não no ruído ψ. Em 13 das
29 gavetas, τ_k² > ψ_k. Quando há gavetas observadas no dia, a covariância condicional Σ_FA já
encolheu a parte de W z que a pessoa explica, e subtrair T − V em cima disso tira variância que
já não está lá. O limite c impede que a matriz deixe de ser válida, mas não devolve a calibração:
em parte das refeições, Σ_novo encosta em 10% da Σ_FA.

**Versões testadas durante o desenvolvimento:**
- **Sem limite e com projeção para semidefinida positiva (v2):** acrescentava de 3,7 a 5,0% de
  traço e chegava a uma NLL da imputação de 2,7·10⁵.
- **Variação entre pessoas fora do ruído antes de condicionar (v3):** ψ* = max(ψ − τ², piso) + v.
  Calibra a imputação, com PICP de 0,90 a 0,91 e NLL em log de 0,79 na taxa 0,5 do H1. Em troca,
  conta τ² duas vezes nas 13 gavetas em que o piso é acionado, com MAE de 0,3 a 0,6 g pior e
  sobrecobertura na regressão.

**Trabalho futuro:** o reajuste conjunto da FA com o efeito da pessoa, x = b(u) + δ_p + W z + ε,
estimado por EM com as duas fontes de variação separadas. Ele resolveria as duas pontas.

**Outras limitações:**
- O histórico atualiza δ, mas as gavetas observadas do próprio dia avaliado não atualizam.
- τ² e σ² são estimados por momentos no treino e tratados como fixos.
