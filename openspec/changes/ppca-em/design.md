# Design

## Context

Motivação em `proposal.md`. Estado atual que orienta o desenho:

- Contratos `Imputer` (`fit`, `sample`) e `ProbRegressor`; baseline `PorcaoMedia`; experimento `experiments/propagacao.py` com um imputador fixo (`IMPUTADOR = "porcao_media"`), escolha de S antes da curva e ~80 s por taxa e semente com S = 100.
- `imputar_treino_cruzado(novo_imputador, …)` ajusta os 5 imputadores das dobras dentro do laço de taxa e semente. O ajuste usa o treino completo das outras dobras, sem máscara; só a amostragem usa `M`. Com o baseline (0,11 s por ajuste), isso não pesava.
- Dados: K = 29 gavetas, ~213 mil refeições de treino e ~45 mil de validação; 89% das células são zero (gaveta ausente); poucas gavetas presentes por refeição.
- `amostras_em_gramas` já processa a validação em blocos pelo orçamento de memória (`ORCAMENTO_BLOCO_BYTES`).

## Goals / Non-Goals

**Goals:**
- PPCA-EM correto com conjuntos de observados diferentes por refeição, vetorizado em NumPy, sem laço Python por refeição.
- O PPCA como nova linha do experimento de propagação, sem mudar o código dos braços.
- Métricas da Parte 1 que sirvam a qualquer imputador pelo contrato.

**Non-Goals:**
- FA (Análise Fatorial), que reaproveita este código na próxima change.
- Extensão B+ (média dependendo de `P` e `C`): registrada em `backlog/decisions.md`.
- Escolha de q (Parte 3); avaliação no teste.

## Decisions

### 1. Modelo e notação

Por refeição n, `x_n ∈ R^K` em `log1p(gramas)`; `O_n` = gavetas observadas (`P = 1` e `M = 0`), `H_n` = ocultas (`M = 1`). Modelo: `x = Wz + μ + ε`, `z ~ N(0, I_q)`, `ε ~ N(0, σ²I_K)`, W de K × q.

Ausentes como condicionantes: gavetas com `P = 0` não entram na verossimilhança, que é a marginal nas observadas, `x_O ~ N(μ_O, W_O W_Oᵀ + σ²I)`. O "dado que a gaveta está presente" está em quais coordenadas entram, não num valor zero.

O conjunto observado vem de `P & ~M`, e não de `isnan(X)`: os valores fora de `O_n` são trocados por 0 antes de qualquer conta, e nada lê `X` nessas células (testes "X[M] preenchido por engano" e "Ausentes fora da verossimilhança").

### 2. Passo E

Para cada refeição, com `r_n = x_{O_n} − μ_{O_n}`:

- `A_n = σ²I_q + W_{O_n}ᵀ W_{O_n}` (q × q)
- `E[z_n] = A_n⁻¹ W_{O_n}ᵀ r_n`
- `Cov[z_n] = σ² A_n⁻¹`; `E[z_n z_nᵀ] = Cov[z_n] + E[z_n]E[z_n]ᵀ`

Vetorização com `O` (N × K, 0/1):
- `W_Oᵀ W_O = einsum('nk,kij->nij', O, W ⊗ W)`, com `(W ⊗ W)_k = w_k w_kᵀ` (K × q × q);
- `W_Oᵀ r_n = ((X − μ) ⊙ O) @ W`;
- `np.linalg.solve` em lote sobre (N, q, q).

Com q = 5 e N ≈ 213 mil: ~150 milhões de operações por iteração, da ordem de 1 s. O agrupamento por padrão de presença fica como plano B (Decisão 8).

### 3. Passo M por gaveta

Cada gaveta k tem seu sistema, somado só nas refeições em que ela é observada. Com `z̃ = [z; 1]`, para atualizar `w_k` e `μ_k` juntos:

- `[w_k; μ_k] = (Σ_{n: k∈O_n} E[z̃_n z̃_nᵀ])⁻¹ Σ_{n: k∈O_n} x_nk E[z̃_n]`, com `E[z̃z̃ᵀ] = [[E[zzᵀ], E[z]], [E[z]ᵀ, 1]]`;
- `σ² = (1/Σ_n |O_n|) Σ_n Σ_{k∈O_n} [(x_nk − w_kᵀE[z_n] − μ_k)² + w_kᵀ Cov[z_n] w_k]`.

As somas por gaveta saem de `einsum('nk,nij->kij', O, E[z̃z̃ᵀ])` e `(O ⊙ X)ᵀ E[z̃]`. Atualizar μ separado de W (μ como média dos observados) foi descartado: com faltantes, essa média não é a de máxima verossimilhança e enviesa W.

Alternativa descartada: uma soma única de `E[zzᵀ]` sobre todas as refeições, que só vale sem faltantes.

### 4. Log-verossimilhança, convergência e inicialização

- Log-verossimilhança dos observados por refeição, pelo lema do determinante e por Woodbury, só com matrizes q × q:
  - `log|C_O| = (|O| − q) log σ² + log|A_n|`;
  - `r_nᵀ C_O⁻¹ r_n = (r_nᵀ r_n − r_nᵀ W_O A_n⁻¹ W_Oᵀ r_n) / σ²`.
  - Refeições sem observados contribuem 0.
- Parada: melhora relativa por célula observada abaixo de `1e-6`, ou 500 iterações. O histórico fica em `historico_log_verossimilhanca_`, junto com `n_iteracoes_`.
- Inicialização com `default_rng(semente)`: μ = média dos observados por gaveta; W com entradas N(0, 0,1²); σ² = média das variâncias dos observados por gaveta.
- Construtor: `PPCA(q=5, semente=0, tolerancia=1e-6, max_iteracoes=500)`. A semente no construtor mantém `fit(X, P, M, C)` como no contrato.

### 5. Amostragem

Para cada refeição, sorteia-se o latente pela posterior e depois as ocultas:

- `z ~ N(E[z_n], σ² A_n⁻¹)`, via Cholesky de `A_n = L Lᵀ`: `z = E[z_n] + σ L⁻ᵀ u`, com `u ~ N(0, I_q)`;
- `x_H = μ_H + W_H z + σ v`, com `v ~ N(0, I)`, e truncamento em 0.

Isso dá a condicional exata `x_H | x_O` sem montar covariâncias de tamanho variável. O teste confere contra o complemento de Schur da covariância `WWᵀ + σ²I`. Sem observados, a posterior é a priori, e as amostras seguem a marginal (taxa 1). `E[z_n]` e `L` são calculados uma vez por bloco e reaproveitados nas S amostras.

### 6. Métricas de imputação

`metricas.avaliar_imputacao(G_verdadeiro, amostras_log, M)` recebe as amostras em `log1p` das células ocultas e devolve MAE em gramas e em log, NLL e CRPS em log, PICP e MPIW em gramas (spec `metricas`). `metricas.imputacao_por_gaveta(...)` faz o mesmo agrupado por gaveta.

- O CRPS vem de `scoringrules.crps_ensemble`, já instalado.
- O intervalo pelos quantis empíricos vale igual em log e em gramas, porque `expm1` é monótona; a MPIW é reportada em gramas.
- O experimento sorteia as amostras da Parte 1 com um fluxo de gerador próprio e as consome em blocos, guardando só as células de `M`.
- Alternativa descartada: um método de momentos condicionais exatos no contrato (ver `proposal.md`).

### 7. Experimento com o imputador como dimensão

`IMPUTADORES = {"porcao_media": PorcaoMedia, "ppca": lambda: PPCA(q=5, semente=0)}`. Por imputador:

1. ajuste no treino completo e nas 5 dobras, uma vez, com o tempo registrado em `results/imputadores_ajuste.csv` (imputador, etapa, segundos, iterações);
2. laço de semente e taxa, com os mesmos geradores para todos os imputadores (`default_rng((semente, taxa, fluxo))`);
3. três braços e métricas da Parte 2; métricas da Parte 1 nas taxas > 0, num fluxo de gerador separado.

Mudanças nas peças existentes:

- `imputar_treino_cruzado` passa a receber os imputadores das dobras já ajustados (`Sequence[Imputer]`), e não um construtor. A função `ajustar_dobras(novo_imputador, G, P, C, dobras)` faz os 5 ajustes. Os testes de `test_propagacao.py` passam a chamar as duas.
- A referência da taxa 0 por número de ocultas (braço `referencia_taxa_0`, imputador `nenhum`) é calculada uma vez por taxa e semente, com `por_ocultas` sobre a previsão do regressor de referência com as gramas completas.
- `escolher_S` sai; `S = 100` vira constante.
- Figuras: `facet_wrap` por imputador, com as cores de braço atuais. A legenda deixa de citar um imputador fixo.
- CSVs com a coluna `imputador`; o resumo agrupa por imputador, taxa e braço.

### 8. Medição de custo e ponto de decisão

Antes da execução completa, um ajuste do PPCA no treino completo mede tempo por iteração, iterações e o tempo de 100 amostras da validação.

- Ajuste até 5 minutos: segue o desenho.
- Acima disso, a execução para e a decisão volta para aprovação: subamostra estratificada do treino, ou agrupamento das refeições por padrão de observados (as refeições do mesmo padrão dividem `A_n`).

Estimativa: 6 ajustes por execução (completo + 5 dobras), mais a curva de 15 rodadas por imputador.

### 9. Organização do código

```
src/amaprob/
  imputation/  ppca.py         (PPCA)
  regression/  propagation.py  (+ ajustar_dobras; imputar_treino_cruzado com dobras ajustadas)
  evaluation/  metricas.py     (+ avaliar_imputacao, imputacao_por_gaveta)
  experiments/ propagacao.py   (imputadores como dimensão)
tests/
  test_ppca.py                 (forma fechada, recuperação, monotonia, condicional, semente)
  test_imputacao.py            (contrato parametrizado nos dois imputadores)
```

## Risks / Trade-offs

- [Ruído isotrópico: o PPCA usa um σ² para todas as gavetas, enquanto o baseline tem um desvio por gaveta; algumas gavetas podem ficar pior calibradas que no baseline] → É o contraste PPCA × FA (ruído diagonal), que a próxima change mede. As métricas por gaveta mostram onde acontece; não deve ser lido como erro de implementação.
- [Gavetas que quase nunca aparecem juntas deixam a covariância entre elas mal determinada] → Só afeta refeições em que as duas estão presentes, que são raras pela mesma razão. A recuperação sintética usa ausência ao acaso.
- [Na taxa 1, sem observados, o PPCA usa a marginal: a média é próxima da do baseline, e o ganho, se houver, vem da correlação entre as ocultas da mesma refeição] → Leitura esperada; a extensão B+ só volta se a taxa 1 continuar larga.
- [EM lento quando autovalores estão próximos] → Limite de 500 iterações e histórico registrado; o teste de forma fechada usa autovalores bem separados.
- [Execução completa mais longa, com dois imputadores] → Sem a comparação de S (−14 min) e com as dobras ajustadas uma vez; tempo total registrado.
- [MC nas métricas da Parte 1 com S = 100] → Erro pequeno diante das diferenças entre imputadores; o mesmo S da Parte 2.

## Migration Plan

1. Nenhuma dependência nova; nenhum dado regravado em `data/processed/`.
2. Os CSVs e as figuras de `results/propagacao_*` são regravados com a coluna `imputador`; `results/propagacao_S.csv` não é tocado.
