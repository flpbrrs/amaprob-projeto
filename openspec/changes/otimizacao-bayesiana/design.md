# Design

## Context

Motivação em `proposal.md`. Estado atual que orienta o desenho:

- Valores de partida:
  - q = 5 no PPCA e na FA, com e sem média condicional;
  - α = 1 no ridge da média condicional (`imputation/media_condicional.py`), também na porção média condicional;
  - M = 256 no SVGP, com 60 épocas.
- O construtor de `PPCA`, `FA` e `PorcaoMedia` aceita `q` (quando há) e `alfa`; o do `SVGP` aceita `n_indutores`.
- Na validação, com a regressão linear e S = 100, o CRPS da propagação na taxa 0,5 é:

  | Imputador | CRPS |
  |---|---|
  | FA condicional | 11,48 |
  | PPCA condicional | 11,64 |
  | Porção média condicional | 11,99 |

  As diferenças entre imputadores são da ordem de décimos de grama, e o ganho esperado da busca é da mesma ordem.
- Custos medidos no treino completo (213 mil refeições):

  | Etapa | Tempo |
  |---|---|
  | Ajuste da FA condicional | 110 s |
  | Ajuste do PPCA condicional | 99 s |
  | Ajuste da porção média condicional | 0,7 s |
  | Ajuste do SVGP (M = 256) | 232 s |
  | Braço de propagação na validação inteira (S = 100), com as métricas | ~65 s |

  O custo do SVGP por passo cresce com M².
- A regressão linear bayesiana ajusta α e β do prior pela evidência (`regression/blr.py`) e não entra na busca.
- `experiments/propagacao.py` tem os geradores por semente, taxa e fluxo (`gerador`) e o carregamento de treino e validação (`treino_e_validacao`), reaproveitados aqui.
- Máquina: CPU de 6 núcleos, 16 GB, sem GPU.

## Goals / Non-Goals

**Goals:**
- Escolher q e α (PPCA e FA condicionais), α (controle) e M (SVGP) só na validação, com otimização bayesiana e busca aleatória do mesmo orçamento.
- Medir o ganho da configuração escolhida contra a de partida no treino completo.
- Gravar a configuração para a rodada no teste.

**Non-Goals:**
- Rodar de novo a curva de propagação com a configuração adotada. Isso fica para a change da rodada no teste, que lê `configs/hiperparametros.json` e, antes do teste, roda a curva de validação em todas as taxas para os modelos cuja configuração mudou (o PPCA, com e sem média condicional, adotou q = 12), porque a busca e a confirmação só olham a taxa 0,5.
- Buscar as versões sem média condicional. Elas herdam o q da versão condicional, para que a comparação com e sem média condicional mude uma coisa só.
- Buscar as épocas, a taxa de aprendizado ou o tamanho do lote do SVGP. As 60 épocas foram medidas com M = 256 e valem para a faixa buscada, como suposição conferida na medição (Decisão 6).

## Decisions

### 1. Candidatos

| Modelo | Botões | Candidatos |
|---|---|---|
| `ppca_condicional`, `fa_condicional` | q, α | q ∈ {1, …, 15} × α ∈ 21 valores logarítmicos de 10⁻² a 10³ (4 por década, inclui α = 1): 315 |
| `porcao_condicional` | α | os mesmos 21 valores de α |
| `svgp` | M | M ∈ {32, 48, …, 512} (passo 16, inclui 256): 31 |

- α é a precisão do prior nos coeficientes padronizados. α → ∞ volta à média fixa; α → 0 é mínimos quadrados sem freio.
- Grade discreta: q e M são inteiros, e a aquisição é calculada em todos os candidatos não avaliados. Não há otimização contínua, o que deixa a escolha exata e reprodutível.
- Para o surrogate, cada botão é levado a [0, 1]: q de forma linear, log₁₀ α e log₂ M.

### 2. Objetivo

`objective(params) -> NLL`, como no contrato do `AGENTS.md` §7. Ele devolve também CRPS, PICP e MPIW para o registro.

- **Imputadores:**
  1. ajuste na amostra do treino (Decisão 4), com o α do candidato (Decisão 3);
  2. braço de propagação de `prever_bracos` com a regressão linear bayesiana, ajustada uma vez no treino completo, na taxa 0,5, semente 0, com o gerador do experimento (`gerador(0, t, FLUXO_VALIDACAO)`) e S = 100;
  3. `avaliar_mistura` na validação inteira.
- **SVGP:**
  1. `SVGP(n_indutores=M, semente=0)` ajustado no treino completo, com as gramas completas;
  2. previsão da validação com as gramas completas (taxa 0);
  3. métricas da preditiva gaussiana.

  Como a taxa 0 não tem máscara, a NLL não depende da semente.
- Com as sementes fixas, o objetivo é determinístico (Regra 6: só treino e validação).

### 3. α sem conversão pela amostra

Na média condicional, o passo M resolve `(Σ E[z̃z̃ᵀ] + s_k α D) θ = Σ x E[z̃]`: é o máximo a posteriori com prior N(0, α⁻¹) nos coeficientes, em forma de soma. As somas crescem com o número de refeições e a penalização não, então com menos dados o prior pesa mais, que é o comportamento esperado de um prior. Com o modelo bem especificado, o α ótimo é a precisão do prior e não depende de N; na regressão ridge, o λ ótimo na forma de soma é pσ²/‖β‖², também sem N. Converter por `n / N` vem da perda escrita como média e, aqui, inflaria o α cerca de 4 vezes.

Por isso o candidato usa o α direto na amostra, e é esse α que vai para o registro e para o arquivo de configuração. Como o modelo não é bem especificado, a transferência é conferida na confirmação: a escolhida também é reajustada com `α · N / n` (Decisão 7), o mesmo freio relativo aos dados que ela teve na amostra. Se essa variante ganhar nas três sementes, a decisão volta para aprovação.

### 4. Amostra estratificada

- Fração de partida: 25% das refeições do treino (~53 mil), estratificada pela ocasião (colunas de ocasião de `C`), com `default_rng(0)`. A mesma amostra serve a todas as avaliações dos imputadores.
- A fração é confirmada na medição (Decisão 6).
- O SVGP não usa amostra. O custo dele é por passo de gradiente, e uma amostra com as mesmas 60 épocas daria menos passos, o que muda o ajuste que está sendo avaliado.

### 5. Busca

`src/amaprob/tuning/busca.py`, sem dependência de dados nem de modelos:

- `Avaliador`: guarda o objetivo e um cache por candidato. Um candidato já avaliado por qualquer método é reaproveitado, com a marca e o tempo original.
- `busca_aleatoria(candidatos, avaliador, n, rng)`: sorteia sem reposição.
- `otimizacao_bayesiana(candidatos, avaliador, n, iniciais, semente)`:
  - começa pelos candidatos `iniciais`, que são os 4 primeiros da busca aleatória;
  - a cada passo:
    1. ajusta um `SingleTaskGP` do BoTorch (padrões da biblioteca: kernel com um comprimento por dimensão e saída padronizada) às entradas normalizadas e a −NLL, com `fit_gpytorch_mll`;
    2. calcula `LogExpectedImprovement` em todos os candidatos não avaliados;
    3. avalia o máximo.
  - `torch.manual_seed(semente + passo)` antes de cada ajuste do surrogate, para que a sequência seja reprodutível.
- Orçamento: 12 avaliações por método e por modelo.

### 6. Medição de custo e ponto de decisão

Antes da rodada:
- tempo de ajuste da FA e do PPCA condicionais na amostra de 25%, com q = 15 (o mais caro), e de um objetivo completo;
- segundos por época do SVGP com M = 512, em 3 épocas, extrapolados para 60;
- curva da ELBO com M = 512, para conferir se 60 épocas ainda assentam.

Estimativa, a confirmar: imputadores ~1 h 30 no total; SVGP de 1 a 3 h, conforme os M visitados. Se a estimativa total passar de 4 h, a execução para e a decisão volta para aprovação (fração menor, teto de M menor ou orçamento menor no SVGP).

**Medido** (amostra de 53.309 refeições):

| Etapa | q = 5 / M = 32 | q = 15 / M = 512 |
|---|---|---|
| Ajuste da FA condicional | 30 s | 163 s |
| Ajuste do PPCA condicional | 46 s | 266 s, no limite de 500 iterações |
| Medida do objetivo | 67 s | 67 s |
| SVGP, por época | 1,3 s | 11,8 s |

Com cerca de 21 ajustes distintos por busca, a estimativa é de ~3 h nos imputadores (com as confirmações) e de 2 a 4 h no SVGP: ~5 h no total, até ~8 h se a busca for para q ou M altos. Passou de 4 h. A decisão foi rodar em duas partes com `--modelos` (imputadores e SVGP), sem mudar o desenho.

### 7. Execução, registro e confirmação

`uv run python -m amaprob.experiments.otimizacao [--modelos ...]`. Para cada modelo:

1. Avalia a configuração de partida (referência).
2. Roda a busca aleatória e depois a otimização bayesiana.
3. Escolhe a menor NLL entre a partida e os dois métodos. No SVGP, escolhe o menor M avaliado (a partida e os dois métodos) cuja NLL fica a até 0,01 da melhor: a NLL tende a cair sempre um pouco com M, e o custo de ajuste e de previsão cresce com ele.
4. Confirma a escolha:
   - **Imputadores:** a configuração de partida e a escolhida são reajustadas no treino completo; nos condicionais com α, também a escolhida com `α · N / n` (`escolhida_alfa_convertido`), como diagnóstico da Decisão 3. O braço de propagação com a regressão linear é medido na taxa 0,5, nas sementes 0, 1 e 2. A linha de partida deve repetir os números de `propagacao_metricas.csv` (mesma configuração e mesmos geradores), o que confere o código do objetivo. A semente 0 também foi usada na escolha; as sementes 1 e 2 são máscaras novas.
   - **SVGP:** a busca já usa o treino completo e a taxa 0, então a confirmação é a das avaliações de partida e escolhida, com as quatro métricas.

Arquivos gravados:
- `results/otimizacao_avaliacoes.csv`: modelo, método (`partida`, `aleatoria`, `bayesiana`), avaliação, q, α, M, NLL, CRPS, PICP, MPIW, segundos, reaproveitada e melhor NLL até ali.
- `results/otimizacao_confirmacao.csv`: modelo, configuração (`partida`, `escolhida` ou `escolhida_alfa_convertido`), hiperparâmetros, semente e as quatro métricas.
- `results/figuras/otimizacao_convergencia.png`: melhor NLL pelo número da avaliação, um painel por modelo, uma linha por método e uma linha horizontal na NLL de partida. Plotnine com o tema comum.
- `configs/hiperparametros.json`: a configuração adotada de cada modelo buscado (ver Adoção). O PPCA e a FA sem média condicional recebem o q adotado nas versões condicionais.

Com `--modelos`, as linhas dos outros modelos nos CSVs ficam como estão, como na junção da propagação. `--so-confirmacao` refaz só a variante de α convertido, a partir da escolhida gravada.

**Adoção** (decidida depois da confirmação da FA): a busca com uma semente otimizou o ruído da máscara. A escolhida da FA (q = 6, α = 1000) ganhou 0,014 de NLL na semente 0 e perdeu 0,006 e 0,015 nas sementes 1 e 2. A oscilação da comparação entre máscaras é de ~0,015. Por isso, um imputador só adota a escolhida se ela melhorar a NLL em cada semente nova (1 e 2) e a média delas em pelo menos 0,015; senão, fica a de partida. O SVGP adota a escolhida, porque a busca dele não tem máscara e a partida (M = 256) concorre na regra do menor M. `configs/hiperparametros.json` é derivado da confirmação.

**Desfecho:** o PPCA condicional adotou q = 12, α = 17,8 (melhora firme nas duas sementes novas); a FA condicional e o controle ficaram na partida. O SVGP escolheu M = 288 pela regra fixada antes da rodada; a diferença para M = 256 (0,004 de NLL) é do tamanho do ruído do ajuste e, sob a exigência de ganho firme dos imputadores, nenhum M passaria. A regra fixada foi mantida, e a curva de validação do SVGP não é refeita por essa troca, que fica declarada.

**Origem dos resultados:** as buscas e as confirmações dos três imputadores foram gravadas antes de duas correções no código: a variante de α convertido usava `α · n / N` e a escolha não incluía a partida entre os concorrentes. As linhas da variante foram regravadas com `--so-confirmacao`, já com `α · N / n`. A escolha não muda, porque em todas as buscas a partida foi pior que o melhor candidato. O SVGP já rodou com o código final.

**Pendente para a change da rodada no teste:** o PPCA sem média condicional herda q = 12 e nunca foi ajustado com esse q; com q = 5, algumas dobras já paravam no limite de 500 iterações, e o PPCA condicional com q = 12 levou 660 s no treino completo. As estimativas de tempo da curva de validação e da rodada no teste não têm medição por baixo. A primeira tarefa daquela change é um ajuste cronometrado de cada PPCA com q = 12 no treino completo, com o número de iterações; parar no limite é ponto de parada, porque um EM sem convergir no teste vira ressalva no artigo e pode inflar o ganho da média condicional no PPCA. Com a partida adotada, as curvas de validação existentes continuam valendo em todas as taxas.

**Borda da grade:** a FA escolheu α = 1000, o teto. A checagem de borda foi descartada: qualquer critério dela ficaria abaixo do ruído entre máscaras. Com a regra de adoção, a borda só importaria se a escolhida fosse adotada. O limite α → ∞ é a versão sem média condicional, já medida e pior que α = 1, então o ótimo é finito.

## Risks / Trade-offs

- **Ganho pequeno.** As diferenças atuais são de décimos de grama. O valor principal da Parte 3 é a escolha feita só na validação e a curva de convergência; o ganho é reportado como sair.
- **q e α escolhidos numa amostra.** Pela teoria, o α transfere sem conversão (Decisão 3), e a confirmação confere isso. Um q maior pode ser sustentado pelo treino completo e não pela amostra; a confirmação mostra o efeito, e a regra de adoção decide.
- **Grades pequenas.** No controle (21 candidatos) e no SVGP (31), 12 avaliações cobrem boa parte da grade, e a vantagem da otimização bayesiana sobre a aleatória tende a aparecer pouco. Fica declarado; a comparação mais informativa é a dos imputadores em duas dimensões.
- **Objetivo com uma semente e uma taxa.** É barato, mas a escolha pode se ajustar à semente 0. Por isso a confirmação mede também as sementes 1 e 2.
- **60 épocas fixas para todo M.** M maior pode pedir mais épocas. Isso é conferido na medição com M = 512; se a ELBO ainda subir, a decisão volta para aprovação.
