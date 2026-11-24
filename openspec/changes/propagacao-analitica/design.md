# Design

## Context

O experimento oficial (`experiments/propagacao.py`) compara três braços com a mesma regressão linear bayesiana: plug-in, plug-in recalibrado e propagação. O plug-in preenche as ocultas com a média das S amostras em gramas e prevê uma vez; a propagação prevê para cada amostra e combina as preditivas na mistura `(1/S) Σ N(μ_s, σ²_s)`.

A Parte 1 entrega a condicional completa das ocultas, e não só amostras: com o posterior do latente, a média e a covariância de `x | x_O` saem em forma fechada para o PPCA (análise de componentes principais probabilística, `σ²` isotrópico), a FA (análise fatorial, `Ψ` diagonal) e a porção média (diagonal pura). A regressão linear bayesiana tem preditiva gaussiana e posterior dos pesos em forma fechada (`regression/blr.py`).

A combinação das duas dá uma preditiva exata, sem sorteio. O interesse é duplo: separar erro de Monte Carlo de erro de modelo na propagação e ter um braço de custo desprezível para varrer taxas.

## Goals / Non-Goals

**Goals:**
- Prever o carboidrato com a incerteza da entrada em forma fechada, para os seis imputadores.
- Comparar com o plug-in e com a propagação por amostragem nas mesmas máscaras e sementes.
- Separar erro de Monte Carlo de diferença de modelo com S = 10, 30 e 100.
- Deixar o protótipo pronto para a decisão de entrar no artigo.

**Non-Goals:**
- Alterar o experimento oficial, seus CSVs ou os modelos existentes.
- Cobrir o SVGP, que não tem preditiva em forma fechada com entrada aleatória.
- Medir no teste. A rodada no teste depende de aprovação.
- Modelar o truncamento em 0, cujo efeito é de no máximo 1 g por célula oculta.

## Decisions

### 1. Momentos pela condicional exata, sem tocar nos imputadores

`momentos_ocultos` lê os internos dos imputadores (`_passo_e`, `_media`, `W_`, `sigma2_`, `psi_`, `media_`, `desvio_`, `coeficientes_media_`, `padronizacao_`) em vez de estender o contrato `Imputer`, que só expõe amostras. A vantagem é não mexer em modelo fechado e não abrir um método novo no contrato para um braço de protótipo. O custo é acoplamento a internos, e é por isso que os testes comparam os momentos com as amostras: uma mudança nos internos aparece neles.

*Alternativa descartada:* estimar os momentos a partir de muitas amostras. Trocaria um erro de Monte Carlo por outro e perderia o ponto do braço.

### 2. Conversão de log para gramas pela log-normal

A condicional é gaussiana em `log1p(gramas)`. Com `g = exp(x) − 1` e `x ~ N(μ, Σ)`,

    E[g_k] = exp(μ_k + Σ_kk/2) − 1
    Cov(g_i, g_j) = exp(μ_i + μ_j + (Σ_ii + Σ_jj)/2) (exp(Σ_ij) − 1)

Como `1 + E[g_k] = exp(μ_k + Σ_kk/2)`, a covariância sai como `a_i a_j (exp(Σ_ij) − 1)`, com `a = 1 + E[g]`, o que dispensa exponenciar duas vezes.

A covariância entre gavetas ocultas da mesma refeição entra na conta. Ela não é, porém, o que distingue este braço da propagação: `Imputer.sample` sorteia os fatores comuns por refeição e só depois aplica o ruído de cada gaveta, então as amostras de uma mesma refeição já saem correlacionadas pela construção. O que distingue o braço analítico é não ter erro de Monte Carlo: ele usa a condicional inteira, numa passada e sem gerador.

### 3. Só as K colunas de gramas são aleatórias

`P` e `C` são sempre observados, e as colunas de presença e de contexto entram com variância 0. Por isso a fórmula usa os blocos `m_K` e `S_KK` e `Cφ = Cg / (s sᵀ)`, com `s` o desvio de padronização das gramas. Com `Cg = 0`, a expressão se reduz ao `predict`, o que é o teste de redução.

### 4. Truncamento em 0 ignorado, e declarado

`Imputer.sample` corta as amostras em `log1p(g) ≥ 0`, isto é, em `g ≥ 0`. Sem o corte, `g = expm1(x)` cairia no intervalo `(−1, 0)`, então o truncamento muda cada célula oculta em no máximo 1 g, contra porções que vão de dezenas de gramas a quilos. O efeito não é a causa de nenhuma diferença entre os braços e não justifica tratamento. A forma fechada não trunca, e isso fica declarado na docstring do módulo.

### 5. Experimento separado, reaproveitando as peças do oficial

`experiments/propagacao_analitica.py` importa `treino_e_medida`, `IMPUTADORES`, `gerador`, `FLUXO_VALIDACAO` e `S` do experimento oficial e ajusta cada imputador uma vez no treino completo, com máscara vazia. Não usa dobras nem o plug-in recalibrado, que existem para o braço de margem fixa.

O experimento oficial e seus CSVs não mudam. As tabelas novas têm nome próprio e a junção por imputador permite rodar em duas etapas.

*Alternativa descartada:* acrescentar uma coluna de braço ao experimento oficial. Misturaria um protótipo aos resultados fechados e obrigaria a reexecutar tudo.

### 6. S = 10 e 30 ao lado de S = 100

A propagação com S = 100 tem erro de Monte Carlo. Sem uma referência, uma diferença entre ela e a forma fechada não se atribui a modelo ou a sorteio. Com três valores de S, a convergência aparece: se S = 10, 30 e 100 caminham para a analítica, a diferença é de amostragem. Cada S usa fluxo próprio de gerador; o de S = 100 é o do experimento oficial, o que permite conferir as linhas de plug-in e propagação contra `results/propagacao_metricas.csv`.

### 7. Taxa 0 fora

Na taxa 0 não há oculta, `Cg = 0` e a analítica coincide com o plug-in. Rodar acrescentaria linhas idênticas.

### 8. Salvaguarda de plausibilidade

`momentos_em_gramas` levanta `ValueError` com a gaveta quando a média em gramas ou o desvio-padrão em gramas passam de 1 t, ou deixam de ser finitos. A salvaguarda só detecta: não corta valor nem altera amostra.

O teto vale para quantidades em gramas, então a covariância entra pela raiz da diagonal. Aplicado à variância em g², o mesmo teto proibiria desvio acima de 1 kg e barraria o modelo adotado em gavetas de porção grande: as bebidas chegam a 9,5 kg observados e têm média observada de 881 g, e a FA condicional imputa bebida oculta com desvio na casa do quilo sem nada de degenerado.

### 9. Ordem da medição

Primeiro a FA com média condicional, a melhor das taxas 0,25 a 0,75 no experimento oficial, e a porção média, o controle sem fatores. Se a forma fechada não ajudar nesses dois casos, não há razão para gastar tempo nos outros quatro. Depois os quatro restantes.

### 10. Teste fora do escopo

A medição no teste só entra com aprovação, porque o teste é uma rodada única e congelada (Regra 6). O protótipo mede só na validação.

### 11. Rodadas em paralelo e junção com conferência

Os seis imputadores custam, cada um, cerca de 20 minutos de medição, quase todos na propagação S = 100. `--sufixo` põe um nome no fim de cada CSV e permite rodar até três processos ao mesmo tempo, sem disputa de arquivo. `--taxas` e `--sementes` restringem a rodada, para reconferir uma combinação isolada e para comparar com a rodada em paralelo.

`experiments/juntar_propagacao_analitica.py` lê o arquivo base mais os de cada sufixo e falha, em vez de juntar, quando as colunas diferem entre dois arquivos ou quando a chave se repete. A chave é (imputador, taxa, semente, braço, S); nas tabelas com mais de uma linha por braço, entram a classe de ocultas e a etapa. O paralelismo não muda número: com semente fixa por (semente de máscara, taxa, fluxo), cada rodada produz o que produziria sozinha, e a conferência de uma combinação isolada contra a rodada em paralelo mede isso.

### 12. O que a medição mostrou

Na validação, a NLL da analítica empata com a de S = 100 na FA condicional, dentro de ±0,02 para os dois lados (4,492 contra 4,476 a 0,5 e 4,733 contra 4,746 a 0,75). Na porção média ela fica levemente pior: de 0,02 a 0,05 acima nas taxas 0,25 a 0,75 (4,602 contra 4,552 a 0,5) e empata na taxa 1,0 (5,046 contra 5,045). O padrão é coerente com a hipótese da forma, em que mais incerteza em log significa mais assimetria e a forma fechada usa só média e variância. O diagnóstico da forma confirma ou descarta essa leitura.

O ganho sobre o plug-in é grande e aparece em todas as taxas (5,270 para 4,492 na FA condicional a 0,5). No CRPS a analítica fica um pouco atrás, com cobertura um pouco acima.

O texto não deve apresentar a analítica como melhor que a propagação. Ela é mais barata (2,0 s contra 50,2 s na FA condicional, 25 vezes; 62,9 vezes na porção média), não consome gerador aleatório, empata em NLL na FA condicional e fica levemente atrás na porção média.

## Riscos

- **Diferença de CRPS:** a analítica fica um pouco atrás da S = 100 no CRPS (11,85 contra 11,48 na FA condicional a 0,5) e acima em PICP90 (94,1% contra 93,5%), com MPIW maior. O truncamento não é a causa, já que vale no máximo 1 g. A explicação provável é a assimetria: a forma fechada usa só média e variância, e a mistura por amostras carrega o terceiro momento. O efeito é medido no diagnóstico da forma.
- **Custo de memória:** a covariância em log é `N × K × K` por taxa; o cálculo é em blocos, mas a matriz inteira da validação (45.639 refeições, K = 29) ocupa cerca de 300 MB se guardada de uma vez. O experimento processa em blocos de 4.096 refeições.
- **Acoplamento a internos:** declarado na decisão 1 e coberto pelos testes.
