# AGENTS.md — Projeto Final de Aprendizagem de Máquina Probabilística (2026.2)

Este arquivo orienta assistentes de código (Claude Code, Codex, Copilot, Cursor e similares) e pessoas que contribuem com o projeto. Leia-o inteiro antes de propor ou escrever código.

---

## 1. Contexto

**Título:** Estimação probabilística do carboidrato de refeições brasileiras a partir de registros alimentares incompletos.

No cuidado em diabetes tipo 1, o carboidrato da refeição orienta a dose de insulina, mas o registro do paciente raramente é completo: ele informa quais alimentos comeu e, às vezes, a quantidade de alguns. Com registro completo, o carboidrato é calculável por tabela. Com informação faltando, é preciso inferir o que falta e quantificar a confiabilidade da estimativa. **O projeto investiga esse segundo regime.**

A identificação dos alimentos (que, num sistema real, viria de imagens) está abstraída: a lista de alimentos vem do dataset.

**Exigências da disciplina** (descrição oficial do projeto):
- Trio de pós-graduação: as **três partes** são obrigatórias — (1) dados faltantes, (2) regressão com predições probabilísticas, (3) ajuste de hiperparâmetros por otimização bayesiana.
- Dados reais; mais de um modelo por parte, com comparação de desempenho; pacotes prontos são permitidos.
- O artigo deve indicar **como cada participante contribuiu**.
- Entregáveis: artigo (Introdução, Fundamentação teórica, Metodologia, Experimentos, Conclusão; LaTeX em modelo de congresso, ex.: NeurIPS), código (notebooks e/ou repositório) e apresentação.
- Datas: acompanhamento de 19/11 a 26/11; **entrega final em 30/11/2026** (SIGAA); apresentações em 01, 03, 08 ou 10/12.

---

## 2. Dados

**Fonte:** microdados de consumo alimentar individual da POF 2017–2018 (Pesquisa de Orçamentos Familiares, IBGE), versão 20230713. Dados reais, até dois dias de registro por pessoa. Instruções de download no `README.md`.

**Verificado no dicionário de variáveis e nos dados:**
- Arquivo `CONSUMO_ALIMENTAR.txt`, largura fixa, uma linha por item (1,18 milhão de itens, 46.164 pessoas). Leitor em `src/amaprob/data/pof.py`.
- `QTD` = gramas consumidas = `V9005` (quantidade da medida caseira) × `GRAMATURA1` (gramas da medida). Ver Regra 1.
- `CHOTOT` = carboidrato total do item (g), calculado pelo IBGE com a TBCA 7.0 (Tabela Brasileira de Composição de Alimentos). **`y` = soma de `CHOTOT` na refeição**; não é preciso cruzar com tabela externa.
- `V9017` = ocasião (1 café da manhã, 2 almoço, 3 lanche, 4 jantar, 5 ceia, 6 outra, 9 não determinado); `V9015` = horário.
- Pessoa = `COD_UPA` + `NUM_DOM` + `NUM_UC` + `COD_INFORMANTE`.
- Adições (açúcar, azeite, margarina…) são linhas próprias, sem medida caseira. O açúcar de mesa responde por 8,5% de todo o carboidrato.

**Unidade de análise — refeição:** itens da mesma pessoa, no mesmo dia, na mesma ocasião e no mesmo horário. Entram só refeições que pedem insulina: ocasiões 1 a 5 com pelo menos 10 g de carboidrato (304.670 refeições). Código em `src/amaprob/data/refeicoes.py`.

**Gavetas (grupos alimentares):** derivadas da classificação do IBGE (20 grupos, 116 subgrupos), versionada em `configs/classificacao_ibge.csv`. O açúcar de mesa é gaveta própria. **K = 29** (Farinhas e massas dividida em subgrupos; Bebidas em 4 perfis nutricionais). Método e justificativa em `docs/piso-de-erro.md`.

**Representação** (para N refeições e K gavetas):

| Objeto | Forma | Conteúdo |
|---|---|---|
| `X` | N × K | `log1p(gramas)` por gaveta; 0 quando a gaveta está ausente |
| `P` | N × K | presença da gaveta (0/1) — **sempre observada** |
| `M` | N × K | máscara de quantidades ocultas; `M ⊆ P` |
| `C` | N × 13 | contexto, só do arquivo de consumo alimentar: ocasião (5 indicadoras), hora (seno e cosseno), macrorregião (5 indicadoras) e `log1p` da renda. Sexo e idade (arquivo `MORADOR`, a verificar) ficam fora |
| `y` | N | carboidrato total da refeição (g) |
| `person_id` | N | identificador do indivíduo (split) |

**Escala log só na imputação.** A imputação trabalha em `log1p(gramas)`; a regressão recebe as quantidades de volta em gramas. Motivo: o carboidrato é quase linear nas gramas (R² de 0,81 em gramas contra 0,58 em log, com 20 grupos e regressão linear).

**Fatos da coleta** (Manual do Agente de Pesquisa da POF 2017–2018, em `Documentacao_20230713/`):
- `QUADRO` 72 e 73 são o primeiro e o segundo dia recordado, com pelo menos um dia de intervalo (p. 304).
- **Açúcar adicionado é estimativa do IBGE:** na coleta só se marca se houve adição (sim ou não), sem quantidade (p. 312 e 314); os gramas foram estimados depois (dicionário, `V9001`). A regra da estimativa não está documentada (146 valores distintos, concentrados em 5, 20 e 30 g). O artigo declara isso como limitação; ver `backlog/decisions.md`.

---

## 3. Regras invioláveis

Violações destas regras invalidam o experimento. Sinalize imediatamente qualquer código que as quebre.

1. **A medida caseira nunca é input.** As gramas da POF vêm de conversão da medida caseira; usá-la torna o problema determinístico. O leitor não carrega `V9005`, `V9007`, `GRAMATURA1` nem `COD_UNIDADE_MEDIDA_FINAL`.
2. **Quantidades mascaradas nunca vazam.** Nenhuma etapa de imputação ou regressão pode ler `X[M]`; elas só são usadas para avaliação. Os imputadores recebem `X` com `NaN` em `M`. Nutrientes (`CHOTOT` e outros) são `QTD` × composição: só servem como alvo, nunca como entrada.
3. **Split por indivíduo.** Os dois dias de uma pessoa ficam no mesmo conjunto (treino, validação ou teste). Split por refeição causa vazamento.
4. **Máscaras geradas antes da modelagem**, com semente fixa, e persistidas em disco. Todos os modelos usam as mesmas máscaras.
5. **Dados reais como base.** O mascaramento é aplicado sobre registros reais; nenhum dado sintético entra como dado principal.
6. **Nada é ajustado no teste.** Otimização bayesiana, escolha de gavetas e escolhas de modelo usam apenas a validação.

---

## 4. Protocolo experimental

- **Taxas de mascaramento:** `{0.0, 0.25, 0.5, 0.75, 1.0}` sobre as quantidades de gavetas **presentes**. A taxa 1.0 é o cenário "sei o que comi, não sei quanto".
- **Conjuntos mascarados:** treino, validação e teste, com o mesmo sorteio.
- **Ajuste no treino completo:** o imputador e o regressor de referência são ajustados uma vez com as gramas completas do treino; as máscaras definem o que fica oculto na validação e no teste. A máscara do treino só serve ao braço plug-in recalibrado (§6), com imputação por ajuste cruzado por pessoa em 5 dobras. Motivo: na taxa 1, o treino mascarado não teria quantidade observada para o imputador aprender. Detalhes em `docs/mascaramento.md`.
- **Mecanismo:** aleatório (MCAR), declarado como limitação. O mascaramento é por gaveta (esconde todos os itens dela de uma vez), também declarado.
- **Repetições:** 3 sementes por taxa; reportar média e desvio.
- **Conjunto de medida:** a validação, para todas as escolhas, em `results/`; o teste só numa rodada única, com modelos e hiperparâmetros congelados (`--conjunto teste`), em `results/teste/`. O ajuste é sempre só no treino.
- **Split:** treino / validação / teste = 70/15/15 por `person_id`, semente 0 (`src/amaprob/data/splits.py`).
- **Escala:** os imputadores da otimização bayesiana são ajustados numa amostra de 25% do treino, estratificada pela ocasião. O SVGP roda no treino e na validação inteiros (ajuste de ~4 min com M = 256, previsão de ~36 s por rodada da curva). Treino em CPU.

---

## 5. Parte 1 — Dados faltantes (imputação probabilística)

**Objetivo:** imputar as quantidades ocultas com distribuição, explorando coocorrências e contexto.

**Contrato** (`src/amaprob/imputation/base.py`):

```python
class Imputer(Protocol):
    def fit(self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray) -> "Imputer": ...
    def sample(self, X: np.ndarray, P: np.ndarray, M: np.ndarray, C: np.ndarray,
               n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Retorna (n_samples, N, K). Entradas observadas são copiadas; só X[M] é amostrado."""
```

**Modelos:**

| Modelo | Escopo | Implementação sugerida |
|---|---|---|
| Porção média por gaveta | Baseline de referência | NumPy |
| PPCA (Análise de Componentes Principais Probabilística) via EM com dados faltantes | Obrigatório | NumPy, do zero (o scikit-learn não trata faltantes); `imputation/ppca.py`, q = 5 de partida |
| FA (Análise Fatorial) via EM com dados faltantes | Obrigatório | NumPy, do zero; `imputation/fa.py`, q = 5 de partida. Difere do PPCA só no ruído (diagonal em vez de isotrópico); módulo próprio, para não mudar o caminho numérico do PPCA. Piso no ruído de cada gaveta: 0,1% da variância observada, contra ψ → 0 (caso de Heywood) |
| Média condicional (porção média, PPCA e FA) | Extra | Opção `media_condicional`; contas comuns em `imputation/media_condicional.py`. A média de cada gaveta vira uma função linear de `P` e `C` (padronizados), com ridge (α = 1) nos coeficientes; no PPCA e na FA, EM do máximo a posteriori (ECM). A porção média condicional é o controle: média condicional sem fatores |
| VAE (Autoencoder Variacional) com máscara | Extra | PyTorch; entrada = `X` observado + `P` + `M` + `C` |

**Gavetas ausentes como condicionantes** (PPCA e FA): a verossimilhança de cada refeição usa só as gavetas observadas (`P = 1` e `M = 0`); gavetas ausentes não entram como zero. O modelo aprende o tamanho das porções dado que a gaveta está presente, como o baseline. Motivos: 89% das células são zero, e toda gaveta imputada está presente (`M ⊆ P`); com zeros, as porções imputadas sairiam puxadas para zero.

**Média condicional:** com a média de cada gaveta fixa, a porção esperada não muda com as outras gavetas presentes nem com o contexto, e na taxa 1 (nenhuma quantidade observada) PPCA e FA ficam perto da porção média; o EM também trata a ausência de uma gaveta como faltante ao acaso, e a média de gavetas ligadas a outras (açúcar e bebidas) escorrega. Com a média condicional, a média dessas gavetas bate com a observada e a cauda das gramas imputadas encolhe. Na validação, com os dois regressores, a FA condicional é a melhor das taxas 0,25 a 0,75, com o PPCA condicional logo atrás, e as duas passam da versão sem média condicional em todas as taxas; na taxa 1, os três condicionais empatam no CRPS, ou seja, sem quantidades observadas o ganho vem de usar `P` e `C`, e a correlação entre gavetas só acrescenta quando há quantidades conhecidas. O controle cobre menos na taxa 1 (92% contra 94% da FA e do PPCA condicionais), com intervalo mais estreito.

**Métricas** (por taxa > 0, nas células ocultas da validação, globais e por gaveta; `evaluation/metricas.py`): MAE em gramas e em log, NLL e CRPS em log, cobertura e largura do intervalo de 90%. Tudo sai das S amostras de `Imputer.sample`, para valer para qualquer imputador sem mudar o contrato: a NLL é a da gaussiana com a média e a variância das amostras de cada célula, e o intervalo vem dos quantis empíricos. Na taxa 0.0 não há o que imputar.

---

## 6. Parte 2 — Regressão probabilística

**Objetivo:** prever o carboidrato da refeição com intervalo a partir do registro incompleto.

**Contrato** (`src/amaprob/regression/base.py`):

```python
class ProbRegressor(Protocol):
    def fit(self, Z: np.ndarray, y: np.ndarray) -> "ProbRegressor": ...
    def predict(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Média e variância da preditiva gaussiana, cada uma de forma (N,)."""
```

`Z` = concatenação de quantidades em gramas (observadas ou imputadas), `P` e `C` (`montar_Z`). A preditiva gaussiana por refeição dá a NLL exata; na propagação, as S amostras da imputação geram a mistura `(1/S) Σ N(μ_s, σ²_s)`.

**Modelos:**

| Modelo | Escopo | Implementação |
|---|---|---|
| Regressão linear bayesiana | Obrigatório | Forma fechada conjugada em NumPy (`regression/blr.py`) |
| Processo Gaussiano Variacional Esparso (SVGP) | Obrigatório | GPyTorch (`regression/svgp.py`): média linear + kernel RBF com um comprimento por coluna, sobre entradas padronizadas; M pontos indutores aprendidos (256 de partida; 288 adotado na Parte 3, §7), 60 épocas de minibatch na ELBO, `float64` |

A média linear faz do SVGP uma "regressão linear + correção não linear", e a comparação com a regressão linear bayesiana responde se a não linearidade ajuda. Com 10 épocas, a ELBO ainda subia e o ruído ficava no dobro do da regressão linear; com 60, ela assenta.

**Experimento central** (`src/amaprob/regression/propagation.py`, `src/amaprob/experiments/propagacao.py`), com o regressor (regressão linear bayesiana e SVGP) e o imputador (porção média, PPCA e FA, cada um com e sem média condicional) como dimensões, com os hiperparâmetros de `configs/hiperparametros.json` (§7), as mesmas máscaras e geradores (`--regressores` e `--imputadores` rodam só alguns pares e juntam as linhas deles aos CSVs existentes do mesmo conjunto; `--conjunto teste` mede no teste e grava em `results/teste/`), e S = 100 amostras fixo (comparação com S = 500 em `results/propagacao_S.csv`). Os três braços usam as mesmas S amostras de `Imputer.sample`, convertidas para gramas por `expm1`:
- **Plug-in:** preenche cada quantidade oculta com a média das amostras **em gramas** e prevê uma vez. A média do log convertida (`expm1` da média) subestimaria as gramas, e o plug-in perderia por viés de conversão, não pela incerteza.
- **Plug-in recalibrado (margem fixa):** a mesma entrada, num regressor ajustado com o treino mascarado e imputado da mesma forma (ver §4). A variância de ruído absorve o erro de imputação como folga igual para toda refeição. Só com a regressão linear bayesiana. Os imputadores das 5 dobras são ajustados uma vez por imputador: sem máscara, o ajuste não depende da taxa nem da semente.
- **Propagação:** prevê para cada amostra e combina as preditivas na mistura `(1/S) Σ N(μ_s, σ²_s)`.

Hipótese: a propagação produz intervalos calibrados; o plug-in subestima a incerteza. Pergunta do artigo: **"o modelo sabe quando não sabe?"**, ou seja, se a incerteza por refeição separa as estimativas confiáveis das que pediriam mais informação, melhor que uma margem fixa. Por isso as métricas também são lidas por número de quantidades ocultas na refeição (0, 1, 2, 3, 4+), que o app conhece no uso, ao lado da referência da taxa 0 (previsão com as gramas completas, agrupada pelas mesmas classes), que separa o efeito das ocultas do efeito do tamanho da refeição.

**Leitura esperada:** com taxa 0.0, erro próximo do piso (~8 g com K = 29; teste de sanidade). Com o aumento da taxa, a largura do intervalo cresce de forma calibrada. **Essa curva é o resultado principal.**

**Métricas** (`src/amaprob/evaluation/metricas.py`): MAE, NLL, CRPS (`scoringrules`), cobertura do intervalo de 90% (PICP), largura média do intervalo (MPIW) e cobertura por quintil da previsão. Métrica clínica secundária: % de refeições com erro de até 7,5 g (meia unidade de insulina com relação 1 U : 15 g, o passo de uma caneta comum; justificativa em `docs/piso-de-erro.md`).

**Resultado da taxa 0** (validação, `experiments/taxa_zero.py`): MAE de 7,87 g, PICP de 93,1% e MPIW de 45,4 g. A cobertura vai de 97,6% nas refeições pequenas a 84,1% nas grandes (caudas pesadas; ver §11). Por isso, a curva de calibração é lida em relação à PICP da taxa 0, e não só em relação a 90%.

---

## 7. Parte 3 — Otimização bayesiana

**Alvos** (`src/amaprob/experiments/otimizacao.py`), numa grade discreta de candidatos:

| Modelo | Botões | Objetivo (NLL do carboidrato na validação) |
|---|---|---|
| PPCA e FA com média condicional | q ∈ {1, …, 15} e α (21 valores logarítmicos de 10⁻² a 10³) | Braço de propagação com a regressão linear, taxa 0,5, semente 0; ajuste em 25% do treino |
| Porção média condicional (controle) | α | Idem; buscado para a comparação com a FA condicional continuar justa |
| SVGP | M ∈ {32, 48, …, 512} | Taxa 0 (gramas completas); ajuste no treino completo |

As versões sem média condicional herdam o q da condicional, para a comparação com e sem média condicional mudar uma coisa só. A porção média não tem botão, e a regressão linear ajusta α e β do prior pela evidência, em forma fechada: a otimização bayesiana serve aos botões que mudam a estrutura do modelo e custam um ajuste inteiro por tentativa.

**Contrato:**

```python
def objective(params: dict) -> float:
    """Treina no treino, retorna a NLL na validação. Determinística dada a semente."""
```

- **Método** (`src/amaprob/tuning/busca.py`): BoTorch, com surrogate de processo gaussiano ajustado às avaliações e Expected Improvement em log calculado em todos os candidatos não avaliados.
- **Comparação:** busca aleatória com o mesmo orçamento (12 avaliações por modelo); as 4 primeiras são comuns aos dois métodos, e um candidato repetido não é recalculado.
- **α na amostra:** usado direto, sem conversão pelo número de refeições. α é a precisão de um prior, e o ótimo dela não depende de N; a confirmação mede a variante convertida como diagnóstico.
- **Escolha:** menor NLL; no SVGP, o menor M a até 0,01 da menor NLL, porque a NLL tende a cair sempre um pouco com M e o custo cresce com ele.
- **Confirmação e adoção:** a escolhida e a de partida reajustadas no treino completo, nas 3 sementes da taxa 0,5 (o SVGP já é buscado no treino completo). Um imputador só adota a escolhida se ela melhorar a NLL nas duas sementes que não entraram na escolha e a média delas em pelo menos 0,015, a oscilação da comparação entre máscaras; senão, fica a de partida.
- **Registro:** todas as avaliações (parâmetros, métricas, tempo, melhor NLL até ali), a confirmação, a curva "melhor NLL × número de avaliações" e a configuração adotada em `configs/hiperparametros.json`, lida pela rodada no teste.

**Resultado** (validação): a configuração de partida está num platô, e os hiperparâmetros pesam pouco perto da estrutura (média condicional, propagação).

| Modelo | Escolhida pela busca | Confirmação (sementes 1 e 2, não usadas na escolha) | Adotada |
|---|---|---|---|
| PPCA condicional | q = 12, α = 17,8 | NLL −0,030 e −0,012; CRPS −0,15 g nas 3 sementes; alcança a FA condicional | **q = 12, α = 17,8** |
| FA condicional | q = 6, α = 1000 (teto da grade) | NLL +0,006 e +0,015: o ganho da semente 0 era ruído da máscara | q = 5, α = 1 |
| Porção média condicional | α = 178 | NLL −0,0002; CRPS −0,006 g | α = 1 |
| SVGP (taxa 0) | M = 288 (o melhor, M = 512, fica na borda) | NLL −0,004 em relação a M = 256 | **M = 288** |

O α quase não importa: α × 4 dá a mesma NLL até a 3ª casa nos três condicionais, inclusive acima do teto da grade da FA (α = 4000), e o limite α → ∞ é a versão sem média condicional, pior. O PPCA sem média condicional herda q = 12. `experiments/propagacao.py` lê a configuração adotada de `configs/hiperparametros.json`.

**Curva de validação com a configuração adotada.** Como a escolha usou só a taxa 0,5, os dois PPCA com q = 12 são medidos em todas as taxas da validação, com os dois regressores (o SVGP com M = 288), antes da rodada no teste. Regra fixada antes da curva: o q dos dois PPCA volta a 5 se, em alguma taxa de 0,25 a 1, a NLL média das 3 sementes do PPCA condicional (propagação com a regressão linear) ficar pior que a de q = 5 por mais de 0,015; o PPCA sem média condicional segue o q do condicional, e a variação dele é só reportada. Um ajuste que pare no teto de 500 iterações do EM tem o teto como ponto de parada declarado. Nos CSVs da validação, as linhas do SVGP com os outros imputadores continuam medidas com M = 256.

**Resultado da curva:** q = 12 fica nos dois PPCA. No PPCA condicional, a configuração adotada contra a de partida (mudam q e α) piora a NLL no máximo em +0,008, na taxa 1, dentro do ruído; de 0,25 a 0,75, melhora a NLL (−0,019 a −0,025) e o CRPS (−0,11 a −0,16 g), e o intervalo fica mais estreito em todas as taxas (−0,9 a −1,7 g), com a cobertura mantida. O PPCA sem média condicional, em que só o q muda, melhora em todas as taxas. Com q = 12, o EM parou no teto em 5 dos 6 ajustes do PPCA e em 4 dos 6 do PPCA condicional (com q = 5, o condicional convergia em ~214 iterações). Ajuste dos dois PPCA com q = 12: ~52 min cada (treino completo e 5 dobras). Otimização bayesiana e busca aleatória se alternam: a bayesiana acha o melhor na FA, no controle e no SVGP, e a aleatória no PPCA; com 12 avaliações e objetivo ruidoso, nenhuma domina.

---

## 8. Estrutura do repositório

```
amaprob-projeto/
├── AGENTS.md                # instruções do projeto (este arquivo)
├── CLAUDE.md                # só importa AGENTS.md (o Claude Code lê CLAUDE.md)
├── README.md
├── pyproject.toml / uv.lock
├── configs/                 # classificação do IBGE e configurações de experimento
├── data/
│   ├── raw/                 # microdados POF (fora do git)
│   ├── interim/             # tabelas lidas (fora do git)
│   └── processed/           # X, P, C, y, person_id, máscaras, splits (fora do git)
├── docs/                    # métodos e decisões para revisão (ex.: piso-de-erro.md)
├── src/amaprob/
│   ├── data/                # leitura POF, gavetas, refeições, splits, máscaras
│   ├── imputation/          # Imputer + baseline, PPCA-EM, FA-EM
│   ├── regression/          # ProbRegressor + modelos + propagação
│   ├── tuning/              # otimização bayesiana
│   ├── evaluation/          # métricas e calibração
│   └── experiments/         # scripts de execução (gravam em results/)
├── notebooks/               # só orquestração e figuras
├── results/                 # métricas em CSV + figuras (validação); teste/ só com a rodada no teste
└── tests/                   # máscaras, splits, métricas, contratos
```

Subpacotes são criados quando o código da etapa é escrito, não antes.

**Ambiente:** `uv` com Python 3.12. Dependências entram sob demanda, discutidas antes (`uv add`). Stack prevista: NumPy, pandas, SciPy, scikit-learn, PyTorch, GPyTorch, BoTorch, PyMC/NumPyro + ArviZ, scoringrules, plotnine (figuras, com o tema comum de `evaluation/tema.py`), pytest. Para ler as planilhas do IBGE uma única vez: `uv run --with openpyxl`.

---

## 9. Diretrizes para o assistente

**Arquitetura (aplicar com pragmatismo de código de pesquisa):**
- Lógica em `src/`; notebooks apenas orquestram e plotam.
- Modelos dependem dos contratos (`Imputer`, `ProbRegressor`), nunca uns dos outros. Trocar um modelo não deve exigir mudar o experimento.
- Uma responsabilidade por módulo: leitura de dados, mascaramento, modelagem e avaliação ficam separados.
- Questione escolhas que adicionem abstração sem problema concreto. Quando houver tensão, explique o custo-benefício.

**Rigor:**
- Toda função aleatória recebe `rng` ou semente explícita.
- Testes obrigatórios para: geração de máscaras (`M ⊆ P`, taxa correta), split por indivíduo (nenhum `person_id` em dois conjuntos), métricas (casos conhecidos) e ausência de leitura de `X[M]` na imputação.
- Comece pelo caso mais simples que valide o pipeline de ponta a ponta (baseline + regressão linear bayesiana em taxa 0.0) antes de sofisticar.
- PPCA-EM e FA-EM são implementados do zero, com derivação comentada. Motivo: as versões prontas (`PCA` e `FactorAnalysis` do scikit-learn) não aceitam dados faltantes, e imputar as quantidades de `M` exige estimar o modelo com as entradas ausentes. O EM com faltantes faz exatamente isso: no passo E, condiciona cada refeição só nas gavetas observadas. Também fazem parte do aprendizado da disciplina.
- Prefira implementar a regressão linear bayesiana do zero, em forma fechada, com derivação comentada.

**Escrita:**
- Código que fala por si: nomes descritivos no lugar de comentários. Comentário ou docstring só para o porquê que o código não mostra. A exceção são os modelos (PPCA, FA, regressão linear bayesiana), que levam a derivação comentada.
- Comentários, docstrings e documentação em tom neutro e impessoal: sem nomes de pessoas, datas de decisão ou atribuições.
- Expanda siglas na primeira ocorrência em comentários e documentação.
- Sinalize suposições sobre os dados da POF como "a verificar".
- Commits só com aprovação, após mostrar o que será incluído.

---

## 10. Ordem de implementação e cronograma

| Semana | Entrega |
|---|---|
| 02–08/11 | `pytest`; máscaras persistidas + testes (máscaras, split); baseline + regressão linear bayesiana + métricas em taxa 0.0 (pipeline de ponta a ponta); Parte 1: PPCA-EM em todas as taxas |
| 09–15/11 | Parte 1: FA-EM (reaproveita o PPCA-EM); Parte 2: SVGP; plug-in vs. propagação em todas as taxas (**resultado principal**); definir o número de amostras da propagação |
| 16–22/11 | Parte 3: otimização bayesiana; resultados parciais prontos para o acompanhamento de 19/11; início do artigo |
| 23–30/11 | Tabelas, figuras, artigo, revisão e entrega (30/11); slides até 01/12 |

Depois do pipeline de ponta a ponta, as Partes 1 e 2 podem avançar em paralelo, porque os modelos dependem só dos contratos.

Escopo: só o mínimo descrito nas seções 5 a 7. Os extras (VAE e análises adicionais) ficam suspensos e só voltam à discussão se houver folga; a busca aleatória entrou na Parte 3 como comparação.

---

## 11. Riscos declarados

- **Excesso de zeros:** vetores de refeição são esparsos e modelos gaussianos ficam mal especificados. Por isso o mascaramento atua só sobre gavetas presentes e a imputação usa o log das gramas. A omissão de itens inteiros é trabalho futuro.
- **Mecanismo de ausência:** mascaramento aleatório vs. omissão real sistemática — limitação declarada.
- **Gramas estimadas, não pesadas:** as gramas da POF vêm de medidas caseiras convertidas por tabela, com porções em degraus. Os modelos aprendem, em parte, essa conversão (ver Regra 1).
- **População geral:** a POF não é uma amostra de pessoas com diabetes tipo 1 — limitação de validade externa.
- **Modelos sofisticados podem não superar o baseline** em precisão; o foco do trabalho é a calibração da incerteza.
- **Escala:** subamostrar quando necessário; treino em CPU. Amostras da imputação ocupam S × N × K valores (100 amostras de toda a validação ≈ 1 GB): amostrar e prever em blocos de refeições.
- **Caudas pesadas no erro da regressão:** o resíduo na taxa 0 tem curtose em excesso de ~30, e o ruído gaussiano homocedástico cobre demais as refeições pequenas e de menos as grandes. Mitigação: cobertura por faixa reportada e leitura relativa à taxa 0. O modelo de ruído alternativo está em `backlog/decisions.md`.

---

## 12. Decisões em aberto

Nenhuma no momento.
