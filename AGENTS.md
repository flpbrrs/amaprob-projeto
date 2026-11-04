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

**A verificar:**
- `QUADRO` 72 e 73 = dia 1 e dia 2 de registro;
- regra usada pelo IBGE para estimar os gramas do açúcar adicionado (146 valores distintos, concentrados em 5, 20 e 30 g).

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
- **Conjuntos mascarados:** treino, validação e teste, com o mesmo protocolo.
- **Mecanismo:** aleatório (MCAR), declarado como limitação. O mascaramento é por gaveta (esconde todos os itens dela de uma vez), também declarado.
- **Repetições:** 3 sementes por taxa; reportar média e desvio.
- **Split:** treino / validação / teste = 70/15/15 por `person_id`, semente 0 (`src/amaprob/data/splits.py`).
- **Escala:** amostra estratificada de refeições para os laços pesados (SVGP, otimização bayesiana). Treino em CPU.

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
| PPCA (Análise de Componentes Principais Probabilística) via EM com dados faltantes | Obrigatório | NumPy, do zero (o scikit-learn não trata faltantes) |
| FA (Análise Fatorial) via EM com dados faltantes | Obrigatório | NumPy; difere do PPCA só no ruído (diagonal em vez de isotrópico), reaproveita o código |
| VAE (Autoencoder Variacional) com máscara | Extra | PyTorch; entrada = `X` observado + `P` + `M` + `C` |

**Métricas** (por taxa > 0): erro de imputação (MAE em gramas e em log), log-verossimilhança nos valores mascarados, cobertura do intervalo de 90%. Na taxa 0.0 não há o que imputar.

**Decisão em aberto:** gavetas ausentes como zeros observados no PPCA (má especificação conhecida) ou como condicionantes. Proposta: condicionantes.

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

| Modelo | Escopo | Implementação sugerida |
|---|---|---|
| Regressão linear bayesiana | Obrigatório | Forma fechada conjugada em NumPy |
| Processo Gaussiano esparso (SVGP) | Obrigatório | GPyTorch |

**Experimento central** (`src/amaprob/regression/propagation.py`):
- **Plug-in:** imputa com a média da Parte 1 e prevê uma vez.
- **Propagação:** para cada amostra `s` de `Imputer.sample`, prevê; combina as amostras preditivas como mistura.

Hipótese: a propagação produz intervalos calibrados; o plug-in subestima a incerteza.

**Leitura esperada:** com taxa 0.0, erro próximo do piso (~8 g com K = 29; teste de sanidade). Com o aumento da taxa, a largura do intervalo cresce de forma calibrada. **Essa curva é o resultado principal.**

**Métricas** (`src/amaprob/evaluation/metricas.py`): MAE, NLL, CRPS (`scoringrules`), cobertura do intervalo de 90% (PICP), largura média do intervalo (MPIW) e cobertura por quintil da previsão. Métrica clínica secundária: % de refeições com erro de até 7,5 g (meia unidade de insulina com relação 1 U : 15 g; fonte a verificar).

**Resultado da taxa 0** (validação, `experiments/taxa_zero.py`): MAE de 7,87 g, PICP de 93,1% e MPIW de 45,4 g. A cobertura vai de 97,6% nas refeições pequenas a 84,1% nas grandes (caudas pesadas; ver §11). Por isso, a curva de calibração é lida em relação à PICP da taxa 0, e não só em relação a 90%.

---

## 7. Parte 3 — Otimização bayesiana

**Alvos:** dimensão latente do PPCA/FA; número de pontos indutores do SVGP.

**Contrato:**

```python
def objective(params: dict) -> float:
    """Treina no treino, retorna a NLL na validação. Determinística dada a semente."""
```

- **Método:** surrogate de Processo Gaussiano (BoTorch, ou `gp_minimize` do scikit-optimize).
- **Extra:** busca aleatória com o mesmo orçamento de avaliações, para comparação.
- **Registro:** salvar todas as avaliações (parâmetros, NLL, tempo) para gerar a curva "melhor NLL × número de avaliações".

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
├── results/                 # métricas em CSV + figuras
└── tests/                   # máscaras, splits, métricas, contratos
```

Subpacotes são criados quando o código da etapa é escrito, não antes.

**Ambiente:** `uv` com Python 3.12. Dependências entram sob demanda, discutidas antes (`uv add`). Stack prevista: NumPy, pandas, SciPy, scikit-learn, PyTorch, GPyTorch, BoTorch (ou scikit-optimize), PyMC/NumPyro + ArviZ, properscoring, matplotlib, pytest. Para ler as planilhas do IBGE uma única vez: `uv run --with openpyxl`.

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

Escopo: só o mínimo descrito nas seções 5 a 7. Os extras (VAE, busca aleatória e análises adicionais) ficam suspensos e só voltam à discussão se houver folga.

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

- Gavetas ausentes no PPCA/FA: zeros observados ou condicionantes (proposta: condicionantes).
