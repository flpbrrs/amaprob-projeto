# CLAUDE.md — Projeto Final de Aprendizagem de Máquina Probabilística (2026.2)

Este arquivo orienta o Claude Code na implementação do projeto. Leia-o inteiro antes de propor ou escrever código.
Fonte da proposta: Claude Doc "Proposta de Projeto Final — Aprendizagem de Máquina Probabilística".

---

## 1. Contexto

**Título:** Estimação probabilística do carboidrato de refeições brasileiras a partir de registros alimentares incompletos.

No cuidado em diabetes tipo 1, o carboidrato da refeição orienta a dose de insulina, mas o registro do paciente raramente é completo: ele informa quais alimentos comeu e, às vezes, a quantidade de alguns. Com registro completo, o carboidrato é calculável por tabela. Com informação faltando, é preciso inferir o que falta e quantificar a confiabilidade da estimativa. **O projeto investiga esse segundo regime.**

A identificação dos alimentos (que na pesquisa de mestrado vem de imagens) está abstraída: a lista de alimentos vem do dataset.

**Entregáveis da disciplina:** artigo (Introdução, Fundamentação, Metodologia, Experimentos, Conclusão), código (notebooks e/ou repositório) e apresentação. Entrega final: 30/11/2026. Exigências: dados reais, mais de um modelo por parte, comparação de desempenho e pelo menos um algoritmo de fora da ementa.

---

## 2. Dados

**Fonte:** microdados de consumo alimentar individual da POF 2017–2018 (Pesquisa de Orçamentos Familiares, IBGE). Dados reais, dois dias de registro por pessoa.

**Unidade de análise:** a refeição, ou seja, os itens da mesma pessoa, no mesmo dia e na mesma ocasião.

**Representação** (para N refeições e K grupos alimentares):

| Objeto | Forma | Conteúdo |
|---|---|---|
| `X` | N × K | `log1p(gramas)` por grupo; 0 quando o grupo está ausente |
| `P` | N × K | presença do grupo (0/1) — **sempre observada** |
| `M` | N × K | máscara de quantidades ocultas; `M ⊆ P` |
| `C` | N × d | contexto: ocasião, horário, região, sexo, idade, renda |
| `y` | N | carboidrato total da refeição (g) |
| `person_id` | N | identificador do indivíduo (hierarquia e split) |

**A verificar antes de codificar** (não assumir):
- layout de largura fixa e dicionário de variáveis do IBGE para o arquivo de consumo alimentar;
- como a ocasião/horário da refeição está codificada;
- nome da coluna de carboidrato e tabela de composição usada pelo IBGE;
- classificação de grupos alimentares a adotar (definir K entre ~15 e 30).

---

## 3. Regras invioláveis

Violações destas regras invalidam o experimento. Sinalize imediatamente qualquer código que as quebre.

1. **A medida caseira nunca é input.** As gramas da POF vêm de conversão da medida caseira; usá-la torna o problema determinístico.
2. **Quantidades mascaradas nunca vazam.** Nenhuma etapa de imputação ou regressão pode ler `X[M]`. Elas só são usadas para avaliação.
3. **Split por indivíduo.** Os dois dias de uma pessoa ficam no mesmo conjunto (treino, validação ou teste). Split por refeição causa vazamento.
4. **Máscaras geradas antes da modelagem**, com semente fixa, e persistidas em disco. Todos os modelos usam as mesmas máscaras.
5. **Dados reais como base.** O mascaramento é aplicado sobre registros reais; nenhum dado sintético entra como dado principal.
6. **Nada é ajustado no teste.** Otimização bayesiana e escolhas de modelo usam apenas a validação.

---

## 4. Protocolo experimental

- **Taxas de mascaramento:** `{0.0, 0.25, 0.5, 0.75, 1.0}` sobre as quantidades de itens **presentes**. A taxa 1.0 é o cenário "sei o que comi, não sei quanto".
- **Mecanismo:** aleatório (MCAR), declarado como limitação.
- **Repetições:** pelo menos 5 sementes por taxa; reportar média e desvio.
- **Split:** treino / validação / teste por `person_id` (ex.: 70/15/15).
- **Escala:** subamostrar (uma região ou amostra estratificada) se o volume inviabilizar o Processo Gaussiano ou o VAE.

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

| Modelo | Papel | Implementação sugerida |
|---|---|---|
| Porção padrão / média por grupo | Baseline determinístico (variância zero ou empírica) | NumPy |
| PPCA via EM com dados faltantes | Ementa | NumPy, do zero (o scikit-learn não trata faltantes) |
| VAE com máscara | Ementa | PyTorch; entrada = `X` observado + `P` + `M` + `C` |
| Análise Fatorial ou GPLVM | Fora da ementa | FA via EM em NumPy, ou Bayesian GPLVM no GPyTorch |

**Métricas** (por taxa): erro de imputação (MAE em gramas e em log), log-verossimilhança nos valores mascarados, cobertura do intervalo de 90%.

**Decisão a registrar:** grupos ausentes entram como zeros observados no PPCA (má especificação conhecida) ou como condicionantes. Documentar a escolha.

---

## 6. Parte 2 — Regressão probabilística

**Objetivo:** prever o carboidrato da refeição com intervalo a partir do registro incompleto.

**Contrato** (`src/amaprob/regression/base.py`):

```python
class ProbRegressor(Protocol):
    def fit(self, Z: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> "ProbRegressor": ...
    def predict_samples(self, Z: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Retorna (n_samples, N) da preditiva."""
```

`Z` = concatenação de quantidades (observadas ou imputadas), `P` e `C`.

**Modelos:**

| Modelo | Implementação sugerida |
|---|---|
| Regressão linear bayesiana | Forma fechada conjugada em NumPy |
| Regressão bayesiana hierárquica (efeito por indivíduo) | PyMC ou NumPyro |
| Processo Gaussiano esparso (SVGP) | GPyTorch |

**Experimento central** (`src/amaprob/regression/propagation.py`):
- **Plug-in:** imputa com a média da Parte 1 e prevê uma vez.
- **Propagação:** para cada amostra `s` de `Imputer.sample`, prevê; combina as amostras preditivas como mistura.

Hipótese: a propagação produz intervalos calibrados; o plug-in subestima a incerteza.

**Leitura esperada:** com taxa 0.0, erro baixo (teste de sanidade). Com o aumento da taxa, a largura do intervalo cresce de forma calibrada. **Essa curva é o resultado principal.**

**Métricas:** MAE, NLL, CRPS (via `properscoring` sobre amostras), cobertura do intervalo de 90% (PICP), largura média do intervalo (MPIW).

---

## 7. Parte 3 — Otimização bayesiana

**Alvos:** dimensão latente e arquitetura do VAE; kernel e número de pontos indutores do SVGP; escalas das priors da hierarquia.

**Contrato:**

```python
def objective(params: dict) -> float:
    """Treina no treino, retorna a NLL na validação. Determinística dada a semente."""
```

- **Método:** surrogate de Processo Gaussiano (BoTorch, ou `gp_minimize` do scikit-optimize).
- **Comparação obrigatória:** busca aleatória com o mesmo orçamento de avaliações.
- **Registro:** salvar todas as avaliações (parâmetros, NLL, tempo) para gerar a curva "melhor NLL × número de avaliações".

---

## 8. Estrutura do repositório

```
amaprob-projeto/
├── CLAUDE.md
├── pyproject.toml
├── configs/                 # experimentos em YAML (taxas, sementes, splits, hiperparâmetros)
├── data/
│   ├── raw/                 # microdados POF (fora do git)
│   ├── interim/             # tabelas lidas
│   └── processed/           # X, P, C, y, person_id, máscaras, splits (parquet/npz)
├── src/amaprob/
│   ├── domain/              # conceitos: refeição, grupo alimentar, especificação de máscara
│   ├── data/                # leitura POF, montagem de refeições, grupos, splits, máscaras
│   ├── imputation/          # Imputer + baselines, PPCA-EM, FA/GPLVM, VAE
│   ├── regression/          # ProbRegressor + modelos + propagação
│   ├── tuning/              # otimização bayesiana, busca aleatória, espaços de busca
│   ├── evaluation/          # métricas e calibração
│   └── experiments/         # scripts de execução por parte (lêem configs, gravam results/)
├── notebooks/               # só orquestração e figuras: 01_eda, 02_parte1, 03_parte2, 04_parte3, 05_figuras
├── results/                 # métricas por (modelo, taxa, semente) em parquet/CSV + figuras
└── tests/                   # máscaras, splits, métricas, contratos
```

**Stack:** Python 3.11+, NumPy, pandas, SciPy, scikit-learn, PyTorch, GPyTorch, BoTorch (ou scikit-optimize), PyMC/NumPyro + ArviZ, properscoring, matplotlib. Pergunte antes de adicionar dependências fora desta lista.

---

## 9. Diretrizes para o assistente

**Arquitetura (aplicar com pragmatismo de código de pesquisa):**
- Lógica em `src/`; notebooks apenas orquestram e plotam.
- Modelos dependem dos contratos (`Imputer`, `ProbRegressor`), nunca uns dos outros. Trocar um modelo não deve exigir mudar o experimento (DIP, OCP).
- `domain/` não importa PyTorch, GPyTorch ou PyMC.
- Uma responsabilidade por módulo: leitura de dados, mascaramento, modelagem e avaliação ficam separados.
- Questione ativamente escolhas que violem DDD, Clean Architecture, SOLID ou Object Calisthenics, mas sem impor abstração que não se pague num projeto de sete semanas. Quando houver tensão, explique o custo-benefício.

**Rigor:**
- Toda função aleatória recebe `rng` ou semente explícita.
- Testes obrigatórios para: geração de máscaras (`M ⊆ P`, taxa correta), split por indivíduo (nenhum `person_id` em dois conjuntos), métricas (casos conhecidos) e ausência de leitura de `X[M]` na imputação.
- Ao implementar um modelo, comece pelo caso mais simples que valide o pipeline de ponta a ponta (baseline + regressão linear bayesiana em taxa 0.0) antes de sofisticar.
- Prefira implementar PPCA-EM e a regressão linear bayesiana do zero, com derivação comentada: fazem parte do aprendizado da disciplina.

**Comunicação:**
- Expanda siglas na primeira ocorrência em comentários e documentação.
- Sinalize suposições sobre os dados da POF como "a verificar".

---

## 10. Ordem de implementação

1. Leitura da POF, montagem de refeições, grupos alimentares e análise exploratória.
2. Splits por indivíduo e máscaras persistidas, com testes.
3. Baselines + regressão linear bayesiana em taxa 0.0 (pipeline de ponta a ponta).
4. Parte 1: PPCA-EM → VAE → FA/GPLVM.
5. Parte 2: hierárquica → SVGP → plug-in vs. propagação em todas as taxas.
6. Parte 3: otimização bayesiana vs. busca aleatória.
7. Tabelas e figuras finais para o artigo.

---

## 11. Riscos declarados

- **Excesso de zeros:** vetores de refeição são esparsos e modelos gaussianos ficam mal especificados. Por isso o mascaramento atua só sobre itens presentes e usa-se o log das gramas. A omissão de itens inteiros é trabalho futuro.
- **Mecanismo de ausência:** mascaramento aleatório vs. omissão real sistemática — limitação declarada.
- **Circularidade:** gramas derivadas da medida caseira (ver Regra 1).
- **Escala:** subamostrar quando necessário.
