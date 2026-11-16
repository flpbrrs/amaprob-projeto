# Design

## Context

O experimento de propagação (`experiments/propagacao.py`) ajusta no treino e mede na validação, com q = 5, α = 1 e M = 256 fixos no módulo. A Parte 3 gravou a configuração adotada em `configs/hiperparametros.json`: PPCA condicional q = 12, α = 17,8; PPCA q = 12 (herdado); FA e FA condicional q = 5; porção média condicional α = 1; SVGP M = 288. A porção média não tem entrada.

Os CSVs são gravados só no fim da execução, e a junção parcial (`juntar`) lê e grava em `results/`. `experiments/otimizacao.py` importa `treino_e_validacao`, `gerador`, `Conjunto`, `S` e `FLUXO_VALIDACAO` deste módulo. A seleção dos índices vem de `taxa_zero.separar`, cujo contrato é nunca devolver o teste.

Tempos medidos que orientam o plano: ajuste do PPCA condicional com q = 12 no treino completo em 660 s (99 s com q = 5); PPCA com q = 5 em 194 s no treino completo e 791 s nas 5 dobras, com duas dobras no teto de 500 iterações; SVGP com M = 256 em 232 s; ~80 s por imputador e rodada com a regressão linear e ~56 s a mais com o SVGP.

## Goals / Non-Goals

**Goals:**
- Deixar a rodada no teste pronta para ser executada, sem mais escolhas.
- Conferir a configuração adotada do PPCA em todas as taxas da validação, com uma regra fixada antes.
- Fazer com que uma falha na rodada no teste custe só uma parte dela.

**Non-Goals:**
- Rodar o teste: isso é feito depois desta change, com os comandos da seção "Rodada no teste".
- Salvar imputadores ajustados para reaproveitar entre execuções. Com a rodada num dia só, isso não compensa.
- Refazer a curva de validação dos imputadores que não mudaram. As linhas do SVGP com esses imputadores ficam com M = 256.
- Mudar o teto de iterações do EM.

## Decisions

### 1. Conjunto escolhido por `--conjunto {validacao, teste}`

O padrão é `validacao`, e assim nada muda para quem já roda o experimento. A seleção do teste fica numa função nova do próprio `propagacao.py`; `taxa_zero.separar` e `treino_e_validacao` não mudam, porque `otimizacao.py` depende deles e o contrato "nunca devolve o teste" continua valendo ali. A medida só troca o recorte `va` pelo do teste: ajuste dos regressores, dos imputadores e das dobras continua só com `tr`.

*Alternativa descartada:* um script separado para o teste. Duplicaria o laço de medida, e uma diferença entre os dois invalidaria a comparação entre validação e teste.

### 2. Pasta de saída por conjunto

`results/` para a validação e `results/teste/` para o teste, com os mesmos nomes de arquivo e uma subpasta `figuras/` em cada um. A pasta é decidida uma vez, em `main`, e passada para a junção e para a gravação. Assim, uma execução parcial no teste junta só com os arquivos do teste. A legenda das figuras e a linha de abertura do log nomeiam o conjunto.

### 3. Configuração lida de `configs/hiperparametros.json`

O arquivo é lido uma vez, e as fábricas dos imputadores e dos regressores são montadas a partir dele. Um modelo sem entrada, ou uma entrada sem um dos botões, usa o valor de partida do construtor. Os rótulos de `NOMES_DOS_IMPUTADORES` e `NOMES_DOS_REGRESSORES` passam a mostrar o valor de cada modelo ("PPCA cond. (q = 12)", "FA (q = 5)", "SVGP (M = 288)"). O caminho do arquivo é o mesmo que `otimizacao.py` grava.

*Alternativa descartada:* passar q e M pela linha de comando. Abriria espaço para a rodada no teste usar algo diferente do que a validação escolheu.

### 4. Iterações no log e no registro de ajuste

`imputadores_ajuste.csv` já tem a coluna `iteracoes` por dobra. Ela ganha a coluna `teto_iteracoes`, e a linha do log de cada imputador passa a mostrar as iterações do treino completo e a maior entre as dobras. Isso dispensa um ajuste cronometrado à parte: a fase de ajuste da curva de validação já mede tempo e iterações. **Decisão de Felipe:** uma parada no teto é aceita como ponto de parada e declarada.

### 5. Curva de validação do PPCA com q = 12

`--imputadores ppca ppca_condicional`, com os dois regressores. O PPCA vem primeiro, porque é o único sem tempo medido com q = 12. Estimativa: ajustes de ~1 h 40 a ~2 h 45 (o PPCA domina a incerteza), SVGP com M = 288 em ~5 min e curva com ~1 h 10, num total de ~3 h a ~4 h. **Ponto de parada:** se o ajuste do PPCA no treino completo passar de 30 min, a execução é interrompida e o custo volta para a decisão.

### 6. Regra da curva (fixada antes de ver os resultados)

**Decisão de Felipe:** para cada PPCA, compara-se a NLL média das 3 sementes, no braço de propagação com a regressão linear, contra as linhas de q = 5 já gravadas (`results/propagacao_metricas.csv` do `HEAD` antes da execução), nas taxas 0,25 a 1. Se ela piorar mais de 0,015 em alguma taxa, o q volta a 5. A taxa 0 fica de fora, porque nela os braços não dependem do imputador.

**Como a regra se aplica ao par (sugestão do Claude, a confirmar):** o q é escolhido pelo PPCA condicional, e o PPCA sem média condicional herda esse q, para a comparação entre os dois mudar uma coisa só. Por isso, a regra decide pelo PPCA condicional e vale para os dois. A piora do PPCA sem média condicional é reportada, mas não troca o q sozinha.

**Se o q voltar a 5:** `configs/hiperparametros.json` passa a ter q = 5 nos dois PPCA, e as linhas deles nos CSVs da validação voltam às do `HEAD`, sem reexecução. Essas linhas foram medidas com q = 5 e, no SVGP, com M = 256. A troca fica declarada no design e no `AGENTS.md`.

A comparação é feita num trecho curto de pandas dentro da tarefa, e não num módulo novo, porque roda uma vez só.

### 7. Conferência do teste sem ler métricas

Antes da rodada, carrega-se o recorte do teste e imprimem-se só o número de refeições e de pessoas, a ausência de pessoas em comum com o treino e a validação, `M ⊆ P` e a taxa efetiva de ocultas por taxa e semente. Nem `y` nem previsão nenhuma é lida, e assim o caminho é testado sem ver resultado do teste (Regra 6).

### 8. Rodada no teste em 3 chamadas

**Decisão de Felipe.** Os comandos, na ordem:

```
uv run python -m amaprob.experiments.propagacao --conjunto teste --imputadores porcao_media porcao_condicional fa fa_condicional
uv run python -m amaprob.experiments.propagacao --conjunto teste --imputadores ppca
uv run python -m amaprob.experiments.propagacao --conjunto teste --imputadores ppca_condicional
```

A primeira cria os arquivos de `results/teste/`, e as outras juntam a eles. Se uma falhar, só ela é refeita. O SVGP é reajustado em cada chamada (~5 min cada); com semente fixa, o ajuste dá o mesmo resultado, o que a junção já pressupõe (cenário "Mesmo resultado que a execução completa"). Estimativa: ~2 h 35, ~1 h 25 a ~2 h 30 e ~1 h 35, num total de ~5 h 35 a ~6 h 40, refeito depois da curva de validação com os tempos medidos.

## Risks / Trade-offs

- [O modo teste grava fora de `results/teste/`] → teste automático que roda o caminho de gravação numa pasta temporária e confere que só ela muda; conferência de `git status` depois da primeira chamada.
- [A curva com q = 12 piora numa taxa alta perto do limite de 0,015] → a regra é fixa e numérica; o resultado é reportado como está, sem segunda rodada.
- [Rodada longa interrompida pelo Windows (suspensão, atualização)] → desativar a suspensão antes; as 3 chamadas limitam a perda.
- [Linhas do SVGP na validação com M diferente entre imputadores] → declarado no `AGENTS.md` §6 e no artigo; os números do artigo vêm do teste, onde todos usam M = 288.
- [Amostragem com q = 12 mais lenta que os ~80 s por rodada medidos com q = 5] → o log da curva mede o tempo por rodada, e a estimativa do teste é refeita a partir dele.

## Migration Plan

Sem migração: o padrão do `--conjunto` mantém o comportamento atual, com exceção dos valores dos botões, que agora vêm do arquivo. Para voltar atrás, basta o `git revert` dos commits da change e a restauração dos CSVs da validação do `HEAD` anterior.
