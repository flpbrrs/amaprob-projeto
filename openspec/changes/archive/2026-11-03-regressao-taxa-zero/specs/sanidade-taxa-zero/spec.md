# Spec Delta

## Purpose

Executar o pipeline de ponta a ponta na taxa 0 (nenhuma quantidade oculta), como teste de sanidade: o erro da regressão deve ficar perto do piso medido com as gavetas adotadas, e as métricas e os tempos de ajuste ficam registrados em disco.

## ADDED Requirements

### Requirement: Ajuste no treino, medida na validação
O experimento SHALL ajustar cada modelo só nas refeições de treino e medir só nas de validação. Ele MUST NOT ler as refeições de teste (Regra 6).

#### Scenario: Conjuntos usados
- **WHEN** o experimento roda
- **THEN** nenhuma linha com conjunto `teste` entra em ajuste nem em métrica

### Requirement: Resultado da taxa 0
Na taxa 0, a regressão SHALL receber `Z` com todas as gramas observadas. O experimento SHALL gravar em `results/` as métricas (MAE, NLL, CRPS, PICP, MPIW e percentual de meia dose) e a cobertura por faixa da previsão, para a regressão linear bayesiana e para a referência de mínimos quadrados com resíduo gaussiano (o piso).

#### Scenario: Perto do piso
- **WHEN** o experimento roda com o dataset da versão 20230713
- **THEN** o MAE da regressão linear bayesiana na validação fica no máximo 0,5 g acima do piso de 8,1 g medido com K = 29

#### Scenario: Arquivos gravados
- **WHEN** o experimento termina
- **THEN** `results/taxa_zero_metricas.csv` (uma linha por modelo) e `results/taxa_zero_cobertura.csv` (uma linha por modelo e faixa) existem

### Requirement: Tempo de ajuste
O experimento SHALL medir e gravar o tempo de parede do ajuste da regressão linear bayesiana no treino e, para o baseline, do ajuste e da geração de 100 amostras, sobre a máscara de taxa 0,5 e semente 0. O tempo do baseline é só medição de custo: suas amostras não entram em métrica.

#### Scenario: Tempos registrados
- **WHEN** o experimento termina
- **THEN** o tempo de cada modelo, em segundos, aparece na saída e em `results/tempos_ajuste.csv`
