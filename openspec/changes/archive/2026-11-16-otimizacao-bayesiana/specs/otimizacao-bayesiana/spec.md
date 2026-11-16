# Spec Delta

## ADDED Requirements

### Requirement: Objetivo avaliado só na validação
Cada avaliação SHALL ajustar o modelo com os hiperparâmetros candidatos no conjunto de ajuste do treino e devolver uma NLL medida na validação. Para os imputadores, a NLL SHALL ser a do carboidrato no braço de propagação com a regressão linear bayesiana, na taxa 0,5 e semente 0. Para o SVGP, SHALL ser a NLL do carboidrato na taxa 0. A busca MUST NOT ler as refeições de teste (Regra 6).

#### Scenario: Conjuntos usados
- **WHEN** uma busca roda
- **THEN** nenhuma refeição de teste entra em ajuste nem em métrica

#### Scenario: Determinística
- **WHEN** o mesmo candidato é avaliado duas vezes com a mesma semente
- **THEN** a NLL devolvida é idêntica

### Requirement: Candidatos discretos fixados antes da busca
Cada modelo SHALL ter um conjunto finito de candidatos fixado antes da busca: q de 1 a 15 e α numa grade logarítmica para os imputadores condicionais, α na mesma grade para o controle e M numa grade de 32 a 512 para o SVGP. Os dois métodos SHALL escolher só candidatos desse conjunto, e um método MUST NOT avaliar duas vezes o mesmo candidato.

#### Scenario: Sem repetição
- **WHEN** um método termina o orçamento
- **THEN** os candidatos avaliados por ele são todos distintos e pertencem ao conjunto do modelo

### Requirement: Otimização bayesiana e busca aleatória com o mesmo orçamento
Para cada modelo, a busca aleatória SHALL sortear candidatos sem reposição, com semente fixa. A otimização bayesiana SHALL começar pelos mesmos primeiros candidatos da busca aleatória e, a partir deles, escolher cada candidato seguinte pelo máximo de uma função de aquisição sobre um surrogate de processo gaussiano ajustado às avaliações feitas. Os dois métodos SHALL ter o mesmo número de avaliações.

#### Scenario: Início comum
- **WHEN** as duas buscas de um modelo terminam
- **THEN** as primeiras avaliações dos dois métodos têm os mesmos candidatos e as mesmas NLL

#### Scenario: Aproxima o mínimo de uma função conhecida
- **WHEN** a otimização bayesiana roda com 12 avaliações sobre uma tigela quadrática de duas variáveis numa grade pequena, com semente fixa
- **THEN** o melhor candidato encontrado fica a no máximo um passo da grade do mínimo, e a melhor NLL dela não é pior que a da busca aleatória com a mesma semente

#### Scenario: Reprodutível pela semente
- **WHEN** a mesma busca roda duas vezes com a mesma semente e o mesmo objetivo
- **THEN** a sequência de candidatos é idêntica

### Requirement: Avaliação repetida entre métodos não é recalculada
Quando um candidato já avaliado por um método é escolhido pelo outro, a NLL SHALL ser reaproveitada sem novo ajuste, e o registro SHALL marcar a avaliação como reaproveitada, com o tempo da avaliação original.

#### Scenario: Reaproveitamento
- **WHEN** os dois métodos escolhem o mesmo candidato
- **THEN** o objetivo é calculado uma única vez para ele

### Requirement: Amostra estratificada do treino para os imputadores
As avaliações dos imputadores SHALL ajustar o modelo numa amostra das refeições do treino, estratificada pela ocasião, com fração e semente fixas e a mesma amostra para todas as avaliações. O SVGP SHALL ser ajustado no treino completo.

#### Scenario: Proporção por ocasião
- **WHEN** a amostra é sorteada
- **THEN** contém só refeições do treino, e a proporção de cada ocasião difere da do treino em menos de 1 ponto percentual

#### Scenario: Mesma amostra
- **WHEN** a amostra é sorteada duas vezes com a mesma fração e semente
- **THEN** as refeições são as mesmas

### Requirement: Registro das avaliações e curva de convergência
Cada avaliação SHALL ser gravada com o modelo, o método, o número da avaliação, os hiperparâmetros, a NLL, o tempo, a marca de reaproveitada e a melhor NLL até ali. A configuração de partida SHALL ser avaliada com o mesmo objetivo, como referência. SHALL ser gerada a figura da melhor NLL pelo número de avaliações, com um painel por modelo, uma linha por método e a referência de partida.

#### Scenario: Melhor até aqui
- **WHEN** o registro de um método é lido em ordem
- **THEN** a melhor NLL até ali nunca aumenta

### Requirement: Confirmação no treino completo
Para cada imputador, a configuração escolhida e a de partida SHALL ser reajustadas no treino completo e medidas na validação na taxa 0,5, nas 3 sementes de máscara, com NLL, CRPS, PICP e MPIW. Para o SVGP, cuja busca já usa o treino completo e a taxa 0, a confirmação SHALL vir das avaliações de partida e escolhida. O resultado SHALL ser gravado por modelo, configuração e semente.

#### Scenario: Comparação gravada
- **WHEN** a confirmação termina
- **THEN** cada imputador tem linhas da configuração de partida e da escolhida nas 3 sementes, e o SVGP tem as duas configurações, todas com as quatro métricas

#### Scenario: Partida igual ao experimento de propagação
- **WHEN** a configuração de partida de um imputador é confirmada
- **THEN** as métricas são iguais às do braço de propagação com a regressão linear, na taxa 0,5, gravadas pelo experimento de propagação nas mesmas sementes

### Requirement: Escolha do M pelo menor dentro da tolerância
No SVGP, a configuração escolhida SHALL ser o menor M entre os avaliados (a partida e os dois métodos) cuja NLL fica a até 0,01 da menor NLL avaliada. Nos imputadores, SHALL ser a de menor NLL entre as mesmas avaliações.

#### Scenario: M menor dentro da tolerância
- **WHEN** M = 512 tem a menor NLL e M = 320 fica 0,006 acima dela, sem M menor dentro de 0,01
- **THEN** o M escolhido é 320

### Requirement: Diagnóstico da transferência do α
Nos imputadores condicionais, a confirmação SHALL incluir também a configuração escolhida com α multiplicado pela razão entre as refeições do treino e as da amostra (o mesmo freio relativo aos dados que na amostra), medida como as demais. O α escolhido SHALL continuar o do candidato, sem conversão.

#### Scenario: Variante gravada
- **WHEN** a confirmação de um imputador condicional termina
- **THEN** há linhas da escolhida com α direto e com α convertido, nas 3 sementes, e o arquivo de configuração traz o α direto

### Requirement: Adoção só com melhora nas máscaras novas
Um imputador SHALL adotar a configuração escolhida só se ela melhorar a NLL da confirmação em cada semente que não entrou na escolha (1 e 2) e a média dessas sementes em pelo menos 0,015, a oscilação entre máscaras medida; senão, SHALL adotar a de partida. O SVGP SHALL adotar a escolhida, porque a busca dele não usa máscara e a de partida concorre na regra do menor M.

#### Scenario: Melhora só na semente da escolha
- **WHEN** a escolhida melhora a NLL só na semente 0 e piora nas sementes 1 e 2
- **THEN** o imputador adota a configuração de partida

#### Scenario: Melhora firme nas máscaras novas
- **WHEN** a escolhida melhora a NLL nas sementes 1 e 2, com média de pelo menos 0,015
- **THEN** o imputador adota a escolhida

### Requirement: Configuração adotada gravada
A configuração adotada de cada modelo SHALL ser gravada num arquivo de configuração, derivado da confirmação. Para o PPCA e a FA sem média condicional, o arquivo SHALL registrar o q adotado na versão condicional correspondente.

#### Scenario: Herança do q
- **WHEN** a configuração é gravada
- **THEN** o q do PPCA e o da FA sem média condicional são iguais aos adotados nas versões condicionais
