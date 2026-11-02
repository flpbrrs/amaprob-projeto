# Spec Delta

## Purpose

Atribuir cada linha (em geral, uma refeição) a treino, validação ou teste sorteando pessoas, de modo que os dois dias de registro de um indivíduo nunca fiquem em conjuntos diferentes.

## ADDED Requirements

### Requirement: Rótulo por linha, sorteado por pessoa
A divisão SHALL receber o `person_id` de cada linha e devolver, alinhado a essas linhas, um rótulo entre `treino`, `validacao` e `teste`. Todas as linhas de um mesmo `person_id` SHALL receber o mesmo rótulo.

#### Scenario: Pessoa com várias refeições
- **WHEN** uma pessoa tem refeições nos dois dias de registro
- **THEN** todas essas refeições recebem o mesmo rótulo

#### Scenario: Nenhuma pessoa em dois conjuntos
- **WHEN** a divisão termina
- **THEN** cada `person_id` aparece em exatamente um dos três conjuntos

### Requirement: Frações aplicadas sobre pessoas
As frações SHALL valer sobre o número de pessoas distintas, não de linhas, com padrão 70/15/15. O treino SHALL receber `round(f_treino × n_pessoas)` pessoas, a validação `round(f_validacao × n_pessoas)` e o teste as restantes.

#### Scenario: Contagem na versão 20230713
- **WHEN** a divisão recebe as refeições da versão 20230713 (46.141 pessoas) com as frações padrão
- **THEN** o treino recebe 32.299 pessoas, a validação 6.921 e o teste 6.921

#### Scenario: Proporção de refeições não é garantida
- **WHEN** as pessoas têm números diferentes de refeições
- **THEN** a proporção de linhas em cada conjunto fica apenas próxima das frações (na versão 20230713: 69,99% / 14,98% / 15,03% das refeições)

### Requirement: Reprodutível pela semente
A atribuição SHALL depender só da semente (padrão 0) e do conjunto de pessoas distintas recebido. Ela não depende da ordem das linhas nem do número de linhas por pessoa. Se o conjunto de pessoas mudar (por exemplo, com outro filtro de refeições), nada garante que uma pessoa mantenha o rótulo.

#### Scenario: Mesma semente, ordem diferente
- **WHEN** as mesmas linhas são passadas embaralhadas, com a mesma semente
- **THEN** cada pessoa recebe o mesmo rótulo da primeira execução
