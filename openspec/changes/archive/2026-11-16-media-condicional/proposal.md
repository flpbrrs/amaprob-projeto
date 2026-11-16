# Proposal

## Why

No PPCA e na FA, a média de cada gaveta é única: o tamanho esperado da porção não muda com as outras gavetas presentes (`P`) nem com o contexto (`C`). Na taxa 1, sem nenhuma quantidade observada, a posterior do latente é a priori, e a porção imputada fica perto da do baseline. Por isso PPCA e FA perdem para a porção média na taxa 1 (CRPS de 18,23 da FA contra 17,97, com MPIW de 124 g contra 111 g). Além disso, o EM trata a ausência de uma gaveta como faltante ao acaso, e a média de quatro gavetas escorrega (Bebidas açucaradas, Bebidas sem açúcar, Açúcar e Molhos), justamente as que ficam subcobertas na taxa 1. Na taxa 1, `P` e `C` são toda a informação disponível, e os modelos não a usam para a porção.

## What Changes

- **Média condicional no PPCA e na FA:** a média de cada gaveta passa a ser uma função linear de `u = [1; P; C]` (colunas padronizadas pelo ajuste), e o passo M vira uma regressão por gaveta em `[z; u]`, com a mesma estrutura de hoje.
  - **Decisão tomada:** penalização ridge pequena nos coeficientes de `u` (fora o intercepto), porque o próprio `P_k`, as indicadoras de `C` que somam 1 e as gavetas que quase não aparecem juntas deixam o sistema sem solução única. O EM vira um EM do máximo a posteriori, e a parada e a monotonia passam a valer para a log-verossimilhança penalizada.
  - Opção no construtor, desligada por padrão. Sem ela, PPCA e FA seguem idênticos (o caminho numérico atual não muda).
- **Dois imputadores novos no experimento**, ao lado dos atuais: PPCA e FA com média condicional. A comparação com e sem média condicional é o resultado.
- **Controle (decisão tomada): porção média condicional**, a porção média de hoje com a média dependendo de `P` e `C` pela mesma regressão ridge, sem fatores. Separa o ganho de usar `P` e `C` do ganho da correlação entre gavetas, que é o que os modelos da disciplina acrescentam.
- **Critério de sucesso, definido antes da rodada (decisão tomada):** na validação, a FA com média condicional na taxa 1 tem CRPS da propagação abaixo do da porção média (17,97), com cobertura da propagação de pelo menos 92%, sem piorar o CRPS nas taxas 0,25 a 0,75. Leituras de apoio: cobertura da Parte 1 em Açúcar e Molhos, média das quatro gavetas problemáticas e cauda das gramas imputadas.
- **Rodada em duas etapas (decisão tomada):** primeiro a FA com média condicional e o controle, com a regressão linear, como ponto de decisão. Se o critério passar, o PPCA com média condicional e o SVGP rodam em seguida. Se não passar, a change fecha com o resultado negativo, sem a segunda etapa.

## Capabilities

### New Capabilities

Nenhuma.

### Modified Capabilities

- `imputacao`: requisito da média condicional (PPCA, FA e o controle na porção média), com redução ao modelo atual, recuperação em dados sintéticos e monotonia do objetivo penalizado; contrato comum também para as variantes novas.
- `propagacao`: os imputadores com média condicional entram na curva.

## Impact

- Código novo: `src/amaprob/imputation/media_condicional.py` (contas comuns aos três imputadores).
- Código alterado: `src/amaprob/imputation/ppca.py`, `fa.py` e `baseline.py` (opção `media_condicional`, com a derivação comentada), `src/amaprob/experiments/propagacao.py` (imputadores novos e largura das figuras).
- Testes: `tests/test_media_condicional.py` (novo) e o contrato em `tests/test_imputacao.py` com as variantes novas. Os testes atuais do PPCA e da FA não mudam.
- Resultados: linhas dos imputadores novos acrescentadas aos CSVs de propagação e da Parte 1; as linhas existentes ficam iguais.
- Sem dependência nova.
- `AGENTS.md` §5, `README.md` e `backlog/decisions.md` (a entrada da média condicional sai do backlog).
