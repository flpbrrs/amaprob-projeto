# Tasks

## 1. Modelo

- [x] 1.1 Implementar a opção `media_condicional` no `ppca.py` (design, Decisões 1 a 4): padronização de `u`, iniciação por regressão ridge, passo M em dois tempos com a penalização, histórico do objetivo penalizado e amostragem com a média de cada refeição; derivação comentada no topo; verificar com `uv run pytest tests/test_ppca.py` sem mudança nos testes atuais
- [x] 1.2 Implementar a mesma opção no `fa.py`, com `s_k = ψ_k` na penalização e o piso do ruído mantido; derivação comentada; verificar com `uv run pytest tests/test_fa.py` sem mudança nos testes atuais
- [x] 1.3 Escrever `tests/test_media_condicional.py` com os cenários da spec `imputacao` para PPCA e FA (redução ao modelo atual, recuperação em dados sintéticos, colunas colineares, ausentes fora da verossimilhança, semente, objetivo penalizado monótono) e acrescentar as variantes ao contrato de `tests/test_imputacao.py`; verificar com `uv run pytest tests/test_media_condicional.py tests/test_imputacao.py tests/test_ppca.py tests/test_fa.py`
- [x] 1.4 Controle: opção `media_condicional` na porção média (`baseline.py`), no contrato e em `tests/test_media_condicional.py` (redução à porção média e recuperação da média); verificar com `uv run pytest tests/test_media_condicional.py tests/test_imputacao.py`

## 2. Medição antes da rodada

- [x] 2.1 Com os dados reais, ajustar PPCA e FA com média condicional no treino completo (tempo, iterações, curva do objetivo) e, na taxa 1 da validação (semente 0), medir a cauda das gramas imputadas (percentis 99, 99,9 e máximo, contra os do modelo sem média condicional) e a média das gavetas Bebidas açucaradas, Bebidas sem açúcar, Açúcar e Molhos contra a observada. Se o ajuste levar mais que o dobro do atual ou a cauda for muito maior, parar e levar a decisão para aprovação

## 3. Experimento

- [x] 3.1 Acrescentar `porcao_condicional`, `ppca_condicional` e `fa_condicional` a `experiments/propagacao.py`, com rótulos e largura das figuras proporcional ao número de imputadores; verificar com `uv run pytest`
- [x] 3.2 Executar `--regressores blr --imputadores porcao_condicional fa_condicional`; verificar que as linhas existentes de cada CSV são iguais às do `HEAD` e que não há duplicatas; aplicar o critério (design, Decisão 6) e parar para a decisão de Felipe
- [x] 3.3 Se o critério passar e Felipe aprovar: executar `--regressores blr --imputadores ppca_condicional` e `--regressores svgp --imputadores porcao_condicional ppca_condicional fa_condicional`, com as mesmas conferências da 3.2
- [x] 3.4 Conferir a leitura: critério por taxa, diferença entre a FA com média condicional e o controle, PICP da propagação perto da taxa 0, cobertura da Parte 1 em Açúcar e Molhos, média das quatro gavetas e cauda das gramas imputadas

## 4. Documentação

- [x] 4.1 Atualizar o `AGENTS.md` §5 (média condicional implementada, imputadores novos no experimento), o `README.md` (comandos e tempos) e o `backlog/decisions.md` (a entrada da média condicional sai do backlog com o resultado); verificar que os comandos rodam como escritos

## 5. Integração

- [x] 5.1 Rodar `uv run pytest` com todos os testes passando e `openspec validate media-condicional --strict` sem erros

## Workflow follow-up

- Mostrar o resultado do critério, as figuras, os tempos e o diff para aprovação antes de commitar.
- Arquivar a change (`/opsx:archive`) depois da revisão.
- Propor a entrada do Diário do projeto no vault para aprovação.
