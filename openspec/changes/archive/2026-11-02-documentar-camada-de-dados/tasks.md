# Tasks

Change só de documentação: nenhum código muda. As tarefas conferem se as specs descrevem fielmente o código e os dados atuais.

## 1. Conferência das specs

- [x] 1.1 Revisar cada requisito de `leitura-pof`, `gavetas`, `refeicoes` e `split-por-pessoa` contra `src/amaprob/data/` e confirmar que nenhum descreve comportamento ausente do código. Verificação: a revisão termina sem requisito marcado como divergente.
- [x] 1.2 Reexecutar, a partir dos microdados da versão 20230713, as contagens citadas nos cenários (1.175.390 itens lidos; 968.908 itens, 304.670 refeições e 46.141 pessoas após a montagem; 21 e 117 gavetas; matriz 304.670 × 21; 32.299 / 6.921 / 6.921 pessoas no split), com um script descartável fora do repositório. Verificação: todos os números conferem.
- [x] 1.3 Rodar `openspec validate documentar-camada-de-dados --strict`. Verificação: o comando informa que a change é válida, sem avisos.

## Workflow follow-up

- Arquivar a change depois da revisão do autor, para gerar as quatro specs em `openspec/specs/`.
- Conferir que o archive preencheu o `## Purpose` de cada spec principal, sem o placeholder `TBD`.
