# Tasks

## 1. Configuração adotada no experimento

- [ ] 1.1 Ler `configs/hiperparametros.json` em `propagacao.py` e montar as fábricas de imputadores e regressores com q, α e M de cada modelo; modelo ou botão sem entrada usa o valor de partida. Verificar com um teste que monta as fábricas a partir de um JSON temporário (PPCA cond. q = 12 e α = 17,8; SVGP M = 288; porção média sem entrada) e confere os atributos dos objetos construídos
- [ ] 1.2 Rótulos das figuras por modelo ("PPCA cond. (q = 12)", "SVGP (M = 288)"); verificar no mesmo teste que os rótulos refletem o JSON
- [ ] 1.3 Conferir que `otimizacao.py` continua importando e rodando: `uv run pytest tests/test_otimizacao.py` passa

## 2. Conjunto de medida e saída por conjunto

- [ ] 2.1 Opção `--conjunto {validacao, teste}` (padrão `validacao`) e função de seleção do teste em `propagacao.py`, sem mudar `taxa_zero.separar` nem `treino_e_validacao`. Verificar com `test_conjuntos_usados` (validação) intacto e um teste novo para o teste: o recorte medido só tem refeições de teste, o de ajuste só de treino, e nenhuma refeição da validação aparece
- [ ] 2.2 Pasta de saída decidida em `main` (`results/` ou `results/teste/`, com `figuras/`) e passada para `juntar` e para a gravação; legenda das figuras e log nomeiam o conjunto. Verificar com um teste que grava numa pasta temporária no modo teste e confere que só ela muda, e que uma junção parcial nela não toca os arquivos da validação
- [ ] 2.3 Rodar `uv run pytest` inteiro e verificar que passa

## 3. Iterações do EM no registro

- [ ] 3.1 Coluna `teto_iteracoes` em `imputadores_ajuste.csv` e iterações (treino completo e maior entre as dobras) na linha do log de cada imputador. Verificar com um teste de `_diagnostico` sobre um PPCA ajustado com `max_iteracoes` pequeno: iterações = teto

## 4. Documentação

- [ ] 4.1 Atualizar `AGENTS.md`: §4 (conjunto de medida e `results/teste/`), §6 (comando com `--conjunto`; M = 288 na rodada; linhas do SVGP da validação com M = 256 nos imputadores que não mudaram), §7 (tirar "roda a configuração de partida até a change da rodada no teste"; regra da curva e teto do EM aceito), §8 (`results/teste/`); README com os 3 comandos da rodada no teste. Verificar que os comandos do README rodam com `--help`

## 5. Curva de validação do PPCA com q = 12

- [ ] 5.1 Guardar as linhas de q = 5 do `HEAD` para a regra (`git show HEAD:results/propagacao_metricas.csv` para um arquivo temporário fora de `results/`) e verificar que ele tem as linhas de `ppca` e `ppca_condicional` com `blr`
- [ ] 5.2 Rodar `uv run python -m amaprob.experiments.propagacao --imputadores ppca ppca_condicional` em segundo plano e acompanhar o log; ponto de parada: ajuste do PPCA no treino completo acima de 30 min interrompe a execução e volta para decisão. Verificar no fim que `results/imputadores_ajuste.csv` traz iterações e tempo dos dois PPCA com q = 12, e que as linhas dos outros imputadores nos CSVs da validação ficaram idênticas às do `HEAD`
- [ ] 5.3 Aplicar a regra (design, decisão 6): NLL média das 3 sementes, propagação com a regressão linear, taxas 0,25 a 1, q = 12 contra q = 5; registrar a tabela de diferenças e o veredito. Se o q voltar a 5, restaurar as linhas dos dois PPCA a partir do `HEAD` e gravar q = 5 em `configs/hiperparametros.json`; verificar com `git diff` que só essas linhas mudaram
- [ ] 5.4 Registrar no design e no `AGENTS.md` §7 o resultado da regra, as iterações (teto declarado, se for o caso) e a leitura da taxa 1; verificar que as figuras da validação foram regravadas com os rótulos novos

## 6. Pronto para o teste

- [ ] 6.1 Conferência do teste sem métricas (design, decisão 7): imprimir refeições, pessoas, pessoas em comum com treino e validação (zero), `M ⊆ P` e taxa efetiva por taxa e semente, sem ler `y` nem prever; verificar que nada é gravado em `results/teste/`
- [ ] 6.2 Refazer a estimativa das 3 chamadas da rodada no teste com os tempos medidos em 5.2 (ajuste e tempo por rodada) e registrá-la no design

## Workflow follow-up

- Mostrar os commits da change e fazer cada um só com o ok de Felipe; arquivar a change depois.
- Entrada do Diário e `tasks.md` do vault só com o ok dele (zona protegida).
- Rodada no teste (fora desta change): desativar a suspensão do Windows e rodar os 3 comandos do design, na ordem.
