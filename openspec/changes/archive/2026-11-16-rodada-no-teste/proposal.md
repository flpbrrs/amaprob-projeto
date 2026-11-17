# Proposal

## Why

Os números do artigo saem de uma rodada única no teste (Regra 6), com modelos e hiperparâmetros congelados pela validação. Hoje o experimento de propagação só mede na validação e usa a configuração de partida (q = 5, α = 1, M = 256), e não a adotada na Parte 3 (`configs/hiperparametros.json`). Falta também conferir a configuração adotada fora da taxa 0,5, que foi a única usada na escolha. Esta change deixa tudo pronto para a rodada no teste ser só executada.

## What Changes

- **Configuração adotada lida pelo experimento:** imputadores e regressores passam a ser construídos com os valores de `configs/hiperparametros.json` (q, α, M), nos dois conjuntos. Os rótulos das figuras mostram o valor de cada modelo.
- **Conjunto de medida escolhido na execução:** validação (padrão) ou teste. O ajuste continua só no treino. O modo teste grava em `results/teste/` (CSVs e figuras), e nada da validação é sobrescrito.
- **Execução parcial também no teste:** a junção por `--regressores` e `--imputadores` vale dentro da pasta do conjunto. **Decisão tomada:** a rodada no teste é feita em 3 chamadas (imputadores rápidos; PPCA; PPCA condicional), para que uma falha perca só uma parte.
- **Iterações do EM no registro de ajuste** (treino completo e maior valor entre as dobras), para conferir a convergência durante a execução.
- **Curva de validação do PPCA com q = 12**, as duas versões. **Decisão tomada:** com os dois regressores (regressão linear e SVGP com M = 288).
- **Regra da curva, fixada antes de vê-la. Decisão tomada:** o q do PPCA volta a 5 se, em alguma taxa de 0,25 a 1, a NLL média das 3 sementes (propagação com a regressão linear) ficar pior que a de q = 5 já gravada por mais de 0,015. Senão, q = 12 fica.
- **Teto do EM. Decisão tomada:** se o PPCA com q = 12 parar nas 500 iterações, o teto vira o ponto de parada declarado; o limite não muda.
- **Conferência do caminho do teste sem ler métricas:** tamanhos, `M ⊆ P` e taxas das máscaras do teste, antes da rodada.
- **Estimativa da rodada no teste refeita** com os tempos medidos do PPCA com q = 12.
- A rodada no teste em si **não** faz parte desta change: ela roda depois, sobre o código desta change.

## Capabilities

### New Capabilities

Nenhuma.

### Modified Capabilities

- `propagacao`: medida na validação ou no teste, com saída por conjunto; configuração adotada lida do arquivo; iterações do EM no registro de ajuste.

## Impact

- Código: `src/amaprob/experiments/propagacao.py` e `tests/test_propagacao.py`. As funções que `experiments/otimizacao.py` importa (`treino_e_validacao`, `gerador`, `Conjunto`, `S`, `FLUXO_VALIDACAO`) mantêm o comportamento.
- Resultados da validação: as linhas de `ppca` e `ppca_condicional` (dos dois regressores) passam a ser as de q = 12, e as do SVGP desses dois, as de M = 288. As linhas dos outros imputadores com o SVGP continuam com M = 256 (mistura declarada).
- Resultados novos: `results/teste/` (só na rodada no teste, fora desta change).
- Docs: `AGENTS.md` §4, §6, §7 e §8 e README.
- Sem dependência nova.
