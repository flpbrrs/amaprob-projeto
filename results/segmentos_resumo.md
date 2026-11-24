# Análise segmentada e cenários difíceis: resumo

Validação da POF 2017–2018 (45.639 refeições); o conjunto de teste não é lido. Gerado a partir de `results/segmentos_*.csv` (`experiments/segmentos.py`). Este resumo destaca local, ocasião, período e tamanho; os CSVs têm também dia da semana, dia atípico, região, quintil de renda e quintil do carboidrato.

Siglas: BLR (regressão linear bayesiana), SVGP (processo gaussiano variacional esparso), FA (análise fatorial); métricas MAE (erro absoluto médio), NLL (log-verossimilhança negativa), CRPS (continuous ranked probability score), PICP90 (cobertura do intervalo de 90%) e MPIW (largura média do intervalo de 90%).

## Notas

1. **SVGP:** o SVGP desta análise usa M = 288 e reproduz a linha M = 288 do CSV oficial (diferença de 0,001 g no MAE). O CSV oficial mistura M = 256 e M = 288 na taxa 0, conforme o imputador da linha.
2. **Variância analítica × mistura:** a variância do braço analítico bate com a da mistura de S = 100 preditivas (razão mediana 1,000 no diagnóstico completo da validação, `results/diagnostico_forma_resumo.csv`). A diferença de largura do intervalo vem da assimetria da mistura (o intervalo analítico é gaussiano e simétrico), não do truncamento em 0 da amostragem, que desloca as quantidades ocultas em no máximo 1 g.

## Definições adotadas neste worktree

- **Local pelo carboidrato:** a classe (dentro, fora, misto, não informado) e a fração de fora usam o carboidrato dos itens, não as gramas.
- **Quintis por posto:** quintis de renda e do carboidrato pelo posto `rank(method="first")`, que quebra empates pela ordem das refeições; os quintis ficam com 9.127 a 9.128 refeições.
- **Etapa D só com a semente 0:** a decomposição da variância por gaveta usa a máscara da semente 0 na taxa 1,0.

## Códigos

Contagens de itens no arquivo bruto e nas refeições válidas (ocasiões 1 a 5, ao menos 10 g de carboidrato); % sobre as refeições válidas.

**V9018: local de consumo**

| código | rótulo | itens (bruto) | itens (refeições válidas) | % |
|---:|---:|---:|---:|---:|
| 1 | Em casa ou levado de casa | 1.039.380 | 867.190 | 89,5 |
| 2 | Merenda escolar | 6.878 | 6.319 | 0,7 |
| 3 | Restaurante a quilo | 3.661 | 3.528 | 0,4 |
| 4 | Cantina, bar, lanchonete, fast food | 9.987 | 8.697 | 0,9 |
| 5 | Restaurante, outros | 10.197 | 9.728 | 1,0 |
| 6 | Vendedores ambulantes ou de rua | 1.830 | 1.588 | 0,2 |
| 7 | Fora de casa, outros | 96.691 | 71.539 | 7,4 |
| 9 | Não determinado | 6.766 | 319 | 0,0 |

**DIA_ATIPICO: dia atípico**

| código | rótulo | itens (bruto) | itens (refeições válidas) | % |
|---:|---:|---:|---:|---:|
| 1 | atipico | 114.267 | 92.028 | 9,5 |
| 2 | tipico | 1.061.123 | 876.880 | 90,5 |

**V9017: ocasião**

| código | rótulo | itens (bruto) | itens (refeições válidas) | % |
|---:|---:|---:|---:|---:|
| 1 | Café da manhã | 255.092 | 239.538 | 24,7 |
| 2 | Almoço | 353.428 | 347.824 | 35,9 |
| 3 | Lanche | 158.454 | 142.737 | 14,7 |
| 4 | Jantar | 239.709 | 231.087 | 23,9 |
| 5 | Ceia | 10.395 | 7.722 | 0,8 |
| 6 | Outra ocasião | 130.534 | 0 | 0,0 |
| 9 | Não determinado | 27.778 | 0 | 0,0 |

Classes de local por refeição: `dentro` = só itens com código 1; `fora` = só itens 2 a 7; `misto` = carboidrato dividido entre dentro e fora (fração média de fora: 0,486); `nao_informado` = só itens com código 9 (6 refeições). O código 9 não entra na fração.

## Etapa A: taxa 0 (gramas completas)

MAE relativo = MAE / carboidrato médio do segmento; % meia dose = refeições com erro de até 7,5 g.

### Regressão linear bayesiana

| segmento | valor | n | carb. (g) | MAE (g) | MAE rel. | NLL | CRPS (g) | PICP90 | MPIW (g) | % meia dose |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local | dentro | 40.561 | 62,7 | 7,78 | 0,124 | 4,013 | 6,52 | 0,932 | 45,4 | 68,0 |
| local | fora | 4.774 | 64,3 | 8,65 | 0,135 | 4,156 | 7,12 | 0,917 | 45,4 | 65,1 |
| local | misto | 298 | 67,9 | 8,07 | 0,119 | 3,956 | 6,49 | 0,930 | 45,4 | 66,4 |
| local | nao_informado | 6 | 44,5 | 8,78 | 0,197 | 3,784 | 5,72 | 1,000 | 45,4 | 33,3 |
| ocasiao | cafe | 11.279 | 57,6 | 6,47 | 0,112 | 3,856 | 5,67 | 0,952 | 45,4 | 74,0 |
| ocasiao | almoco | 12.634 | 79,0 | 7,82 | 0,099 | 4,031 | 6,54 | 0,933 | 45,4 | 67,7 |
| ocasiao | lanche | 9.957 | 48,1 | 8,87 | 0,184 | 4,134 | 7,19 | 0,920 | 45,4 | 62,5 |
| ocasiao | jantar | 11.112 | 64,2 | 8,31 | 0,129 | 4,093 | 6,91 | 0,918 | 45,4 | 66,8 |
| ocasiao | ceia | 657 | 43,2 | 10,52 | 0,243 | 4,196 | 8,12 | 0,898 | 45,4 | 53,9 |
| periodo | madrugada | 256 | 59,7 | 7,05 | 0,118 | 3,936 | 6,07 | 0,949 | 45,4 | 71,5 |
| periodo | manha | 14.855 | 58,0 | 6,79 | 0,117 | 3,879 | 5,84 | 0,949 | 45,4 | 71,9 |
| periodo | tarde | 17.593 | 67,6 | 8,24 | 0,122 | 4,080 | 6,82 | 0,927 | 45,4 | 65,8 |
| periodo | noite | 12.935 | 62,1 | 8,63 | 0,139 | 4,129 | 7,12 | 0,914 | 45,4 | 65,3 |
| tamanho | 1 | 7.039 | 38,8 | 10,84 | 0,280 | 4,379 | 8,52 | 0,888 | 45,4 | 54,7 |
| tamanho | 2 | 8.923 | 49,0 | 8,16 | 0,166 | 4,042 | 6,81 | 0,922 | 45,4 | 66,8 |
| tamanho | 3 | 12.761 | 61,3 | 6,91 | 0,113 | 3,910 | 5,93 | 0,944 | 45,4 | 72,2 |
| tamanho | 4 | 10.229 | 71,4 | 6,00 | 0,084 | 3,850 | 5,47 | 0,959 | 45,4 | 76,5 |
| tamanho | 5+ | 6.687 | 96,6 | 9,07 | 0,094 | 4,134 | 7,18 | 0,920 | 45,4 | 60,5 |

### SVGP (M = 288)

| segmento | valor | n | carb. (g) | MAE (g) | MAE rel. | NLL | CRPS (g) | PICP90 | MPIW (g) | % meia dose |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local | dentro | 40.561 | 62,7 | 7,17 | 0,114 | 3,915 | 5,95 | 0,939 | 42,4 | 70,0 |
| local | fora | 4.774 | 64,3 | 8,38 | 0,130 | 4,114 | 6,80 | 0,913 | 42,6 | 65,4 |
| local | misto | 298 | 67,9 | 7,73 | 0,114 | 3,907 | 6,20 | 0,933 | 42,6 | 66,1 |
| local | nao_informado | 6 | 44,5 | 6,95 | 0,156 | 3,677 | 4,96 | 1,000 | 42,5 | 50,0 |
| ocasiao | cafe | 11.279 | 57,6 | 6,16 | 0,107 | 3,755 | 5,14 | 0,961 | 42,3 | 75,2 |
| ocasiao | almoco | 12.634 | 79,0 | 7,14 | 0,090 | 3,951 | 6,01 | 0,937 | 42,5 | 70,9 |
| ocasiao | lanche | 9.957 | 48,1 | 8,31 | 0,173 | 4,059 | 6,71 | 0,918 | 42,4 | 64,3 |
| ocasiao | jantar | 11.112 | 64,2 | 7,60 | 0,118 | 3,978 | 6,30 | 0,926 | 42,5 | 67,7 |
| ocasiao | ceia | 657 | 43,2 | 9,83 | 0,227 | 4,147 | 7,52 | 0,912 | 43,0 | 55,4 |
| periodo | madrugada | 256 | 59,7 | 6,26 | 0,105 | 3,800 | 5,19 | 0,977 | 42,5 | 75,4 |
| periodo | manha | 14.855 | 58,0 | 6,41 | 0,111 | 3,783 | 5,33 | 0,956 | 42,4 | 73,2 |
| periodo | tarde | 17.593 | 67,6 | 7,60 | 0,112 | 3,998 | 6,30 | 0,928 | 42,5 | 68,5 |
| periodo | noite | 12.935 | 62,1 | 7,94 | 0,128 | 4,028 | 6,53 | 0,922 | 42,6 | 66,6 |
| tamanho | 1 | 7.039 | 38,8 | 10,16 | 0,262 | 4,271 | 7,90 | 0,886 | 42,4 | 55,0 |
| tamanho | 2 | 8.923 | 49,0 | 7,66 | 0,156 | 3,944 | 6,24 | 0,928 | 42,5 | 67,6 |
| tamanho | 3 | 12.761 | 61,3 | 6,28 | 0,102 | 3,805 | 5,35 | 0,955 | 42,4 | 74,2 |
| tamanho | 4 | 10.229 | 71,4 | 5,57 | 0,078 | 3,766 | 5,01 | 0,963 | 42,4 | 79,1 |
| tamanho | 5+ | 6.687 | 96,6 | 8,41 | 0,087 | 4,080 | 6,71 | 0,922 | 42,7 | 63,8 |

## Etapa B: cobertura por quintil do carboidrato previsto (taxa 0)

PICP90 em cada quintil da média prevista, dentro de cada segmento. Q5 = maiores previsões.

### BLR

| segmento | valor | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| local | dentro | 0,976 | 0,966 | 0,947 | 0,927 | 0,846 |
| local | fora | 0,979 | 0,945 | 0,940 | 0,920 | 0,801 |
| local | misto | 0,983 | 0,983 | 0,950 | 0,864 | 0,867 |
| local | nao_informado | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| ocasiao | cafe | 0,988 | 0,981 | 0,981 | 0,927 | 0,881 |
| ocasiao | almoco | 0,979 | 0,976 | 0,957 | 0,922 | 0,833 |
| ocasiao | lanche | 0,982 | 0,975 | 0,925 | 0,909 | 0,810 |
| ocasiao | jantar | 0,958 | 0,965 | 0,924 | 0,917 | 0,827 |
| ocasiao | ceia | 0,985 | 0,970 | 0,931 | 0,847 | 0,758 |
| periodo | madrugada | 1,000 | 0,922 | 0,980 | 0,882 | 0,961 |
| periodo | manha | 0,989 | 0,975 | 0,977 | 0,928 | 0,877 |
| periodo | tarde | 0,976 | 0,954 | 0,948 | 0,929 | 0,829 |
| periodo | noite | 0,961 | 0,958 | 0,928 | 0,908 | 0,818 |

### SVGP

| segmento | valor | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---:|---:|---:|---:|---:|
| local | dentro | 0,987 | 0,961 | 0,947 | 0,935 | 0,863 |
| local | fora | 0,982 | 0,925 | 0,938 | 0,919 | 0,802 |
| local | misto | 1,000 | 0,966 | 0,917 | 0,932 | 0,850 |
| local | nao_informado | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| ocasiao | cafe | 0,994 | 0,977 | 0,977 | 0,957 | 0,902 |
| ocasiao | almoco | 0,975 | 0,976 | 0,956 | 0,928 | 0,849 |
| ocasiao | lanche | 0,997 | 0,976 | 0,904 | 0,901 | 0,814 |
| ocasiao | jantar | 0,974 | 0,968 | 0,930 | 0,914 | 0,845 |
| ocasiao | ceia | 0,985 | 0,985 | 0,916 | 0,832 | 0,841 |
| periodo | madrugada | 1,000 | 1,000 | 0,961 | 0,961 | 0,961 |
| periodo | manha | 0,994 | 0,971 | 0,969 | 0,952 | 0,894 |
| periodo | tarde | 0,986 | 0,941 | 0,941 | 0,931 | 0,842 |
| periodo | noite | 0,975 | 0,960 | 0,929 | 0,910 | 0,836 |

## Etapa C: FA condicional + BLR, taxas 0,5 e 1,0

Média das 3 sementes. Plug-in = média das 100 imputações em gramas; propagação = mistura de S = 100 preditivas; analítica = momentos exatos (log-normal + lei da variância total). Δ = PICP do braço menos a PICP do mesmo segmento na taxa 0 (BLR).

### Global

| taxa | braço | MAE (g) | NLL | CRPS (g) | PICP90 | MPIW (g) | % meia dose |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0,5 | analitica | 15,60 | 4,492 | 11,85 | 0,941 | 81,9 | 41,5 |
| 0,5 | plugin | 15,67 | 5,270 | 12,26 | 0,780 | 45,4 | 41,6 |
| 0,5 | propagacao | 15,67 | 4,476 | 11,48 | 0,935 | 76,7 | 41,6 |
| 1,0 | analitica | 24,66 | 4,968 | 18,12 | 0,946 | 124,6 | 19,7 |
| 1,0 | plugin | 24,80 | 7,050 | 19,52 | 0,590 | 45,4 | 19,9 |
| 1,0 | propagacao | 24,80 | 4,981 | 17,34 | 0,939 | 114,3 | 19,9 |

### Taxa 0,5

| segmento | valor | n | PICP taxa 0 | PICP plug-in | PICP propag. | PICP analít. | Δ plug-in | Δ propag. | Δ analít. | MAE analít. (g) | MPIW propag. (g) | MPIW analít. (g) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local | dentro | 40.561 | 0,932 | 0,781 | 0,937 | 0,943 | -0,151 | 0,004 | 0,010 | 15,49 | 76,6 | 81,7 |
| local | fora | 4.774 | 0,917 | 0,771 | 0,924 | 0,931 | -0,146 | 0,007 | 0,014 | 16,44 | 77,2 | 83,2 |
| local | misto | 298 | 0,930 | 0,753 | 0,944 | 0,946 | -0,177 | 0,015 | 0,017 | 16,64 | 79,4 | 83,8 |
| local | nao_informado | 6 | 1,000 | 0,833 | 1,000 | 1,000 | -0,167 | 0,000 | 0,000 | 17,41 | 93,2 | 130,0 |
| ocasiao | cafe | 11.279 | 0,952 | 0,810 | 0,951 | 0,955 | -0,142 | -0,001 | 0,003 | 13,44 | 74,5 | 79,3 |
| ocasiao | almoco | 12.634 | 0,933 | 0,742 | 0,933 | 0,943 | -0,191 | -0,000 | 0,009 | 17,41 | 82,8 | 87,1 |
| ocasiao | lanche | 9.957 | 0,920 | 0,814 | 0,929 | 0,932 | -0,106 | 0,009 | 0,012 | 14,66 | 70,4 | 76,8 |
| ocasiao | jantar | 11.112 | 0,918 | 0,761 | 0,927 | 0,935 | -0,157 | 0,009 | 0,017 | 16,60 | 77,9 | 83,0 |
| ocasiao | ceia | 657 | 0,898 | 0,788 | 0,933 | 0,938 | -0,110 | 0,035 | 0,040 | 15,18 | 74,0 | 82,3 |
| periodo | madrugada | 256 | 0,949 | 0,773 | 0,938 | 0,934 | -0,176 | -0,012 | -0,016 | 15,57 | 73,6 | 78,4 |
| periodo | manha | 14.855 | 0,949 | 0,809 | 0,948 | 0,953 | -0,140 | -0,001 | 0,003 | 13,88 | 74,1 | 78,8 |
| periodo | tarde | 17.593 | 0,927 | 0,767 | 0,931 | 0,938 | -0,160 | 0,004 | 0,011 | 16,42 | 78,6 | 83,9 |
| periodo | noite | 12.935 | 0,914 | 0,764 | 0,926 | 0,933 | -0,151 | 0,012 | 0,019 | 16,47 | 77,3 | 82,8 |
| tamanho | 1 | 7.039 | 0,888 | 0,812 | 0,909 | 0,909 | -0,076 | 0,021 | 0,021 | 15,37 | 68,5 | 76,2 |
| tamanho | 2 | 8.923 | 0,922 | 0,793 | 0,936 | 0,941 | -0,129 | 0,014 | 0,020 | 14,52 | 72,3 | 77,1 |
| tamanho | 3 | 12.761 | 0,944 | 0,784 | 0,943 | 0,948 | -0,160 | -0,001 | 0,004 | 15,27 | 76,9 | 81,6 |
| tamanho | 4 | 10.229 | 0,959 | 0,787 | 0,949 | 0,956 | -0,172 | -0,010 | -0,003 | 14,93 | 78,6 | 83,0 |
| tamanho | 5+ | 6.687 | 0,920 | 0,711 | 0,927 | 0,942 | -0,209 | 0,007 | 0,022 | 18,95 | 88,2 | 93,1 |

### Taxa 1,0

| segmento | valor | n | PICP taxa 0 | PICP plug-in | PICP propag. | PICP analít. | Δ plug-in | Δ propag. | Δ analít. | MAE analít. (g) | MPIW propag. (g) | MPIW analít. (g) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local | dentro | 40.561 | 0,932 | 0,589 | 0,940 | 0,947 | -0,344 | 0,007 | 0,015 | 24,60 | 114,2 | 124,3 |
| local | fora | 4.774 | 0,917 | 0,601 | 0,933 | 0,936 | -0,316 | 0,016 | 0,019 | 25,10 | 114,5 | 126,4 |
| local | misto | 298 | 0,930 | 0,534 | 0,963 | 0,973 | -0,396 | 0,034 | 0,044 | 25,46 | 118,4 | 127,6 |
| local | nao_informado | 6 | 1,000 | 0,833 | 1,000 | 1,000 | -0,167 | 0,000 | 0,000 | 18,00 | 127,2 | 151,6 |
| ocasiao | cafe | 11.279 | 0,952 | 0,608 | 0,952 | 0,956 | -0,343 | -0,000 | 0,004 | 22,78 | 113,1 | 123,9 |
| ocasiao | almoco | 12.634 | 0,933 | 0,525 | 0,936 | 0,948 | -0,408 | 0,002 | 0,014 | 27,64 | 126,1 | 134,0 |
| ocasiao | lanche | 9.957 | 0,920 | 0,670 | 0,938 | 0,938 | -0,251 | 0,018 | 0,018 | 21,81 | 100,5 | 113,9 |
| ocasiao | jantar | 11.112 | 0,918 | 0,567 | 0,930 | 0,942 | -0,351 | 0,012 | 0,024 | 25,95 | 114,8 | 124,3 |
| ocasiao | ceia | 657 | 0,898 | 0,671 | 0,959 | 0,960 | -0,227 | 0,061 | 0,062 | 20,70 | 106,2 | 123,3 |
| periodo | madrugada | 256 | 0,949 | 0,556 | 0,901 | 0,895 | -0,393 | -0,048 | -0,055 | 26,47 | 108,7 | 117,8 |
| periodo | manha | 14.855 | 0,949 | 0,620 | 0,949 | 0,954 | -0,329 | -0,000 | 0,005 | 22,78 | 110,9 | 120,9 |
| periodo | tarde | 17.593 | 0,927 | 0,574 | 0,936 | 0,944 | -0,353 | 0,009 | 0,016 | 25,62 | 117,5 | 128,1 |
| periodo | noite | 12.935 | 0,914 | 0,577 | 0,932 | 0,942 | -0,338 | 0,018 | 0,028 | 25,47 | 113,8 | 124,3 |
| tamanho | 1 | 7.039 | 0,888 | 0,735 | 0,930 | 0,929 | -0,154 | 0,041 | 0,041 | 20,13 | 91,6 | 107,8 |
| tamanho | 2 | 8.923 | 0,922 | 0,648 | 0,942 | 0,946 | -0,274 | 0,020 | 0,025 | 21,82 | 101,1 | 110,2 |
| tamanho | 3 | 12.761 | 0,944 | 0,575 | 0,941 | 0,947 | -0,369 | -0,003 | 0,002 | 24,54 | 113,1 | 122,2 |
| tamanho | 4 | 10.229 | 0,959 | 0,538 | 0,941 | 0,953 | -0,421 | -0,018 | -0,005 | 26,32 | 123,9 | 133,5 |
| tamanho | 5+ | 6.687 | 0,920 | 0,465 | 0,939 | 0,953 | -0,455 | 0,019 | 0,033 | 30,88 | 143,2 | 152,4 |

## Etapa D: ranking, descrição e decomposição

### Ranking (BLR, taxa 0, segmentos com n ≥ 100)

Três mais difíceis e três mais fáceis por MAE relativo e por desvio absoluto da PICP em relação a 90%, com a descrição do segmento.

| critério | grupo | segmento | valor | n | carb. (g) | MAE rel. | PICP90 | gavetas médias | fração c/ bebida | fração c/ açúcar | var. resíduo (g²) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mae_relativo | dificil | tamanho | 1 | 7.039 | 38,8 | 0,280 | 0,888 | 1,00 | 0,061 | 0,000 | 319 |
| mae_relativo | dificil | quintil_carboidrato | Q1 | 9.128 | 20,3 | 0,264 | 0,979 | 2,02 | 0,443 | 0,200 | 66 |
| mae_relativo | dificil | ocasiao | ceia | 657 | 43,2 | 0,243 | 0,898 | 1,84 | 0,391 | 0,266 | 249 |
| mae_relativo | facil | ocasiao | almoco | 12.634 | 79,0 | 0,099 | 0,933 | 4,02 | 0,459 | 0,138 | 186 |
| mae_relativo | facil | tamanho | 5+ | 6.687 | 96,6 | 0,094 | 0,920 | 5,53 | 0,783 | 0,399 | 225 |
| mae_relativo | facil | tamanho | 4 | 10.229 | 71,4 | 0,084 | 0,959 | 4,00 | 0,715 | 0,454 | 117 |
| desvio_picp | dificil | quintil_carboidrato | Q1 | 9.128 | 20,3 | 0,264 | 0,979 | 2,02 | 0,443 | 0,200 | 66 |
| desvio_picp | dificil | quintil_carboidrato | Q5 | 9.128 | 129,4 | 0,106 | 0,829 | 4,05 | 0,681 | 0,388 | 517 |
| desvio_picp | dificil | quintil_carboidrato | Q2 | 9.128 | 36,9 | 0,164 | 0,966 | 2,66 | 0,518 | 0,250 | 86 |
| desvio_picp | facil | tamanho | 1 | 7.039 | 38,8 | 0,280 | 0,888 | 1,00 | 0,061 | 0,000 | 319 |
| desvio_picp | facil | regiao | norte | 6.408 | 64,2 | 0,136 | 0,909 | 2,89 | 0,548 | 0,231 | 219 |
| desvio_picp | facil | ocasiao | ceia | 657 | 43,2 | 0,243 | 0,898 | 1,84 | 0,391 | 0,266 | 249 |

### Variância do resíduo na taxa 0 (BLR)

Média do erro quadrático por segmento. A variância preditiva da BLR é quase constante (≈ 190 g²: 1/β mais um termo de parâmetro pequeno), então segmentos acima de 190 ficam com cobertura abaixo da global e os abaixo, acima.

| segmento | valor | n | gavetas médias | carb. (g) | fração c/ bebida | fração c/ açúcar | var. resíduo (g²) |
|---|---|---:|---:|---:|---:|---:|---:|
| local | dentro | 40.561 | 3,09 | 62,7 | 0,563 | 0,303 | 179,0 |
| local | fora | 4.774 | 3,05 | 64,3 | 0,606 | 0,237 | 233,7 |
| local | misto | 298 | 3,62 | 67,9 | 0,745 | 0,383 | 157,1 |
| local | nao_informado | 6 | 1,83 | 44,5 | 0,167 | 0,000 | 91,5 |
| ocasiao | cafe | 11.279 | 3,08 | 57,6 | 0,862 | 0,591 | 119,3 |
| ocasiao | almoco | 12.634 | 4,02 | 79,0 | 0,459 | 0,138 | 185,8 |
| ocasiao | lanche | 9.957 | 2,05 | 48,1 | 0,528 | 0,281 | 225,0 |
| ocasiao | jantar | 11.112 | 3,05 | 64,2 | 0,442 | 0,193 | 209,4 |
| ocasiao | ceia | 657 | 1,84 | 43,2 | 0,391 | 0,266 | 248,5 |
| periodo | madrugada | 256 | 2,91 | 59,7 | 0,832 | 0,578 | 149,4 |
| periodo | manha | 14.855 | 2,98 | 58,0 | 0,739 | 0,478 | 127,7 |
| periodo | tarde | 17.593 | 3,32 | 67,6 | 0,509 | 0,208 | 204,6 |
| periodo | noite | 12.935 | 2,90 | 62,1 | 0,448 | 0,203 | 223,3 |
| tamanho | 1 | 7.039 | 1,00 | 38,8 | 0,061 | 0,000 | 318,5 |
| tamanho | 2 | 8.923 | 2,00 | 49,0 | 0,550 | 0,145 | 190,0 |
| tamanho | 3 | 12.761 | 3,00 | 61,3 | 0,631 | 0,386 | 139,8 |
| tamanho | 4 | 10.229 | 4,00 | 71,4 | 0,715 | 0,454 | 116,9 |
| tamanho | 5+ | 6.687 | 5,53 | 96,6 | 0,783 | 0,399 | 225,1 |

### Decomposição da variância preditiva por gaveta (taxa 1,0, braço analítico, semente 0)

Fração da variância preditiva média do segmento atribuída à incerteza da quantidade de cada gaveta (os 5 maiores postos). `var. pred.` = variância preditiva média; `var. entrada` = parte que vem das quantidades ocultas.

**tamanho = 1** (n = 7.039; var. pred. = 1719 g²; var. entrada = 1529 g²)

| posto | gaveta | fração da var. preditiva | fração da var. de entrada |
|---:|---|---:|---:|
| 1 | Doces | 0,574 | 0,646 |
| 2 | Panificados | 0,082 | 0,092 |
| 3 | Laticínios | 0,062 | 0,070 |
| 4 | Frutas | 0,042 | 0,047 |
| 5 | Pizzas, salgados e sanduíches | 0,036 | 0,040 |

**ocasiao = ceia** (n = 657; var. pred. = 2210 g²; var. entrada = 2020 g²)

| posto | gaveta | fração da var. preditiva | fração da var. de entrada |
|---:|---|---:|---:|
| 1 | Doces | 0,478 | 0,523 |
| 2 | Laticínios | 0,174 | 0,191 |
| 3 | Panificados | 0,104 | 0,113 |
| 4 | Açúcar | 0,054 | 0,059 |
| 5 | Bebidas / açucaradas | 0,022 | 0,024 |

**local = fora** (n = 4.774; var. pred. = 1806 g²; var. entrada = 1615 g²)

| posto | gaveta | fração da var. preditiva | fração da var. de entrada |
|---:|---|---:|---:|
| 1 | Doces | 0,254 | 0,283 |
| 2 | Panificados | 0,139 | 0,155 |
| 3 | Cereais | 0,097 | 0,109 |
| 4 | Pizzas, salgados e sanduíches | 0,076 | 0,085 |
| 5 | Leguminosas | 0,075 | 0,083 |

**local = dentro** (n = 40.561; var. pred. = 1673 g²; var. entrada = 1483 g²)

| posto | gaveta | fração da var. preditiva | fração da var. de entrada |
|---:|---|---:|---:|
| 1 | Panificados | 0,203 | 0,229 |
| 2 | Doces | 0,162 | 0,182 |
| 3 | Cereais | 0,122 | 0,138 |
| 4 | Leguminosas | 0,096 | 0,108 |
| 5 | Açúcar | 0,054 | 0,061 |

## Suposições e dúvidas

- **Dicionário fora do checkout.** Os códigos de V9018 e V9017 foram conferidos no instrumento POF 7 (Bloco de Consumo Alimentar Pessoal) e na distribuição dos microdados, não no dicionário de variáveis (a verificar).
- **DIA_ATIPICO:** 1 = atípico, 2 = típico (a verificar no dicionário). Dia atípico responde por 9,7% dos itens no arquivo bruto e 9,5% nas refeições válidas.
- **Comparação com a versão do worktree `amaprob-projeto-parte1`:** lá o local é classificado pelas gramas (misto: 372 refeições contra 298 aqui), os empates de renda têm outro tratamento e a Etapa D faz a média das 3 sementes. Nos segmentos com o mesmo n, BLR, SVGP e braço analítico batem (até 2e-15); plug-in e propagação diferem dentro do erro de Monte Carlo, porque o sorteio das imputações percorre as refeições em outros blocos.
- **Período** pela hora de V9015 (0–5, 6–11, 12–17, 18–23 h); a ocasião declarada e o horário podem discordar, e nenhum dos dois é corrigido.
- **Ranking na taxa 0 com n ≥ 100.** O corte tira o local não informado (6 refeições). O ranking na taxa 1,0 do braço analítico fica em aberto.
- **Heterocedasticidade:** a cobertura cai de ~98% no quintil inferior da previsão para 82–88% no superior em todos os segmentos (Etapa B), e a variância do resíduo vai de ~117 a ~318 g² entre segmentos contra ~190 g² previstos. O ruído homocedástico da BLR é a causa provável.
