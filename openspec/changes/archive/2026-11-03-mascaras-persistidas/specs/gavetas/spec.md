# Spec Delta

## ADDED Requirements

### Requirement: Configuração adotada de 29 gavetas
A camada de dados SHALL expor a configuração de gavetas adotada pelo projeto: dividir `Bebidas` e `Farinhas e massas` e fundir os subgrupos de `Bebidas` em 4 perfis nutricionais (`sem açúcar`, `açucaradas`, `lácteas`, `cerveja e vinho`). Os demais grupos ficam inteiros e o açúcar de mesa mantém a gaveta própria.

#### Scenario: Número de gavetas
- **WHEN** as gavetas são montadas com a configuração adotada a partir da classificação versionada
- **THEN** há 29 gavetas distintas

#### Scenario: Perfis de Bebidas
- **WHEN** a configuração adotada é usada
- **THEN** os subgrupos `Água`, `Café`, `Chá`, `Refrigerantes diet/light`, `Refrescos/sucos industrializados diet/light` e `Destiladas` recebem `Bebidas / sem açúcar`; `Sucos`, `Refrescos/sucos industrializados`, `Refrigerantes`, `Outras bebidas não-alcoólicas` e `Bebidas à base de soja` recebem `Bebidas / açucaradas`; `Bebidas lácteas` recebe `Bebidas / lácteas`; `Cerveja` e `Vinho` recebem `Bebidas / cerveja e vinho`

#### Scenario: Farinhas e massas por subgrupo
- **WHEN** a configuração adotada é usada
- **THEN** cada código de `Farinhas e massas` recebe a gaveta `Farinhas e massas / <subgrupo>`
