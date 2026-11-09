# Saturno — etapa Duolingo

Relatório incremental. Uma seção por fase.

## Legenda de fontes

| Sigla | O que é |
|---|---|
| `exp:N` | `experiment.py`, linha N — repo `duolingo/halflife-regression`, commit `0041df0` (2017-12-14) |
| `eval:N` | `evaluation.r`, linha N — mesmo commit |
| `README` | `README.md` do mesmo commit |
| `§x` | seção do artigo (Settles & Meeder, ACL 2016, pp. 1848–1858) |
| `DV` | API do Dataverse, `data/dataverse-metadata.json` |
| `inv` | `data/inventory.json`, gerado por `src/data/inventory.js` |

---

## Fase A — Dado e referência

### A1. Arquivo e licença

| Item | Valor | Fonte |
|---|---|---|
| Dataset | "Replication Data for: A Trainable Spaced Repetition Model…" | DV |
| Versão | 1.0, publicada 2017-12-14 | DV |
| Arquivo | `settles.acl16.learning_traces.13m.csv.gz` | DV |
| id do arquivo | 3091087 | DV |
| Tamanho | 379 004 009 bytes | DV |
| MD5 esperado | `0a1cae5eb7ad4b0bd9c0de91d74fcced` | DV |
| MD5 obtido | `0a1cae5eb7ad4b0bd9c0de91d74fcced` — **confere** | `md5` local; `gzip -t` ok |
| Licença dos dados | **CC BY-NC 4.0** (atribuição, não comercial) | DV, campo `termsOfUse` |
| Licença do código | MIT | `LICENSE.md`, cabeçalho `exp:2` |
| Acesso | público, sem pedido de acesso | DV, `fileAccessRequest=false` |

Uso aqui: pessoal, local, não comercial. Cabe na licença.
Citar o artigo ao usar (README, "When using this data set…").

### A2. Referência: o que o código oficial faz

#### Split

| Item | Valor | Fonte |
|---|---|---|
| Regra | primeiros 90% das linhas = treino; últimos 10% = teste | `exp:257-258` |
| Ordem | a do arquivo; **sem embaralhar antes do split** | `exp:218-258` |
| Embaralha | só o treino, antes da passada de SGD | `exp:118` |
| Artigo | "first 90% for training and remaining 10% for testing" | §4.1 |

#### Rótulo e alvo derivado

| Variável | Cálculo | Fonte |
|---|---|---|
| p | `p_recall`, cortado em [0,0001; 0,9999] | `exp:221`, `exp:175-177` |
| Δ (t) | `delta` / 86 400 → dias | `exp:222` |
| h "observada" | −Δ / log₂(p), cortada em [15 min; 274 dias] | `exp:223`, `exp:20-21`, `exp:180-182` |
| acertos / erros | `history_correct` / `history_seen − history_correct` | `exp:229-231` |

#### Features por modelo

| Modelo | Features | Fonte |
|---|---|---|
| HLR | √(1+acertos), √(1+erros), bias, indicador do lexema | `exp:244-252` |
| HLR −lex | igual, sem o lexema (flag `-l`) | `exp:251` |
| LR | as do HLR + Δ em dias (`time`) | `exp:247-248` |
| Leitner | acertos − erros; h = 2^(diff) | `exp:238`, `exp:63` |
| Pimsleur | acertos + erros (= `history_seen`); h = 2^(2,35·n − 16,46) | `exp:240`, `exp:70` |

Lexema: a chave é `learning_language:lexeme_string` (`exp:252`).
O README diz que o original usava só `lexeme_id` (README, "Only the lexeme_id column…").

#### Saída e loss

| Modelo | Saída | Loss | Fonte |
|---|---|---|---|
| HLR | ĥ = 2^(Θ·x); p̂ = 2^(−Δ/ĥ) | (p−p̂)² + α(h−ĥ)² + λ‖Θ‖² | `exp:49-59`, §3.3, Apêndice A.3 |
| LR | p̂ = sigmoide(Θ·x) | (p−p̂)² + λ‖Θ‖² | `exp:75-77`, `exp:102-113` |
| Leitner, Pimsleur | fórmula fixa, sem treino | — | `exp:100-101`, `exp:116-117` |

#### Hiperparâmetros

| Parâmetro | Valor | Fonte |
|---|---|---|
| η (taxa) | 0,001 | `exp:37`, §4.1 |
| α (peso de h) | 0,01 | `exp:37`, §4.1 |
| λ (L2) | 0,1 | `exp:37`, §4.1 |
| σ | 1 | `exp:37` |
| Otimizador | AdaGrad simplificado: taxa / √(1 + nº de vezes que a feature apareceu) | `exp:89`, Apêndice A.3 |
| Épocas | 1 passada | `exp:115-120` |
| Ajuste | escolhidos no primeiro 1M de linhas | §4.1 |

#### Métricas

| Métrica | Como | Fonte |
|---|---|---|
| MAE | média de \|p − p̂\| | `exp:185-187`, `eval:9`, §4.1 |
| AUC | ROC com rótulo = `round(p)`, escore = p̂ | `eval:11`, §4.1 |
| COR_h | Spearman entre h e ĥ | `eval:14` (`method='s'`), §4.1 |

#### Números publicados (Tabela 2, §4.1)

| Modelo | MAE ↓ | AUC ↑ | COR_h ↑ |
|---|---|---|---|
| HLR | 0,128 | 0,538 | 0,201 |
| HLR −lex | 0,128 | 0,537 | 0,160 |
| HLR −h | 0,350 | 0,528 | −0,143 |
| HLR −lex −h | 0,350 | 0,528 | −0,142 |
| Leitner | 0,235 | 0,542 | −0,098 |
| Pimsleur | 0,445 | 0,510 | −0,132 |
| LR | 0,211 | 0,513 | n/a |
| LR −lex | 0,212 | 0,514 | n/a |
| Constante p̄ = 0,859 | 0,175 | n/a | n/a |

#### Divergências código × artigo (importam para replicar)

| # | Divergência | Fonte |
|---|---|---|
| D1 | A função `spearmanr` do Python calcula **Pearson** (não ranqueia). O número do artigo vem do R, que é Spearman. | `exp:195-206` × `eval:14` |
| D2 | A taxa do HLR tem um fator extra 1/(1+p), ausente no artigo. | `exp:89` × Apêndice A.3 |
| D3 | L2 no código: λ·θ; no artigo: 2λ·θ. E é aplicado a cada aparição da feature, não uma vez por passo. | `exp:97` × Apêndice A.3 |
| D4 | Pimsleur: código 2,35 e −16,46; artigo 2,4 e −16,5 (arredondado). | `exp:70` × Apêndice A.2 |
| D5 | LR devolve h aleatório; por isso COR_h = n/a. | `exp:78` |
| D6 | AUC: no R, `round(0.5)` = 0 (arredonda para o par). Linhas com p = 0,5 viram classe 0. | `eval:11` |
| D7 | Tabela 1 do artigo mostra a feature x_n (total visto); o código usa só acertos e erros. | §3.4, Tabela 1 × `exp:242-245` |
| D8 | AUC: `pROC::roc()` usa `direction = "auto"` por padrão. Se a mediana do escore nos negativos passa a dos positivos, ele **inverte** a comparação. Uma AUC abaixo de 0,5 vira 1 − AUC. (Achado na B1.) | `eval:11` × pROC `R/roc.R:139, 350-355` (commit `29ffd11`; lógica presente desde `5d15efd`, 2011) |

Decisão para a Fase B: replicar **o código** (é o que gerou a Tabela 2), e medir COR_h como Spearman (como o R).

### A3. Inventário do CSV

Uma passada em streaming: 37 s, pico de memória 145 MB. Fonte de tudo abaixo: `inv`.

#### Colunas

| Coluna | Tipo | Observação |
|---|---|---|
| p_recall | float | 77 valores distintos; = session_correct / session_seen em 100% das linhas |
| timestamp | int | Unix, segundos |
| delta | int | segundos; nunca 0 |
| user_id | texto | 115 222 aprendizes |
| learning_language, ui_language | texto | ver abaixo |
| lexeme_id | texto | 19 279 ids; cada id tem um só lexeme_string |
| lexeme_string | texto | 18 781 strings distintas |
| history_seen, history_correct | int | nenhum correct > seen |
| session_seen, session_correct | int | nenhum correct > seen; máx. 20 |

#### Volume

| Item | Valor |
|---|---|
| Linhas | 12 854 226 (artigo: "12.9 million", §4.1) |
| Período | 2013-02-28 → 2013-03-12 (UTC) — ~2 semanas, como em §4.1 |
| Ponto do split | linha 11 568 803 |
| Treino termina / teste começa | ambos em 2013-03-11 13:45:39 UTC; **nenhuma** linha de teste é anterior ao fim do treino |

O split do artigo é, na prática, **temporal**. O arquivo não é estritamente ordenado (225 347 recuos locais de timestamp), mas a fronteira é limpa.

#### Idiomas (learning ← ui)

| Par ui→learning | Linhas | % |
|---|---|---|
| es→en | 3 641 179 | 28,3 |
| en→es | 3 407 689 | 26,5 |
| en→fr | 1 873 734 | 14,6 |
| en→de | 1 452 597 | 11,3 |
| **pt→en** | **949 460** | **7,4** |
| en→it | 793 935 | 6,2 |
| it→en | 424 152 | 3,3 |
| en→pt | 311 480 | 2,4 |

- Idiomas aprendidos: de, en, es, fr, it, pt. **Sem ja, sem ru.**
- ui_language = pt: 949 460 linhas, **todas aprendendo inglês**. Não há fr nem es com interface pt.

#### p_recall

| Faixa | Linhas | % |
|---|---|---|
| 0 | 906 340 | 7,1 |
| (0; 0,25] | 1 872 | 0,0 |
| (0,25; 0,5] | 478 634 | 3,7 |
| (0,5; 0,75] | 488 757 | 3,8 |
| (0,75; 1) | 195 681 | 1,5 |
| 1 | 10 782 942 | 83,9 |

- Média: 0,896. Linhas com p = 0,5 exato: 449 133 (3,5%) — afetadas por D6.
- 59% das linhas têm session_seen = 1, então p só pode ser 0 ou 1.

#### delta (faixas alinhadas à escada 1-3-7-16-35-75)

| Δ | Linhas | % |
|---|---|---|
| < 1 h | 4 969 348 | 38,7 |
| 1 h – 1 d | 1 839 882 | 14,3 |
| 1 – 3 d | 1 834 804 | 14,3 |
| 3 – 7 d | 1 531 271 | 11,9 |
| 7 – 16 d | 1 178 082 | 9,2 |
| 16 – 35 d | 751 502 | 5,8 |
| 35 – 75 d | 440 261 | 3,4 |
| 75 – 274 d | 292 814 | 2,3 |
| > 274 d | 16 262 | 0,1 |

Máximo: 467 dias. Quase 40% das linhas são revisões no mesmo dia.

#### history_seen

| Vezes vistas antes | Linhas | % |
|---|---|---|
| 1 | 559 900 | 4,4 |
| 2 | 1 156 364 | 9,0 |
| 3–5 | 3 945 144 | 30,7 |
| 6–10 | 2 841 800 | 22,1 |
| 11–20 | 2 007 423 | 15,6 |
| 21–50 | 1 476 156 | 11,5 |
| 51–100 | 492 312 | 3,8 |
| > 100 | 375 127 | 2,9 |

Média 22, máximo 13 518. Nunca 0: toda linha já tem histórico.

#### lexeme_string

Formato `superfície/lema<pos><mod>…` (README). Exemplos reais:

| Idioma | Exemplos |
|---|---|
| fr | `la/le<det><def><f><sg>` · `mes/mon<det><pos><mf><pl>` · `c'/ce<prn><tn><nt><sg>` |
| es | `niños/niño<n><m><pl>` · `leo/leer<vblex><pri><p1><sg>` |
| pt | `temos/ter<vblex><pri><p1><pl>` · `vermelhas/vermelho<adj><f><pl>` |
| it | `dell'/di<pr>+il<det><def><mf><sg>` (contração: dois lemas) |

| Caso | Linhas | % |
|---|---|---|
| Superfície curinga `<*sf>` (vale para qualquer flexão) | 431 410 | 3,4 |
| Qualquer curinga `<*…>` | 538 246 | 4,2 |
| Expressão fixa como pos (`@common_phrases:il_y_a`, `@adv:por_favor`…) | poucos milhares | <0,1 |

Classes mais frequentes (pos): n 5,47 M · vblex 2,24 M · det 1,35 M · prn 1,05 M · adj 0,71 M · vbser 0,55 M.

#### Ponto em aberto

| # | Fato | Por que importa |
|---|---|---|
| O1 | Média de p no arquivo: 0,896. O artigo usa p̄ = 0,859 na linha "Constant" (Tabela 2). | Não sei de onde vem 0,859. Investigo na B0 antes de comparar MAE. |

### A4. Amostra de trabalho

#### Proposta: 1 M de linhas

| Item | Valor |
|---|---|
| Regra | Bernoulli por índice de linha, taxa 1 000 000 / 12 854 226 = 0,0778, hash determinístico, semente 0 |
| Split | a partição vem do índice **no arquivo completo** (< 11 568 803 = treino). A amostra é um subconjunto do treino e do teste do artigo. |
| Treino | 899 782 linhas |
| Teste | 99 471 linhas |
| Validação (B3) | proposta: as linhas de treino com índice ≥ 81% do arquivo (fatia temporal final do treino) |
| Arquivo | `data/sample.csv` (107 MB, gitignored), gerado em 9 s |

#### A amostra representa o arquivo?

| Idioma | % no arquivo | % na amostra |
|---|---|---|
| en | 39,01 | 39,03 |
| es | 26,51 | 26,49 |
| fr | 14,58 | 14,59 |
| de | 11,30 | 11,32 |
| it | 6,18 | 6,18 |
| pt | 2,42 | 2,39 |

- Média de p (cortado): treino 0,8958, teste 0,8952.
- Lexemas no treino da amostra: 12 764. Linhas de teste com lexema nunca visto: 362 (0,4%) → índice 0, "desconhecido".
- Para a Fase C: fr 132 k / 14 k, es 239 k / 25 k, en 349 k / 41 k (treino / teste). ui=pt: 73 940 linhas (todas en).

#### Tempo de 1 época (tfjs 4.22, backend cpu, Node 26)

| Modelo | Parâmetros | Lote | s / época (900 k) |
|---|---|---|---|
| Regressão logística, 3 features | 4 | 256 | 0,75 |
| Regressão logística, 3 features | 4 | 1 024 | 0,46 |
| Regressão logística, 3 features | 4 | 4 096 | 0,41 |
| MLP 32, sem lexema (medido em 20 k, extrapolado) | 161 | 1 024 | ~7 |
| MLP + `tf.layers.embedding` 16 | 204 913 | 256 | **> 12 min sem terminar** |
| MLP + embedding próprio 16 (vocab da amostra: 12 765) | 204 913 | 1 024 | **25,9** (medido) |

Achado: no backend cpu puro, o gradiente do `tf.layers.embedding` cresce com o tamanho do vocabulário (19 mil) a cada passo, então fica inviável [I: medido, causa não verificada no código do tfjs].
Solução: `src/model/embedding.js` — `gather` na ida, `scatterND` na volta (`tf.customGrad`). Gradiente idêntico ao do `gather` nativo (teste em `test/embedding.test.mjs`). Ainda é embedding, não one-hot.

Leitura: 1 M sobra para B0–B2 (segundos por época). B3 com 20 épocas ≈ 9 min. Dá para subir a amostra depois, se a B3 pedir; o loader aceita qualquer taxa sem quebrar o split.

### A5. Código entregue

| Arquivo | Papel |
|---|---|
| `src/data/duolingo.js` | streaming (.csv e .gz), parse do lexema, instância como `read_data`, features por método, split, amostra, vocabulário |
| `src/data/inventory.js` | inventário em uma passada |
| `src/data/sample.js` | grava e lê a amostra de trabalho |
| `src/model/bench.js` | mede 1 época (`lr` ou `mlp`) |
| `src/model/embedding.js` | embedding com gradiente esparso |
| `test/duolingo.test.mjs` | 11 testes, fixture de 10 linhas em `test/fixtures/duolingo-mini.csv` |
| `test/embedding.test.mjs` | 1 teste: ida e gradiente contra o `gather` nativo, conta feita à mão |

### A6. Decisões pedidas a você

| # | Pergunta | Minha proposta |
|---|---|---|
| Q1 | Amostra de 1 M (taxa 0,0778, semente 0)? | Sim |
| Q2 | Validação da B3 = fatia final do treino (índice ≥ 81% do arquivo)? | Sim (temporal, como o split) |
| Q3 | Replicar o **código** oficial (D2, D3), não a fórmula do artigo? | Sim; reporto as duas onde divergirem |
| Q4 | COR_h como Spearman (R), não o "spearmanr" do Python (que é Pearson, D1)? | Sim; reporto o Pearson ao lado |
| Q5 | AUC com rótulo `round(p)` do R (0,5 → 0, D6)? | Sim, e também com p ≥ 0,5 → 1 ao lado |
| Q6 | Fase C com ui=pt: só existe para en. Ok? | Sim; fr e es ficam só com ui=en |

### A7. Respostas (2026-10-01) e proposta para Q2

| # | Decisão |
|---|---|
| Q1 | Amostra de 1 M, taxa 0,0778, semente 0 |
| Q2 | Validação = linhas com índice em [81%; 90%) do arquivo completo. Nunca toca o teste. Protocolo abaixo. |
| Q3 | Replicar o código; reportar código e artigo onde divergem |
| Q4 | COR_h: Spearman principal, Pearson ao lado |
| Q5 | AUC: regra do R principal (0,5 → 0); p ≥ 0,5 → 1 ao lado |
| Q6 | ui=pt só para en, sem esforço extra |

#### Q2 — proposta: opção (b)

| Modelo | Treina em | Usa a validação para |
|---|---|---|
| B0 (constante, Leitner, Pimsleur) | média do treino [0; 90%) | nada |
| B1 LR, B2 HLR | [0; 90%), 1 passada, como `exp:115-120` | nada (hiperparâmetros do artigo, §4.1) |
| B3 MLP, passo 1 | [0; 81%) | escolher a época E* (early stopping) e reportar a métrica de validação |
| B3 MLP, passo 2 | [0; 90%), do zero, mesma semente, E* épocas | — ; avalia no teste uma vez |

Por que (b), e não (a):

- Com (b), **todos os modelos veem as mesmas linhas de treino** no fim. A regra 2 ("mesmo split") vale sem asterisco.
- Com (a), LR e HLR perderiam 10% do treino só por causa da B3, e se afastariam do protocolo da Tabela 2.
- Custo de (b): E* foi escolhido com 10% menos dados. É o desvio padrão desse método; reporto E* e a curva de validação.
- Se B1 ou B2 em TF.js precisarem de mais de uma época (ex.: Adam), seguem o mesmo protocolo de dois passos.

Status: **aprovada em 2026-10-01.** Registro:

- **MLP (B3) tem early stopping** na validação [81%; 90%) e retreina em [0; 90%) com E* épocas.
- **LR (B1) e HLR (B2) usam os hiperparâmetros do artigo** (η = 0,001, α = 0,01, λ = 0,1; `exp:37`, §4.1), sem validação e sem early stopping.

#### Checagem de mutação do `embeddingLookup`

| Passo | Resultado |
|---|---|
| Mutação: gradiente com `tensorScatterUpdate` (sobrescreve índice repetido) no lugar de `scatterND` (soma) | linha 2 do gradiente: [2, 20] em vez de [3, 30] |
| `test/embedding.test.mjs` | **falha** (1 de 1) |
| Reversão | `git diff` vazio; 12 de 12 passam |

---

## Fase B — Modelos

Código: `src/model/metrics.js`, `src/model/baselines.js`, `src/model/b0.js`. Resultado bruto: `data/b0-results.json`.
Métricas como o R: previsões arredondadas a 4 casas antes da AUC e do COR_h (`exp:162` grava o `.preds` assim).

### B0 — baselines sem treino

#### Réplica no teste completo do artigo (1 285 423 linhas)

| Modelo | MAE ↓ | AUC ↑ (R) | COR_h ↑ (Spearman) | Artigo (Tabela 2) |
|---|---|---|---|---|
| Constante = média do treino (0,896) | 0,174 | 0,500 | n/a | 0,175 · n/a · n/a |
| Leitner | **0,235** | **0,542** | **−0,098** | 0,235 · 0,542 · −0,098 |
| Pimsleur | **0,445** | **0,510** | **−0,132** | 0,445 · 0,510 · −0,132 |

Leitner e Pimsleur batem o artigo nas três casas, nas três métricas.
Isso valida o loader, o split, as fórmulas e as métricas.

#### Variantes (Q4, Q5) no mesmo teste

| Modelo | AUC (p ≥ 0,5 → 1) | COR_h (Pearson) |
|---|---|---|
| Leitner | 0,551 | −0,105 |
| Pimsleur | 0,515 | −0,112 |

As variantes **não** batem o artigo. Confirma que a Tabela 2 usou a regra do R (D6) e Spearman (D1).

#### O1 — fechado: inconsistência do artigo

| Constante c | Origem | MAE no teste |
|---|---|---|
| 0,896 | média de p no treino (90%) | 0,174 |
| 0,895 | média de p no primeiro 1 M (o conjunto de ajuste, §4.1) | 0,175 |
| 0,905 | Σ session_correct / Σ session_seen | 0,169 |
| 0,859 | o valor escrito no artigo | 0,199 |

- A MAE da linha "Constant" (0,175, Tabela 2) bate com a **média do treino**.
- O rótulo "p̄ = 0,859" (Tabela 2 e nota 7) não é reproduzível com este arquivo: dá MAE 0,199. Nenhum dos 12 candidatos testados chega a 0,859.
- Decisão (2026-10-01): **inconsistência do artigo. Não investigar mais.** A constante aqui é a média de p no treino.

#### Na amostra de 1 M (a régua para B1–B3)

| Modelo | MAE ↓ | AUC ↑ (R) | COR_h ↑ | AUC (≥) | COR_h (Pearson) |
|---|---|---|---|---|---|
| Constante (0,8958) | **0,1747** | 0,500 | n/a | 0,500 | n/a |
| Leitner | 0,2349 | **0,5395** | **−0,097** | 0,551 | −0,104 |
| Pimsleur | 0,4457 | 0,5057 | −0,133 | 0,513 | −0,111 |

Teste da amostra: 99 471 linhas. Treino: 899 782. Tempo total da B0 (arquivo completo + amostra): 14 s. Parâmetros treinados: 0.

#### Critério de vitória (aprovado em 2026-10-01)

| Regra | Como |
|---|---|
| Escada | Leitner → LR → HLR → MLP. Cada degrau só se compara ao **anterior**. |
| Sempre | Reportar também contra a constante. |
| Vencer | MAE menor que o degrau anterior, com **IC 95% da diferença de MAE excluindo zero** |
| Bootstrap | Pareado por aprendiz (user_id): reamostra aprendizes com reposição, cada um com todas as suas linhas. 1000 reamostras, semente 20261001, quantil tipo 7 do R. |
| Venceu na MAE e piorou na AUC | Marcar **"venceu sem discriminar"** e parar para decisão |
| **Efeito mínimo (2026-10-02), só para o futuro** | Na Fase C e no dado próprio, vencer exige **também ΔMAE ≥ 0,005**, fixado antes de rodar. **Não se aplica retroativamente** a B1–B3. |
| **Emenda (2026-10-02)** | "Piorou na AUC" = **IC 95% inteiro da diferença de AUC abaixo de zero** (mesmo bootstrap pareado). **A emenda é posterior ao resultado da B2** e foi reaplicada à B1. |

Régua na amostra (o primeiro degrau a vencer é o Leitner):

| Métrica | Leitner | Constante |
|---|---|---|
| MAE | 0,2349 | 0,1747 |
| AUC (R) | 0,5395 | 0,500 |

Aviso, como hipótese: no artigo a LR tem MAE 0,211 (Tabela 2). Ela venceria o Leitner (0,235) e perderia para a constante (0,175).

Código: `src/model/bootstrap.js`. Teste à mão em `test/bootstrap.test.mjs`: 2 aprendizes, 4 linhas; IC [−0,1; +0,2], mediana −0,025, ponto −0,025.
Checagem de mutação: ponderar por aprendiz em vez de por linha → mediana +0,05 → o teste **falha**. Revertido.

#### Testes da B0

`test/metrics.test.mjs`: 12 testes. `test/bootstrap.test.mjs`: 7 testes. MAE, AUC (com e sem empate), postos, Pearson e Spearman contra contas feitas à mão; Leitner e Pimsleur contra a fórmula à mão, incluindo os cortes de 15 min e 274 dias.

### B1 — regressão logística (um neurônio)

Código: `src/model/lr.js`, `src/model/b1.js`, `src/data/columns.js`. Resultado bruto: `data/b1-results.json`.

#### O que é

| Item | Valor | Fonte |
|---|---|---|
| Neurônio | p̂ = sigmoide(w·x + peso do lexema) | `exp:75-77` |
| Features | √(1+acertos), √(1+erros), Δ em dias, bias, um peso por lexema | `exp:244-252` |
| Treino | 1 passada embaralhada, lote 1, taxa η/√(1 + aparições da feature), erro (p̂ − p)·x, depois L2 | `exp:102-120` |
| Hiperparâmetros | η = 0,001, λ = 0,1, σ = 1 (do artigo, sem early stopping) | `exp:37`, §4.1 |
| Em TF.js | soma ponderada, sigmoide, corte e atualização são ops de tensor; o lexema é um vetor [V] (embedding de dimensão 1) | `src/model/lr.js` |
| Conferência | a mesma conta em JS puro (float64) | `trainLrJs` |

#### Réplica no arquivo completo (réplica JS, 11,57 M treino, 1,29 M teste)

| Modelo | MAE | AUC (R, crua) | AUC (pROC, direção auto) | Artigo (Tabela 2) |
|---|---|---|---|---|
| LR | **0,211** | 0,488 | **0,512** | 0,211 · 0,513 |
| LR −lex | **0,212** | 0,486 | **0,514** | 0,212 · 0,514 |

- A MAE bate. A AUC só bate com a **inversão automática do pROC** (D8).
- Leitura: a LR do artigo **ranqueia ao contrário** (AUC crua < 0,5). O 0,513 publicado é 1 − 0,487.
- Leitner e Pimsleur não são afetados: neles as duas AUCs são iguais (B0 rodado de novo).

#### Na amostra (o degrau da escada)

| Modelo | MAE ↓ | AUC crua ↑ | AUC pROC | Parâmetros | Treino |
|---|---|---|---|---|---|
| Constante | 0,1747 | 0,500 | 0,500 | 0 | — |
| Leitner (degrau anterior) | 0,2349 | 0,5395 | 0,5395 | 0 | — |
| **LR (TF.js)** | **0,2191** | **0,4782** | 0,5218 | 12 768 | 13,0 s |
| LR −lex (TF.js) | 0,2193 | 0,4775 | 0,5225 | 4 | 7,1 s |

- TF.js × JS puro: maior diferença de previsão 0,0007 (float32 × float64 em 900 mil passos). MAE do JS: 0,2193.
- Pesos da LR: acertos +0,387, erros +0,174, Δ **+0,0076**, bias +0,195. Mais erros e mais tempo **sobem** p̂ [V: pesos]. É o sinal contrário ao da memória. Hipótese: a taxa encolhe com 1/√(passo), então 1 passada não converge [I].
- MAE pior que no arquivo completo (0,219 × 0,211). Hipótese: com 13× menos passos, os pesos andam menos [I].

#### Veredito (bootstrap pareado, 19 064 aprendizes, 1000 reamostras, semente 20261001)

| Comparação | Δ MAE (LR − outro) | IC 95% | Veredito |
|---|---|---|---|
| LR × Leitner (degrau anterior) | −0,0158 | [−0,0188; −0,0126] | MAE menor, IC exclui zero |
| LR × constante | +0,0444 | [+0,0434; +0,0455] | não venceu |

**Resultado: "venceu sem discriminar".** A LR erra menos que o Leitner na MAE, mas ranqueia pior: AUC 0,478 × 0,540 (crua) ou 0,522 × 0,540 (pROC). Pela regra, paro aqui.

#### Decisões (2026-10-01)

| # | Decisão |
|---|---|
| — | **B1 não aprovada**: "venceu sem discriminar". Resultado negativo. |
| Emenda | Reaplicada em 2026-10-02 (ver abaixo): **veredito confirmado**. |
| Q7 | A B2 decide contra o **Leitner** (último degrau aprovado). Reporta também contra LR e constante. |
| Q8 | Decide a **AUC calculada (direção fixa)**, inclusive o flag. A do pROC fica ao lado, só para comparar com o artigo. |

#### Regra emendada reaplicada à B1 (2026-10-02)

| Comparação | Δ AUC (LR − outro) | IC 95% | Veredito |
|---|---|---|---|
| LR × Leitner | −0,061 | [−0,070; −0,053] | IC todo abaixo de zero → **venceu sem discriminar (confirmado)** |
| LR × constante | −0,022 | [−0,030; −0,013] | não venceu (MAE pior) |

#### Diagnóstico dos sinais (fora da escada; não muda o veredito)

LR na amostra, réplica JS (= TF.js, B1), **10 épocas, taxa constante η = 0,001**, semente 20261001 + época. Código: `src/model/b1-diag.js`. Bruto: `data/b1-diag.json`.

Peso de "erros" (√(1+erros)) e de Δ (dias) ao fim de cada época, LR com lexema:

| Época | erros, λ = 0,1 | Δ, λ = 0,1 | erros, λ = 0 | Δ, λ = 0 | bias, λ = 0 |
|---|---|---|---|---|---|
| 1 | +0,091 | +0,189 | −0,403 | +0,174 | 2,26 |
| 2 | +0,139 | +0,068 | −0,405 | +0,064 | 2,29 |
| 3 | +0,126 | +0,114 | −0,413 | +0,083 | 2,24 |
| 4 | +0,085 | +0,042 | −0,402 | +0,024 | 2,26 |
| 5 | +0,097 | +0,005 | −0,402 | **−0,009** | 2,24 |
| 6 | +0,084 | +0,006 | −0,471 | **−0,002** | 2,29 |
| 7 | +0,113 | +0,089 | −0,388 | +0,080 | 2,24 |
| 8 | +0,119 | +0,026 | −0,393 | +0,015 | 2,27 |
| 9 | +0,114 | +0,051 | −0,395 | +0,037 | 2,21 |
| 10 | +0,129 | +0,078 | −0,375 | +0,075 | 2,26 |

LR −lex: mesmo padrão (`data/b1-diag.json`).

O teste pedido (λ = 0,1, o do código): **os dois sinais persistem** nas 10 épocas.
O teste extra (λ = 0, sem L2) separa as duas features:

| Feature | λ = 0,1 | λ = 0 | Leitura |
|---|---|---|---|
| erros | + em 10/10 | **− em 10/10** | Artefato do L2, não seleção [V: tabela] |
| Δ | + em 10/10 | + em 8/10, ≈ 0 em 2/10 | Não resolvido |

Por que o L2 inverte "erros" [I: mecanismo, coerente com os números]: o L2 do código encolhe todo peso a cada passo (`exp:111`). O bias não chega ao logit da média (logit 0,896 ≈ 2,15; com λ = 0 o bias vai a 2,26). Toda feature sempre positiva (√(1+erros) ≥ 1, Δ > 0) vira um "intercepto substituto" e ganha peso positivo.

O dado cru, sem modelo (treino da amostra):

| Δ | <1h | 1h–1d | 1–3d | 3–7d | 7–16d | 16–35d | 35–75d | >75d |
|---|---|---|---|---|---|---|---|---|
| p médio | 0,909 | 0,895 | 0,892 | 0,889 | 0,885 | 0,879 | 0,872 | 0,861 |

| Erros anteriores (só history_seen 6–10) | 0 | 1 | 2 | 3–5 | >5 |
|---|---|---|---|---|---|
| p médio | 0,920 | 0,895 | 0,863 | 0,815 | 0,632 |

- A recordação **cai** com Δ e com erros, como a memória prevê.
- Mas a queda com Δ é **rasa**: de menos de 1 hora a mais de 75 dias, só 0,048.

#### H-sel — hipótese registrada (não é regra)

**H-sel:** o agendador do Duolingo só deixa Δ crescer para itens que o aprendiz já domina. Então, neste dado, Δ longo vem junto com item forte, e a curva de esquecimento observada fica achatada.

| Evidência | A favor | Contra |
|---|---|---|
| Sinal de Δ na LR | positivo e persistente com λ = 0,1; positivo em 8/10 com λ = 0 | oscila até ≈ 0; Δ não está em escala, SGD com taxa fixa é ruidoso |
| Dado cru | queda de 0,048 em 75+ dias: rasa demais para um item médio [I] | a queda existe e é monótona |
| "erros" | — | o sinal positivo de "erros" era L2, não seleção |

**Impacto previsto na Fase C** [I]:

| Degrau da escada | Efeito esperado de H-sel |
|---|---|
| 1, 3, 7 dias | pouco viés: Δ curto é comum para itens de toda força (52% das linhas têm Δ < 1 dia, `inv`) |
| 16, 35, 75 dias | p_recall **superestimado**: estimado em "sobreviventes". A escada vai parecer mais segura do que é. |

**Aceita em 2026-10-02 para Δ** (não para "erros"). Na Fase C, os degraus **16, 35 e 75 dias são limite superior**.

Onde a escada cai no dado (`inv`, arquivo completo; contagem no treino da amostra):

| Faixa de Δ | % das linhas | Linhas no treino da amostra | Degrau da escada |
|---|---|---|---|
| < 1 h | **38,7%** | 348 643 | — |
| 1 h – 1 d | 14,3% | 129 285 | 1 |
| 1 – 3 d | 14,3% | 129 295 | 3 |
| 3 – 7 d | 11,9% | 105 448 | 7 |
| 7 – 16 d | 9,2% | 82 970 | 16 |
| 16 – 35 d | 5,8% | 52 467 | 35 |
| 35 – 75 d | 3,4% | 30 647 | 75 |
| > 75 d | 2,4% | 21 027 | — |

Os degraus longos caem na **região esparsa**: 16–75 dias somam 9,2% das linhas, contra 38,7% abaixo de 1 hora.
Menos dado e mais seleção no mesmo lugar.

Como a Fase C deve tratar isso [proposta]: ler as estimativas dos degraus longos como **limite superior**, condicionar no histórico (acertos, erros) e reportar o n de linhas por degrau.

Observação fora da escada: com λ = 0, a LR chega a MAE 0,150–0,181 e AUC 0,51–0,61. Não muda o veredito da B1 (o protocolo é o do código).

#### Testes da B1

| Arquivo | Testes | O que confere |
|---|---|---|
| `test/lr.test.mjs` | 4 | 1 passo à mão (w = 0,001 · 0,4999 · (1 − 0,0001)) no TF.js e no JS; −lex; embaralhamento determinístico; TF.js = JS na fixture |
| `test/metrics.test.mjs` | +1 | AUC com direção auto do pROC, 4 casos à mão |

`trainLrJs` ganhou `epochs`, `constantRate` e `onEpoch`, só para o diagnóstico. Os padrões continuam os do código (1 época, taxa decrescente).

### B2 — HLR (half-life regression)

Código: `src/model/hlr.js`, `src/model/b2.js`. Bruto: `data/b2-results.json`.

#### O que é

| Item | Valor | Fonte |
|---|---|---|
| Saída da rede | log2(ĥ) = Θ·x; ĥ = 2^(Θ·x), cortada em [15 min; 274 d]; p̂ = 2^(−Δ/ĥ) | `exp:49-59` |
| Features | √(1+acertos), √(1+erros), bias, um peso por lexema | `exp:244-252` |
| Loss | (p − p̂)² + α(h − ĥ)² + λ‖Θ‖², com o termo da meia-vida | §3.3, Apêndice A.3 |
| Passo | a derivada exata dessa loss em Θ·x (`exp:86-87`). **Conferido contra `tf.grad`** num ponto sem corte. | `test/hlr.test.mjs` |
| Treino | 1 passada embaralhada, lote 1, taxa (1/(1+p))·η/√(1+aparições), depois L2 | `exp:89-99` (D2, D3) |
| Hiperparâmetros | η = 0,001, α = 0,01, λ = 0,1 (do artigo, sem early stopping) | `exp:37`, §4.1 |

#### Réplica no arquivo completo (réplica JS, 11,57 M treino, 1,29 M teste)

| Modelo | MAE | AUC | COR_h | Artigo (Tabela 2) |
|---|---|---|---|---|
| HLR | 0,129 | **0,538** | **0,201** | 0,128 · 0,538 · 0,201 |
| HLR −lex | 0,129 | **0,537** | **0,160** | 0,128 · 0,537 · 0,160 |
| HLR −h | **0,350** | **0,528** | **−0,143** | 0,350 · 0,528 · −0,143 |
| HLR −lex −h | **0,350** | **0,528** | **−0,142** | 0,350 · 0,528 · −0,142 |

- 11 de 12 números batem nas três casas. A MAE do HLR fica 0,001 acima.
- Ruído do embaralhamento (o Python embaralha sem semente, `exp:118`): 9 sementes dão MAE de 0,1286 a 0,1291 no HLR.
- Com a taxa do artigo (sem o fator 1/(1+p), L2 2λθ): MAE 0,1273, AUC 0,536, COR_h 0,197.
- O 0,128 publicado fica entre as duas taxas. Diferença ≤ 0,0014: **não é "longe"**. Não investiguei mais.
- Aqui a direção automática do pROC não muda nada: AUC crua = AUC pROC nas 4 variantes.

#### Na amostra (o degrau da escada)

| Modelo | MAE ↓ | AUC ↑ | COR_h ↑ | Parâmetros | Treino |
|---|---|---|---|---|---|
| Constante | 0,1747 | 0,500 | n/a | 0 | — |
| Leitner (último aprovado) | 0,2349 | **0,5395** | −0,097 | 0 | — |
| LR (não aprovada) | 0,2191 | 0,4782 | n/a | 12 768 | 13 s |
| **HLR (TF.js)** | **0,1283** | 0,5368 | **0,170** | 12 767 | 21,7 s |
| HLR −lex (TF.js) | 0,1284 | 0,5363 | 0,107 | 3 | 15,3 s |
| HLR −h (JS) | 0,4269 | 0,5280 | −0,144 | 12 767 | 0,05 s |
| HLR −lex −h (JS) | 0,4270 | 0,5280 | −0,143 | 3 | 0,08 s |

- TF.js × JS puro: MAE igual (0,1283). Maior diferença de previsão: 0,023 (float32 em 2^x, 900 mil passos).
- O termo da meia-vida é o que faz o HLR funcionar: sem ele, a MAE triplica (0,128 → 0,427), como no artigo.
- Pesos do HLR (TF.js): acertos +0,012, erros −0,171, bias 7,33. O bias sozinho dá ĥ = 2^7,33 ≈ 161 dias. Os acertos quase não pesam [V: pesos].
- Por que o bias é tão alto [V: contagem]: 50,8% das linhas de treino têm h-alvo cortado em 274 dias (p = 1 com Δ > ~1 hora). Outras 4,5% estão no piso de 15 min. A loss puxa ĥ para o teto.

#### Veredito (bootstrap pareado, 19 064 aprendizes, 1000 reamostras, semente 20261001)

Resultado pela regra original (2026-10-01): **"venceu sem discriminar"** contra o Leitner. Paramos e pedimos decisão (Q9).
A regra foi emendada **depois** desse resultado. Abaixo, o veredito pela regra emendada:

| Comparação | Δ MAE | IC 95% | Δ AUC | IC 95% | Veredito (emendada) |
|---|---|---|---|---|---|
| **HLR × Leitner (decide)** | −0,1066 | [−0,1094; −0,1037] | −0,0027 | [−0,0068; +0,0014] | **venceu** |
| HLR × LR | −0,0908 | [−0,0924; −0,0892] | +0,059 | [+0,049; +0,069] | venceu |
| HLR × constante | −0,0464 | [−0,0474; −0,0453] | +0,037 | [+0,030; +0,043] | venceu |

- O HLR é o primeiro modelo que **vence a constante** na MAE.
- A AUC fica 0,003 abaixo da do Leitner, mesma ordem do artigo (§4.1, Tabela 2). O IC inclui zero.

#### Decisão (2026-10-02)

| # | Decisão |
|---|---|
| Q9 | **HLR aprovada** sob a regra emendada. A emenda é **posterior** ao resultado da B2. Degrau a vencer na B3: **HLR**. |

#### Auditoria do bug do `Int32Array`

| Pergunta | Resposta | Como |
|---|---|---|
| O bug existia em B0 ou B1? | **Não.** `pairedBootstrapStat` nasceu na B2 e não estava em nenhum commit. | `git grep` em `HEAD` (8807e3c) |
| Algum `.map` sobre typed array com dado em B0/B1? | **Não.** O único `.map` sobre linhas (`b0.js:18`) é sobre array comum. | `git grep "\.map("` em `src/` |
| Os números commitados mudam? | **Não.** B0 e B1 rodados de novo: 0 diferenças em `b0-results.json` e `b1-results.json`, fora os tempos. | comparação campo a campo |

#### Testes da B2

| Arquivo | Testes | O que confere |
|---|---|---|
| `test/hlr.test.mjs` | 4 | passo do código = `tf.grad` da loss do artigo; 1 passo à mão no JS e no TF.js (com e sem termo h); TF.js = JS na fixture; ĥ = 3 e p̂ = 0,5 à mão |
| `test/bootstrap.test.mjs` | +3 | bootstrap de estatística dá o IC do caso à mão; o ponto não trunca escore (bug achado e corrigido nesta etapa: `.map` de `Int32Array` zerava a AUC do ponto); `aucDiffBootstrap` à mão; veredito pela regra emendada |

### B3 — MLP: proposta (não implementada)

Degrau a vencer: **HLR** (MAE 0,1283, AUC 0,5368, COR_h 0,170 na amostra).

#### Alvo: p direto ou log2 h?

| | (A) p direto: sigmoide + MSE | (B) log2 ĥ, como a HLR |
|---|---|---|
| Saída | p̂ = sigmoide(f(x, Δ)) | log2 ĥ = f(x); p̂ = 2^(−Δ/ĥ) |
| Prós | otimiza quase a métrica (MSE ≈ MAE); não usa o h-alvo cortado (50,8% em 274 d); não impõe curva exponencial; mais simples | mesma cabeça e mesma loss da HLR → a diferença vem só da rede; p̂ **cai com Δ por construção**; dá ĥ para o COR_h e para a Fase C (p̂ em 1, 3, 7… dias sai de ĥ) |
| Contras | Δ tem de entrar como feature livre. A LR mostrou o risco: com H-sel, a rede pode aprender p̂ **subindo** com Δ, inútil para a escada. COR_h fica n/a. Compara arquiteturas diferentes. | herda a curva exponencial; herda o h-alvo cortado, que puxa ĥ para o teto; o termo α(h − ĥ)² domina a loss (h em dias, até 274) |

**Proposta: (B).** Motivos:
1. A Fase C precisa de p̂ monótono em Δ para ler a escada.
2. Comparar com a HLR exige a mesma cabeça e a mesma loss.
3. Δ fora da entrada impede que a seleção (H-sel) entre direto em ĥ.

(A) fica como possível etapa futura, não nesta escada.

#### Arquitetura

| Camada | Detalhe | Parâmetros (etapa 2) |
|---|---|---|
| Entrada densa | √(1+acertos), √(1+erros) — as mesmas da HLR | — |
| Embedding do lexema | dim 8, gradiente esparso (`src/model/embedding.js`); `<unk>` = vetor 0 fixo | 12 765 × 8 = 102 120 |
| Oculta | 10 → 32, ReLU, dropout 0,2 | 352 |
| Saída | 32 → 1 = z; log2 ĥ = log2(15 min) + (log2 274 − log2(15 min))·sigmoide(z) | 33 |
| **Total** | | **102 505** |
| −lex (ablação) | 2 → 32 → 1, sem embedding | 129 |

- O corte suave (sigmoide entre 15 min e 274 d) substitui o `hclip` duro: tem gradiente em toda parte. Na avaliação, p̂ ainda passa por `pclip`.
- As features são as mesmas da HLR. Assim o ganho, se houver, vem da não linearidade e do embedding, não de dado novo.
- Loss: média no lote de (p − p̂)² + α(h − ĥ)², com α = 0,01 (do artigo). Sem L2 explícito; a regularização é o dropout e o early stopping.
- Otimizador: Adam, taxa 0,001, lote 1024, semente 20261001, embaralhamento por época com semente + época.

Risco conhecido: o termo de h domina a loss (α·(274 − 161)² ≈ 128 contra ≤ 1 do termo de p).
Proposta: manter α = 0,01, para comparar com a HLR. Não ajustar α nesta etapa (menos hipóteses testadas).

#### Plano da opção (b)

| Etapa | Linhas (amostra) | O quê |
|---|---|---|
| 1 | treino: índice < 10 411 923 (0–81%): **810 072** | treina; vocabulário só dessas linhas (12 428 lexemas) |
| 1 | validação: 81–90%: **89 710** (2013-03-10 10:45 → 03-11 13:45 UTC) | MAE de p ao fim de cada época; 361 linhas com lexema novo → `<unk>` |
| 1 | parada | até 30 épocas, paciência 3 → **E\*** = época de menor MAE de validação |
| 2 | treino: 0–90%: **899 782** | do zero, mesma semente, **E\* épocas**; vocabulário refeito (12 764) |
| 2 | teste: **99 471** | avalia uma vez; nunca tocado na etapa 1 |

Reporto E\*, a curva de validação por época, e o MAE de validação da etapa 1.

#### Ablação (obrigatória)

| Modelo | Etapas 1 e 2 | Compara com |
|---|---|---|
| MLP (com lexema) | sim, E\* próprio | **HLR — decide** (MAE + regra emendada da AUC) |
| MLP −lex | sim, E\* próprio | HLR −lex (informativo) |

Ambos também contra Leitner, LR e constante (informativo), com o bootstrap pareado de MAE e de AUC.

#### Custo estimado

Base medida na Fase A: MLP com embedding 16 + oculta 32 = **25,9 s por época de 900 mil linhas** (TF.js, cpu).

| Passo | Estimativa |
|---|---|
| Etapa 1, com lexema (810 k linhas, ~25–30 s/época, 8–30 épocas) | 4–15 min |
| Etapa 2, com lexema (E\* × ~28–33 s) | 4–15 min |
| −lex, etapas 1 e 2 (sem embedding, ~5–8 s/época) | 2–6 min |
| Bootstraps de MAE e AUC (4 comparações × 2 modelos) | ~5 min |
| **Total** | **~15–40 min** de máquina |

[I: extrapolado da medição da Fase A; o 2^x da cabeça e a validação por época somam algo.]

#### Testes previstos (antes de rodar)

- Cabeça: corte suave dentro de [15 min; 274 d]; ĥ = 3 e Δ = 3 → p̂ = 0,5, à mão.
- Gradiente da loss em z: derivação manual × `tf.grad`.
- Early stopping: E\* e parada por paciência numa curva de validação feita à mão.
- Partições: etapa 1 nunca vê índice ≥ 81%; etapa 2 nunca vê o teste.
- Checagem de mutação em um deles.

Status: **aprovada em 2026-10-02**, com 4 ajustes: early stopping pela MAE de validação (loss ao lado); etapa 2 com sementes 0, 1, 2 e veredito só se as 3 vencerem; HLR −lex para comparar com MLP −lex; testes e mutação antes de rodar. Etapa 1 com semente 0.

### B3 — MLP: resultados

Código: `src/model/mlp.js`, `src/model/b3.js`. Bruto: `data/b3-results.json`. Rodada de 2026-10-02, 20 min de máquina.

#### Testes e mutação (antes de rodar)

| Teste (`test/mlp.test.mjs`) | O que confere |
|---|---|
| cabeça limitada | ĥ ∈ [15 min; 274 d] para z de −10⁴ a +10⁴; z = 0 → meio do intervalo em log2 |
| p̂ à mão | ĥ = 3, Δ = 3 → 0,5; Δ = 6 → 0,25 |
| gradiente | derivação manual de ∂L/∂z = `tf.grad`, 3 pontos |
| early stopping | curva à mão → melhor época 3, para na 6; empate não conta como melhora |
| limites do split | 80 → etapa 1; 81 e 89 → validação; 90 → teste (N = 100) e nos índices reais (10 411 922/3, 11 568 802/3) |
| MLP na fixture | nº de parâmetros, `<unk>` = 0, 1 época roda, previsões nos limites |

| Mutação | Teste que falhou |
|---|---|
| empate conta como melhora (`<` → `<=`) | early stopping, empate |
| paciência fora por um (`>=` → `>`) | as 2 de early stopping |
| validação começa em 80% | limites do split |

As 3 revertidas: arquivo idêntico ao original, 50 de 50 passando.

#### Etapa 1: early stopping (semente 0)

| Época | MLP: MAE val | MLP: loss val | MLP −lex: MAE val | MLP −lex: loss val |
|---|---|---|---|---|
| 1 | 0,12779 | 151,5 | 0,12785 | 153,7 |
| 2 | 0,12715 | 148,8 | 0,12728 | 150,6 |
| 3 | 0,12714 | 147,7 | 0,12732 | 149,2 |
| 4 | 0,12692 | 147,4 | 0,12714 | 148,8 |
| 5 | 0,12722 | 147,2 | 0,12736 | 148,7 |
| 6 | 0,12717 | 146,9 | 0,12738 | 148,6 |
| 7 | 0,12689 | 146,7 | 0,12709 | 148,5 |
| **8** | **0,12665** | 146,8 | **0,12693** | 148,6 |
| 9 | 0,12708 | 146,6 | 0,12725 | 148,5 |
| 10 | 0,12706 | 146,7 | 0,12731 | 148,5 |
| 11 (para) | 0,12705 | 146,8 | 0,12716 | 148,5 |

- **E\* = 8** nos dois.
- A loss quase só mede o termo de h (≈ 147; o termo de p é ≤ 1). Ela continua caindo enquanto a MAE estaciona. Era o risco previsto.
- Etapa 1: 142 s (MLP), 35 s (−lex). Cerca de 13 s e 3 s por época.

#### Etapa 2: teste, 3 sementes

| Modelo | Semente | MAE ↓ | AUC ↑ | COR_h ↑ | Treino |
|---|---|---|---|---|---|
| MLP | 0 | 0,12755 | 0,5378 | 0,212 | 118 s |
| MLP | 1 | 0,12736 | 0,5383 | 0,212 | 119 s |
| MLP | 2 | 0,12751 | 0,5373 | 0,212 | 119 s |
| **MLP (mediana)** | | **0,1275** | **0,5378** | **0,212** | |
| MLP −lex | 0 | 0,12764 | 0,5369 | 0,168 | 28 s |
| MLP −lex | 1 | 0,12756 | 0,5367 | 0,166 | 28 s |
| MLP −lex | 2 | 0,12772 | 0,5368 | 0,167 | 28 s |
| **MLP −lex (mediana)** | | **0,1276** | **0,5368** | **0,167** | |

Parâmetros na etapa 2: MLP **102 505** (99 817 na etapa 1, vocabulário menor); MLP −lex **129**.
AUC crua = AUC pROC em todas as sementes.

#### Veredito contra a HLR (decide) — 3 de 3 sementes

| Semente | Δ MAE (MLP − HLR) | IC 95% | Δ AUC | IC 95% | Veredito |
|---|---|---|---|---|---|
| 0 | −0,00077 | [−0,00085; −0,00069] | +0,0011 | [−0,0006; +0,0034] | venceu |
| 1 | −0,00097 | [−0,00105; −0,00088] | +0,0015 | [−0,0005; +0,0044] | venceu |
| 2 | −0,00082 | [−0,00090; −0,00074] | +0,0006 | [−0,0008; +0,0026] | venceu |

**Resultado: MLP vence a HLR nas 3 sementes. Aprovável pela regra.**

Contra os outros degraus (informativo; as 3 sementes têm o mesmo veredito):

| Contra | Δ MAE (faixa nas 3 sementes) | Δ AUC (faixa) | Veredito |
|---|---|---|---|
| Leitner | −0,1073 a −0,1075 | −0,0022 a −0,0012 (IC cruza zero) | venceu |
| LR | −0,0916 a −0,0918 | +0,059 a +0,060 | venceu |
| Constante | −0,0471 a −0,0473 | +0,037 a +0,038 | venceu |

#### Ablação de lexema (obrigatória)

| Comparação | Δ MAE (3 sementes) | Δ AUC | Veredito |
|---|---|---|---|
| MLP −lex × HLR −lex | −0,00070 a −0,00087 | IC cruza zero nas 3 | venceu nas 3 |
| MLP −lex × HLR (com lexema) | −0,00060 a −0,00077 | IC cruza zero nas 3 | venceu nas 3 |
| MLP × MLP −lex (medianas) | −0,0001 | +0,001 | — (sem bootstrap direto) |

#### Leitura (frases curtas; as marcadas [I] são hipótese)

1. **O ganho é real, mas pequeno.** MAE 0,1283 → 0,1275: −0,0008, ou 0,6%. O IC exclui zero porque o bootstrap é pareado e o n é grande, não porque o ganho seja grande.
2. **O lexema quase não ajuda a MAE.** Sem lexema, 129 parâmetros chegam a 0,1276. O ganho vem de outra coisa que não o embedding.
3. **O lexema ajuda o COR_h.** 0,212 com lexema × 0,167 sem. Mesma direção que na HLR (0,170 × 0,107) e no artigo (0,201 × 0,160).
4. **Confundidor aberto [I]:** o MLP difere da HLR em duas coisas ao mesmo tempo, a arquitetura (não linear) e o otimizador (Adam, 8 épocas, lote 1024 × 1 passada de SGD com taxa decrescente). O MLP −lex com 129 parâmetros vencer sugere que boa parte do ganho pode ser **otimização**, não rede.
   - Teste que separaria: HLR linear treinada como o MLP (Adam, mesmo E\*, mesmo lote). Não rodei: está fora do pedido.
5. A loss é dominada pelo termo de h (α = 0,01 fixo). A MAE estaciona na época 2–4, e o resto da curva é ruído de ±0,0003.

#### Custo real × estimado

| Passo | Estimado | Real |
|---|---|---|
| Etapa 1, MLP | 4–15 min | 2,4 min (11 épocas) |
| Etapa 2, MLP, 3 sementes | 4–15 min (1 semente) | 6 min |
| −lex, etapas 1 e 2 | 2–6 min | 2 min |
| Degraus anteriores + bootstraps | ~5 min | ~9 min |
| **Total** | 15–40 min | **20 min** |

#### Decisões pedidas

| # | Pergunta | Minha proposta |
|---|---|---|
| Q10 | Aprovar o MLP como degrau (venceu nas 3 sementes, ganho de 0,6%)? | Aprovar pela regra, e registrar no relatório que o ganho é pequeno. |

#### Decisão (2026-10-02)

| # | Decisão |
|---|---|
| Q10 | **MLP aprovado pela regra.** Ganho **estatisticamente claro, praticamente irrelevante**: ΔMAE −0,0008 (0,6%). O critério da escada **não tinha efeito mínimo**. |
| Q11 | Rodar o controle "HLR linear treinada como o MLP" (ver §B3-controle). |
| Q11 | Rodar o controle "HLR linear com Adam, E\* = 8, lote 1024" para separar arquitetura de otimização? | Sim, antes da Fase C. Custa ~2 min. Muda qual modelo a Fase C deve usar: se o controle empatar com o MLP, a HLR com Adam é mais simples e interpretável. |
| Q12 | Modelo da Fase C | Decidir depois do Q11. |

Status: **aprovado (Q10).**

### B3-controle — HLR linear treinada como o MLP (Q11)

Código: `arch: 'linear'` em `src/model/mlp.js`, `src/model/b3c.js`, `src/model/b3c-diag.js`. Bruto: `data/b3c-results.json`, `data/b3c-diag.json`. Rodada de 2026-10-02.

#### O que é

| Item | Controle | MLP (B3) |
|---|---|---|
| Modelo | log2 ĥ = Θ·x, corte duro em [15 min; 274 d] (= `hclip` da HLR) | 10 → 32 ReLU → 1, corte suave |
| Features | √(1+acertos), √(1+erros), bias, peso escalar por lexema | as mesmas + embedding 8 |
| Início | tudo em 0 (como a HLR) | Glorot; embedding N(0; 0,05) |
| Treino | **igual**: Adam 0,001, lote 1024, loss (p − p̂)² + 0,01·(h − ĥ)², early stopping pela MAE em 81–90% (até 30, paciência 3), etapa 2 com sementes 0, 1, 2 | |
| Parâmetros | 12 768 (−lex: 3) | 102 505 (−lex: 129) |

#### Testes e mutação (`test/control.test.mjs`)

| Teste | O que confere |
|---|---|
| corte duro | z = ±100 → limites; z = 1,5 → 1,5 |
| gradiente | `tf.grad` da loss do controle em z = `dLossDdp` da HLR (B2), 3 pontos dentro do corte |
| 1 passo de Adam à mão | g ≈ −4,025 → todo parâmetro anda +0,001·\|g\|/(\|g\| + 10⁻⁷); `<unk>` fica 0 |

| Mutação | Teste que falhou |
|---|---|
| sem corte duro | corte duro |
| controle ignora o lexema | passo de Adam (L₁ fica 0) |
| passo com sinal trocado | passo de Adam |

As 3 revertidas; 53 de 53 passando.

#### Resultado pelo protocolo aprovado: **inválido**

| Modelo | E\* | MAE val em E\* | MAE teste (mediana) | AUC | COR_h |
|---|---|---|---|---|---|
| Controle | **2** | 0,13191 | 0,1325 | 0,5305 | **−0,142** |
| Controle −lex | **3** | 0,13039 | 0,1312 | 0,5294 | **−0,151** |
| MLP | 8 | 0,12665 | 0,1275 | 0,5378 | 0,212 |
| MLP −lex | 8 | 0,12693 | 0,1276 | 0,5368 | 0,167 |

O MLP retreinado reproduz a B3 **bit a bit** nas 3 sementes, com e sem lexema.

| Comparação (3 sementes) | Δ MAE | Δ AUC | Veredito |
|---|---|---|---|
| MLP × controle | −0,0050 a −0,0052 | +0,007 a +0,008 | MLP venceu nas 3 |
| MLP −lex × controle −lex | −0,0034 a −0,0036 | +0,007 | MLP venceu nas 3 |
| Controle × HLR (B2) | +0,0042 a +0,0043 | −0,006 | não venceu |
| Controle −lex × HLR −lex (B2) | +0,0025 a +0,0028 | −0,007 | não venceu |

**Por que é inválido** [V: curvas]:

- O controle começa com bias 0 (ĥ = 1 dia). O Adam move cada peso ~0,001 por passo, e o bias precisa chegar a ~7.
- Nesse trajeto, a loss cai sem parar (364 → 233), mas a MAE de validação **piora** depois da época 2.
- O early stopping pela MAE para no vale do transitório: E\* = 2 e 3.
- Sintoma: COR_h negativo (−0,14), o mesmo padrão da HLR sem o termo de h. A meia-vida não chegou a ser aprendida.
- O MLP não sofre disso: a cabeça suave começa no meio do intervalo, e a escala da sigmoide (×14,7) amplia cada passo.

Logo, **as linhas acima não respondem ao Q11.** O "MLP venceu o controle" é um artefato de convergência.

#### Diagnóstico (fora do veredito): 30 épocas fixas, só validação, semente 0

O teste não foi tocado.

| Época | Controle: MAE val | bias | Controle −lex: MAE val | bias |
|---|---|---|---|---|
| 2 | 0,13191 | 1,88 | 0,13178 | 2,00 |
| 5 | 0,13412 | 3,72 | 0,13105 | 3,47 |
| 8 | 0,13048 | 6,07 | 0,13322 | 6,05 |
| 10 | 0,12782 | 6,76 | 0,12792 | 7,39 |
| 12 | 0,12728 | 6,87 | 0,12715 | 7,61 |
| 16 | 0,12710 | 6,93 | 0,12703 | 7,64 |
| 20 | 0,12714 | 6,96 | 0,12713 | 7,64 |
| 25 | 0,12703 | 6,97 | 0,12704 | 7,64 |
| 30 | 0,12722 | 6,98 | 0,12724 | 7,64 |

| Na validação (épocas 12–30) | Faixa da MAE | Melhor |
|---|---|---|
| Controle convergido | 0,1270–0,1273 | 0,12703 |
| Controle −lex convergido | 0,1270–0,1273 | 0,12703 |
| MLP (B3, épocas 2–11) | 0,1266–0,1272 | 0,12665 |
| MLP −lex (B3, épocas 2–11) | 0,1269–0,1274 | 0,12693 |

- Leitura [I]: convergido, o controle fica a ~0,0004 do MLP na validação. É da ordem do ruído entre épocas (±0,0002).
- Isso **sugere** que a maior parte do ganho da B3 sobre a HLR do código (0,0008) é **otimização**, não arquitetura.
- Não é prova: uma semente, só validação, e o número de épocas foi escolhido olhando a curva.
- Pesos do controle convergido: acertos ≈ −0,01, erros ≈ −0,14, bias 6,98 (ĥ base ≈ 126 dias). **Os acertos quase não pesam**, como na HLR da B2 (+0,012) [V: pesos].

#### Regra de efeito mínimo (2026-10-02)

Só para o futuro (Fase C, dado próprio). Para referência, nenhuma diferença da B3 chega a 0,005: MLP × HLR = −0,0008.

#### Decisões pedidas

| # | Pergunta | Minha proposta |
|---|---|---|
| Q11b | Refazer o controle de forma válida? | Sim, com o conserto **fixado agora, antes de rodar**: bias iniciado na média de log2 h-alvo (cortado) das linhas de treino da etapa 1. Nada mais muda (Adam, lote, α, early stopping pela MAE, paciência 3, 3 sementes). Não usa a validação. Custo ~3 min. |
| Q12 | Modelo da Fase C | Ver abaixo |

**Proposta para o Q12**

| Ponto | Proposta |
|---|---|
| Família | A cabeça log2 ĥ (HLR/MLP): p̂ cai com Δ por construção, e p̂ em 1, 3, 7, 16, 35, 75 dias sai direto de ĥ |
| Principal | **MLP (B3, semente mediana)**. É o topo aprovado da escada. A Fase C pergunta p̂ "em função do histórico de acertos", e os modelos lineares (HLR da B2 e controle) dão peso ≈ 0 aos acertos: pela forma, quase não conseguem mostrar essa dependência. |
| Sensibilidade | **HLR (B2)** ao lado, em toda tabela da Fase C. Se os dois discordarem em mais de 0,05 de p̂ em algum degrau, o degrau é marcado. |
| Ressalvas que viajam junto | degraus 16, 35, 75 = limite superior (H-sel); n de linhas por degrau; ganho do MLP < 0,005 (abaixo do efeito mínimo novo) |
| Dependência | Se o Q11b mostrar o controle empatado com o MLP, troco o principal pela HLR linear com Adam (mais simples, mesmo desempenho) e verifico antes se ela também dá peso ≈ 0 aos acertos. |

Status: §B3-controle escrita. Q12 adiado (2026-10-02).

#### Antes do Q11b (2026-10-02)

**1. Diff de `src/model/mlp.js` desde a9e3f42**

| Mudança | Efeito no MLP |
|---|---|
| `arch: 'mlp'` em `MLP_DEFAULTS`; ramo `arch === 'linear'` em `createMlp` e `zOf` | nenhum: o ramo só roda com `arch: 'linear'` |
| `hardLog2H` e `headOf(m)`; `lossFromZ` ganha o parâmetro `head` (padrão `boundedLog2H`) | nenhum: para o MLP, `headOf` devolve `boundedLog2H` |
| `trainEpoch` e `predictMlp` chamam `headOf(m)` | nenhum: mesma cabeça |

- Prova: o MLP retreinado no `b3c.js` reproduz a B3 **bit a bit** (3 sementes, ±lex).
- Por que fica: o controle precisa usar **o mesmo laço** de treino, a mesma loss, a mesma previsão e o mesmo early stopping que o MLP. Só assim a única diferença é o modelo. Uma cópia do laço em outro arquivo poderia divergir em silêncio.
- **Decisão: mantido, não revertido.**

**2. Como o MLP inicializa o bias de saída**

| Item | MLP (B3) | Controle (proposta do Q11b) |
|---|---|---|
| Bias de saída | `b2 = 0` (`mlp.js:58`) | média de log2 h-alvo na etapa 1 = **5,49** |
| ĥ inicial | z = W2·ReLU(W1·x + b1) + 0 ≠ 0. Depende da semente (Glorot): ĥ mediano **1,4 d** (semente 0, a que escolhe E\*), **48 d** (semente 1), **0,05 d** (semente 2) | **45 d**, igual para todo exemplo |
| Passo efetivo em log2 ĥ | cabeça sigmoide × 14,7: na origem, ×3,67 por unidade de z; e 32 unidades ocultas movem z juntas | ×1, um bias + 2 pesos + 1 peso por lexema |

[V: medido no início do treino, 200 mil linhas da etapa 1, 3 sementes, ±lex]

**Não é equivalente.** A proposta daria ao controle um início na média do dado (45 d), que o MLP da etapa 1 não tem (1,4 d).
No passo, a assimetria vai no sentido oposto: o MLP anda ~4× mais rápido em log2 ĥ.

Como igualar, sem mexer no que já foi aprovado:

| Opção | O que muda | Mexe no MLP? | Resolve a convergência do controle? |
|---|---|---|---|
| **(A) Início igual na média** | os dois começam com log2 ĥ = 5,49. MLP: `b2 = logit((5,49 − LO)/(HI − LO)) = 1,53`. Controle: bias = 5,49. | **sim** (só o `b2` inicial) | sim |
| (B) Controle no início do MLP | controle com bias = 0,48 (ĥ 1,4 d) | não | **não**: parte de quase o mesmo ponto da corrida inválida |
| (C) Aquecimento no early stopping para os dois | ex.: só contar paciência a partir da época 12 | não no modelo, sim no protocolo aprovado | sim, mas é um ajuste escolhido olhando a curva |
| (D) Aceitar a assimetria | controle começa em 5,49; MLP como está | não | sim |

**Proposta: (A), como um par novo** ("MLP-início-média" × "controle-início-média"), mesmo protocolo (MAE de validação, paciência 3, 3 sementes, ±lex).
- A B3 aprovada **não é sobrescrita**: o par só responde ao Q11.
- Custo: ~15 min (MLP ±lex: ~10; controle ±lex: ~3; bootstraps).
- Teste antes de rodar: com `b2` inicial = 1,53 e W2 = 0, a cabeça devolve log2 ĥ = 5,49 (conta à mão). Mais uma mutação.

**Parado aqui:** igualar exige mudança no MLP. O Q11b e a dependência parcial esperam essa decisão.

**Decisão (2026-10-02):** diff do `mlp.js` mantido; **opção (A) aprovada**, como par novo. Último controle do Q11; qualquer resultado é registrado como está.

### Q11b — par novo com início igual (opção A, 2026-10-02)

Código: opção `initLog2h` em `src/model/mlp.js` (sem ela, tudo como na B3), `src/model/b3c2.js`. Bruto: `data/b3c2-results.json`. A corrida inválida (§B3-controle) continua registrada acima.

#### Teste e mutação antes de rodar

| Teste (`test/control.test.mjs`) | Conta à mão |
|---|---|
| MLP com `b2` = 1,53 e W2 = 0 → log2 ĥ = 5,49 | u = (5,49 + 6,5850)/14,6831 = 0,82237; b2 = ln(0,82237/0,17763) = 1,5325 |
| Controle com bias 5,49 → log2 ĥ = 5,49 | direto |
| Sem a opção: `b2` = 0 e bias = 0 (B3 intacta) | — |

| Mutação | Resultado |
|---|---|
| `b2` sem logit (usa u = 0,822) | teste falha |
| LO e HI trocados | teste falha |

As 2 revertidas; 54 de 54 passando.

#### Início

Média de log2 h-alvo nas 810 072 linhas da etapa 1: **5,4921** → ĥ = 45,0 d. `b2` do MLP = **1,5335**.

ĥ inicial mediano por semente (etapa 1; a etapa 2 dá o mesmo), E\* e teste:

| Modelo | ĥ inicial (sem. 0 / 1 / 2), dias | E\* (sem. 0 / 1 / 2) | MAE teste (sem. 0 / 1 / 2) | MAE mediana | AUC | COR_h | Parâmetros |
|---|---|---|---|---|---|---|---|
| Controle | 45,0 / 45,0 / 45,0 | 8/6/6 | 0,12754 / 0,12741 / 0,12746 | 0,12746 | 0,5358 | 0,192 | 12768 |
| MLP | 40,1 / 177,5 / 1,107 | 4/3/6 | 0,12763 / 0,12715 / 0,12745 | 0,12745 | 0,5374 | 0,212 | 102505 |
| Controle −lex | 45,0 / 45,0 / 45,0 | 7/6/10 | 0,12747 / 0,12760 / 0,12753 | 0,12753 | 0,5360 | 0,162 | 3 |
| MLP −lex | 40,3 / 189,5 / 0,683 | 8/4/6 | 0,12767 / 0,12755 / 0,12765 | 0,12765 | 0,5371 | 0,168 | 129 |

- O E\* que decide é o da semente 0. Os das sementes 1 e 2 são diagnóstico (fora do veredito).
- O controle começa em 45 d em toda semente. O MLP começa centrado em 45 d no `b2`, mas o termo W2·ReLU(…) desloca: 40 d, 177–189 d, 0,7–1,1 d. Essa assimetria residual não some sem zerar W2, e zerar W2 mudaria o MLP além do aprovado.
- O E\* varia entre sementes (MLP: 3 a 6; controle: 6 a 8). A MAE de validação é um platô ruidoso depois da época ~4.

#### Veredito do Q11: MLP × controle, mesmo início, mesma semente

| Semente | Δ MAE (MLP − controle) | IC 95% | IC Δ AUC | Veredito |
|---|---|---|---|---|
| 0 | 0,00009 | [0,00006; 0,00012] | [0,0001; 0,0037] | não venceu |
| 1 | -0,00026 | [-0,00031; -0,00021] | [0,0003; 0,0051] | venceu |
| 2 | -0,00001 | [-0,00004; 0,00003] | [-0,0000; 0,0029] | não venceu |

| −lex, semente | Δ MAE | IC 95% | IC Δ AUC | Veredito |
|---|---|---|---|---|
| 0 | 0,00020 | [0,00015; 0,00024] | [0,0003; 0,0042] | não venceu |
| 1 | -0,00005 | [-0,00010; -0,00001] | [0,0000; 0,0024] | venceu |
| 2 | 0,00012 | [0,00008; 0,00016] | [-0,0000; 0,0025] | não venceu |

**Resultado: o MLP não vence o controle** (vence em 1 de 3 sementes, com e sem lexema; a regra pede 3 de 3).

- MAE mediana: MLP 0,12745 × controle 0,12746. Diferenças por semente de −0,00026 a +0,00020.
- O MLP ranqueia um pouco melhor: Δ AUC +0,001 a +0,002 (IC acima de zero em 2 sementes); COR_h 0,212 × 0,192.
- **Resposta ao Q11:** o ganho de MAE da B3 sobre a HLR do código (0,0008) vem do **treino** (Adam, épocas, lote, início), **não da arquitetura**.

#### Contra a HLR da B2 (informativo)

| Semente | Controle × HLR: Δ MAE | IC Δ AUC | Veredito | MLP × HLR: Δ MAE | Veredito |
|---|---|---|---|---|---|
| 0 | -0,00078 | [-0,0011; -0,0006] | venceu sem discriminar | -0,00069 | venceu |
| 1 | -0,00091 | [-0,0012; -0,0007] | venceu sem discriminar | -0,00118 | venceu |
| 2 | -0,00087 | [-0,0012; -0,0007] | venceu sem discriminar | -0,00088 | venceu |

O controle erra menos que a HLR do código, mas ranqueia um pouco pior (AUC −0,001, IC abaixo de zero): "venceu sem discriminar" nas 3 sementes.

### Dependência parcial (só validação)

Como: cada uma das 89 710 linhas da validação recebe (Δ, acertos, erros) da grade e mantém o seu lexema; a tabela mostra a **média** do recall previsto.
Modelos que **não viram** a validação: MLP e controle do par novo (etapa 1, semente 0, pesos da época E\*); HLR com o algoritmo da B2, refeita em 0–81%.

⚠ = degrau longo (16, 35, 75 d): **limite superior** (H-sel, §B1). Ali o dado é esparso (16–75 d = 9,2% das linhas) e selecionado.

#### MLP (par novo)

| Erros | Acertos | Δ 1 d | Δ 3 d | Δ 7 d | Δ 16 d ⚠ | Δ 35 d ⚠ | Δ 75 d ⚠ |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0,996 | 0,989 | 0,974 | 0,941 | 0,876 | 0,753 |
| 0 | 3 | 0,996 | 0,988 | 0,972 | 0,937 | 0,867 | 0,737 |
| 0 | 5 | 0,996 | 0,987 | 0,970 | 0,933 | 0,859 | 0,722 |
| 0 | 10 | 0,996 | 0,987 | 0,969 | 0,931 | 0,855 | 0,716 |
| 1 | 1 | 0,996 | 0,988 | 0,972 | 0,937 | 0,868 | 0,739 |
| 1 | 3 | 0,996 | 0,987 | 0,970 | 0,933 | 0,859 | 0,723 |
| 1 | 5 | 0,995 | 0,986 | 0,968 | 0,929 | 0,852 | 0,710 |
| 1 | 10 | 0,995 | 0,986 | 0,968 | 0,928 | 0,850 | 0,706 |
| 3 | 1 | 0,995 | 0,986 | 0,968 | 0,928 | 0,849 | 0,705 |
| 3 | 3 | 0,995 | 0,985 | 0,966 | 0,924 | 0,840 | 0,690 |
| 3 | 5 | 0,995 | 0,985 | 0,964 | 0,920 | 0,834 | 0,679 |
| 3 | 10 | 0,995 | 0,985 | 0,966 | 0,923 | 0,839 | 0,687 |

#### Controle (par novo)

| Erros | Acertos | Δ 1 d | Δ 3 d | Δ 7 d | Δ 16 d ⚠ | Δ 35 d ⚠ | Δ 75 d ⚠ |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0,996 | 0,987 | 0,971 | 0,935 | 0,863 | 0,730 |
| 0 | 3 | 0,996 | 0,987 | 0,971 | 0,935 | 0,863 | 0,729 |
| 0 | 5 | 0,996 | 0,987 | 0,971 | 0,935 | 0,863 | 0,729 |
| 0 | 10 | 0,996 | 0,987 | 0,971 | 0,934 | 0,862 | 0,728 |
| 1 | 1 | 0,996 | 0,987 | 0,970 | 0,933 | 0,859 | 0,722 |
| 1 | 3 | 0,996 | 0,987 | 0,970 | 0,932 | 0,858 | 0,721 |
| 1 | 5 | 0,996 | 0,987 | 0,970 | 0,932 | 0,858 | 0,720 |
| 1 | 10 | 0,996 | 0,987 | 0,970 | 0,932 | 0,857 | 0,719 |
| 3 | 1 | 0,995 | 0,986 | 0,968 | 0,929 | 0,852 | 0,709 |
| 3 | 3 | 0,995 | 0,986 | 0,968 | 0,929 | 0,851 | 0,708 |
| 3 | 5 | 0,995 | 0,986 | 0,968 | 0,929 | 0,851 | 0,708 |
| 3 | 10 | 0,995 | 0,986 | 0,968 | 0,928 | 0,850 | 0,707 |

#### HLR (B2, refeita em 0–81%)

| Erros | Acertos | Δ 1 d | Δ 3 d | Δ 7 d | Δ 16 d ⚠ | Δ 35 d ⚠ | Δ 75 d ⚠ |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 0,995 | 0,986 | 0,968 | 0,929 | 0,851 | 0,708 |
| 0 | 3 | 0,995 | 0,986 | 0,968 | 0,929 | 0,851 | 0,707 |
| 0 | 5 | 0,995 | 0,986 | 0,968 | 0,929 | 0,850 | 0,707 |
| 0 | 10 | 0,995 | 0,986 | 0,968 | 0,928 | 0,849 | 0,705 |
| 1 | 1 | 0,995 | 0,986 | 0,967 | 0,925 | 0,844 | 0,696 |
| 1 | 3 | 0,995 | 0,986 | 0,967 | 0,925 | 0,844 | 0,695 |
| 1 | 5 | 0,995 | 0,985 | 0,966 | 0,925 | 0,843 | 0,694 |
| 1 | 10 | 0,995 | 0,985 | 0,966 | 0,925 | 0,842 | 0,693 |
| 3 | 1 | 0,995 | 0,985 | 0,964 | 0,920 | 0,834 | 0,677 |
| 3 | 3 | 0,995 | 0,984 | 0,964 | 0,920 | 0,833 | 0,676 |
| 3 | 5 | 0,995 | 0,984 | 0,964 | 0,920 | 0,833 | 0,676 |
| 3 | 10 | 0,995 | 0,984 | 0,964 | 0,919 | 0,832 | 0,674 |

#### Direção dos efeitos

Contagem de fatias em que o recall previsto **sobe**, **desce** ou é **misto** ao aumentar acertos (1 → 3 → 5 → 10; 18 fatias = 6 Δ × 3 níveis de erros) e erros (0 → 1 → 3; 24 fatias = 6 Δ × 4 níveis de acertos). Tamanho do efeito com erros = 0 (acertos 1 → 10) e acertos = 1 (erros 0 → 3), em Δ = 16 / 75 d:

| Modelo | Acertos: direção | Acertos 1 → 10: efeito em 16 / 75 d | Erros: direção | Erros 0 → 3: efeito em 16 / 75 d |
|---|---|---|---|---|
| MLP | desce 12/18, misto 6/18 | -0,010 / -0,038 | desce 24/24 | -0,013 / -0,048 |
| Controle | desce 18/18 | -0,001 / -0,002 | desce 24/24 | -0,006 / -0,021 |
| HLR (B2) | desce 18/18 | -0,001 / -0,003 | desce 24/24 | -0,009 / -0,031 |

Leitura:

1. **Δ domina.** De 1 para 75 dias, o recall previsto cai de ~0,996 para ~0,68–0,75 nos três modelos.
2. **Erros: os três concordam.** Mais erros → recall menor, em todas as fatias.
3. **Acertos: os três dizem "não sobe".** Nos lineares, o efeito é praticamente nulo (≤ 0,003). No MLP, **desce** de forma visível nos degraus longos (−0,038 em 75 d, com erros = 0).
4. Isso contraria a intuição ("mais acertos, memória mais forte"). Hipótese **H-exp** [I]: no Duolingo, muitos acertos acumulados marcam palavras que o sistema continua mostrando, ou seja, palavras que ainda precisam de prática. A contagem também inclui repetições dentro da mesma lição. É seleção, como H-sel, mas pela exposição.
5. **Para a Fase C:** "recall em função do histórico de acertos" **não é robusto** entre modelos. A dependência de Δ e de erros é robusta.

### Proposta revista para o Q12

| Ponto | Proposta |
|---|---|
| Modelo de referência | **Controle (HLR linear com Adam, início na média)**: mesma MAE do MLP (0,12746 × 0,12745), 12 768 parâmetros contra 102 505, pesos legíveis (ĥ = 2^(Θ·x)), p̂ cai com Δ por construção. Pela regra de efeito mínimo (≥ 0,005), nenhum modelo da cabeça log2 ĥ é melhor que outro aqui. |
| Ressalva | O controle não é um degrau aprovado da escada (é um controle). Se a escada formal pesar, a alternativa é o MLP (B3), com a mesma tabela. |
| Como reportar na Fase C | As **três tabelas lado a lado** (controle, MLP, HLR B2). Conclusão só onde as três concordam (efeito de Δ, efeito de erros). Onde discordam (acertos), fica "não resolvido". |
| Marcação | Degrau em que os três diferem em mais de 0,05 de recall: marcado. Degraus 16/35/75: limite superior (H-sel). Todo degrau com n de linhas. |
| Escopo | Recall da população Duolingo 2013, de, en, es, fr, it, pt. **Hipótese** para o app, nunca regra (P8, P10). |

Status: **Q11b, dependência parcial e proposta do Q12 escritos. Parado antes do commit.**

