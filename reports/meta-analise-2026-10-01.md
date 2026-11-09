# Meta-análise do instrutor-idiomas — 2026-10-01

Somente leitura. Único arquivo escrito: este.

## Legenda

| Sigla | Caminho |
|---|---|
| `js/` | `squads/language-instructor-squad/tools/study-app/js/` |
| `ex/` | `squads/language-instructor-squad/tools/study-app/js/exercises/` |
| `content/` | `squads/language-instructor-squad/tools/study-app/content/` |
| `sq/` | `squads/language-instructor-squad/` |
| `srv` | `bin/sweep-server.py` |

| Etiqueta | Significado |
|---|---|
| [V arq:linha] | VERIFICADO no arquivo |
| [I: motivo] | INFERIDO |
| [NV] | NÃO VERIFICADO |

---

## Fase 1 — Inventário

### 1.1 Árvore (sem node_modules)

| Pasta | Conteúdo | Etiqueta |
|---|---|---|
| `bin/` | servidor, backup, vault push/pull, 2 plists, transcribe.sh, check-es-m2.py | [V `ls bin`] |
| `state/` | `progress/` (ja, fr, ru .json) · `tracks/` (5 .md) · `sessions/` · `reviews/` (`_inbox/`, `_processed/`) | [V `git ls-files state`] |
| `sq/` | agents (8), tasks (18), workflows (4), data (4 yaml), config (3), checklists (2), templates (4), tools | [V `ls -R`] |
| `content/` | ja, fr, ru (units), ja/fr/es (nexos), ja/fr/es/en (passages), ru/sweeps, sky | [V `find -type d`] |
| `media/` | probes, whisper-models, voice-dataset (fora do git) | [V `.gitignore:7`] |
| `ml/saturno/` | package.json (tfjs), `src/data/events.js`, `test/smoke.js`, `data/events.jsonl` (pessoal, gitignored, não publicado), `src/model` e `src/serve` vazios | [V `ls`, `.gitignore:19-20`] |
| `docs/` | prd (1 arquivo), stories (epic-01 + 4 stories Draft), guides, research | [V `ls docs`] |

### 1.2 Módulos JS do app — contagem real: 27 (16 em `js/` + 11 em `ex/`)

| Módulo | Propósito | Etiqueta |
|---|---|---|
| `main.js` (1147) | boot, Home, roteamento de todas as atividades, gravações | [V js/main.js:1-12] |
| `scheduler.js` (351) | SRS: escada, schedule, decay, teto diário, load/save | [V js/scheduler.js:1-27] |
| `session.js` (183) | monta fila diária (dívida → novo, aquecimento) | [V js/session.js:1-4] |
| `friction.js` (153) | sinais F1–F4 | [V js/friction.js:1-23] |
| `report.js` (170) | patch de tracks, log de sessão, milestone, degrau | [V js/report.js:1-10] |
| `corpus.js` (278) | carrega units/passages/nexos/sweeps/probe/conversa | [V js/corpus.js:1-7] |
| `speech.js` (244) | TTS, Web Speech, gravação + `/transcribe` | [V js/speech.js:1-5,200-244] |
| `text-match.js` (126) | normalização, Levenshtein, romaji/cirílico | [V js/text-match.js:1-3,119-126] |
| `ui.js` (91) | troca de tela, feedback não-modal | [V js/ui.js:1-2] |
| `sound.js` (36) | chime de acerto (Web Audio) | [V js/sound.js:1-4] |
| `home-saturn.js` (533) | Home orbital (luas = idiomas) | [V js/home-saturn.js:1-6] |
| `saturn-geometry.js` (229) | geometria do Saturno | [V js/saturn-geometry.js:1-6] |
| `backdrop.js` (67) | fundo de anéis | [V js/backdrop.js:1-6] |
| `sky.js` (205) | mapa do céu por idioma | [V js/sky.js:1-6] |
| `sky-geometry.js` (60) | projeção de estrelas | [V js/sky-geometry.js:1-6] |
| `ex/context.js` … `ex/corrija.js` | 11 exercícios (ver Fase 2) | [V `ls js/exercises`] |

### 1.3 Endpoints de `srv` (single-thread: `socketserver.TCPServer`)

[V srv:700] — requisições são serializadas; não há corrida dentro do processo.

| Método | Rota | Lê | Escreve | Validação / allowlist | Etiqueta |
|---|---|---|---|---|---|
| GET | `/state/progress/{lang}` | progress/{lang}.json | — | `^[a-z]{2}$`; JSON corrompido → 500 | [V srv:214-229] |
| POST | `/state/progress/{lang}` | rev atual | arquivo INTEIRO, atômico (tmp+replace) | exige `items` objeto; `rev` deve bater senão 409; rev+1 | [V srv:231-281] |
| GET | `/state/tracks` | frontmatter dos 5 tracks | — | parser próprio, 1 nível | [V srv:283-334] |
| POST | `/state/tracks/{lang}` | track .md | só linhas das chaves enviadas, atômico | `TRACK_WRITABLE_KEYS` (17 chaves); chave inexistente → 409; `\n` no valor → rejeita | [V srv:47-67,390-484] |
| POST | `/state/sessions/{date}` | sessão .md | append de 1 linha (reescreve arquivo via tmp) | data `YYYY-MM-DD`, `lang`, `activity` obrigatórios | [V srv:486-527] |
| POST | `/save` | — | `_inbox/{ts}_{batchId}.json` | JSON ≤2MB; kind=sweep retira lote do manifest | [V srv:157-206] |
| POST | `/save-audio` | — | `media/probes/` | ≤25MB | [V srv:573-599] |
| POST | `/transcribe` | — | clipe + `voice-dataset/{lang}/manifest.jsonl` | lang ∈ 5; ≤5MB; ffmpeg + whisper-cli (até 85 s de timeout somado) | [V srv:611-668] |
| GET | `/state/activity-counts` | _inbox + _processed | — | dedup por hash sem timestamp | [V srv:336-388] |
| GET/POST | `/sun/messages`, `/sun/message` | sun-conversation.jsonl | append | texto obrigatório | [V srv:529-571] |

- Docstring omite `/transcribe` e `/activity-counts` [V srv:3-20].
- Bind padrão 127.0.0.1; plist não define `STUDY_APP_HOST` [V srv:697; bin/com.instrutor-idiomas.server.plist].
- Acesso do iPhone via `tailscale serve --bg 8000` (proxy local) [V notas de sessão]; configuração ativa [NV].

### 1.4 Princípios, regras e agentes

| Artefato | Local | Etiqueta |
|---|---|---|
| P1–P10 | `sq/config/design-principles.md` linhas 6,18,25,42,52,65,74,82,89,95 | [V] |
| R1–R8 (+R9) | `sq/data/languages.yaml` linhas 88–109 (R9 em 98) | [V `grep -n "id: R"`] |
| F1–F5, decay, teto | `sq/data/pacing-rules.yaml:13-42` | [V] |

| Agente | Entrada | Saída declarada | Etiqueta |
|---|---|---|---|
| srs-scheduler | track_id, budget_minutes | `state/reviews/{track}.md` | [V sq/tasks/schedule-reviews.md:7-20] |
| progress-tracker | session_card, self_report (facil/na-medida/perdido) | `state/tracks/*.md` | [V sq/tasks/track-progress.md:7-20] |

- Algoritmo do scheduler: escada 1·3·7·16·35·75, lapso volta 2 degraus [V sq/agents/srs-scheduler-agent.md:23-27].
- Tracker observa F1–F5 [V sq/agents/progress-tracker-agent.md:25-30].
- Na prática, o app (não o agente) grava `state/progress/*.json` [V js/scheduler.js:24-27].

### 1.5 Testes

| Item | Situação | Etiqueta |
|---|---|---|
| App / servidor | nenhum teste | [V `find -name "*test*"`] |
| ml/saturno | só `test/smoke.js` (4 linhas, imprime tensor) | [V ml/saturno/test/smoke.js:1-4] |
| `npm test` | script padrão que falha | [V ml/saturno/package.json:7] |

---

## Fase 2 — Fluxo de dados por exercício

Abreviações: SRS = `state/progress/{lang}.json` · INB = `_inbox` · SES = `state/sessions/*.md` · TRK = `state/tracks/*.md`.

| Tipo | Arquivo:linha | Pistas ANTES da resposta | Captura | Nota | Confirmação do aprendiz? | Persistido | DESCARTADO | schedule()? |
|---|---|---|---|---|---|---|---|---|
| contexto | ex/context.js:19-50 | frase L2, kanji, leitura, áudio automático; tradução+emoji após 600 ms | botão Continuar | sempre 1 | não | SRS (via schedule) | tempo, replays | sim [V js/main.js:600-605] |
| cloze | ex/cloze.js:31-128 | **áudio da frase INTEIRA (inclui a resposta)**, tradução L1, emoji; banco opcional | texto digitado | answerMatches → 1; banco → 0,5; erro 0 | não | SRS | texto digitado, uso do banco, áudio ouvido, replays | sim |
| ditado | ex/ditado.js:21-134 | só áudio (normal/lento); banco opcional | texto digitado | Levenshtein: 0 → 1 (0,5 se banco); ≤20% → 0,5; senão 0; "Não sei" → 0 | não | SRS | texto, distância | sim — **fora da rotação** [V js/session.js:132-143] |
| parear | ex/parear.js:20-114 | palavra+emoji, significados | toques | 1; erro antes → 0,5 | não | SRS por lexema | sequência de toques, nº de erros | sim [V js/main.js:506-509,900-905] |
| fala | ex/fala.js:47-127 | frase, kanji, leitura, L1+emoji, áudio do modelo | áudio → Whisper (fallback Web Speech) | rel ≤0,25 → 1; "Continuar" → 0,5/0; "Li certo" → 1 | sim ("Li certo") | SRS; manifest só do áudio | **flag selfReported**, texto ouvido, rel | sim [V js/main.js:602] |
| shadow | ex/shadow.js:47-93 | frase completa, L1+emoji, áudio | botão Travei/Consegui (mic opcional) | 1/0 autorrelato | é só autorrelato | `progress.ladder.cleanStreak`, TRK `output_ladder_step` | nota por item, texto ouvido | **não** [V js/main.js:579-588] |
| fala RU (noSchedule) | js/session.js:153-162 | idem fala/shadow | idem | idem | idem | só `modesPassed[modality]++` | resto | **não** [V js/main.js:589-599] |
| leitura | ex/leitura.js:26-82 | texto L2, tradução por toque | botão Continuar | nenhuma | não | INB (EN/ES), SES, TRK `ultima_leitura` | **tentativas de tradução digitadas**, linhas reveladas | só flashcards (parear) JA/FR [V js/main.js:900-906] |
| varredura | ex/varredura.js:18-67 | palavra + **emoji** + áudio | botão R/N | autorrelato | não (sem desfazer) | INB itens; SRS só os R | N no SRS; tempo | **não** — `introduceRecognized` [V js/main.js:805-807] |
| nexo | ex/nexo.js:16-81 | par mínimo, pergunta, vocabulário | textarea | nenhuma (check-es-m2 externo) | não | INB `text` | o próprio "skip" | não |
| probe (lição) | ex/probe.js:28-216 | faixa, glossário, tom, pares de registro | chips + textarea | autorrelato travou/fluido | é autorrelato | INB rating, notes, text | notes quando passo 4 é pulado (ver Fase 6) | não |
| probe (áudio) | ex/probe.js:218-369 | prompt | MediaRecorder | autorrelato | é autorrelato | INB + `media/probes/` | — | não |
| conversa | js/main.js:706-730 + probe.js | prompt falado por TTS | MediaRecorder/texto | autorrelato | é autorrelato | INB, TRK `ultima_conversa` | — | não |
| Corrija | ex/corrija.js:28-89 | frase errada, L1, dica opcional | texto | canon igual a uma certa → 1 (0,5 com dica); **unchanged → 0** | não | INB `{id, grade, typed}` | uso da dica, flag unchanged | não |
| autorrelato | js/main.js:1068-1090 | — | 3 botões | — | — | SES + `progress.recentSelfReports` | — | não |
| externo (Duolingo etc.) | srv:344-349 | — | relato ao agente | — | — | INB kind=external (sem itens) | — | não |

Evidências-chave:
- Cloze toca a frase inteira, contradizendo o próprio comentário [V ex/cloze.js:36-37 vs 73-74].
- Nenhuma tentativa da fila SRS vai para o INB: `runMainQueue` só chama saveProgress/pushSessionLog [V js/main.js:541-627]; kinds no INB = external, maintenance-*, sweep, quiz, match [V script `volume.py`].

---

## Fase 3 — Verificação das afirmações

| # | Veredito | Evidência |
|---|---|---|
| C1 | **REFUTADA** | Outros escritores: `decay` [V js/scheduler.js:145-157] chamado em toda montagem de sessão, inclusive na prévia da Home [V js/session.js:45-46; js/main.js:221]; `introduceRecognized` [V js/scheduler.js:245-253]; `getOrCreateItem`→`newIntroducedToday` [V js/scheduler.js:261-275,323-328]; noSchedule `modesPassed++` [V js/main.js:593-597]; `ladderUpdate` [V js/report.js:105-119]; `commitFriction` [V js/friction.js:108-122]; `recordSelfReport` [V js/friction.js:127-139]; `recordSessionTiming` [V js/friction.js:145-153]; `sessionOpen` [V js/main.js:645,927]; `rev` no servidor [V srv:271]; edição manual (P9) [V sq/config/design-principles.md:89-93]. |
| C2 | **REFUTADA** | Jitter = `Math.random()` sem semente [V js/scheduler.js:231]. `today` padrão = data UTC do relógio [V js/scheduler.js:37-39]. Nudge depende de `recentGrades` [V js/scheduler.js:188-189]. Replay só reproduz se o intervalo sorteado for gravado no evento ou o jitter virar hash determinístico [I: dedução do código]. |
| C3 | **CONFIRMADA** (com nuance) | Janela = 4 [V js/scheduler.js:87,221-222]. Contadores vitalícios sobrevivem (vistas, lapsos, modes, modesPassed) [V js/scheduler.js:56-83]. Tentativas SRS não vão a nenhum log [V js/main.js:541-627]. Soma de `modes` = `vistas` [redigido: dado pessoal de estudo]. |
| C4 | **CONFIRMADA** | `unchanged` → `ok=false` → grade 0 [V ex/corrija.js:56-58]; resultado `{id, grade, typed}` sem flag [V ex/corrija.js:67]. Única rodada real confirmou o efeito [redigido: dado pessoal de estudo]. Recuperável comparando `typed` com `errada` enquanto o conteúdo não mudar [I: depende do conteúdo atual]. |
| C5 | **CONFIRMADA** | R/N, revela, avança em 900 ms, sem desfazer [V ex/varredura.js:16,48-62]. Emoji aparece antes do R/N [V ex/varredura.js:31]. Só R entra no SRS [V js/main.js:805-807]. Itens N nunca foram retestados [redigido: dado pessoal de estudo]. |
| C6 | **PARCIAL** | Reescreve o arquivo inteiro [V srv:276-279]. Proteção existe: `rev` otimista → 409 [V srv:252-271] + servidor single-thread [V srv:700]. Não há last-write-wins silencioso no disco, MAS o cliente perdedor só faz `console.warn` e segue sem recarregar; todas as gravações seguintes também falham [V js/scheduler.js:121-132]; `main.js` ignora o retorno [V js/main.js:603]. A Home grava progress ao renderizar [V js/session.js:46]: `rev` sobe sem mudar `vistas` [redigido: dado pessoal de estudo]. Perda silenciosa da sessão no outro aparelho [I: combinação dos dois]. |
| C7 | **PARCIAL** (sobrevive ao push; não ao pull) | Direção principal repo→vault: `rsync -a --delete` [V bin/vault-push.sh:10], diário às 22:00 via launchd [V bin/com.instrutor-idiomas.backup.plist]. Conflito no push: pula se o vault tiver arquivo mais novo [V bin/state-backup.sh:53-63]. Pull manual sem `-u` nem checagem [V bin/vault-pull.sh:15] → sobrescreve arquivo do repo com cópia mais velha do vault [I: semântica do rsync]. Um JSONL append-only perderia linhas num pull fora de hora [I]. Cópias de conflito do iCloud [NV]. |
| C8 | **PARCIAL** | Id de frase "estável pra sempre", sem duplicatas [V js/corpus.js:12; script]. Chave SRS = `lang:token` (forma de superfície) [V js/corpus.js:45-47]: editar o token cria item novo e órfão o antigo [I]. Nenhum campo de versão no conteúdo [V `grep version content/`]. Nenhuma frase editada sob o mesmo id no git [V script]. |
| C9 | **REFUTADA** | `selfReported: true` [V ex/fala.js:94] é descartado: só `grade`/`modality` vão a `schedule` [V js/main.js:602]. Manifest grava `expected/heard/lexId`, não a decisão [V srv:662-664]. Os 3 manifests estão com 0 bytes (mtime 2026-09-29 15:09) [V `ls -la media/voice-dataset`]; causa [NV]. |
| C10 | **CONFIRMADA** (lista) | `track.last_touch` (F3) [V js/friction.js:47]; `progress.reentry` [V :49,58]; `progress.sessionOpen` (F2) [V :73]; `friction.cooldownUntil` (F4) [V :83]; `friction.lightenNextSession` (F1) [V :90]. Fontes: last_touch ← report.js [V js/report.js:40,53,63] + manual; sessionOpen ← main.js [V :645,927]; cooldown ← autorrelato [V js/main.js:1080]; lighten ← média móvel 0,7/0,3 [V js/friction.js:152], não "média de 14 dias" [V sq/data/pacing-rules.yaml:29]. |
| C11 | **CONFIRMADA** (descrição) | Regex `fala:N`/`recall:N` [V js/report.js:27]; conta ITENS com `modesPassed[produce|recall] > 0` [V js/report.js:31]; arredonda 2 casas, teto 1. `modesPassed` sobe com qualquer grade ≥1, inclusive "Li certo" e "Consegui" do fallback [V js/scheduler.js:191-196; ex/fala.js:94,115] e pelo caminho noSchedule [V js/main.js:593-597]. EN/ES: `fluido_count` por autorrelato [V js/main.js:686-690]. Valores batem com progress [redigido: dado pessoal de estudo]. |
| C12 | **PARCIAL** (misto) | Commits de `state/` misturados em feat/fix; automático só a partir do backup diário [V bin/state-backup.sh]. Cadência irregular; poucos snapshots por revisão [redigido: dado pessoal de estudo]. |

---

## Fase 4 — Qualidade e volume do dado

### 4.1–4.3 Volume e inconsistências

> Redigido na versão pública: estas seções mediam o histórico pessoal de estudo do autor (contagens de sessões, tentativas por semana, arquivos de lote). Os dados não estão incluídos neste repositório.

### 4.4 Esquema e migrações

| Achado | Etiqueta |
|---|---|
| 15 campos por item, estável em todos os snapshots do git | [V script sobre `git show`] |
| JA: 2 itens sem `modesPassed/recentGrades/window*`; 5 sem `heldAtTier` | [V script] |
| RU: 18 sem `modesPassed`; 28 sem `window*` | [V script] |
| Migração é preguiçosa (inicializa no próximo schedule) | [V js/scheduler.js:171-184] |
| Campos de topo só em JA/FR até 09-26; RU ganhou `avgItemMs/friction/ladder…` em 09-26 | [V script] |
| INB: `maintenance-probe` tem 3 formatos (audioFile / notes+text / text) | [V script] |

---

## Fase 5 — Aderência da proposta

### (a) Log append-only `state/attempts/{lang}/{AAAA-MM}.jsonl`

| Aspecto | Achado | Etiqueta |
|---|---|---|
| Viável? | Sim. Servidor serializa requisições, append é seguro | [V srv:700] |
| Complica | Cliente descarta erro de rede sem fila de reenvio | [V ex/probe.js:54-56; js/main.js:342-344] |
| Complica | Reenvios já duplicam → precisa `attempt_id` | [V script] |
| Complica | vault-pull pode truncar o arquivo | [V bin/vault-pull.sh:15] |
| Complica | Data UTC × local | [V js/scheduler.js:37-39; srv:164] |
| Princípios | P9 pede "Markdown+frontmatter" — JSON já é exceção aceita (progress) | [V sq/config/design-principles.md:91; I] |
| Custo | srv (+1 endpoint), main.js (~8 pontos de gravação), 1 módulo novo de cliente | [I: contagem de call sites] |
| Mudar | `attempt_id` UUID + `ts_utc` + `local_date`; endpoint idempotente; vault-pull excluir `attempts/` ou usar `-u` | [I] |

### (b) Contrato único de tentativa

| Aspecto | Achado | Etiqueta |
|---|---|---|
| Viável? | Sim, com ajuste. Hoje são 7 assinaturas de retorno diferentes | [V ex/*.js; js/main.js:502,763,802,873] |
| Falta capturar | texto digitado (cloze/ditado), `selfReported`, unchanged, uso de dica/banco, replays, tempo por item | [V Fase 2] |
| Pistas | Emoji+L1 visíveis em cloze/fala/shadow; áudio da frase inteira no cloze | [V ex/cloze.js:62,73-74] |
| Princípios | Exibir nota bruta violaria P5 — o log não pode virar placar na UI | [V design-principles.md:60-61] |
| Custo | 11 módulos de exercício + main.js | [I] |
| Mudar | Registrar `cues` como lista do que ESTAVA na tela (não "o que ele viu"); `label_source ∈ {auto, self, override}`; "11 exercícios" inclui 3 sem nota (contexto, leitura, nexo) — marcar como exposição, não tentativa | [I] |

### (c) progress.json como projeção por replay

| Aspecto | Achado | Etiqueta |
|---|---|---|
| Viável como descrito? | **Não.** progress não é função só de tentativas | [V C1] |
| Bloqueia | jitter aleatório | [V js/scheduler.js:231] |
| Bloqueia | `decay` depende do dia em que o app foi ABERTO, inclusive prévia | [V js/session.js:45; js/main.js:221] |
| Bloqueia | estado de fricção, teto diário, `sessionOpen`, ladder | [V C1] |
| Bloqueia | histórico não existe: notas de cloze erradas pelo app e sem texto bruto | [V js/text-match.js:44-49] |
| Princípios | P9: edição manual de progress.json seria apagada no próximo replay | [V design-principles.md:92-93] |
| Custo | scheduler.js, session.js, friction.js, report.js, main.js; motor de replay novo | [I] |
| Mudar | Gênesis = snapshot atual do progress como evento; gravar o intervalo sorteado no evento; decay calculado na leitura (puro) em vez de mutar; evento `override` para edição manual; replay como VERIFICAÇÃO (diff), não fonte de escrita, até provar paridade | [I] |

### (d) Catálogo com ID estável e versão

| Aspecto | Achado | Etiqueta |
|---|---|---|
| Viável? | Sim. Id de frase já existe e não tem duplicata | [V js/corpus.js:12; script] |
| Complica | Chave SRS é o token, não um id de lexema | [V js/corpus.js:45-47] |
| Complica | Migrar as chaves existentes quebra o vínculo com histórico | [V script; I] |
| Custo | corpus.js + validador; migração só se trocar a chave | [I] |
| Mudar | Manter `lang:token` como chave; anexar `sentence_id` + hash do conteúdo em cada tentativa (versão implícita, sem campo manual) | [I] |

### (e) ML em modo sombra (TF.js)

| Aspecto | Achado | Etiqueta |
|---|---|---|
| Viável? | Infraestrutura sim (tfjs instalado) | [V ml/saturno/package.json:14] |
| Dado | volume semanal pequeno [redigido: dado pessoal de estudo]; rótulos de cloze inválidos até 09-28; Corrija com falsos 0 | [V Fase 4] |
| Dado | `events.js` lê só o INB → não vê nenhuma tentativa SRS | [V ml/saturno/src/data/events.js:19-26] |
| Dado | timestamp é do lote, não do item | [V ml/saturno/src/data/events.js:31] |
| Coerência | O próprio scheduler descarta HLR por falta de dado | [V js/scheduler.js:1-7] |
| Princípios | Sombra não fere P9 se não decide; exibir previsão como % fere P5 | [V design-principles.md:52-63] |
| Princípios | Sem story para isto (AIOX Art. III) | [V `ls docs/stories`] |
| Mudar | Adiar treino até ≥ ~2–3 meses de log limpo; começar com baseline não-neural (regressão logística / HLR) avaliado offline contra a escada | [I] |

---

## Fase 6 — Riscos e lacunas não pedidos

### 6.1 Bugs e perdas de dado

| # | Achado | Etiqueta |
|---|---|---|
| 1 | Cloze toca a frase inteira (com a resposta) antes de digitar — "recall" mede escuta | [V ex/cloze.js:36-37 × 73-74] |
| 2 | 409 → sessão inteira some em silêncio (só console) | [V js/scheduler.js:121-132] |
| 3 | Data em UTC em scheduler, friction, report, main: depois das 21:00 BRT "hoje" é amanhã | [V js/scheduler.js:37-39; js/friction.js:27-29; js/report.js:13-15; js/main.js:131-133] |
| 4 | `/transcribe` bloqueia o servidor single-thread por até ~85 s; o outro aparelho trava | [V srv:644-655,700] |
| 5 | Probe-lição: chips "travou" e tom se perdem se o passo 4 for pulado (`notesText` só é setado no clique do passo 4) | [V ex/probe.js:165,173-175] |
| 6 | Leitura: tentativas de tradução digitadas nunca são lidas | [V ex/leitura.js:55-56,80] |
| 7 | Voice-dataset: manifests com 0 bytes | [V `ls -la`] |
| 8 | Distratores do cloze = sempre os 3 primeiros itens do progress (comentário promete "tamanho parecido") | [V js/main.js:555-557 × ex/cloze.js:51] |
| 9 | `/save` nomeia por segundo + batchId: dois envios no mesmo segundo sobrescrevem | [V srv:164-169; I: colisão] |
| 10 | Corrija: itens próprios além de 4 são descartados mesmo com poucos genéricos | [V js/main.js:752-754] |
| 11 | `showDoneScreen` regrava `current.progress` antigo após leitura do mesmo idioma → 409 provável | [V js/main.js:644-646; I: rev desatualizado] |

### 6.2 Contradições comentário × código / doc × código

| # | Achado | Etiqueta |
|---|---|---|
| 1 | Lapso: agente/task dizem 2 degraus; RU usa 1 | [V sq/agents/srs-scheduler-agent.md:25; sq/tasks/schedule-reviews.md:42 × js/scheduler.js:169] |
| 2 | Task diz gravar `state/reviews/{track}.md`; app grava `state/progress/*.json` | [V sq/tasks/schedule-reviews.md:19 × js/scheduler.js:24] |
| 3 | Itens/min: task 8 (reconhecimento) e 3 (evocação); app 1,1 | [V sq/tasks/schedule-reviews.md:37 × js/session.js:10] |
| 4 | "Palavra solta não entra na fila" × palavras isoladas no progress RU | [V sq/agents/srs-scheduler-agent.md:21 × js/main.js:805-807] |
| 5 | F1 "média de 14 dias" × média móvel exponencial | [V pacing-rules.yaml:29 × js/friction.js:152] |
| 6 | P4 manda encurtar sessão na closing lane; app só zera item novo | [V design-principles.md:46-47 × js/session.js:65-83] |
| 7 | Card de closing aparece com progresso ≥1; sessão usa `<1` | [V js/main.js:403 × js/session.js:66] |
| 8 | ES `active-production` (R9) × app trata ES como maintenance | [V languages.yaml:98 × js/main.js:36] |
| 9 | speech.js diz que manda a frase como prompt; servidor não manda | [V js/speech.js:181 × srv:651-654] |
| 10 | `loadProbeConfig` diz "1 objeto"; é lista (en 4, es 5) | [V js/corpus.js:253 × script] |
| 11 | Referências a "PRD §7.3/§8.3/§13" sem arquivo correspondente | [V js/scheduler.js:2; js/session.js:1; `find -iname "*prd*"`] |
| 12 | Comentários citam `saturn.js`, que não existe | [V js/main.js:433; `ls js`] |

### 6.3 Código morto e acoplamentos

| # | Achado | Etiqueta |
|---|---|---|
| 1 | Ramo `lang === "ru"` de describeActivity é inalcançável | [V js/main.js:44,219,243-255] |
| 2 | `ditado` importado e registrado, fora da rotação | [V js/main.js:21,48; js/session.js:132-143] |
| 3 | `ruTrackUpdates` só é chamada pela varredura, que não roda mais | [V js/main.js:828] |
| 4 | Prévia da Home tem efeito colateral (decay + POST) — "prévia pura" só vale para fricção | [V js/session.js:45-46 × js/friction.js:19-23] |
| 5 | Contagem de atividade do Saturno depende do nome de arquivo do INB | [V srv:342-361] |

---

## Prova de somente-leitura

Ver `git status` ao final da sessão (impresso no terminal).
