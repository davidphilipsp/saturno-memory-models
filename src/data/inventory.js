// Inventário do CSV completo em uma passada de streaming. Saída: data/inventory.json + resumo no terminal.
import fs from 'node:fs';
import path from 'node:path';
import { RAW_FILE, DATA_DIR, readRows, parseLexeme, splitPoint } from './duolingo.js';

const DAY = 86400;
// Faixas de Δ alinhadas à escada do scheduler (1, 3, 7, 16, 35, 75 dias) + teto de meia-vida (274)
const DELTA_EDGES = [[1 / 24, '<1h'], [1, '1h–1d'], [3, '1–3d'], [7, '3–7d'], [16, '7–16d'], [35, '16–35d'],
  [75, '35–75d'], [274, '75–274d'], [Infinity, '>274d']];
const SEEN_EDGES = [[1, '1'], [2, '2'], [5, '3–5'], [10, '6–10'], [20, '11–20'], [50, '21–50'], [100, '51–100'], [Infinity, '>100']];
const P_EDGES = [[0, '0'], [0.25, '(0,.25]'], [0.5, '(.25,.5]'], [0.75, '(.5,.75]'], [0.9999999, '(.75,1)'], [Infinity, '1']];

const bucket = (edges, x) => edges.find(([e]) => x <= e)[1];
const inc = (o, k, n = 1) => { o[k] = (o[k] || 0) + n; };

export async function inventory(file = RAW_FILE) {
  const s = {
    file: path.basename(file), header: null, rows: 0, types: {},
    learning: {}, ui: {}, pair: {}, uiPtByLearning: {},
    p: {}, pSum: 0, pValues: {}, delta: {}, deltaZero: 0, deltaMaxDays: 0,
    seen: {}, seenMax: 0, seenSum: 0, sessionSeen: {}, sessionSeenMax: 0,
    anomalies: { correctGtSeen: 0, sessionCorrectGtSeen: 0, pOutOfRange: 0, pMismatchSession: 0 },
    tsMin: Infinity, tsMax: -Infinity, tsDecreases: 0,
    lexeme: { wildcardSurface: 0, wildcardAny: 0, noSlash: 0, posCounts: {} },
  };
  const users = new Set(), lexIds = new Map(), lexStrings = new Set();
  const tsByIndex = [];   // amostra do relógio ao longo do arquivo (a cada 100k)
  let prevTs = -Infinity, idTo1 = true;
  for await (const r of readRows(file)) {
    if (!s.header) s.header = Object.keys(r).filter(k => k !== '_index');
    s.rows++;
    for (const k of s.header) {
      const v = r[k], t = v === '' ? 'empty' : /^-?\d+$/.test(v) ? 'int' : /^-?\d*\.?\d+(e-?\d+)?$/i.test(v) ? 'float' : 'string';
      s.types[k] ??= {}; inc(s.types[k], t);
    }
    const p = Number(r.p_recall), d = Number(r.delta) / DAY, seen = Number(r.history_seen), ss = Number(r.session_seen);
    const ts = Number(r.timestamp);
    inc(s.learning, r.learning_language); inc(s.ui, r.ui_language); inc(s.pair, `${r.ui_language}->${r.learning_language}`);
    if (r.ui_language === 'pt') inc(s.uiPtByLearning, r.learning_language);
    inc(s.p, bucket(P_EDGES, p)); s.pSum += p;
    if (Object.keys(s.pValues).length < 5000 || s.pValues[r.p_recall]) inc(s.pValues, r.p_recall);
    if (p < 0 || p > 1) s.anomalies.pOutOfRange++;
    if (Math.abs(p - Number(r.session_correct) / ss) > 1e-6) s.anomalies.pMismatchSession++;
    if (d === 0) s.deltaZero++;
    inc(s.delta, bucket(DELTA_EDGES, d)); s.deltaMaxDays = Math.max(s.deltaMaxDays, d);
    inc(s.seen, bucket(SEEN_EDGES, seen)); s.seenMax = Math.max(s.seenMax, seen); s.seenSum += seen;
    inc(s.sessionSeen, bucket(SEEN_EDGES, ss)); s.sessionSeenMax = Math.max(s.sessionSeenMax, ss);
    if (Number(r.history_correct) > seen) s.anomalies.correctGtSeen++;
    if (Number(r.session_correct) > ss) s.anomalies.sessionCorrectGtSeen++;
    s.tsMin = Math.min(s.tsMin, ts); s.tsMax = Math.max(s.tsMax, ts);
    if (ts < prevTs) s.tsDecreases++;
    prevTs = ts;
    if (r._index % 100000 === 0) tsByIndex.push([r._index, ts]);
    users.add(r.user_id); lexStrings.add(r.lexeme_string);
    const prevLex = lexIds.get(r.lexeme_id);
    if (prevLex === undefined) lexIds.set(r.lexeme_id, r.lexeme_string); else if (prevLex !== r.lexeme_string) idTo1 = false;
    const ls = r.lexeme_string;
    if (!ls.includes('/')) s.lexeme.noSlash++;
    else {
      const lx = parseLexeme(ls);
      if (lx.surface.startsWith('<*')) s.lexeme.wildcardSurface++;
      inc(s.lexeme.posCounts, lx.pos);
    }
    if (ls.includes('<*')) s.lexeme.wildcardAny++;
  }
  s.pMean = s.pSum / s.rows; s.seenMean = s.seenSum / s.rows;
  s.users = users.size; s.lexemeIds = lexIds.size; s.lexemeStrings = lexStrings.size; s.lexemeIdIsOneToOne = idTo1;
  s.splitPoint = splitPoint(s.rows);
  s.tsByIndex = tsByIndex;
  s.pDistinctValues = Object.keys(s.pValues).length;
  s.pTopValues = Object.entries(s.pValues).sort((a, b) => b[1] - a[1]).slice(0, 12);
  delete s.pValues; delete s.pSum; delete s.seenSum;
  return s;
}

// Segunda passada barata: timestamps do treino × teste (o split é temporal?) e exemplos de lexema
export async function splitClock(file, total) {
  const sp = splitPoint(total);
  let trainMax = -Infinity, testMin = Infinity, testBeforeTrainMax = 0, trainMaxTs = null;
  const examples = {};
  for await (const r of readRows(file)) {
    const ts = Number(r.timestamp);
    if (r._index < sp) trainMax = Math.max(trainMax, ts);
    else { testMin = Math.min(testMin, ts); if (ts < trainMax) testBeforeTrainMax++; }
    examples[r.learning_language] ??= new Set();
    if (examples[r.learning_language].size < 6) examples[r.learning_language].add(r.lexeme_string);
  }
  return { splitPoint: sp, trainMaxTs: trainMax, testMinTs: testMin, testRowsBeforeTrainMax: testBeforeTrainMax,
    lexemeExamples: Object.fromEntries(Object.entries(examples).map(([k, v]) => [k, [...v]])) };
}

if (process.argv[1] === import.meta.filename) {
  const t0 = Date.now();
  const s = await inventory();
  s.seconds = (Date.now() - t0) / 1000;
  Object.assign(s, await splitClock(RAW_FILE, s.rows));
  s.secondsTotal = (Date.now() - t0) / 1000;
  fs.writeFileSync(path.join(DATA_DIR, 'inventory.json'), JSON.stringify(s, null, 1));
  console.log(JSON.stringify({ rows: s.rows, users: s.users, lexemeIds: s.lexemeIds, seconds: s.secondsTotal }));
}
