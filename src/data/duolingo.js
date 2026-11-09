// Loader do dataset Settles & Meeder (2016) — espelha read_data() de
// github.com/duolingo/halflife-regression experiment.py:209-258 (commit 0041df0).
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import zlib from 'node:zlib';

export const DATA_DIR = path.resolve(import.meta.dirname, '../../data');
export const RAW_FILE = path.join(DATA_DIR, 'settles.acl16.learning_traces.13m.csv.gz');

// experiment.py:20-21
export const MIN_HALF_LIFE = 15 / (24 * 60);   // 15 minutos, em dias
export const MAX_HALF_LIFE = 274;              // 9 meses, em dias
export const TRAIN_FRACTION = 0.9;             // experiment.py:257

export const COLUMNS = ['p_recall', 'timestamp', 'delta', 'user_id', 'learning_language', 'ui_language',
  'lexeme_id', 'lexeme_string', 'history_seen', 'history_correct', 'session_seen', 'session_correct'];

export const pclip = p => Math.min(Math.max(p, 0.0001), 0.9999);                // experiment.py:175-177
export const hclip = h => Math.min(Math.max(h, MIN_HALF_LIFE), MAX_HALF_LIFE);  // experiment.py:180-182

// CSV mínimo: vírgula, aspas duplas opcionais com "" de escape
export function splitCsv(line) {
  if (!line.includes('"')) return line.split(',');
  const out = [];
  let cur = '', q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') q = false;
      else cur += c;
    } else if (c === '"') q = true;
    else if (c === ',') { out.push(cur); cur = ''; }
    else cur += c;
  }
  out.push(cur);
  return out;
}

// Linhas cruas, em streaming; .gz descomprime no caminho. Nunca carrega o arquivo inteiro.
export async function* readRows(file, { maxLines = Infinity } = {}) {
  let input = fs.createReadStream(file);
  if (file.endsWith('.gz')) input = input.pipe(zlib.createGunzip());
  const rl = readline.createInterface({ input, crlfDelay: Infinity });
  let header = null, i = 0;
  for await (const line of rl) {
    if (!line) continue;
    const cells = splitCsv(line);
    if (!header) { header = cells; continue; }
    if (i >= maxLines) break;
    const row = {};
    for (let k = 0; k < header.length; k++) row[header[k]] = cells[k];
    row._index = i++;
    yield row;
  }
  rl.close();
  input.destroy();
}

// "surface-form/lemma<pos>[<modifiers>...]" — README.md, seção "Data Set and Format"
export function parseLexeme(s) {
  const slash = s.indexOf('/');
  const surface = s.slice(0, slash);
  const rest = s.slice(slash + 1);
  const lt = rest.indexOf('<');
  const lemma = lt === -1 ? rest : rest.slice(0, lt);
  const tags = lt === -1 ? [] : rest.slice(lt + 1, -1).split('><');
  return { surface, lemma, pos: tags[0] ?? '', modifiers: tags.slice(1) };
}

// Uma linha → instância com os mesmos campos derivados de experiment.py:221-233
export function toInstance(row) {
  const p = pclip(Number(row.p_recall));
  const t = Number(row.delta) / (60 * 60 * 24);
  const h = hclip(-t / Math.log2(p));
  const seen = Number(row.history_seen);
  const right = Number(row.history_correct);
  const sessionSeen = Number(row.session_seen);
  const sessionRight = Number(row.session_correct);
  return {
    index: row._index,
    p, t, h,
    pRaw: Number(row.p_recall),
    seen, right, wrong: seen - right,
    sessionSeen, sessionRight,
    lang: `${row.ui_language}->${row.learning_language}`,
    learning: row.learning_language,
    ui: row.ui_language,
    lexemeId: row.lexeme_id,
    lexeme: row.lexeme_string,
    lexemeKey: `${row.learning_language}:${row.lexeme_string}`,  // experiment.py:252
    userId: row.user_id,
    ts: Number(row.timestamp),
  };
}

// Features densas por método — experiment.py:237-250. O lexema vai à parte (lexemeKey → vocab).
export function denseFeatures(inst, method) {
  if (method === 'leitner') return [inst.right - inst.wrong];         // :238
  if (method === 'pimsleur') return [inst.right + inst.wrong];        // :240
  const f = [Math.sqrt(1 + inst.right), Math.sqrt(1 + inst.wrong)];   // :244-245
  if (method === 'lr') f.push(inst.t);                                // :247-248
  f.push(1);                                                          // bias, :249-250
  return f;
}

// Split do artigo: ordem do arquivo, primeiros 90% treino (experiment.py:257-258)
export const splitPoint = total => Math.floor(TRAIN_FRACTION * total);
export const partitionOf = (index, total) => (index < splitPoint(total) ? 'train' : 'test');

// Hash determinístico de inteiro → [0,1). Amostra estável sem guardar estado.
export function unitHash(i, seed = 0) {
  let x = (i ^ Math.imul(seed + 0x9e3779b9, 0x85ebca6b)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

// Amostra que não quebra o split: a partição vem do índice no arquivo completo;
// a amostragem é Bernoulli(rate) por índice, independente da partição.
export async function* sampleInstances(file, { total, rate, seed = 0, maxLines } = {}) {
  for await (const row of readRows(file, { maxLines })) {
    if (unitHash(row._index, seed) >= rate) continue;
    const inst = toInstance(row);
    inst.split = partitionOf(row._index, total);
    yield inst;
  }
}

// Vocabulário de lexemas só com o treino (o teste pode ter lexema novo → índice 0, "desconhecido")
export function buildVocab(instances) {
  const vocab = new Map([['<unk>', 0]]);
  for (const inst of instances) if (inst.split !== 'test' && !vocab.has(inst.lexemeKey)) vocab.set(inst.lexemeKey, vocab.size);
  return vocab;
}

export async function countLines(file) {
  let n = 0;
  for await (const _ of readRows(file)) n++;
  return n;
}
