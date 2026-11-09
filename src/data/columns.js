// Dado em colunas (typed arrays) para treino e avaliação. Duas fontes:
//   fromRows(rows)   — amostra já em memória (loadSample)
//   loadFullColumns() — o CSV completo em streaming, sem objetos por linha
import { RAW_FILE, readRows, toInstance, partitionOf } from './duolingo.js';

const TOTAL = 12854226;   // inv: rows

function alloc(n) {
  return {
    n, p: new Float64Array(n), t: new Float64Array(n), h: new Float64Array(n),
    right: new Int32Array(n), wrong: new Int32Array(n), lex: new Int32Array(n), user: new Int32Array(n),
    train: new Uint8Array(n), vocabSize: 1, users: 0,
  };
}

// O vocabulário de lexemas sai só do treino; lexema novo no teste → 0 (peso nunca treinado = 0, como o defaultdict de exp:40)
function filler(cols) {
  const vocab = new Map([['<unk>', 0]]), users = new Map();
  return (i, inst, isTrain) => {
    cols.p[i] = inst.p; cols.t[i] = inst.t; cols.h[i] = inst.h;
    cols.right[i] = inst.right; cols.wrong[i] = inst.wrong;
    if (isTrain && !vocab.has(inst.lexemeKey)) vocab.set(inst.lexemeKey, vocab.size);
    cols.lex[i] = vocab.get(inst.lexemeKey) ?? 0;
    if (!users.has(inst.userId)) users.set(inst.userId, users.size);
    cols.user[i] = users.get(inst.userId);
    cols.train[i] = isTrain ? 1 : 0;
    cols.vocabSize = vocab.size; cols.users = users.size;
  };
}

// Duas passadas para o vocabulário não depender da ordem treino/teste
export function fromRows(rows) {
  const cols = alloc(rows.length), fill = filler(cols);
  rows.forEach((r, i) => { if (r.split === 'train') fill(i, r, true); });
  rows.forEach((r, i) => { if (r.split !== 'train') fill(i, r, false); });
  return cols;
}

// No arquivo, todo treino vem antes de todo teste (split por índice), então uma passada basta
export async function loadFullColumns(file = RAW_FILE, total = TOTAL) {
  const cols = alloc(total), fill = filler(cols);
  for await (const row of readRows(file)) {
    const inst = toInstance(row);
    fill(row._index, inst, partitionOf(row._index, total) === 'train');
  }
  return cols;
}

export const indicesWhere = (cols, pred) => {
  const out = [];
  for (let i = 0; i < cols.n; i++) if (pred(i)) out.push(i);
  return Int32Array.from(out);
};
