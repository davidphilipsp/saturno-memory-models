// B0 — baselines sem treino + investigação de O1 (de onde vem p̄ = 0,859 da Tabela 2).
// Uso: node src/model/b0.js  → data/b0-results.json
import fs from 'node:fs';
import path from 'node:path';
import { RAW_FILE, DATA_DIR, readRows, toInstance, splitPoint } from '../data/duolingo.js';
import { loadSample } from '../data/sample.js';
import { evaluate, mae } from './metrics.js';
import { leitner, pimsleur } from './baselines.js';

const TOTAL = 12854226;            // inv: rows
const SP = splitPoint(TOTAL);

function runBaselines(test, trainMean) {
  const p = Float64Array.from(test, r => r.p), h = Float64Array.from(test, r => r.h);
  const out = { constant: evaluate({ p, phat: new Float64Array(p.length).fill(trainMean), h, hhat: null }) };
  out.constant.c = trainMean;
  for (const [name, f] of [['leitner', leitner], ['pimsleur', pimsleur]]) {
    const pr = test.map(f);
    out[name] = evaluate({ p, phat: Float64Array.from(pr, x => x.p), h, hhat: Float64Array.from(pr, x => x.h) });
  }
  return out;
}

// 1) Arquivo completo: candidatos a p̄ e o teste do artigo inteiro
async function full() {
  const acc = { train: { n: 0, p: 0, pc: 0, sc: 0, ss: 0 }, test: { n: 0, p: 0, pc: 0, sc: 0, ss: 0 }, first1M: { n: 0, p: 0, pc: 0, sc: 0, ss: 0 }, all: { n: 0, p: 0, pc: 0, sc: 0, ss: 0 } };
  const test = [];
  for await (const row of readRows(RAW_FILE)) {
    const inst = toInstance(row);
    const parts = ['all', row._index < SP ? 'train' : 'test'];
    if (row._index < 1e6) parts.push('first1M');
    for (const k of parts) {
      const a = acc[k];
      a.n++; a.p += inst.pRaw; a.pc += inst.p; a.sc += inst.sessionRight; a.ss += inst.sessionSeen;
    }
    if (row._index >= SP) test.push({ p: inst.p, h: inst.h, t: inst.t, right: inst.right, wrong: inst.wrong });
  }
  const candidates = {};
  for (const [k, a] of Object.entries(acc)) {
    candidates[`${k}: média de p`] = a.p / a.n;
    candidates[`${k}: média de p cortado`] = a.pc / a.n;
    candidates[`${k}: Σ session_correct / Σ session_seen`] = a.sc / a.ss;
  }
  const P = Float64Array.from(test, r => r.p);
  const maeOf = c => mae(P, new Float64Array(P.length).fill(c));
  const o1 = Object.fromEntries(Object.entries(candidates).map(([k, c]) => [k, { c, testMae: maeOf(c) }]));
  o1['artigo: 0,859'] = { c: 0.859, testMae: maeOf(0.859) };
  // que constante dá MAE 0,175 no teste? (MAE de constante é linear por partes em c)
  const grid = [];
  for (let c = 0.80; c <= 0.95 + 1e-9; c += 0.001) grid.push([+c.toFixed(3), maeOf(c)]);
  return { o1, grid, testRows: test.length, baselines: runBaselines(test, acc.train.pc / acc.train.n) };
}

// 2) Amostra de 1 M: a referência para B1–B3
async function sample() {
  const rows = await loadSample();
  const train = rows.filter(r => r.split === 'train'), test = rows.filter(r => r.split === 'test');
  const mean = train.reduce((s, r) => s + r.p, 0) / train.length;
  return { trainRows: train.length, testRows: test.length, baselines: runBaselines(test, mean) };
}

const t0 = Date.now();
const res = { full: await full(), sample: await sample(), seconds: (Date.now() - t0) / 1000 };
fs.writeFileSync(path.join(DATA_DIR, 'b0-results.json'), JSON.stringify(res, null, 1));
const fmt = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, typeof v === 'number' ? +v.toFixed(4) : v]));
console.log('O1'); console.table(Object.fromEntries(Object.entries(res.full.o1).map(([k, v]) => [k, fmt(v)])));
console.log('full test', res.full.testRows); console.table(Object.fromEntries(Object.entries(res.full.baselines).map(([k, v]) => [k, fmt(v)])));
console.log('sample test', res.sample.testRows); console.table(Object.fromEntries(Object.entries(res.sample.baselines).map(([k, v]) => [k, fmt(v)])));
console.log('grid (c, MAE) perto de 0,175:', JSON.stringify(res.full.grid.filter(([, m]) => Math.abs(m - 0.175) < 0.003)));
console.log('seconds', res.seconds);
