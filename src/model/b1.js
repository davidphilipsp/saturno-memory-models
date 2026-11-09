// B1 — LR. Amostra: TF.js (o degrau da escada) + réplica JS (conferência). Arquivo completo: réplica JS × Tabela 2.
// Uso: node src/model/b1.js [--skip-full]  → data/b1-results.json
import fs from 'node:fs';
import path from 'node:path';
import { DATA_DIR } from '../data/duolingo.js';
import { loadSample } from '../data/sample.js';
import { fromRows, loadFullColumns, indicesWhere } from '../data/columns.js';
import { evaluate } from './metrics.js';
import { leitner } from './baselines.js';
import { pairedBootstrap, aucDiffBootstrap, verdict } from './bootstrap.js';
import { trainLrJs, trainLrTf, predictLr, lrParams } from './lr.js';

const pick = (arr, idx) => Float64Array.from(idx, i => arr[i]);
const absErr = (p, phat) => Float64Array.from(p, (v, j) => Math.abs(v - phat[j]));

function timed(f) { const t = Date.now(); const r = f(); return [r, (Date.now() - t) / 1000]; }

async function onSample() {
  const cols = fromRows(await loadSample());
  const train = indicesWhere(cols, i => cols.train[i] === 1), test = indicesWhere(cols, i => cols.train[i] === 0);
  const p = pick(cols.p, test), h = pick(cols.h, test), users = pick(cols.user, test);
  let mean = 0; for (const i of train) mean += cols.p[i]; mean /= train.length;

  const lt = Array.from(test, i => leitner({ t: cols.t[i], right: cols.right[i], wrong: cols.wrong[i] }));
  const leit = { phat: Float64Array.from(lt, x => x.p), hhat: Float64Array.from(lt, x => x.h) };
  const constPhat = new Float64Array(test.length).fill(mean);

  const out = { trainRows: train.length, testRows: test.length, users: new Set(users).size, vocab: cols.vocabSize, models: {} };
  out.models.constant = { ...evaluate({ p, phat: constPhat, h, hhat: null }), c: mean };
  out.models.leitner = evaluate({ p, phat: leit.phat, h, hhat: leit.hhat });

  const runs = {};
  for (const [name, opts] of [['lr', {}], ['lr-lex', { omitLexemes: true }]]) {
    const [mTf, secTf] = timed(() => trainLrTf(cols, train, opts));
    const [mJs, secJs] = timed(() => trainLrJs(cols, train, opts));
    const phat = predictLr(mTf, cols, test), phatJs = predictLr(mJs, cols, test);
    let maxDiff = 0; for (let j = 0; j < phat.length; j++) maxDiff = Math.max(maxDiff, Math.abs(phat[j] - phatJs[j]));
    runs[name] = phat;
    out.models[name] = {
      ...evaluate({ p, phat, h, hhat: null }), trainSecondsTf: secTf, trainSecondsJs: secJs,
      params: lrParams(mTf, cols), weights: Object.fromEntries(['right', 'wrong', 'time', 'bias'].map((k, i) => [k, mTf.w[i]])),
      maxPredDiffTfVsJs: maxDiff, maeJs: evaluate({ p, phat: phatJs, h, hhat: null }).mae,
    };
  }
  // A escada: LR (com lexema, o padrão de exp:251) contra o degrau anterior (Leitner) e contra a constante
  const eLr = absErr(p, runs.lr);
  const vsLeitner = pairedBootstrap({ errNew: eLr, errPrev: absErr(p, leit.phat), groups: users });
  const vsConstant = pairedBootstrap({ errNew: eLr, errPrev: absErr(p, constPhat), groups: users });
  // Regra emendada em 2026-10-02 (depois da B2), reaplicada aqui: o flag usa o IC da diferença de AUC
  const aucVsLeitner = aucDiffBootstrap({ p, phatNew: runs.lr, phatPrev: leit.phat, groups: users });
  const aucVsConstant = aucDiffBootstrap({ p, phatNew: runs.lr, phatPrev: constPhat, groups: users });
  out.ladder = {
    vsLeitner: { ...vsLeitner, aucDiff: aucVsLeitner, verdict: verdict(vsLeitner, aucVsLeitner) },
    vsConstant: { ...vsConstant, aucDiff: aucVsConstant, verdict: verdict(vsConstant, aucVsConstant) },
  };
  return out;
}

async function onFull() {
  const t0 = Date.now();
  const cols = await loadFullColumns();
  const loadSeconds = (Date.now() - t0) / 1000;
  const train = indicesWhere(cols, i => cols.train[i] === 1), test = indicesWhere(cols, i => cols.train[i] === 0);
  const p = pick(cols.p, test), h = pick(cols.h, test);
  const out = { loadSeconds, trainRows: train.length, testRows: test.length, vocab: cols.vocabSize, models: {} };
  for (const [name, opts] of [['lr', {}], ['lr-lex', { omitLexemes: true }]]) {
    const [m, sec] = timed(() => trainLrJs(cols, train, opts));
    out.models[name] = { ...evaluate({ p, phat: predictLr(m, cols, test), h, hhat: null }), trainSecondsJs: sec };
  }
  return out;
}

const res = { sample: await onSample() };
if (!process.argv.includes('--skip-full')) res.full = await onFull();
fs.writeFileSync(path.join(DATA_DIR, 'b1-results.json'), JSON.stringify(res, null, 1));
const r4 = o => Object.fromEntries(Object.entries(o).filter(([, v]) => typeof v !== 'object' || v === null).map(([k, v]) => [k, typeof v === 'number' && !Number.isInteger(v) ? +v.toPrecision(5) : v]));
console.log('AMOSTRA'); console.table(Object.fromEntries(Object.entries(res.sample.models).map(([k, v]) => [k, r4(v)])));
console.log('pesos lr', res.sample.models.lr.weights, 'pesos lr-lex', res.sample.models['lr-lex'].weights);
console.log('escada'); console.table(Object.fromEntries(Object.entries(res.sample.ladder).map(([k, v]) => [k, r4(v)])));
for (const [k, v] of Object.entries(res.sample.ladder)) console.log(k, 'Δ AUC', JSON.stringify(r4(v.aucDiff)));
if (res.full) { console.log('ARQUIVO COMPLETO'); console.table(Object.fromEntries(Object.entries(res.full.models).map(([k, v]) => [k, r4(v)]))); }
