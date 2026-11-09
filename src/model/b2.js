// B2 — HLR. Amostra: TF.js (o degrau) + réplica JS + variantes. Arquivo completo: as 4 variantes da Tabela 2 (JS).
// Escada: decide contra o Leitner (último degrau aprovado); reporta contra LR e constante (decisão de 2026-10-01).
// Uso: node src/model/b2.js [--skip-full]  → data/b2-results.json
import fs from 'node:fs';
import path from 'node:path';
import { DATA_DIR } from '../data/duolingo.js';
import { loadSample } from '../data/sample.js';
import { fromRows, loadFullColumns, indicesWhere } from '../data/columns.js';
import { evaluate } from './metrics.js';
import { leitner } from './baselines.js';
import { pairedBootstrap, aucDiffBootstrap, verdict } from './bootstrap.js';
import { trainLrTf, predictLr } from './lr.js';
import { trainHlrJs, trainHlrTf, predictHlr, hlrParams } from './hlr.js';

const pick = (arr, idx) => Float64Array.from(idx, i => arr[i]);
const absErr = (p, phat) => Float64Array.from(p, (v, j) => Math.abs(v - phat[j]));
function timed(f) { const t = Date.now(); const r = f(); return [r, (Date.now() - t) / 1000]; }
const VARIANTS = [['hlr', {}], ['hlr-lex', { omitLexemes: true }], ['hlr-h', { omitHTerm: true }], ['hlr-lex-h', { omitLexemes: true, omitHTerm: true }]];

async function onSample() {
  const cols = fromRows(await loadSample());
  const train = indicesWhere(cols, i => cols.train[i] === 1), test = indicesWhere(cols, i => cols.train[i] === 0);
  const p = pick(cols.p, test), h = pick(cols.h, test), users = pick(cols.user, test);
  let mean = 0; for (const i of train) mean += cols.p[i]; mean /= train.length;

  const lt = Array.from(test, i => leitner({ t: cols.t[i], right: cols.right[i], wrong: cols.wrong[i] }));
  const prev = {
    constant: new Float64Array(test.length).fill(mean),
    leitner: Float64Array.from(lt, x => x.p),
    lr: predictLr(trainLrTf(cols, train), cols, test),
  };
  const out = { trainRows: train.length, testRows: test.length, users: new Set(users).size, models: {} };
  out.models.constant = { ...evaluate({ p, phat: prev.constant, h, hhat: null }), c: mean };
  out.models.leitner = evaluate({ p, phat: prev.leitner, h, hhat: Float64Array.from(lt, x => x.h) });
  out.models.lr = evaluate({ p, phat: prev.lr, h, hhat: null });

  let hlrPhat;
  for (const [name, opts] of VARIANTS) {
    const [mJs, secJs] = timed(() => trainHlrJs(cols, train, opts));
    const js = predictHlr(mJs, cols, test);
    const row = { params: hlrParams(mJs, cols), trainSecondsJs: secJs };
    if (name === 'hlr' || name === 'hlr-lex') {           // as duas que importam vão também em TF.js
      const [mTf, secTf] = timed(() => trainHlrTf(cols, train, opts));
      const tfp = predictHlr(mTf, cols, test);
      let d = 0; for (let j = 0; j < test.length; j++) d = Math.max(d, Math.abs(tfp.phat[j] - js.phat[j]));
      Object.assign(row, evaluate({ p, phat: tfp.phat, h, hhat: tfp.hhat }), { trainSecondsTf: secTf, maxPredDiffTfVsJs: d,
        maeJs: evaluate({ p, phat: js.phat, h, hhat: js.hhat }).mae, weights: { right: mTf.w[0], wrong: mTf.w[1], bias: mTf.w[2] } });
      if (name === 'hlr') hlrPhat = tfp.phat;
    } else Object.assign(row, evaluate({ p, phat: js.phat, h, hhat: js.hhat }), { weights: { right: mJs.w[0], wrong: mJs.w[1], bias: mJs.w[2] } });
    out.models[name] = row;
  }
  const eH = absErr(p, hlrPhat);
  out.ladder = {};
  for (const k of ['leitner', 'lr', 'constant']) {
    const b = pairedBootstrap({ errNew: eH, errPrev: absErr(p, prev[k]), groups: users });
    const a = aucDiffBootstrap({ p, phatNew: hlrPhat, phatPrev: prev[k], groups: users });
    out.ladder[`vs ${k}`] = { ...b, aucDiff: a, verdict: verdict(b, a), decides: k === 'leitner' };
  }
  return out;
}

async function onFull() {
  const cols = await loadFullColumns();
  const train = indicesWhere(cols, i => cols.train[i] === 1), test = indicesWhere(cols, i => cols.train[i] === 0);
  const p = pick(cols.p, test), h = pick(cols.h, test);
  const out = { trainRows: train.length, testRows: test.length, models: {} };
  for (const [name, opts] of [...VARIANTS, ['hlr (taxa do artigo)', { paperRate: true }]]) {
    const [m, sec] = timed(() => trainHlrJs(cols, train, opts));
    const r = predictHlr(m, cols, test);
    out.models[name] = { ...evaluate({ p, phat: r.phat, h, hhat: r.hhat }), trainSecondsJs: sec, weights: { right: m.w[0], wrong: m.w[1], bias: m.w[2] } };
  }
  return out;
}

const res = { sample: await onSample() };
if (!process.argv.includes('--skip-full')) res.full = await onFull();
fs.writeFileSync(path.join(DATA_DIR, 'b2-results.json'), JSON.stringify(res, null, 1));
const r4 = o => Object.fromEntries(Object.entries(o).filter(([, v]) => typeof v !== 'object' || v === null)
  .map(([k, v]) => [k, typeof v === 'number' && !Number.isInteger(v) ? +v.toPrecision(4) : v]));
const show = (title, o) => { console.log(title); console.table(Object.fromEntries(Object.entries(o).map(([k, v]) => [k, r4(v)]))); };
show('AMOSTRA', res.sample.models);
for (const [k, v] of Object.entries(res.sample.models)) if (v.weights) console.log(k, JSON.stringify(v.weights));
show('ESCADA (HLR TF.js)', res.sample.ladder);
for (const [k, v] of Object.entries(res.sample.ladder)) console.log(k, 'Δ AUC', JSON.stringify(r4(v.aucDiff)));
if (res.full) { show('ARQUIVO COMPLETO', res.full.models); for (const [k, v] of Object.entries(res.full.models)) console.log(k, JSON.stringify(v.weights)); }
