// B3-controle (Q11, 2026-10-02): HLR linear treinada como o MLP. Separa arquitetura de otimização.
// Mesmo protocolo da B3: etapa 1 (semente 0) em 0–81% com early stopping pela MAE em 81–90% (até 30, paciência 3) → E* próprio;
// etapa 2 em 0–90% com sementes 0, 1, 2. ±lex. Compara com o MLP da mesma semente (retreinado: confere a reprodução da B3)
// e com a HLR da B2 (código). Uso: node src/model/b3c.js → data/b3c-results.json
import fs from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { DATA_DIR } from '../data/duolingo.js';
import { loadSample } from '../data/sample.js';
import { fromRows, indicesWhere } from '../data/columns.js';
import { evaluate, mae } from './metrics.js';
import { pairedBootstrap, aucDiffBootstrap, verdict } from './bootstrap.js';
import { trainHlrTf, predictHlr } from './hlr.js';
import { stageOf, EarlyStopper, createMlp, mlpParams, trainEpoch, predictMlp, disposeMlp, MLP_DEFAULTS } from './mlp.js';

const TOTAL = 12854226, MAX_EPOCHS = 30, PATIENCE = 3, SEEDS = [0, 1, 2];
const pick = (arr, idx) => Float64Array.from(idx, i => arr[i]);
const absErr = (p, phat) => Float64Array.from(p, (v, j) => Math.abs(v - phat[j]));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

await tf.ready();
const rows = await loadSample();
const cols1 = fromRows(rows.map(r => ({ ...r, split: stageOf(r.index, TOTAL) === 'step1' ? 'train' : 'held' })));
const step1 = indicesWhere(cols1, i => cols1.train[i] === 1);
const val = Int32Array.from(rows.flatMap((r, i) => (stageOf(r.index, TOTAL) === 'val' ? [i] : [])));
const cols2 = fromRows(rows);
const train = indicesWhere(cols2, i => cols2.train[i] === 1), test = indicesWhere(cols2, i => cols2.train[i] === 0);
const p = pick(cols2.p, test), h = pick(cols2.h, test), users = pick(cols2.user, test), pVal = pick(cols1.p, val);

function twoStage(name, opts) {
  const res = { opts, step1: { curve: [] }, step2: {} };
  let m = createMlp(cols1.vocabSize, opts, 0), opt = tf.train.adam(MLP_DEFAULTS.lrate);
  const es = new EarlyStopper(PATIENCE);
  let t0 = Date.now();
  for (let e = 0; e < MAX_EPOCHS; e++) {
    const trainLoss = trainEpoch(m, opt, cols1, step1, e);
    const v = predictMlp(m, cols1, val, { withLoss: true });
    const row = { epoch: e + 1, trainLoss, valLoss: v.loss, valMae: mae(pVal, v.phat) };
    res.step1.curve.push(row);
    const state = es.update(row.valMae);
    log(name, 'etapa1', JSON.stringify(row), state);
    if (state === 'stop') break;
  }
  Object.assign(res.step1, { bestEpoch: es.bestEpoch, bestValMae: es.best, seconds: (Date.now() - t0) / 1000 });
  disposeMlp(m); opt.dispose();
  for (const seed of SEEDS) {
    m = createMlp(cols2.vocabSize, opts, seed); opt = tf.train.adam(MLP_DEFAULTS.lrate);
    t0 = Date.now();
    for (let e = 0; e < es.bestEpoch; e++) trainEpoch(m, opt, cols2, train, e);
    const r = predictMlp(m, cols2, test);
    res.step2[seed] = { ...evaluate({ p, phat: r.phat, h, hhat: r.hhat }), seconds: (Date.now() - t0) / 1000, params: mlpParams(m), phat: r.phat };
    log(name, 'etapa2 semente', seed, JSON.stringify({ mae: res.step2[seed].mae, auc: res.step2[seed].auc, corH: res.step2[seed].corH }));
    disposeMlp(m); opt.dispose();
  }
  return res;
}

function compare(phatNew, phatPrev) {
  const b = pairedBootstrap({ errNew: absErr(p, phatNew), errPrev: absErr(p, phatPrev), groups: users });
  const a = aucDiffBootstrap({ p, phatNew, phatPrev, groups: users });
  return { maeDiff: b, aucDiff: a, verdict: verdict(b, a) };
}

const hlr = { lex: predictHlr(trainHlrTf(cols2, train), cols2, test).phat, nolex: predictHlr(trainHlrTf(cols2, train, { omitLexemes: true }), cols2, test).phat };
const out = { models: {}, comparisons: {} };
for (const [key, useLex] of [['lex', true], ['nolex', false]]) {
  const ctrl = twoStage(`controle${useLex ? '' : ' −lex'}`, { arch: 'linear', useLex });
  const mlp = twoStage(`mlp${useLex ? '' : ' −lex'}`, { arch: 'mlp', useLex });
  const cmp = {};
  for (const s of SEEDS) {
    cmp[s] = {
      'mlp vs controle': compare(mlp.step2[s].phat, ctrl.step2[s].phat),
      'controle vs hlr (B2)': compare(ctrl.step2[s].phat, hlr[key]),
    };
    log(key, 'semente', s, Object.entries(cmp[s]).map(([k, c]) => `${k}: ΔMAE ${c.maeDiff.point.toFixed(5)} [${c.maeDiff.lo.toFixed(5)}; ${c.maeDiff.hi.toFixed(5)}] ${c.verdict}`).join(' · '));
  }
  for (const r of [ctrl, mlp]) {
    for (const s of SEEDS) delete r.step2[s].phat;
    r.median = Object.fromEntries(['mae', 'auc', 'corH'].map(k => [k, median(SEEDS.map(s => r.step2[s][k]))]));
  }
  out.models[`controle-${key}`] = ctrl; out.models[`mlp-${key}`] = mlp; out.comparisons[key] = cmp;
}
fs.writeFileSync(path.join(DATA_DIR, 'b3c-results.json'), JSON.stringify(out, null, 1));
log('fim');
