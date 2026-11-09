// B3 — MLP (cabeça da HLR), opção (b), ablação ±lex, 3 sementes na etapa 2 (aprovado em 2026-10-02).
// Etapa 1 (semente 0): treino em 0–81%, early stopping pela MAE em 81–90% (até 30 épocas, paciência 3) → E*.
// Etapa 2 (sementes 0, 1, 2): do zero em 0–90%, E* épocas; avalia no teste.
// Veredito: vence a HLR só se as 3 sementes vencerem (IC da ΔMAE < 0; regra emendada da AUC).
// Uso: node src/model/b3.js  → data/b3-results.json (progresso no terminal)
import fs from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { DATA_DIR } from '../data/duolingo.js';
import { loadSample } from '../data/sample.js';
import { fromRows, indicesWhere } from '../data/columns.js';
import { evaluate, mae } from './metrics.js';
import { leitner } from './baselines.js';
import { pairedBootstrap, aucDiffBootstrap, verdict } from './bootstrap.js';
import { trainLrTf, predictLr } from './lr.js';
import { trainHlrTf, predictHlr } from './hlr.js';
import { stageOf, EarlyStopper, createMlp, mlpParams, trainEpoch, predictMlp, disposeMlp, MLP_DEFAULTS } from './mlp.js';

const TOTAL = 12854226, MAX_EPOCHS = 30, PATIENCE = 3, SEEDS = [0, 1, 2];
const pick = (arr, idx) => Float64Array.from(idx, i => arr[i]);
const absErr = (p, phat) => Float64Array.from(p, (v, j) => Math.abs(v - phat[j]));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

await tf.ready();
const rows = await loadSample();
// Etapa 1: só 0–81% é treino (vocabulário incluso); 81–90% e teste ficam fora
const cols1 = fromRows(rows.map(r => ({ ...r, split: stageOf(r.index, TOTAL) === 'step1' ? 'train' : 'held' })));
const step1 = indicesWhere(cols1, i => cols1.train[i] === 1);
const val = Int32Array.from(rows.flatMap((r, i) => (stageOf(r.index, TOTAL) === 'val' ? [i] : [])));
// Etapa 2: split do artigo (0–90% treino)
const cols2 = fromRows(rows);
const train = indicesWhere(cols2, i => cols2.train[i] === 1), test = indicesWhere(cols2, i => cols2.train[i] === 0);
const p = pick(cols2.p, test), h = pick(cols2.h, test), users = pick(cols2.user, test), pVal = pick(cols1.p, val);
log(`etapa1 ${step1.length} · validação ${val.length} · etapa2 ${train.length} · teste ${test.length} · vocab ${cols1.vocabSize}/${cols2.vocabSize}`);

// Degraus anteriores, no mesmo teste (TF.js, como na B1 e na B2)
let mean = 0; for (const i of train) mean += cols2.p[i]; mean /= train.length;
const lt = Array.from(test, i => leitner({ t: cols2.t[i], right: cols2.right[i], wrong: cols2.wrong[i] }));
const prev = {
  constant: { phat: new Float64Array(test.length).fill(mean) },
  leitner: { phat: Float64Array.from(lt, x => x.p), hhat: Float64Array.from(lt, x => x.h) },
  lr: { phat: predictLr(trainLrTf(cols2, train), cols2, test) },
  hlr: predictHlr(trainHlrTf(cols2, train), cols2, test),
  'hlr-lex': predictHlr(trainHlrTf(cols2, train, { omitLexemes: true }), cols2, test),
};
const out = { rows: { step1: step1.length, val: val.length, train: train.length, test: test.length }, models: {}, mlp: {} };
for (const [k, v] of Object.entries(prev)) out.models[k] = evaluate({ p, phat: v.phat, h, hhat: v.hhat ?? null });
log('anteriores', JSON.stringify(Object.fromEntries(Object.entries(out.models).map(([k, v]) => [k, +v.mae.toFixed(4)]))));

function compare(phat, k) {
  const b = pairedBootstrap({ errNew: absErr(p, phat), errPrev: absErr(p, prev[k].phat), groups: users });
  const a = aucDiffBootstrap({ p, phatNew: phat, phatPrev: prev[k].phat, groups: users });
  return { maeDiff: b, aucDiff: a, verdict: verdict(b, a) };
}

for (const [name, useLex, against] of [['mlp', true, ['hlr', 'leitner', 'lr', 'constant']], ['mlp-lex', false, ['hlr-lex', 'hlr']]]) {
  const res = { useLex, decidesAgainst: against[0], step1: { curve: [] }, step2: {} };
  // Etapa 1
  let m = createMlp(cols1.vocabSize, { useLex }, 0), opt = tf.train.adam(MLP_DEFAULTS.lrate);
  const es = new EarlyStopper(PATIENCE);
  let t0 = Date.now();
  for (let e = 0; e < MAX_EPOCHS; e++) {
    const te = Date.now();
    const trainLoss = trainEpoch(m, opt, cols1, step1, e);
    const v = predictMlp(m, cols1, val, { withLoss: true });
    const row = { epoch: e + 1, trainLoss, valLoss: v.loss, valMae: mae(pVal, v.phat), seconds: (Date.now() - te) / 1000 };
    res.step1.curve.push(row);
    const state = es.update(row.valMae);
    log(name, 'etapa1', JSON.stringify(row), state);
    if (state === 'stop') break;
  }
  res.step1.bestEpoch = es.bestEpoch; res.step1.bestValMae = es.best; res.step1.seconds = (Date.now() - t0) / 1000;
  res.params = mlpParams(m);
  disposeMlp(m); opt.dispose();
  // Etapa 2
  for (const seed of SEEDS) {
    m = createMlp(cols2.vocabSize, { useLex }, seed); opt = tf.train.adam(MLP_DEFAULTS.lrate);
    t0 = Date.now();
    const losses = [];
    for (let e = 0; e < es.bestEpoch; e++) losses.push(trainEpoch(m, opt, cols2, train, e));
    const seconds = (Date.now() - t0) / 1000;
    const r = predictMlp(m, cols2, test);
    const ev = evaluate({ p, phat: r.phat, h, hhat: r.hhat });
    const cmp = Object.fromEntries(against.map(k => [k, compare(r.phat, k)]));
    res.step2[seed] = { ...ev, trainLosses: losses, seconds, params: mlpParams(m), vs: cmp };
    log(name, 'etapa2 semente', seed, JSON.stringify({ mae: ev.mae, auc: ev.auc, corH: ev.corH, seconds }),
      Object.entries(cmp).map(([k, c]) => `${k}: ${c.verdict}`).join(' · '));
    disposeMlp(m); opt.dispose();
  }
  const verdicts = SEEDS.map(s => res.step2[s].vs[against[0]].verdict);
  res.verdict = verdicts.every(v => v === 'venceu') ? 'venceu' : 'não venceu (nem todas as 3 sementes venceram)';
  res.verdicts = verdicts;
  res.median = Object.fromEntries(['mae', 'auc', 'corH'].map(k => [k, median(SEEDS.map(s => res.step2[s][k]))]));
  out.mlp[name] = res;
  log(name, 'VEREDITO contra', against[0], res.verdict, JSON.stringify(res.median));
}
fs.writeFileSync(path.join(DATA_DIR, 'b3-results.json'), JSON.stringify(out, null, 1));
log('fim');
