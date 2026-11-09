// Q11b, opção (A), aprovada em 2026-10-02 — último controle do Q11. Par NOVO; a B3 aprovada não é sobrescrita.
// Controle (HLR linear) e MLP começam no mesmo ponto: log2 ĥ = média de log2 h-alvo das linhas da etapa 1.
// Protocolo idêntico ao aprovado: etapa 1 (semente 0) com early stopping pela MAE de validação (até 30, paciência 3) → E*;
// etapa 2 com sementes 0, 1, 2 em E* épocas; ±lex. Diagnóstico fora do veredito: etapa 1 também nas sementes 1 e 2.
// Depois: dependência parcial só na validação, com modelos que não a viram (etapa 1, semente 0, em E*; HLR da B2 refeita em 0–81%).
// Uso: node src/model/b3c2.js → data/b3c2-results.json
import fs from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { DATA_DIR } from '../data/duolingo.js';
import { loadSample } from '../data/sample.js';
import { fromRows, indicesWhere } from '../data/columns.js';
import { evaluate, mae } from './metrics.js';
import { pairedBootstrap, aucDiffBootstrap, verdict } from './bootstrap.js';
import { trainHlrTf, predictHlr } from './hlr.js';
import { stageOf, EarlyStopper, createMlp, mlpParams, trainEpoch, predictMlp, disposeMlp, MLP_DEFAULTS, zForLog2H } from './mlp.js';

const TOTAL = 12854226, MAX_EPOCHS = 30, PATIENCE = 3, SEEDS = [0, 1, 2];
const DELTAS = [1, 3, 7, 16, 35, 75], RIGHTS = [1, 3, 5, 10], WRONGS = [0, 1, 3];
const pick = (arr, idx) => Float64Array.from(idx, i => arr[i]);
const absErr = (p, phat) => Float64Array.from(p, (v, j) => Math.abs(v - phat[j]));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const median = xs => { const a = Float64Array.from(xs).sort(); return a[Math.floor(a.length / 2)]; };

await tf.ready();
const rows = await loadSample();
const cols1 = fromRows(rows.map(r => ({ ...r, split: stageOf(r.index, TOTAL) === 'step1' ? 'train' : 'held' })));
const step1 = indicesWhere(cols1, i => cols1.train[i] === 1);
const val = Int32Array.from(rows.flatMap((r, i) => (stageOf(r.index, TOTAL) === 'val' ? [i] : [])));
const cols2 = fromRows(rows);
const train = indicesWhere(cols2, i => cols2.train[i] === 1), test = indicesWhere(cols2, i => cols2.train[i] === 0);
const p = pick(cols2.p, test), h = pick(cols2.h, test), users = pick(cols2.user, test), pVal = pick(cols1.p, val);
let s0 = 0; for (const i of step1) s0 += Math.log2(cols1.h[i]);
const INIT = s0 / step1.length;
log(`início: média log2 h-alvo (etapa 1) = ${INIT.toFixed(4)} → ĥ ${(2 ** INIT).toFixed(1)} d; b2 do MLP = ${zForLog2H(INIT).toFixed(4)}`);

const initMedian = (cols, idx, opts, seed) => {
  const m = createMlp(cols.vocabSize, opts, seed);
  const r = predictMlp(m, cols, idx.subarray(0, 200000));
  disposeMlp(m);
  return median(r.hhat);
};

function step1Run(name, opts, seed) {
  const m = createMlp(cols1.vocabSize, opts, seed), opt = tf.train.adam(MLP_DEFAULTS.lrate);
  const es = new EarlyStopper(PATIENCE), curve = [];
  let best = null;
  for (let e = 0; e < MAX_EPOCHS; e++) {
    const trainLoss = trainEpoch(m, opt, cols1, step1, e);
    const v = predictMlp(m, cols1, val, { withLoss: true });
    const row = { epoch: e + 1, trainLoss, valLoss: v.loss, valMae: mae(pVal, v.phat) };
    curve.push(row);
    const state = es.update(row.valMae);
    if (es.bestEpoch === e + 1) {          // guarda os pesos da melhor época (para a dependência parcial)
      if (best) Object.values(best).forEach(t => t.dispose());
      best = Object.fromEntries(Object.entries(m.vars).map(([k, t]) => [k, t.clone()]));
    }
    log(name, `etapa1 semente ${seed}`, JSON.stringify({ epoch: row.epoch, valMae: +row.valMae.toFixed(5), valLoss: +row.valLoss.toFixed(2) }), state);
    if (state === 'stop') break;
  }
  disposeMlp(m); opt.dispose();
  return { curve, bestEpoch: es.bestEpoch, bestValMae: es.best, best: { vars: best, opts: { ...MLP_DEFAULTS, ...opts }, seed } };
}

function compare(phatNew, phatPrev) {
  const b = pairedBootstrap({ errNew: absErr(p, phatNew), errPrev: absErr(p, phatPrev), groups: users });
  const a = aucDiffBootstrap({ p, phatNew, phatPrev, groups: users });
  return { maeDiff: b, aucDiff: a, verdict: verdict(b, a) };
}

// Dependência parcial: cada linha da validação recebe (Δ, acertos, erros) da grade e mantém o seu lexema; média de p̂.
function partialDependence(predict) {
  const n = val.length, base = {
    n, p: new Float64Array(n), h: new Float64Array(n), user: new Int32Array(n), train: new Uint8Array(n),
    lex: Int32Array.from(val, i => cols1.lex[i]), vocabSize: cols1.vocabSize,
  };
  const idx = Int32Array.from({ length: n }, (_, i) => i), table = {};
  for (const wrong of WRONGS) for (const right of RIGHTS) for (const d of DELTAS) {
    const c = { ...base, t: new Float64Array(n).fill(d), right: new Int32Array(n).fill(right), wrong: new Int32Array(n).fill(wrong) };
    const ph = predict(c, idx);
    table[`${wrong}|${right}|${d}`] = ph.reduce((a, v) => a + v, 0) / n;
  }
  return table;
}
const direction = seq => seq.every((v, i) => i === 0 || v > seq[i - 1]) ? 'sobe' : seq.every((v, i) => i === 0 || v < seq[i - 1]) ? 'desce' : 'misto';
function directions(table) {
  const right = {}, wrong = {};
  for (const d of DELTAS) for (const w of WRONGS) { const k = direction(RIGHTS.map(r => table[`${w}|${r}|${d}`])); right[k] = (right[k] || 0) + 1; }
  for (const d of DELTAS) for (const r of RIGHTS) { const k = direction(WRONGS.map(w => table[`${w}|${r}|${d}`])); wrong[k] = (wrong[k] || 0) + 1; }
  return { right, wrong };
}

const out = { init: { meanLog2h: INIT, halfLifeDays: 2 ** INIT, b2: zForLog2H(INIT) }, variants: {}, partialDependence: {} };
const hlrFull = { lex: predictHlr(trainHlrTf(cols2, train), cols2, test).phat, nolex: predictHlr(trainHlrTf(cols2, train, { omitLexemes: true }), cols2, test).phat };
const pdModels = {};

for (const [key, useLex] of [['lex', true], ['nolex', false]]) {
  const res = {};
  for (const arch of ['linear', 'mlp']) {
    const name = `${arch === 'linear' ? 'controle' : 'mlp'}${useLex ? '' : ' −lex'} (início na média)`;
    const opts = { arch, useLex, initLog2h: INIT };
    const r = { initialHalfLifeMedian: {}, step1: {}, step2: {} };
    for (const seed of SEEDS) r.initialHalfLifeMedian[seed] = { step1: initMedian(cols1, step1, opts, seed), step2: initMedian(cols2, train, opts, seed) };
    log(name, 'ĥ inicial mediano por semente', JSON.stringify(r.initialHalfLifeMedian));
    for (const seed of SEEDS) {           // semente 0 decide E*; 1 e 2 são diagnóstico
      const s1 = step1Run(name, opts, seed);
      r.step1[seed] = { curve: s1.curve, bestEpoch: s1.bestEpoch, bestValMae: s1.bestValMae };
      if (seed === 0 && useLex) pdModels[arch] = s1.best; else Object.values(s1.best.vars).forEach(t => t.dispose());
    }
    const E = r.step1[0].bestEpoch;
    log(name, 'E* (semente 0) =', E, '· diagnóstico: E* semente 1 =', r.step1[1].bestEpoch, '· semente 2 =', r.step1[2].bestEpoch);
    for (const seed of SEEDS) {
      const m = createMlp(cols2.vocabSize, opts, seed), opt = tf.train.adam(MLP_DEFAULTS.lrate);
      const t0 = Date.now();
      for (let e = 0; e < E; e++) trainEpoch(m, opt, cols2, train, e);
      const pr = predictMlp(m, cols2, test);
      r.step2[seed] = { ...evaluate({ p, phat: pr.phat, h, hhat: pr.hhat }), seconds: (Date.now() - t0) / 1000, params: mlpParams(m), phat: pr.phat };
      log(name, 'etapa2 semente', seed, JSON.stringify({ mae: +r.step2[seed].mae.toFixed(5), auc: +r.step2[seed].auc.toFixed(4), corH: +r.step2[seed].corH.toFixed(3) }));
      disposeMlp(m); opt.dispose();
    }
    res[arch] = r;
  }
  res.comparisons = {};
  for (const seed of SEEDS) {
    res.comparisons[seed] = {
      'mlp vs controle': compare(res.mlp.step2[seed].phat, res.linear.step2[seed].phat),
      'controle vs hlr (B2)': compare(res.linear.step2[seed].phat, hlrFull[key]),
      'mlp vs hlr (B2)': compare(res.mlp.step2[seed].phat, hlrFull[key]),
    };
    log(key, 'semente', seed, Object.entries(res.comparisons[seed]).map(([k, c]) => `${k}: ΔMAE ${c.maeDiff.point.toFixed(5)} [${c.maeDiff.lo.toFixed(5)}; ${c.maeDiff.hi.toFixed(5)}] ΔAUC [${c.aucDiff.lo.toFixed(4)}; ${c.aucDiff.hi.toFixed(4)}] ${c.verdict}`).join(' · '));
  }
  for (const arch of ['linear', 'mlp']) {
    for (const seed of SEEDS) delete res[arch].step2[seed].phat;
    res[arch].median = Object.fromEntries(['mae', 'auc', 'corH'].map(k => [k, median(SEEDS.map(s => res[arch].step2[s][k]))]));
  }
  out.variants[key] = res;
}

// HLR da B2 (mesmo algoritmo, TF.js) refeita em 0–81%, para não ter visto a validação
const hlrStep1 = trainHlrTf(cols1, step1);
const pdPredict = {
  mlp: (c, idx) => predictMlp(pdModels.mlp, c, idx).phat,
  controle: (c, idx) => predictMlp(pdModels.linear, c, idx).phat,
  'hlr (B2, em 0–81%)': (c, idx) => predictHlr(hlrStep1, c, idx).phat,
};
for (const [k, f] of Object.entries(pdPredict)) {
  const table = partialDependence(f);
  out.partialDependence[k] = { table, directions: directions(table) };
  log('dependência parcial', k, JSON.stringify(out.partialDependence[k].directions));
}
fs.writeFileSync(path.join(DATA_DIR, 'b3c2-results.json'), JSON.stringify(out, null, 1));
log('fim');
