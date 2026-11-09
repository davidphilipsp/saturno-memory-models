// Diagnóstico do B3-controle (fora do veredito): a HLR linear com Adam 0,001 converge em 30 épocas?
// Só etapa 1 (0–81%) e validação (81–90%); o teste não é tocado. Semente 0. Registra o bias por época.
// Uso: node src/model/b3c-diag.js → data/b3c-diag.json
import fs from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { DATA_DIR } from '../data/duolingo.js';
import { loadSample } from '../data/sample.js';
import { fromRows, indicesWhere } from '../data/columns.js';
import { mae } from './metrics.js';
import { stageOf, createMlp, trainEpoch, predictMlp, disposeMlp } from './mlp.js';

const TOTAL = 12854226;
await tf.ready();
const rows = await loadSample();
const cols1 = fromRows(rows.map(r => ({ ...r, split: stageOf(r.index, TOTAL) === 'step1' ? 'train' : 'held' })));
const step1 = indicesWhere(cols1, i => cols1.train[i] === 1);
const val = Int32Array.from(rows.flatMap((r, i) => (stageOf(r.index, TOTAL) === 'val' ? [i] : [])));
const pVal = Float64Array.from(val, i => cols1.p[i]);
const out = {};
for (const useLex of [true, false]) {
  const m = createMlp(cols1.vocabSize, { arch: 'linear', useLex }, 0), opt = tf.train.adam(0.001), curve = [];
  for (let e = 0; e < 30; e++) {
    const trainLoss = trainEpoch(m, opt, cols1, step1, e);
    const v = predictMlp(m, cols1, val, { withLoss: true });
    const [wR, wW] = m.vars.w.dataSync(), b = m.vars.b.dataSync()[0];
    curve.push({ epoch: e + 1, trainLoss, valLoss: v.loss, valMae: mae(pVal, v.phat), bias: b, right: wR, wrong: wW });
  }
  out[useLex ? 'controle' : 'controle-lex'] = curve;
  disposeMlp(m);
  console.log(useLex ? 'controle' : 'controle −lex');
  console.table(curve.map(c => Object.fromEntries(Object.entries(c).map(([k, v]) => [k, Number.isInteger(v) ? v : +v.toPrecision(5)]))));
}
fs.writeFileSync(path.join(DATA_DIR, 'b3c-diag.json'), JSON.stringify(out, null, 1));
