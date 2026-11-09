import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { sampleInstances, MIN_HALF_LIFE, MAX_HALF_LIFE } from '../src/data/duolingo.js';
import { fromRows, indicesWhere } from '../src/data/columns.js';
import {
  LO, HI, boundedLog2H, pFromLog2H, lossFromZ, stageOf, EarlyStopper,
  createMlp, mlpParams, trainEpoch, predictMlp, disposeMlp,
} from '../src/model/mlp.js';

await tf.setBackend('cpu');
const FIX = path.resolve(import.meta.dirname, 'fixtures/duolingo-mini.csv');
const close = (a, b, eps) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const num = x => x.dataSync()[0];

test('cabeça limitada: ĥ fica em [15 min; 274 d]; z = 0 → meio do intervalo em log2', () => {
  close(2 ** num(boundedLog2H(tf.scalar(-50))), MIN_HALF_LIFE, 1e-6);
  close(2 ** num(boundedLog2H(tf.scalar(50))), MAX_HALF_LIFE, 1e-3);
  close(num(boundedLog2H(tf.scalar(0))), (LO + HI) / 2, 1e-5);
  for (const z of [-1e4, -3, 0, 3, 1e4]) {
    const h = 2 ** num(boundedLog2H(tf.scalar(z)));
    assert.ok(h >= MIN_HALF_LIFE * (1 - 1e-5) && h <= MAX_HALF_LIFE * (1 + 1e-5), `z=${z} → ${h}`);
  }
});

test('p̂ à mão: ĥ = 3 dias, Δ = 3 → 0,5; Δ = 6 → 0,25', () => {
  close(num(pFromLog2H(tf.scalar(Math.log2(3)), tf.scalar(3))), 0.5, 1e-6);
  close(num(pFromLog2H(tf.scalar(Math.log2(3)), tf.scalar(6))), 0.25, 1e-6);
});

test('gradiente da loss em z: derivação manual = tf.grad', () => {
  // L = (p − p̂)² + α(h − ĥ)²; log2 ĥ = LO + R·σ(z); ĥ' = ĥ·ln2·R·σ(1−σ); ∂p̂/∂ĥ = p̂·ln2·Δ/ĥ²
  const alpha = 0.01, R = HI - LO;
  for (const [z, p, h, t] of [[0.3, 0.9, 20, 4], [-1.2, 0.5, 2, 1.5], [2, 1, 274, 10]]) {
    const s = 1 / (1 + Math.exp(-z)), hh = 2 ** (LO + R * s), ph = 2 ** (-t / hh);
    const dh = hh * Math.LN2 * R * s * (1 - s);
    const manual = (-2 * (p - ph) * ph * Math.LN2 * t / hh ** 2 - 2 * alpha * (h - hh)) * dh;
    const auto = num(tf.grad(zz => lossFromZ(zz, { p: tf.tensor1d([p]), h: tf.tensor1d([h]), t: tf.tensor1d([t]) }, alpha))(tf.tensor1d([z])));
    close(auto, manual, Math.max(1e-3, Math.abs(manual) * 1e-3));
  }
});

test('early stopping: curva à mão, paciência 3 → melhor época 3, para na 6 (nunca vê a 7)', () => {
  const curve = [0.20, 0.18, 0.17, 0.175, 0.171, 0.172, 0.16];
  const es = new EarlyStopper(3);
  let stoppedAt = null;
  for (const v of curve) if (es.update(v) === 'stop') { stoppedAt = es.epoch; break; }
  assert.equal(stoppedAt, 6);
  assert.equal(es.bestEpoch, 3);
  assert.equal(es.best, 0.17);
});

test('early stopping: empate não conta como melhora', () => {
  const es = new EarlyStopper(2);
  assert.equal(es.update(0.2), 'continue');
  assert.equal(es.update(0.2), 'continue');
  assert.equal(es.update(0.2), 'stop');
  assert.equal(es.bestEpoch, 1);
});

test('limites do split (b): 0–81% etapa 1, 81–90% validação, 90%+ teste; disjuntos', () => {
  assert.equal(stageOf(80, 100), 'step1');
  assert.equal(stageOf(81, 100), 'val');
  assert.equal(stageOf(89, 100), 'val');
  assert.equal(stageOf(90, 100), 'test');
  const N = 12854226;
  assert.equal(stageOf(10411922, N), 'step1');
  assert.equal(stageOf(10411923, N), 'val');
  assert.equal(stageOf(11568802, N), 'val');
  assert.equal(stageOf(11568803, N), 'test');
  const by = { step1: 0, val: 0, test: 0 };
  for (let i = 0; i < 1000; i++) by[stageOf(i, 1000)]++;
  assert.deepEqual(by, { step1: 810, val: 90, test: 100 });
});

test('MLP: parâmetros como na proposta; 1 época roda; previsões dentro dos limites', async () => {
  const rows = [];
  for await (const r of sampleInstances(FIX, { total: 10, rate: 1 })) rows.push(r);
  const cols = fromRows(rows);
  const train = indicesWhere(cols, i => cols.train[i] === 1), test = indicesWhere(cols, i => cols.train[i] === 0);
  const m = createMlp(cols.vocabSize, { batchSize: 4 }, 0);
  assert.equal(mlpParams(m), cols.vocabSize * 8 + 10 * 32 + 32 + 32 + 1);
  assert.deepEqual(Array.from(m.vars.E.slice([0, 0], [1, 8]).dataSync()), new Array(8).fill(0));   // <unk> = 0
  const loss = trainEpoch(m, tf.train.adam(0.001), cols, train, 0);
  assert.ok(Number.isFinite(loss));
  const r = predictMlp(m, cols, test, { withLoss: true });
  assert.ok(r.phat[0] >= 0.0001 && r.phat[0] <= 0.9999);
  assert.ok(r.hhat[0] >= MIN_HALF_LIFE * 0.999 && r.hhat[0] <= MAX_HALF_LIFE * 1.001);
  assert.ok(Number.isFinite(r.loss));
  disposeMlp(m);
  const noLex = createMlp(cols.vocabSize, { useLex: false }, 0);
  assert.equal(mlpParams(noLex), 2 * 32 + 32 + 32 + 1);
  disposeMlp(noLex);
});
