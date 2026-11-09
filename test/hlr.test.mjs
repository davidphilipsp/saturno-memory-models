import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { sampleInstances, MAX_HALF_LIFE } from '../src/data/duolingo.js';
import { fromRows } from '../src/data/columns.js';
import { dLossDdp, trainHlrJs, trainHlrTf, predictHlr, hlrParams } from '../src/model/hlr.js';

await tf.setBackend('cpu');
const FIX = path.resolve(import.meta.dirname, 'fixtures/duolingo-mini.csv');
const close = (a, b, eps) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

test('o passo do código é a derivada da loss do artigo em relação a Θ·x (tf.grad, ponto sem corte)', () => {
  // dp = 1 → ĥ = 2; Δ = 1 → p̂ = 2^(−1/2); p = 0,9, h = 3, α = 0,01
  const p = 0.9, h = 3, t = 1, alpha = 0.01;
  const loss = dp => tf.add(tf.square(tf.sub(p, tf.pow(2, tf.div(-t, tf.pow(2, dp))))), tf.mul(alpha, tf.square(tf.sub(h, tf.pow(2, dp)))));
  const auto = tf.grad(loss)(tf.scalar(1)).dataSync()[0];
  const manual = dLossDdp({ phat: 2 ** -0.5, hhat: 2, p, h, t, hlwt: alpha, omitHTerm: false });
  close(manual, auto, 1e-5);
  // sem o termo de h, só a parte de p
  const autoP = tf.grad(dp => tf.square(tf.sub(p, tf.pow(2, tf.div(-t, tf.pow(2, dp))))))(tf.scalar(1)).dataSync()[0];
  close(dLossDdp({ phat: 2 ** -0.5, hhat: 2, p, h, t, hlwt: alpha, omitHTerm: true }), autoP, 1e-5);
});

// Uma linha: 0 acertos, 0 erros → x = [1, 1, 1]; Δ = 1 dia; p = 1 → 0,9999; h = corte de 274; lexema 1.
// Θ = 0 → ĥ = 1, p̂ = 0,5.
// ∂p = 2(0,5 − 0,9999)·ln²2·0,5·(1/1) ; ∂h = 2(1 − 274)·ln2·1 ; g = ∂p + 0,01·∂h
// taxa = (1/1,9999)·0,001 ; w = −taxa·g, depois w·(1 − taxa·0,1)
const one = {
  n: 1, p: Float64Array.of(0.9999), t: Float64Array.of(1), h: Float64Array.of(MAX_HALF_LIFE),
  right: Int32Array.of(0), wrong: Int32Array.of(0), lex: Int32Array.of(1), user: Int32Array.of(0),
  train: Uint8Array.of(1), vocabSize: 2, users: 1,
};

test('HLR, 1 passo à mão (JS e TF.js)', () => {
  const g = 2 * (0.5 - 0.9999) * Math.LN2 ** 2 * 0.5 + 0.01 * 2 * (1 - 274) * Math.LN2;
  const rate = 0.001 / 1.9999;
  const expected = -rate * g * (1 - rate * 0.1);
  close(expected, 0.0020123, 1e-6);           // ordem de grandeza conferida à mão: 0,0005·4,025
  for (const m of [trainHlrJs(one, [0]), trainHlrTf(one, [0])]) {
    for (const v of m.w) close(v, expected, 1e-8);
    close(m.lexW[1], expected, 1e-8);
    assert.equal(m.lexW[0], 0);
  }
  const noH = trainHlrJs(one, [0], { omitHTerm: true });
  close(noH.w[0], -rate * (2 * (0.5 - 0.9999) * Math.LN2 ** 2 * 0.5) * (1 - rate * 0.1), 1e-12);
});

test('HLR: TF.js e JS puro dão o mesmo modelo na fixture', async () => {
  const rows = [];
  for await (const r of sampleInstances(FIX, { total: 10, rate: 1 })) rows.push(r);
  const cols = fromRows(rows);
  const train = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const a = trainHlrJs(cols, train, { lrate: 0.05 }), b = trainHlrTf(cols, train, { lrate: 0.05 });
  for (let k = 0; k < 3; k++) close(a.w[k], b.w[k], 1e-5);
  for (let l = 0; l < cols.vocabSize; l++) close(a.lexW[l], b.lexW[l], 1e-5);
  const pa = predictHlr(a, cols, [9]), pb = predictHlr(b, cols, [9]);
  close(pa.phat[0], pb.phat[0], 1e-5);
  close(pa.hhat[0], pb.hhat[0], 1e-4);
  assert.equal(hlrParams(a, cols), 3 + cols.vocabSize - 1);
});

test('predictHlr: Θ·x = log2 3 e Δ = 3 → ĥ = 3, p̂ = 0,5', () => {
  const cols = { ...one, t: Float64Array.of(3) };
  const r = predictHlr({ w: Float64Array.of(0, 0, Math.log2(3)), lexW: new Float64Array(2), omitLexemes: false }, cols, [0]);
  close(r.hhat[0], 3, 1e-12); close(r.phat[0], 0.5, 1e-12);
});
