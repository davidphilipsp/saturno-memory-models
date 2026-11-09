import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { sampleInstances } from '../src/data/duolingo.js';
import { fromRows } from '../src/data/columns.js';
import { trainLrJs, trainLrTf, predictLr, shuffled, lrParams } from '../src/model/lr.js';

await tf.setBackend('cpu');
const FIX = path.resolve(import.meta.dirname, 'fixtures/duolingo-mini.csv');
const close = (a, b, eps) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

// Uma linha: 0 acertos, 0 erros, Δ = 1 dia, p = 1 (→ 0,9999), lexema 1 → x = [1, 1, 1, 1]
const one = {
  n: 1, p: Float64Array.of(0.9999), t: Float64Array.of(1), h: Float64Array.of(1),
  right: Int32Array.of(0), wrong: Int32Array.of(0), lex: Int32Array.of(1), user: Int32Array.of(0),
  train: Uint8Array.of(1), vocabSize: 2, users: 1,
};

test('LR, 1 passo à mão: p̂ = 0,5; erro −0,4999; w = 0,001·0,4999·(1 − 0,0001) = 0,00049985001', () => {
  const expected = 0.001 * 0.4999 * (1 - 0.001 * 0.1);
  for (const m of [trainLrJs(one, [0]), trainLrTf(one, [0])]) {
    for (const v of m.w) close(v, expected, 1e-9);
    close(m.lexW[1], expected, 1e-9);
    assert.equal(m.lexW[0], 0);            // <unk> nunca treina
  }
});

test('LR −lex: o peso do lexema fica 0 e não conta como parâmetro', () => {
  const m = trainLrJs(one, [0], { omitLexemes: true });
  assert.equal(m.lexW[1], 0);
  assert.equal(lrParams(m, one), 4);
  assert.equal(lrParams(trainLrJs(one, [0]), one), 5);
});

test('shuffled: permutação determinística', () => {
  const a = shuffled([0, 1, 2, 3, 4, 5, 6, 7], 5);
  assert.deepEqual([...a].sort(), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(a, shuffled([0, 1, 2, 3, 4, 5, 6, 7], 5));
});

test('LR: TF.js e JS puro dão o mesmo modelo na fixture (mesma ordem)', async () => {
  const rows = [];
  for await (const r of sampleInstances(FIX, { total: 10, rate: 1 })) rows.push(r);
  const cols = fromRows(rows);
  const train = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const a = trainLrJs(cols, train, { lrate: 0.1 }), b = trainLrTf(cols, train, { lrate: 0.1 });
  for (let k = 0; k < 4; k++) close(a.w[k], b.w[k], 1e-6);
  for (let l = 0; l < cols.vocabSize; l++) close(a.lexW[l], b.lexW[l], 1e-6);
  const pa = predictLr(a, cols, [9]), pb = predictLr(b, cols, [9]);
  close(pa[0], pb[0], 1e-6);
  assert.ok(pa[0] >= 0.0001 && pa[0] <= 0.9999);
});
