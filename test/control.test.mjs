import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import { MAX_HALF_LIFE } from '../src/data/duolingo.js';
import { LO, HI, hardLog2H, lossFromZ, createMlp, mlpParams, trainEpoch, disposeMlp } from '../src/model/mlp.js';
import { dLossDdp } from '../src/model/hlr.js';

await tf.setBackend('cpu');
const close = (a, b, eps) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const num = x => x.dataSync()[0];

test('controle: corte duro = hclip da HLR', () => {
  assert.equal(num(hardLog2H(tf.scalar(100))), Math.fround(HI));
  assert.equal(num(hardLog2H(tf.scalar(-100))), Math.fround(LO));
  close(num(hardLog2H(tf.scalar(1.5))), 1.5, 1e-7);
});

test('controle: tf.grad da loss em z = gradiente da HLR (dLossDdp da B2), dentro do corte', () => {
  for (const [z, p, h, t] of [[1, 0.9, 3, 1], [-2, 0.5, 0.5, 0.2], [5, 1, 274, 30]]) {
    const hh = 2 ** z, ph = 2 ** (-t / hh);
    const auto = num(tf.grad(zz => lossFromZ(zz, { p: tf.tensor1d([p]), h: tf.tensor1d([h]), t: tf.tensor1d([t]) }, 0.01, hardLog2H))(tf.tensor1d([z])));
    const manual = dLossDdp({ phat: ph, hhat: hh, p, h, t, hlwt: 0.01, omitHTerm: false });
    close(auto, manual, Math.max(1e-4, Math.abs(manual) * 1e-4));
  }
});

// Um passo de Adam à mão. Uma linha: 0 acertos, 0 erros → x = [1, 1]; lexema 1; Δ = 1; p = 0,9999; h = 274.
// Θ = 0 → z = 0 → ĥ = 1, p̂ = 0,5. g = ∂L/∂z = 2(0,5 − 0,9999)·ln²2·0,5 + 0,01·2(1 − 274)·ln2 ≈ −4,025.
// ∂L/∂w₁ = ∂L/∂w₂ = ∂L/∂b = ∂L/∂L₁ = g (x = 1).
// Adam, 1º passo: m̂ = g, v̂ = g² → Δθ = −η·g/(|g| + ε) = +0,001·|g|/(|g| + 1e−7).
test('controle: 1 passo de Adam à mão, todos os parâmetros andam +0,001; <unk> fica 0', () => {
  const g = 2 * (0.5 - 0.9999) * Math.LN2 ** 2 * 0.5 + 0.01 * 2 * (1 - 274) * Math.LN2;
  close(g, -4.02476, 1e-4);
  const expected = 0.001 * Math.abs(g) / (Math.abs(g) + 1e-7);
  const cols = {
    n: 1, p: Float64Array.of(0.9999), t: Float64Array.of(1), h: Float64Array.of(MAX_HALF_LIFE),
    right: Int32Array.of(0), wrong: Int32Array.of(0), lex: Int32Array.of(1), user: Int32Array.of(0),
    train: Uint8Array.of(1), vocabSize: 2, users: 1,
  };
  const m = createMlp(2, { arch: 'linear', batchSize: 1 }, 0);
  assert.equal(mlpParams(m), 2 + 1 + 2);
  trainEpoch(m, tf.train.adam(0.001), cols, Int32Array.of(0), 0);
  for (const v of [...m.vars.w.dataSync(), ...m.vars.b.dataSync()]) close(v, expected, 1e-8);
  const L = m.vars.L.dataSync();
  assert.equal(L[0], 0);
  close(L[1], expected, 1e-8);
  disposeMlp(m);
});

// Q11b, opção (A): início na média. Conta à mão: LO = log2(15/1440) = −6,5850; HI = log2 274 = 8,0981.
// u = (5,49 + 6,5850)/14,6831 = 0,82237; b2 = ln(0,82237/0,17763) = 1,5325.
test('início na média: MLP com b2 = 1,53 e W2 = 0 → log2 ĥ = 5,49; controle com bias 5,49 → 5,49', async () => {
  const { zForLog2H, predictMlp } = await import('../src/model/mlp.js');
  close(zForLog2H(5.49), 1.5325, 1e-3);
  const cols = {
    n: 2, p: Float64Array.of(1, 0.5), t: Float64Array.of(1, 3), h: Float64Array.of(274, 3),
    right: Int32Array.of(4, 0), wrong: Int32Array.of(1, 2), lex: Int32Array.of(1, 0), user: Int32Array.of(0, 0),
    train: Uint8Array.of(1, 0), vocabSize: 2, users: 1,
  };
  const m = createMlp(2, { initLog2h: 5.49 }, 0);
  close(m.vars.b2.dataSync()[0], 1.5325, 1e-3);
  m.vars.W2.assign(tf.zeros(m.vars.W2.shape));
  for (const hh of predictMlp(m, cols, Int32Array.of(0, 1)).hhat) close(Math.log2(hh), 5.49, 1e-4);
  disposeMlp(m);
  const c = createMlp(2, { arch: 'linear', initLog2h: 5.49 }, 0);
  close(c.vars.b.dataSync()[0], 5.49, 1e-6);
  for (const hh of predictMlp(c, cols, Int32Array.of(0, 1)).hhat) close(Math.log2(hh), 5.49, 1e-5);
  disposeMlp(c);
  // sem a opção, tudo como na B3 (b2 = 0, bias do controle = 0)
  const d = createMlp(2, {}, 0), e = createMlp(2, { arch: 'linear' }, 0);
  assert.equal(d.vars.b2.dataSync()[0], 0); assert.equal(e.vars.b.dataSync()[0], 0);
  disposeMlp(d); disposeMlp(e);
});
