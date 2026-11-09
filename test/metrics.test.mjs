import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mae, auc, aucAuto, ranks, pearson, spearman, labelR, labelGe, round4, evaluate } from '../src/model/metrics.js';
import { leitner, pimsleur, constant } from '../src/model/baselines.js';
import { MAX_HALF_LIFE, MIN_HALF_LIFE } from '../src/data/duolingo.js';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

test('MAE à mão: |1−0,8| + |0−0,1| + |0,5−0,5| = 0,3 → 0,1', () => {
  close(mae([1, 0, 0.5], [0.8, 0.1, 0.5]), 0.1);
});

test('AUC à mão, sem empate: 3 de 4 pares certos = 0,75', () => {
  // positivos 0,9 e 0,4; negativos 0,6 e 0,1 → (0,9>0,6)(0,9>0,1)(0,4<0,6)(0,4>0,1)
  close(auc([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1]), 0.75);
});

test('AUC à mão, com empate: empate vale 1/2', () => {
  close(auc([1, 0], [0.5, 0.5]), 0.5);
  // positivos 0,8 e 0,5; negativo 0,5 → 1 + 0,5 = 1,5 de 2
  close(auc([1, 1, 0], [0.8, 0.5, 0.5]), 0.75);
  close(auc([1, 1, 0], [0.1, 0.2, 0.9]), 0);
  assert.ok(Number.isNaN(auc([1, 1], [0.1, 0.2])));
});

test('AUC com direção automática do pROC, à mão', () => {
  // negativos 0,8 e 0,9 (mediana 0,85) acima dos positivos 0,1 e 0,2 (mediana 0,15) → inverte: 1 − 0 = 1
  close(aucAuto([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]), 1);
  // medianas iguais → direção "<" (pROC usa <=) → AUC crua
  close(aucAuto([1, 0], [0.5, 0.5]), 0.5);
  // positivos acima → igual à crua
  close(aucAuto([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1]), 0.75);
  // mediana dos negativos (0,55) > a dos positivos (0,5), embora a crua seja 0,5 → 1 − 0,5 = 0,5
  close(aucAuto([1, 1, 0, 0], [0.9, 0.1, 0.5, 0.6]), 0.5);
});

test('rótulo: round do R leva 0,5 a 0; a variante ≥ leva a 1', () => {
  assert.deepEqual([0.4999, 0.5, 0.5001, 1].map(labelR), [0, 0, 1, 1]);
  assert.deepEqual([0.4999, 0.5, 0.5001, 1].map(labelGe), [0, 1, 1, 1]);
});

test('postos: empate recebe posto médio', () => {
  assert.deepEqual(Array.from(ranks([10, 30, 20, 20])), [1, 4, 2.5, 2.5]);
});

test('Pearson à mão: x=1..4, y=1,2,3,100 → 149/√36025', () => {
  close(pearson([1, 2, 3, 4], [1, 2, 3, 100]), 149 / Math.sqrt(36025));
});

test('Spearman à mão: monótona = 1; com empate em x = 4,5/√22,5', () => {
  close(spearman([1, 2, 3, 4], [1, 2, 3, 100]), 1);
  close(spearman([1, 2, 2, 3], [1, 2, 3, 4]), 4.5 / Math.sqrt(22.5));
  close(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1);
});

test('round4: 4 casas como o .preds', () => {
  assert.deepEqual(Array.from(round4([0.123456, 0.99995])), [0.1235, 1]);
});

test('evaluate: sem hhat, COR_h fica null', () => {
  const r = evaluate({ p: [1, 0, 1, 0.5], phat: [0.9, 0.2, 0.7, 0.6], h: [1, 2, 3, 4], hhat: null });
  close(r.mae, (0.1 + 0.2 + 0.3 + 0.1) / 4);
  close(r.auc, 1);        // 0,5 vira negativo; positivos 0,9 e 0,7 > negativos 0,2 e 0,6
  close(r.aucGe, 1);      // 0,5 vira positivo: o único negativo (0,2) fica abaixo de todos
  assert.equal(r.corH, null);
});

test('Leitner à mão: 5 acertos, 1 erro → h = 2^4 = 16; Δ=2 → p = 2^(−1/8)', () => {
  const r = leitner({ t: 2, right: 5, wrong: 1 });
  assert.equal(r.h, 16); close(r.p, 2 ** -0.125);
  assert.equal(leitner({ t: 1, right: 2000, wrong: 0 }).h, MAX_HALF_LIFE);   // overflow → teto
  assert.equal(leitner({ t: 1, right: 0, wrong: 30 }).h, MIN_HALF_LIFE);
});

test('Pimsleur à mão: n = 6 → h = 2^(14,1 − 16,46) = 2^(−2,36)', () => {
  const r = pimsleur({ t: 2, right: 5, wrong: 1 });
  close(r.h, 2 ** -2.36); close(r.p, Math.max(2 ** (-2 / 2 ** -2.36), 0.0001));
  assert.equal(pimsleur({ t: 1, right: 100, wrong: 0 }).h, MAX_HALF_LIFE);
});

test('constante: devolve c e h null', () => {
  assert.deepEqual(constant(0.9)({}), { p: 0.9, h: null });
});
