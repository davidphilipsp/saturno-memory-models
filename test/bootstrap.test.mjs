import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mulberry32, quantile, pairedBootstrap, pairedBootstrapStat, aucDiffBootstrap, verdict } from '../src/model/bootstrap.js';

const close = (a, b, eps = 1e-12) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

test('mulberry32: mesma semente, mesma sequência; valores em [0,1)', () => {
  const a = mulberry32(7), b = mulberry32(7), c = mulberry32(8);
  const xs = Array.from({ length: 5 }, a);
  assert.deepEqual(xs, Array.from({ length: 5 }, b));
  assert.notDeepEqual(xs, Array.from({ length: 5 }, c));
  for (const x of xs) assert.ok(x >= 0 && x < 1);
});

test('quantil tipo 7 à mão: [1,2,3,4], q=0,25 → 1,75; q=0,5 → 2,5', () => {
  close(quantile([1, 2, 3, 4], 0.25), 1.75);
  close(quantile([1, 2, 3, 4], 0.5), 2.5);
  close(quantile([1, 2, 3, 4], 1), 4);
});

// Caso à mão, 2 aprendizes:
//   A: 1 linha, erro novo 0,3, erro anterior 0,1 → soma da diferença +0,2, 1 linha
//   B: 3 linhas, cada uma novo 0,1, anterior 0,2 → soma −0,3, 3 linhas
// Ponto (dado inteiro): (0,2 − 0,3) / 4 = −0,025
// Reamostras possíveis de 2 usuários: AA → 0,4/2 = +0,2 (prob 1/4); BB → −0,6/6 = −0,1 (1/4);
//   AB ou BA → −0,1/4 = −0,025 (1/2) → mediana −0,025 (ponderar por aprendiz daria +0,05).
// Com 1000 reamostras, P(menos de 26 BB ou de 26 AA) ≈ 0 → IC 95% = [−0,1; +0,2], cruza o zero.
const groups = ['A', 'B', 'B', 'B'];
const errNew = [0.3, 0.1, 0.1, 0.1], errPrev = [0.1, 0.2, 0.2, 0.2];

test('bootstrap pareado à mão: ponto −0,025, IC [−0,1; 0,2], cruza zero', () => {
  const r = pairedBootstrap({ errNew, errPrev, groups, B: 1000, seed: 1 });
  close(r.point, -0.025);
  close(r.lo, -0.1);
  close(r.hi, 0.2);
  close(r.mid, -0.025);
  assert.equal(r.users, 2);
  assert.equal(r.excludesZero, false);
  assert.equal(verdict(r, { hi: -0.01 }), 'não venceu');
});

test('bootstrap: só valores possíveis aparecem, nas proporções esperadas', () => {
  // reconta as 1000 reamostras com a mesma semente
  const rand = mulberry32(1);
  const tally = { AA: 0, BB: 0, AB: 0 };
  for (let b = 0; b < 1000; b++) {
    const u = [Math.floor(rand() * 2), Math.floor(rand() * 2)].sort().join('');
    tally[u === '00' ? 'AA' : u === '11' ? 'BB' : 'AB']++;
  }
  // 1/4, 1/4, 1/2 com folga de ±5 desvios-padrão
  assert.ok(Math.abs(tally.AA - 250) < 70 && Math.abs(tally.BB - 250) < 70 && Math.abs(tally.AB - 500) < 80, JSON.stringify(tally));
});

test('bootstrap: diferença igual em todo aprendiz → IC degenerado no ponto, exclui zero', () => {
  const r = pairedBootstrap({ errNew: [0.1, 0.1, 0.2], errPrev: [0.2, 0.2, 0.3], groups: ['x', 'y', 'z'], seed: 3 });
  close(r.point, -0.1); close(r.lo, -0.1, 1e-9); close(r.hi, -0.1, 1e-9);
  assert.equal(r.excludesZero, true);
});

test('bootstrap: mesma semente, mesmo IC', () => {
  assert.deepEqual(pairedBootstrap({ errNew, errPrev, groups, seed: 9 }), pairedBootstrap({ errNew, errPrev, groups, seed: 9 }));
});

test('veredito (regra emendada): "sem discriminar" só com IC da diferença de AUC todo abaixo de zero', () => {
  assert.equal(verdict({ hi: -0.01 }, { hi: 0.002 }), 'venceu');            // AUC pior no ponto, IC cruza zero
  assert.equal(verdict({ hi: -0.01 }, { hi: -0.001 }), 'venceu sem discriminar');
  assert.equal(verdict({ hi: 0.001 }, { hi: 0.5 }), 'não venceu');
});

test('aucDiffBootstrap: novo ranqueia perfeito, anterior invertido → ponto +1', () => {
  const r = aucDiffBootstrap({ p: [1, 1, 0, 0], phatNew: [0.9, 0.8, 0.2, 0.1], phatPrev: [0.1, 0.2, 0.8, 0.9], groups: ['a', 'b', 'c', 'd'], B: 50, seed: 4 });
  close(r.point, 1);
});

test('bootstrap de estatística: o mesmo caso à mão (média por linha da diferença) dá o mesmo IC', () => {
  const d = errNew.map((v, i) => v - errPrev[i]);           // A: +0,2; B: −0,1 ×3
  const mean = rows => rows.reduce((s, i) => s + d[i], 0) / rows.length;
  const r = pairedBootstrapStat({ groups, stat: mean, B: 1000, seed: 1 });
  close(r.point, -0.025); close(r.lo, -0.1); close(r.mid, -0.025); close(r.hi, 0.2);
  const ref = pairedBootstrap({ errNew, errPrev, groups, B: 1000, seed: 1 });
  close(r.lo, ref.lo); close(r.hi, ref.hi);
});

test('bootstrap de estatística: o ponto recebe array comum (escore fracionário não é truncado)', () => {
  const r = pairedBootstrapStat({ groups: ['a', 'b'], stat: rows => rows.map(i => [0.25, 0.75][i]).reduce((s, v) => s + v, 0), B: 10, seed: 2 });
  close(r.point, 1);
});
