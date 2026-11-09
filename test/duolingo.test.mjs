import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import {
  readRows, toInstance, parseLexeme, denseFeatures, splitCsv, splitPoint, partitionOf,
  sampleInstances, buildVocab, unitHash, countLines, pclip, hclip, MIN_HALF_LIFE, MAX_HALF_LIFE,
} from '../src/data/duolingo.js';

const FIX = path.resolve(import.meta.dirname, 'fixtures/duolingo-mini.csv');
const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const all = async gen => { const out = []; for await (const x of gen) out.push(x); return out; };

test('splitCsv: vírgula dentro de aspas', () => {
  assert.deepEqual(splitCsv('1,"a,b/a<det>",2'), ['1', 'a,b/a<det>', '2']);
  assert.deepEqual(splitCsv('x,"say ""hi""",y'), ['x', 'say "hi"', 'y']);
});

test('readRows: 10 linhas, cabeçalho vira chave, índice sequencial', async () => {
  const rows = await all(readRows(FIX));
  assert.equal(rows.length, 10);
  assert.equal(rows[4].lexeme_string, 'a,b/a<det><ind><sg>');
  assert.deepEqual(rows.map(r => r._index), [...Array(10).keys()]);
  assert.equal((await all(readRows(FIX, { maxLines: 3 }))).length, 3);
});

test('readRows: lê .gz igual ao .csv', async () => {
  const gz = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'duo-')), 'mini.csv.gz');
  fs.writeFileSync(gz, zlib.gzipSync(fs.readFileSync(FIX)));
  assert.deepEqual(await all(readRows(gz)), await all(readRows(FIX)));
  assert.equal(await countLines(gz), 10);
});

test('clip: limites de experiment.py', () => {
  assert.equal(pclip(1), 0.9999);
  assert.equal(pclip(0), 0.0001);
  assert.equal(hclip(0), MIN_HALF_LIFE);
  assert.equal(hclip(1e9), MAX_HALF_LIFE);
});

test('toInstance: p, t, h derivados como em read_data', async () => {
  const [r0, r1, r2] = (await all(readRows(FIX))).map(toInstance);
  // p=0.5, Δ=2 dias → h = 2 / 1 = 2
  close(r1.p, 0.5); close(r1.t, 2); close(r1.h, 2);
  assert.equal(r1.right, 5); assert.equal(r1.wrong, 1);
  assert.equal(r1.sessionSeen, 4); assert.equal(r1.sessionRight, 2);
  // p=1 vira 0.9999; h = 1 / 0.000144… ≈ 6931 → teto de 274
  close(r0.p, 0.9999); assert.equal(r0.h, MAX_HALF_LIFE);
  // p=0, Δ=0 → h = 0 → piso de 15 min
  assert.equal(r2.h, MIN_HALF_LIFE);
  assert.equal(r1.lang, 'en->es');
  assert.equal(r1.lexemeKey, 'es:lees/leer<vblex><pri><p2><sg>');
});

test('parseLexeme: superfície, lema, pos, modificadores e curinga', () => {
  assert.deepEqual(parseLexeme('lees/leer<vblex><pri><p2><sg>'),
    { surface: 'lees', lemma: 'leer', pos: 'vblex', modifiers: ['pri', 'p2', 'sg'] });
  assert.deepEqual(parseLexeme('<*sf>/sein<vbser><*numb>'),
    { surface: '<*sf>', lemma: 'sein', pos: 'vbser', modifiers: ['*numb'] });
  assert.deepEqual(parseLexeme('ciao/ciao<ij>'), { surface: 'ciao', lemma: 'ciao', pos: 'ij', modifiers: [] });
});

test('denseFeatures: por método, como experiment.py:237-250', async () => {
  const inst = toInstance((await all(readRows(FIX)))[3]);   // seen 10, correct 7, Δ 0.5 dia
  assert.deepEqual(denseFeatures(inst, 'leitner'), [4]);
  assert.deepEqual(denseFeatures(inst, 'pimsleur'), [10]);
  assert.deepEqual(denseFeatures(inst, 'hlr'), [Math.sqrt(8), Math.sqrt(4), 1]);
  assert.deepEqual(denseFeatures(inst, 'lr'), [Math.sqrt(8), Math.sqrt(4), 0.5, 1]);
});

test('split: primeiros 90% treino, ordem do arquivo', () => {
  assert.equal(splitPoint(10), 9);
  assert.equal(partitionOf(8, 10), 'train');
  assert.equal(partitionOf(9, 10), 'test');
  assert.equal(splitPoint(12854226), 11568803);
});

test('unitHash: determinístico e ~uniforme', () => {
  assert.equal(unitHash(42, 7), unitHash(42, 7));
  assert.notEqual(unitHash(42, 7), unitHash(42, 8));
  let s = 0; const n = 100000;
  for (let i = 0; i < n; i++) s += unitHash(i) < 0.1;
  assert.ok(Math.abs(s / n - 0.1) < 0.005, `taxa ${s / n}`);
});

test('sampleInstances: a partição vem do índice no arquivo completo, não da amostra', async () => {
  const full = await all(sampleInstances(FIX, { total: 10, rate: 1 }));
  assert.deepEqual(full.map(i => i.split), [...Array(9).fill('train'), 'test']);
  const half = await all(sampleInstances(FIX, { total: 10, rate: 0.5, seed: 3 }));
  for (const i of half) assert.equal(i.split, partitionOf(i.index, 10));
  assert.deepEqual(half, await all(sampleInstances(FIX, { total: 10, rate: 0.5, seed: 3 })));
});

test('buildVocab: só treino; 0 = desconhecido', async () => {
  const insts = await all(sampleInstances(FIX, { total: 10, rate: 1 }));
  const v = buildVocab(insts);
  assert.equal(v.get('<unk>'), 0);
  assert.equal(v.size, 1 + 8);       // 9 linhas de treino, uma repete o lexema id1
  assert.ok(!v.has('es:libro/libro<n><m><sg>'));   // só aparece no teste
});
