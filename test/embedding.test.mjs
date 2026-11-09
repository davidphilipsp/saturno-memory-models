import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import { embeddingLookup } from '../src/model/embedding.js';

test('embeddingLookup: ida = gather; gradiente = o do gather, somando índices repetidos', async () => {
  await tf.setBackend('cpu');
  const table = tf.tensor2d([[1, 2], [3, 4], [5, 6], [7, 8]]);
  const idx = tf.tensor1d([2, 0, 2], 'int32');
  assert.deepEqual(embeddingLookup(table, idx).arraySync(), [[5, 6], [1, 2], [5, 6]]);
  const w = tf.tensor2d([[1, 10], [100, 1000], [2, 20]]);
  const g = tf.grad(t => tf.sum(tf.mul(embeddingLookup(t, idx), w)))(table).arraySync();
  // linha 0 recebe w[1]; linha 2 recebe w[0] + w[2]; linhas 1 e 3 nada
  assert.deepEqual(g, [[100, 1000], [0, 0], [3, 30], [0, 0]]);
  const ref = tf.grad(t => tf.sum(tf.mul(tf.gather(t, idx), w)))(table).arraySync();
  assert.deepEqual(g, ref);
});
