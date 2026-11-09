// Fase A, passo 4: tempo de 1 época na amostra.
// Uso: node src/model/bench.js [lr|mlp] [batchSize]
//   lr  — regressão logística, 1 neurônio, features densas de 'lr'
//   mlp — embedding de lexema (16, gradiente esparso) + oculta 32 + dropout: o formato que a B3 vai ter
// Sem embaralhar no mlp: só mede tempo.
import * as tf from '@tensorflow/tfjs';
import { loadSample } from '../data/sample.js';
import { buildVocab } from '../data/duolingo.js';
import { embeddingLookup } from './embedding.js';

const kind = process.argv[2] || 'lr';
const batchSize = Number(process.argv[3] || 1024);
await tf.ready();
const t0 = Date.now();
const rows = (await loadSample()).filter(r => r.split === 'train');
const tLoad = (Date.now() - t0) / 1000;

const n = rows.length;
const vocab = buildVocab(rows);
const X = new Float32Array(n * 3), L = new Int32Array(n), Y = new Float32Array(n);
rows.forEach((r, i) => {
  X.set([Math.sqrt(1 + r.right), Math.sqrt(1 + r.wrong), r.t], i * 3);   // features de 'lr' sem o bias (a Dense tem bias)
  L[i] = vocab.get(r.lexemeKey);
  Y[i] = r.p;
});
const xs = tf.tensor2d(X, [n, 3]), ls = tf.tensor2d(L, [n, 1], 'int32'), ys = tf.tensor2d(Y, [n, 1]);

let fit, params;
if (kind === 'lr') {
  const model = tf.sequential({ layers: [tf.layers.dense({ units: 1, inputShape: [3], activation: 'sigmoid' })] });
  model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
  fit = async () => (await model.fit(xs, ys, { epochs: 1, batchSize, shuffle: true, verbose: 0 })).history.loss[0];
  params = model.countParams();
} else {
  // embedding próprio (gradiente esparso) — tf.layers.embedding não termina 1 época no backend cpu
  const D = 16, H = 32;
  const E = tf.variable(tf.randomNormal([vocab.size, D], 0, 0.05));
  const W1 = tf.variable(tf.randomNormal([3 + D, H], 0, 0.2)), b1 = tf.variable(tf.zeros([H]));
  const W2 = tf.variable(tf.randomNormal([H, 1], 0, 0.2)), b2 = tf.variable(tf.zeros([1]));
  const opt = tf.train.adam(0.01);
  const lsFlat = ls.reshape([n]);
  fit = async () => {
    let last;
    for (let s = 0; s < n; s += batchSize) {
      const b = Math.min(batchSize, n - s);
      last?.dispose();
      last = tf.tidy(() => {
        const x = xs.slice([s, 0], [b, 3]), l = lsFlat.slice([s], [b]), y = ys.slice([s, 0], [b, 1]);
        return opt.minimize(() => {
          const h = tf.dropout(tf.relu(tf.add(tf.matMul(tf.concat([x, embeddingLookup(E, l)], 1), W1), b1)), 0.2);
          return tf.losses.meanSquaredError(y, tf.sigmoid(tf.add(tf.matMul(h, W2), b2)));
        }, true);
      });
    }
    return (await last.data())[0];
  };
  params = [E, W1, b1, W2, b2].reduce((a, v) => a + v.size, 0);
}

const t1 = Date.now();
const loss = await fit();
const tEpoch = (Date.now() - t1) / 1000;
console.log(JSON.stringify({
  model: kind, backend: tf.getBackend(), tfjs: tf.version.tfjs, trainRows: n, vocab: vocab.size, batchSize,
  loadSeconds: tLoad, epochSeconds: tEpoch, rowsPerSecond: Math.round(n / tEpoch), loss, params,
}));
