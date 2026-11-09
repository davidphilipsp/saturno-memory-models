// B1 — regressão logística (um neurônio), algoritmo de experiment.py:75-78, 102-113, 115-120.
// Features: √(1+acertos), √(1+erros), Δ em dias, bias (exp:244-250) + um peso por lexema (exp:251-252).
// Treino: 1 passada embaralhada, lote 1, taxa por feature η/√(1 + nº de vezes que a feature apareceu),
// atualização do erro (p̂ − p)·x e depois L2: w −= taxa·λ·w/σ² (sobre o w já atualizado, como no Python).
import * as tf from '@tensorflow/tfjs';
import { pclip } from '../data/duolingo.js';
import { mulberry32 } from './bootstrap.js';

export const LR_DEFAULTS = { lrate: 0.001, l2wt: 0.1, sigma: 1, seed: 20261001, omitLexemes: false };   // exp:37
export const DENSE = ['right', 'wrong', 'time', 'bias'];

export const denseX = (cols, i) => [Math.sqrt(1 + cols.right[i]), Math.sqrt(1 + cols.wrong[i]), cols.t[i], 1];

// exp:118 — random.shuffle do treino. Aqui Fisher-Yates com semente.
export function shuffled(idx, seed) {
  const a = Int32Array.from(idx), rand = mulberry32(seed);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const sigmoid = z => 1 / (1 + Math.exp(-z));

// Réplica em JS puro (float64). Serve de conferência do TF.js e roda o arquivo completo em segundos.
// Fora do código oficial (diagnóstico da B1): epochs > 1 reembaralha com seed + época;
// constantRate = true troca η/√(1+contagem) por η fixo. onEpoch(e, model) recebe uma cópia ao fim de cada época.
export function trainLrJs(cols, trainIdx, opts = {}) {
  const { lrate, l2wt, sigma, seed, omitLexemes, epochs = 1, constantRate = false, onEpoch } = { ...LR_DEFAULTS, ...opts };
  const w = new Float64Array(4), lexW = new Float64Array(cols.vocabSize), lexCount = new Int32Array(cols.vocabSize);
  const l2 = l2wt / sigma ** 2;
  let step = 0;
  for (let e = 0; e < epochs; e++) {
    const order = shuffled(trainIdx, seed + e);
    for (let s = 0; s < order.length; s++, step++) {
      const i = order[s], x = denseX(cols, i), l = cols.lex[i];
      let dp = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3];
      if (!omitLexemes) dp += lexW[l];
      const err = pclip(sigmoid(dp)) - cols.p[i];
      const rate = constantRate ? lrate : lrate / Math.sqrt(1 + step);   // as 4 densas aparecem em toda linha: contagem = passo
      for (let k = 0; k < 4; k++) { w[k] -= rate * err * x[k]; w[k] -= rate * l2 * w[k]; }
      if (!omitLexemes) {
        const rl = constantRate ? lrate : lrate / Math.sqrt(1 + lexCount[l]++);
        lexW[l] -= rl * err; lexW[l] -= rl * l2 * lexW[l];
      }
    }
    onEpoch?.(e, { w: Float64Array.from(w), lexW: Float64Array.from(lexW), omitLexemes });
  }
  return { w, lexW, omitLexemes };
}

// O mesmo algoritmo em TF.js: o neurônio (soma ponderada → sigmoide) e a atualização são ops de tensor.
export function trainLrTf(cols, trainIdx, opts = {}) {
  const { lrate, l2wt, sigma, seed, omitLexemes } = { ...LR_DEFAULTS, ...opts };
  const w = tf.variable(tf.zeros([4])), lexW = tf.variable(tf.zeros([cols.vocabSize]));
  const lexCount = new Int32Array(cols.vocabSize);
  const order = shuffled(trainIdx, seed), l2 = l2wt / sigma ** 2;
  for (let s = 0; s < order.length; s++) {
    const i = order[s], l = cols.lex[i];
    const rate = lrate / Math.sqrt(1 + s);
    const rl = omitLexemes ? 0 : lrate / Math.sqrt(1 + lexCount[l]++);
    tf.tidy(() => {
      const x = tf.tensor1d(denseX(cols, i)), li = tf.tensor1d([l], 'int32');
      let dp = tf.sum(tf.mul(w, x));
      if (!omitLexemes) dp = tf.add(dp, tf.gather(lexW, li));
      const err = tf.sub(tf.clipByValue(tf.sigmoid(dp), 0.0001, 0.9999), cols.p[i]);
      const w1 = tf.sub(w, tf.mul(tf.mul(err, x), rate));
      w.assign(tf.sub(w1, tf.mul(w1, rate * l2)));
      if (!omitLexemes) {
        const lw1 = tf.sub(tf.gather(lexW, li), tf.mul(err, rl));
        lexW.assign(tf.tensorScatterUpdate(lexW, li.expandDims(1), tf.sub(lw1, tf.mul(lw1, rl * l2))));
      }
    });
  }
  const out = { w: Float64Array.from(w.dataSync()), lexW: Float64Array.from(lexW.dataSync()), omitLexemes };
  w.dispose(); lexW.dispose();
  return out;
}

export function predictLr(model, cols, idx) {
  const out = new Float64Array(idx.length);
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j], x = denseX(cols, i);
    let dp = model.w[0] * x[0] + model.w[1] * x[1] + model.w[2] * x[2] + model.w[3] * x[3];
    if (!model.omitLexemes) dp += model.lexW[cols.lex[i]];
    out[j] = pclip(sigmoid(dp));
  }
  return out;
}

export const lrParams = (model, cols) => 4 + (model.omitLexemes ? 0 : cols.vocabSize - 1);   // −1: o <unk> não treina
