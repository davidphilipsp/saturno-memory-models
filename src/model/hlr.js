// B2 — half-life regression. A rede prevê log2(ĥ) = Θ·x; ĥ = 2^(Θ·x); p̂ = 2^(−Δ/ĥ). experiment.py:49-59, 82-99.
// Features: √(1+acertos), √(1+erros), bias (exp:244-250) + um peso por lexema (exp:251-252).
// Loss: (p − p̂)² + α(h − ĥ)² + λ‖Θ‖² (§3.3, Apêndice A.3). O passo do código é a derivada exata dessa loss
// em relação a Θ·x (conferido em test/hlr.test.mjs contra tf.grad), com os valores cortados de p̂ e ĥ.
// Treino: 1 passada embaralhada, lote 1, taxa (1/(1+p))·η/√(1 + aparições) (exp:89, D2), depois L2 (exp:97, D3).
import * as tf from '@tensorflow/tfjs';
import { pclip, hclip, MIN_HALF_LIFE, MAX_HALF_LIFE } from '../data/duolingo.js';
import { shuffled } from './lr.js';

export const HLR_DEFAULTS = { lrate: 0.001, hlwt: 0.01, l2wt: 0.1, sigma: 1, seed: 20261001, omitLexemes: false, omitHTerm: false };   // exp:37
const LN2 = Math.LN2;

export const hlrX = (cols, i) => [Math.sqrt(1 + cols.right[i]), Math.sqrt(1 + cols.wrong[i]), 1];

// Gradiente do código em relação a dp = Θ·x (exp:86-87); já soma o termo de h com peso α
export function dLossDdp({ phat, hhat, p, h, t, hlwt, omitHTerm }) {
  const dlp = 2 * (phat - p) * LN2 ** 2 * phat * (t / hhat);
  const dlh = 2 * (hhat - h) * LN2 * hhat;
  return omitHTerm ? dlp : dlp + hlwt * dlh;
}

function forward(dp, t) {
  const hhat = hclip(2 ** dp);
  return { hhat, phat: pclip(2 ** (-t / hhat)) };
}

// Réplica em JS puro (float64): conferência do TF.js e o arquivo completo.
// paperRate = true: tira o fator 1/(1+p) (D2) e usa L2 2λθ (D3), como no Apêndice A.3.
export function trainHlrJs(cols, trainIdx, opts = {}) {
  const { lrate, hlwt, l2wt, sigma, seed, omitLexemes, omitHTerm, paperRate = false } = { ...HLR_DEFAULTS, ...opts };
  const w = new Float64Array(3), lexW = new Float64Array(cols.vocabSize), lexCount = new Int32Array(cols.vocabSize);
  const order = shuffled(trainIdx, seed), l2 = (paperRate ? 2 : 1) * l2wt / sigma ** 2;
  for (let s = 0; s < order.length; s++) {
    const i = order[s], x = hlrX(cols, i), l = cols.lex[i], p = cols.p[i];
    let dp = w[0] * x[0] + w[1] * x[1] + w[2] * x[2];
    if (!omitLexemes) dp += lexW[l];
    const { phat, hhat } = forward(dp, cols.t[i]);
    const g = dLossDdp({ phat, hhat, p, h: cols.h[i], t: cols.t[i], hlwt, omitHTerm });
    const pf = paperRate ? 1 : 1 / (1 + p);
    const rate = pf * lrate / Math.sqrt(1 + s);       // as 3 densas aparecem em toda linha
    for (let k = 0; k < 3; k++) { w[k] -= rate * g * x[k]; w[k] -= rate * l2 * w[k]; }
    if (!omitLexemes) {
      const rl = pf * lrate / Math.sqrt(1 + lexCount[l]++);
      lexW[l] -= rl * g; lexW[l] -= rl * l2 * lexW[l];
    }
  }
  return { w, lexW, omitLexemes };
}

// O mesmo algoritmo em TF.js: ida (2^(Θ·x), 2^(−Δ/ĥ), cortes) e passo como ops de tensor.
export function trainHlrTf(cols, trainIdx, opts = {}) {
  const { lrate, hlwt, l2wt, sigma, seed, omitLexemes, omitHTerm } = { ...HLR_DEFAULTS, ...opts };
  const w = tf.variable(tf.zeros([3])), lexW = tf.variable(tf.zeros([cols.vocabSize]));
  const lexCount = new Int32Array(cols.vocabSize);
  const order = shuffled(trainIdx, seed), l2 = l2wt / sigma ** 2;
  for (let s = 0; s < order.length; s++) {
    const i = order[s], l = cols.lex[i], p = cols.p[i], t = cols.t[i], h = cols.h[i];
    const rate = lrate / (1 + p) / Math.sqrt(1 + s);
    const rl = omitLexemes ? 0 : lrate / (1 + p) / Math.sqrt(1 + lexCount[l]++);
    tf.tidy(() => {
      const x = tf.tensor1d(hlrX(cols, i)), li = tf.tensor1d([l], 'int32');
      let dp = tf.sum(tf.mul(w, x));
      if (!omitLexemes) dp = tf.add(dp, tf.gather(lexW, li));
      const hhat = tf.clipByValue(tf.pow(2, dp), MIN_HALF_LIFE, MAX_HALF_LIFE);
      const phat = tf.clipByValue(tf.pow(2, tf.div(-t, hhat)), 0.0001, 0.9999);
      const dlp = tf.mul(tf.mul(tf.sub(phat, p), 2 * LN2 ** 2), tf.mul(phat, tf.div(t, hhat)));
      const g = omitHTerm ? dlp : tf.add(dlp, tf.mul(tf.mul(tf.sub(hhat, h), 2 * LN2 * hlwt), hhat));
      const w1 = tf.sub(w, tf.mul(tf.mul(g, x), rate));
      w.assign(tf.sub(w1, tf.mul(w1, rate * l2)));
      if (!omitLexemes) {
        const lw1 = tf.sub(tf.gather(lexW, li), tf.mul(g, rl));
        lexW.assign(tf.tensorScatterUpdate(lexW, li.expandDims(1), tf.sub(lw1, tf.mul(lw1, rl * l2))));
      }
    });
  }
  const out = { w: Float64Array.from(w.dataSync()), lexW: Float64Array.from(lexW.dataSync()), omitLexemes };
  w.dispose(); lexW.dispose();
  return out;
}

export function predictHlr(model, cols, idx) {
  const phat = new Float64Array(idx.length), hhat = new Float64Array(idx.length);
  for (let j = 0; j < idx.length; j++) {
    const i = idx[j], x = hlrX(cols, i);
    let dp = model.w[0] * x[0] + model.w[1] * x[1] + model.w[2] * x[2];
    if (!model.omitLexemes) dp += model.lexW[cols.lex[i]];
    const f = forward(dp, cols.t[i]);
    phat[j] = f.phat; hhat[j] = f.hhat;
  }
  return { phat, hhat };
}

export const hlrParams = (model, cols) => 3 + (model.omitLexemes ? 0 : cols.vocabSize - 1);
