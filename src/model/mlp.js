// B3 — MLP com a cabeça da HLR: a rede prevê log2(ĥ); p̂ = 2^(−Δ/ĥ). Proposta aprovada em 2026-10-02.
// Entrada: √(1+acertos), √(1+erros) (as mesmas da HLR) + embedding 8 do lexema (gradiente esparso).
// 10 → 32 ReLU → dropout 0,2 → 1 (z). log2 ĥ = LO + (HI − LO)·sigmoide(z): corte suave em [15 min; 274 d].
// Loss: média no lote de (p − p̂)² + α(h − ĥ)², α = 0,01 (§3.3). Adam 0,001, lote 1024.
import * as tf from '@tensorflow/tfjs';
import { MIN_HALF_LIFE, MAX_HALF_LIFE, pclip, splitPoint } from '../data/duolingo.js';
import { embeddingLookup } from './embedding.js';
import { shuffled } from './lr.js';

export const LO = Math.log2(MIN_HALF_LIFE), HI = Math.log2(MAX_HALF_LIFE);
export const MLP_DEFAULTS = { emb: 8, hidden: 32, dropout: 0.2, alpha: 0.01, lrate: 0.001, batchSize: 1024, useLex: true, arch: 'mlp' };
// arch 'linear' = controle do Q11: o modelo da HLR (log2 ĥ = Θ·x, corte duro = hclip, peso escalar por lexema,
// tudo começando em 0) treinado exatamente como o MLP (Adam, lote, loss, α, early stopping, sementes).
// initLog2h (Q11b, opção A, 2026-10-02): início igual para os dois. Controle: bias = initLog2h.
// MLP: b2 = logit((initLog2h − LO)/(HI − LO)), o z em que a cabeça suave devolve initLog2h. Sem a opção: B3 (b2 = 0).

// Cabeça limitada: z → log2 ĥ ∈ (LO, HI)
export const boundedLog2H = z => tf.add(LO, tf.mul(HI - LO, tf.sigmoid(z)));
// p̂ = 2^(−Δ/ĥ)
export const pFromLog2H = (log2h, t) => tf.pow(2, tf.neg(tf.div(t, tf.pow(2, log2h))));

// Cabeça da HLR: corte duro de log2 ĥ em [LO, HI] (= hclip, exp:180-182)
export const hardLog2H = z => tf.clipByValue(z, LO, HI);
export const headOf = m => (m.opts.arch === 'linear' ? hardLog2H : boundedLog2H);

export function lossFromZ(z, { p, h, t }, alpha, head = boundedLog2H) {
  const log2h = head(z);
  const hhat = tf.pow(2, log2h), phat = pFromLog2H(log2h, t);
  return tf.mean(tf.add(tf.square(tf.sub(p, phat)), tf.mul(alpha, tf.square(tf.sub(h, hhat)))));
}

// Partições da opção (b), pelo índice no arquivo completo
export function stageOf(index, total) {
  const v0 = Math.floor(0.81 * total), sp = splitPoint(total);
  return index < v0 ? 'step1' : index < sp ? 'val' : 'test';
}

// Early stopping pela MAE de validação: para depois de `patience` épocas seguidas sem melhora estrita
export class EarlyStopper {
  constructor(patience = 3) { this.patience = patience; this.best = Infinity; this.bestEpoch = 0; this.bad = 0; this.epoch = 0; }
  update(metric) {
    this.epoch++;
    if (metric < this.best) { this.best = metric; this.bestEpoch = this.epoch; this.bad = 0; } else this.bad++;
    return this.bad >= this.patience ? 'stop' : 'continue';
  }
}

// z da cabeça suave que dá log2 ĥ = v
export const zForLog2H = v => { const u = (v - LO) / (HI - LO); return Math.log(u / (1 - u)); };

export function createMlp(vocabSize, opts = {}, seed = 0) {
  const o = { ...MLP_DEFAULTS, ...opts };
  if (o.arch === 'linear') {
    const vars = { w: tf.variable(tf.zeros([2, 1])), b: tf.variable(tf.fill([1], o.initLog2h ?? 0)) };
    if (o.useLex) vars.L = tf.variable(tf.zeros([vocabSize, 1]));
    return { vars, opts: o, seed };
  }
  const inDim = 2 + (o.useLex ? o.emb : 0);
  const vars = {
    W1: tf.variable(tf.initializers.glorotUniform({ seed }).apply([inDim, o.hidden])),
    b1: tf.variable(tf.zeros([o.hidden])),
    W2: tf.variable(tf.initializers.glorotUniform({ seed: seed + 1 }).apply([o.hidden, 1])),
    b2: tf.variable(o.initLog2h === undefined ? tf.zeros([1]) : tf.fill([1], zForLog2H(o.initLog2h))),
  };
  if (o.useLex) {
    // linha 0 = <unk>: zero. Não recebe gradiente porque todo lexema do treino está no vocabulário.
    const e = tf.randomNormal([vocabSize, o.emb], 0, 0.05, 'float32', seed + 2);
    vars.E = tf.variable(tf.concat([tf.zeros([1, o.emb]), e.slice([1, 0], [vocabSize - 1, o.emb])]));
    e.dispose();
  }
  return { vars, opts: o, seed };
}

export const mlpParams = m => Object.values(m.vars).reduce((s, v) => s + v.size, 0);

function zOf(m, dense, lex, training, dropSeed) {
  if (m.opts.arch === 'linear') {
    const { w, b, L } = m.vars;
    let z = tf.add(tf.matMul(dense, w), b);
    if (m.opts.useLex) z = tf.add(z, embeddingLookup(L, lex));
    return z.reshape([-1]);
  }
  const { W1, b1, W2, b2, E } = m.vars;
  const x = m.opts.useLex ? tf.concat([dense, embeddingLookup(E, lex)], 1) : dense;
  let a = tf.relu(tf.add(tf.matMul(x, W1), b1));
  if (training) a = tf.dropout(a, m.opts.dropout, null, dropSeed);
  return tf.add(tf.matMul(a, W2), b2).reshape([-1]);
}

function batchTensors(cols, idx) {
  const n = idx.length, d = new Float32Array(n * 2), l = new Int32Array(n);
  const p = new Float32Array(n), h = new Float32Array(n), t = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const i = idx[j];
    d[2 * j] = Math.sqrt(1 + cols.right[i]); d[2 * j + 1] = Math.sqrt(1 + cols.wrong[i]);
    l[j] = cols.lex[i]; p[j] = cols.p[i]; h[j] = cols.h[i]; t[j] = cols.t[i];
  }
  return { dense: tf.tensor2d(d, [n, 2]), lex: tf.tensor1d(l, 'int32'), p: tf.tensor1d(p), h: tf.tensor1d(h), t: tf.tensor1d(t) };
}

// Uma época: embaralha com (semente, época), lotes de batchSize. Devolve a loss média de treino.
export function trainEpoch(m, opt, cols, trainIdx, epoch) {
  const order = shuffled(trainIdx, m.seed * 1000 + epoch), bs = m.opts.batchSize;
  let sum = 0, n = 0;
  for (let s = 0; s < order.length; s += bs) {
    const idx = order.subarray(s, Math.min(s + bs, order.length));
    const loss = tf.tidy(() => {
      const b = batchTensors(cols, idx);
      return opt.minimize(() => lossFromZ(zOf(m, b.dense, b.lex, true, m.seed * 1e6 + epoch * 1e3 + s / bs), b, m.opts.alpha, headOf(m)), true);
    });
    sum += loss.dataSync()[0] * idx.length; n += idx.length;
    loss.dispose();
  }
  return sum / n;
}

// Previsão sem dropout, em blocos. p̂ passa por pclip como nos outros modelos.
export function predictMlp(m, cols, idx, { withLoss = false } = {}) {
  const phat = new Float64Array(idx.length), hhat = new Float64Array(idx.length);
  let lossSum = 0;
  for (let s = 0; s < idx.length; s += 8192) {
    const part = idx.subarray ? idx.subarray(s, s + 8192) : idx.slice(s, s + 8192);
    tf.tidy(() => {
      const b = batchTensors(cols, part), z = zOf(m, b.dense, b.lex, false);
      const log2h = headOf(m)(z);
      const ph = pFromLog2H(log2h, b.t).dataSync(), hh = tf.pow(2, log2h).dataSync();
      for (let j = 0; j < part.length; j++) { phat[s + j] = pclip(ph[j]); hhat[s + j] = hh[j]; }
      if (withLoss) lossSum += lossFromZ(z, b, m.opts.alpha, headOf(m)).dataSync()[0] * part.length;
    });
  }
  return { phat, hhat, loss: withLoss ? lossSum / idx.length : null };
}

export function disposeMlp(m) { for (const v of Object.values(m.vars)) v.dispose(); }
