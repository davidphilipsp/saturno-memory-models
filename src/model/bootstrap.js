import { auc, round4, labelR } from './metrics.js';

// Bootstrap pareado por aprendiz: reamostra user_id com reposição; cada usuário leva todas as suas linhas.
// Diferença = MAE(novo) − MAE(anterior), ponderada por linha. Negativa = o novo erra menos.

// PRNG com semente (mulberry32): mesma semente, mesmas reamostragens
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Quantil tipo 7 (padrão do R): interpolação linear entre as estatísticas de ordem
export function quantile(sorted, q) {
  const h = (sorted.length - 1) * q, lo = Math.floor(h);
  return lo + 1 < sorted.length ? sorted[lo] + (h - lo) * (sorted[lo + 1] - sorted[lo]) : sorted[lo];
}

// errNew, errPrev: |p − p̂| por linha; groups: user_id por linha
export function pairedBootstrap({ errNew, errPrev, groups, B = 1000, seed = 20261001, alpha = 0.05 }) {
  const idx = new Map();
  const sNew = [], sPrev = [], cnt = [];
  for (let i = 0; i < groups.length; i++) {
    let g = idx.get(groups[i]);
    if (g === undefined) { g = cnt.length; idx.set(groups[i], g); sNew.push(0); sPrev.push(0); cnt.push(0); }
    sNew[g] += errNew[i]; sPrev[g] += errPrev[i]; cnt[g]++;
  }
  const U = cnt.length, rand = mulberry32(seed);
  const diffs = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    let a = 0, c = 0, n = 0;
    for (let u = 0; u < U; u++) {
      const g = Math.floor(rand() * U);
      a += sNew[g]; c += sPrev[g]; n += cnt[g];
    }
    diffs[b] = (a - c) / n;
  }
  const total = cnt.reduce((s, x) => s + x, 0);
  const point = (sNew.reduce((s, x) => s + x, 0) - sPrev.reduce((s, x) => s + x, 0)) / total;
  diffs.sort();
  const lo = quantile(diffs, alpha / 2), hi = quantile(diffs, 1 - alpha / 2);
  return { point, lo, mid: quantile(diffs, 0.5), hi, users: U, B, seed, excludesZero: hi < 0 || lo > 0 };
}

// Veredito da escada. Regra de 2026-10-01, com emenda de 2026-10-02 (posterior ao resultado da B2):
// vence se o IC 95% da diferença de MAE fica todo abaixo de zero; "sem discriminar" só se, além disso,
// o IC 95% da diferença de AUC (novo − anterior, mesmo bootstrap) também fica todo abaixo de zero.
export function verdict(maeBoot, aucDiffBoot) {
  if (!(maeBoot.hi < 0)) return 'não venceu';
  return aucDiffBoot.hi < 0 ? 'venceu sem discriminar' : 'venceu';
}

// Mesmo bootstrap pareado por aprendiz, para qualquer estatística das linhas (ex.: diferença de AUC).
// stat(rows) recebe os índices das linhas da reamostra (com repetição) e devolve um número.
// Usado no veredito para a diferença de AUC (emenda de 2026-10-02).
export function pairedBootstrapStat({ groups, stat, B = 1000, seed = 20261001, alpha = 0.05 }) {
  const byUser = new Map();
  for (let i = 0; i < groups.length; i++) {
    if (!byUser.has(groups[i])) byUser.set(groups[i], []);
    byUser.get(groups[i]).push(i);
  }
  const lists = [...byUser.values()], U = lists.length, rand = mulberry32(seed);
  const all = Array.from({ length: groups.length }, (_, i) => i);   // array comum: .map de typed array truncaria escores
  const vals = new Float64Array(B);
  for (let b = 0; b < B; b++) {
    const rows = [];
    for (let u = 0; u < U; u++) for (const i of lists[Math.floor(rand() * U)]) rows.push(i);
    vals[b] = stat(rows);
  }
  vals.sort();
  return { point: stat(all), lo: quantile(vals, alpha / 2), mid: quantile(vals, 0.5), hi: quantile(vals, 1 - alpha / 2), users: U, B, seed };
}

// IC da diferença de AUC (direção fixa, rótulo do R, previsões com 4 casas como o .preds)
export function aucDiffBootstrap({ p, phatNew, phatPrev, groups, ...opts }) {
  const y = Array.from(round4(p), labelR), a = round4(phatNew), b = round4(phatPrev);
  const sub = (arr, rows) => rows.map(i => arr[i]);
  return pairedBootstrapStat({ groups, ...opts, stat: rows => auc(sub(y, rows), sub(a, rows)) - auc(sub(y, rows), sub(b, rows)) });
}
