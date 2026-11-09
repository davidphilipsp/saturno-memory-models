// Métricas da Tabela 2 (§4.1), como calculadas em evaluation.r.
// Entradas: arrays (ou typed arrays) de mesmo tamanho.

// MAE — exp:185-187, eval:9
export function mae(y, yhat) {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += Math.abs(y[i] - yhat[i]);
  return s / y.length;
}

// Rótulo binário da AUC. eval:11 usa round(p) do R, que arredonda 0,5 para o par (0).
export const labelR = p => (p > 0.5 ? 1 : 0);
export const labelGe = p => (p >= 0.5 ? 1 : 0);

// Postos 1..n, empates recebem o posto médio
export function ranks(x) {
  const n = x.length;
  const order = new Uint32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => x[a] - x[b]);
  const r = new Float64Array(n);
  for (let i = 0; i < n;) {
    let j = i;
    while (j + 1 < n && x[order[j + 1]] === x[order[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[order[k]] = avg;
    i = j + 1;
  }
  return r;
}

// AUC = estatística de Mann-Whitney normalizada; empate conta 1/2 (o mesmo que pROC::roc)
export function auc(labels, scores) {
  const r = ranks(scores);
  let n1 = 0, sum = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] === 1) { n1++; sum += r[i]; }
  const n0 = labels.length - n1;
  if (n1 === 0 || n0 === 0) return NaN;
  return (sum - (n1 * (n1 + 1)) / 2) / (n1 * n0);
}

function median(xs) {
  const a = Float64Array.from(xs).sort(), m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

// AUC como o pROC::roc() calcula por padrão (direction = "auto", pROC R/roc.R:139, 350-355):
// se a mediana do escore nos negativos passa a dos positivos, inverte a comparação → AUC ≥ 0,5 quase sempre.
// Premia ranqueamento invertido. Serve para comparar com a Tabela 2; não decide a escada.
export function aucAuto(labels, scores) {
  const neg = [], pos = [];
  for (let i = 0; i < labels.length; i++) (labels[i] === 1 ? pos : neg).push(scores[i]);
  const a = auc(labels, scores);
  return median(neg) > median(pos) ? 1 - a : a;
}

export function pearson(x, y) {
  const n = x.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx, b = y[i] - my;
    num += a * b; dx += a * a; dy += b * b;
  }
  return num / Math.sqrt(dx * dy);
}

// Spearman = Pearson dos postos (eval:14, method='s'). O "spearmanr" de exp:195-206 é Pearson cru (D1).
export const spearman = (x, y) => pearson(ranks(x), ranks(y));

// O R lê o arquivo .preds, gravado com 4 casas (exp:162). Isso cria empates.
export const round4 = x => Float64Array.from(x, v => Math.round(v * 1e4) / 1e4);

// Pacote da Tabela 2. hhat = null → COR_h n/a.
export function evaluate({ p, phat, h, hhat }) {
  const P = round4(p), PP = round4(phat);
  const out = {
    n: p.length,
    mae: mae(p, phat),
    auc: auc(Array.from(P, labelR), PP),
    aucGe: auc(Array.from(P, labelGe), PP),
    aucAuto: aucAuto(Array.from(P, labelR), PP),
    corH: null, corHPearson: null,
  };
  if (hhat) {
    const H = round4(h), HH = round4(hhat);
    out.corH = spearman(H, HH);
    out.corHPearson = pearson(H, HH);
  }
  return out;
}
