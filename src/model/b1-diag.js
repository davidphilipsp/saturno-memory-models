// Diagnóstico da B1 (fora da escada): LR na amostra, 10 épocas, taxa constante, mesma semente.
// Pergunta: os sinais de "erros" e "Δ" se corrigem (não convergência) ou persistem (viés de seleção, H-sel)?
// Uso: node src/model/b1-diag.js  → data/b1-diag.json
import fs from 'node:fs';
import path from 'node:path';
import { DATA_DIR } from '../data/duolingo.js';
import { loadSample } from '../data/sample.js';
import { fromRows, indicesWhere } from '../data/columns.js';
import { evaluate } from './metrics.js';
import { trainLrJs, predictLr } from './lr.js';

const cols = fromRows(await loadSample());
const train = indicesWhere(cols, i => cols.train[i] === 1), test = indicesWhere(cols, i => cols.train[i] === 0);
const p = Float64Array.from(test, i => cols.p[i]), h = Float64Array.from(test, i => cols.h[i]);
const out = {};
// λ = 0.1 (código) e λ = 0 (sem L2): separa "não convergência / regularização" de "viés de seleção"
for (const [name, opts] of [['lr', {}], ['lr-lex', { omitLexemes: true }], ['lr λ=0', { l2wt: 0 }], ['lr-lex λ=0', { omitLexemes: true, l2wt: 0 }]]) {
  const rows = [];
  const t0 = Date.now();
  trainLrJs(cols, train, {
    ...opts, epochs: 10, constantRate: true,
    onEpoch: (e, m) => {
      const ev = evaluate({ p, phat: predictLr(m, cols, test), h, hhat: null });
      rows.push({ epoch: e + 1, right: m.w[0], wrong: m.w[1], time: m.w[2], bias: m.w[3], mae: ev.mae, auc: ev.auc });
    },
  });
  out[name] = { rows, seconds: (Date.now() - t0) / 1000 };
}
fs.writeFileSync(path.join(DATA_DIR, 'b1-diag.json'), JSON.stringify(out, null, 1));
for (const [k, v] of Object.entries(out)) {
  console.log(k, `${v.seconds}s`);
  console.table(v.rows.map(r => Object.fromEntries(Object.entries(r).map(([a, b]) => [a, Number.isInteger(b) ? b : +b.toPrecision(4)]))));
}
