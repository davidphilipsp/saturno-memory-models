// Amostra de trabalho: Bernoulli(rate) por índice de linha, partição do artigo preservada.
// Uso: node src/data/sample.js <total_de_linhas> <alvo>   → data/sample.csv
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { RAW_FILE, DATA_DIR, sampleInstances, splitCsv } from './duolingo.js';

export const SAMPLE_FILE = path.join(DATA_DIR, 'sample.csv');
const FIELDS = ['index', 'split', 'p', 't', 'h', 'right', 'wrong', 'sessionSeen', 'sessionRight',
  'learning', 'ui', 'lexemeKey', 'userId', 'ts'];
const NUMERIC = new Set(['index', 'p', 't', 'h', 'right', 'wrong', 'sessionSeen', 'sessionRight', 'ts']);
const quote = v => (typeof v === 'string' && /[",]/.test(v) ? `"${v.replaceAll('"', '""')}"` : v);

export async function writeSample({ total, target, seed = 0, file = RAW_FILE, out = SAMPLE_FILE }) {
  const rate = target / total;
  const ws = fs.createWriteStream(out);
  ws.write(FIELDS.join(',') + '\n');
  const counts = { train: 0, test: 0 };
  for await (const inst of sampleInstances(file, { total, rate, seed })) {
    counts[inst.split]++;
    if (!ws.write(FIELDS.map(k => quote(inst[k])).join(',') + '\n')) await new Promise(r => ws.once('drain', r));
  }
  await new Promise(r => ws.end(r));
  return { rate, seed, ...counts };
}

// Lê a amostra inteira para a memória (1M linhas cabem) como lista de objetos
export async function loadSample(file = SAMPLE_FILE, { maxLines = Infinity } = {}) {
  const rl = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
  let header = null;
  const out = [];
  for await (const line of rl) {
    if (!line) continue;
    const c = splitCsv(line);
    if (!header) { header = c; continue; }
    if (out.length >= maxLines) break;
    const o = {};
    header.forEach((k, i) => { o[k] = NUMERIC.has(k) ? Number(c[i]) : c[i]; });
    out.push(o);
  }
  rl.close();
  return out;
}

if (process.argv[1] === import.meta.filename) {
  const [total, target] = process.argv.slice(2).map(Number);
  const t0 = Date.now();
  const r = await writeSample({ total, target });
  console.log(JSON.stringify({ ...r, seconds: (Date.now() - t0) / 1000 }));
}
