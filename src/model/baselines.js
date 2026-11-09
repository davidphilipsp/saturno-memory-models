// B0 — modelos sem treino. Fórmulas de experiment.py:56-74.
import { pclip, hclip } from '../data/duolingo.js';

const fromHalfLife = (t, h) => ({ p: pclip(2 ** (-t / h)), h });

// Leitner: h = 2^(acertos − erros) — exp:61-67, Apêndice A.2
export function leitner({ t, right, wrong }) {
  return fromHalfLife(t, hclip(2 ** (right - wrong)));   // 2**grande = Infinity → teto, como o OverflowError do Python
}

// Pimsleur: h = 2^(2,35·n − 16,46), n = acertos + erros — exp:68-74 (artigo: 2,4 e −16,5, D4)
export function pimsleur({ t, right, wrong }) {
  return fromHalfLife(t, hclip(2 ** (2.35 * (right + wrong) - 16.46)));
}

// Constante: sempre a média do treino; sem meia-vida
export const constant = c => () => ({ p: c, h: null });
