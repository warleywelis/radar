// ml.js: modelo de previsão dos 3 caminhos do preço em 7 dias (0 = alta, 1 = estabilidade, 2 = baixa).
// Regressão logística multinomial (softmax) em JavaScript puro, sem dependências.
const H = 7, NF = 12;
const VERSAO = "2026-10-25";
const MIN_ATR = 1.2; // oscilação diária mínima (%): exclui stablecoins e ativos atrelados, em que "1 oscilação" é só ruído
const NF_EXTRA = 15; // com RSI, ADX e volume de 5 dias
const FEATURES_EXTRA = ["rsi", "adx", "volume5"];
const FEATURES = ["ret1", "ret3", "ret7", "ret20", "dist20h", "volume", "tendencia", "inclinacao50", "oscilacao", "corpo", "btc7", "btcTendencia"];
const CLASSES = ["alta", "estabilidade", "baixa"];

function atrPctAt(cs, i, n = 14) {
  if (i < n) return null;
  let s = 0;
  for (let j = i - n + 1; j <= i; j++) {
    const p = cs[j - 1].c;
    s += Math.max(cs[j].h - cs[j].l, Math.abs(cs[j].h - p), Math.abs(cs[j].l - p));
  }
  return s / n / cs[i].c * 100;
}
function smaAt(cs, i, n) { let s = 0; for (let j = i - n + 1; j <= i; j++) s += cs[j].c; return s / n; }
const clip = (v, a = 5) => Math.max(-a, Math.min(a, v));

// RSI de Wilder (14)
function rsiAt(cs, i, n = 14) {
  const s = Math.max(1, i - 60);
  let ag = 0, al = 0;
  for (let j = s; j < s + n && j <= i; j++) { const d = cs[j].c - cs[j - 1].c; if (d > 0) ag += d; else al -= d; }
  ag /= n; al /= n;
  for (let j = s + n; j <= i; j++) { const d = cs[j].c - cs[j - 1].c; ag = (ag * (n - 1) + Math.max(d, 0)) / n; al = (al * (n - 1) + Math.max(-d, 0)) / n; }
  return al === 0 ? 100 : 100 - 100 / (1 + ag / al);
}
// ADX de Wilder (14)
function adxAt(cs, i, n = 14) {
  const s = Math.max(1, i - 80), dxs = [];
  let tr = 0, p = 0, m = 0;
  for (let j = s; j <= i; j++) {
    const up = cs[j].h - cs[j - 1].h, dn = cs[j - 1].l - cs[j].l, pdm = up > dn && up > 0 ? up : 0, mdm = dn > up && dn > 0 ? dn : 0;
    const t = Math.max(cs[j].h - cs[j].l, Math.abs(cs[j].h - cs[j - 1].c), Math.abs(cs[j].l - cs[j - 1].c));
    if (j < s + n) { tr += t; p += pdm; m += mdm; } else { tr = tr - tr / n + t; p = p - p / n + pdm; m = m - m / n + mdm; }
    if (j >= s + n - 1) { const pdi = 100 * p / tr, mdi = 100 * m / tr; dxs.push(100 * Math.abs(pdi - mdi) / ((pdi + mdi) || 1)); }
  }
  let adx = dxs.slice(0, n).reduce((a, b) => a + b, 0) / n;
  for (let k = n; k < dxs.length; k++) adx = (adx * (n - 1) + dxs[k]) / n;
  return adx;
}
// Contexto do Bitcoin por dia: retorno de 7 dias e tendência, em unidades da oscilação diária do próprio Bitcoin
function btcContext(btc) {
  const m = new Map();
  for (let i = 55; i < btc.length; i++) {
    const a = atrPctAt(btc, i);
    if (!a) continue;
    m.set(btc[i].t, { r7: Math.log(btc[i].c / btc[i - 7].c) / (a / 100), tr: (btc[i].c / smaAt(btc, i, 50) - 1) / (a / 100) });
  }
  return m;
}
// Atributos da moeda no fim do dia i (só usa dados até o dia i)
function features(cs, i, bctx, extra = false) {
  if (i < 56) return null;
  const a = atrPctAt(cs, i);
  if (!a || a < MIN_ATR) return null;
  const af = a / 100, c = cs[i].c;
  let mh = 0, sv = 0;
  for (let j = i - 20; j < i; j++) { if (cs[j].h > mh) mh = cs[j].h; sv += cs[j].v; }
  if (!(sv > 0) || !(cs[i].v > 0)) return null;
  const b = bctx.get(cs[i].t);
  if (!b) return null;
  const s50 = smaAt(cs, i, 50), s50b = smaAt(cs, i - 5, 50), rng = cs[i].h - cs[i].l;
  const x = [
    clip(Math.log(c / cs[i - 1].c) / af), clip(Math.log(c / cs[i - 3].c) / af), clip(Math.log(c / cs[i - 7].c) / af), clip(Math.log(c / cs[i - 20].c) / af),
    clip((c / mh - 1) / af), clip(Math.log(cs[i].v / (sv / 20)), 3), clip((c / s50 - 1) / af), clip((s50 / s50b - 1) / af),
    Math.log(a), rng > 0 ? (c - cs[i].o) / rng : 0, clip(b.r7), clip(b.tr)
  ];
  if (extra) {
    const v5 = (cs[i - 1].v + cs[i - 2].v + cs[i - 3].v + cs[i - 4].v + cs[i - 5].v) / 5;
    x.push(clip((rsiAt(cs, i) - 50) / 25), clip((adxAt(cs, i) - 25) / 15, 3), v5 > 0 ? clip(Math.log(cs[i].v / v5), 3) : 0);
  }
  if (x.some(v => !Number.isFinite(v))) return null;
  return { x, a };
}
// Resultado real: onde o preço terminou 7 dias depois da entrada (abertura do dia seguinte), em unidades da oscilação diária
function label(cs, i, a, thr = 1) {
  const n = (cs[i + H].c / cs[i + 1].o - 1) * 100 / a;
  return n >= thr ? 0 : n <= -thr ? 2 : 1;
}
// Resultado líquido (%) de uma operação com meta, stop sugerido e prazo de 7 dias, já com taxas e slippage
function tradeNet(cs, i, a, P) {
  const sp = Math.min(Math.max(P.k * a, 4), P.maxl);
  const entry = cs[i + 1].o * (1 + P.slip / 100), stopP = entry * (1 - sp / 100), tgtP = entry * (1 + P.tgt / 100);
  let exit = null, k = i + 1;
  for (; k <= i + H; k++) {
    if (cs[k].l <= stopP) { exit = stopP * (1 - P.slip / 100); break; }
    if (cs[k].h >= tgtP) { exit = tgtP; break; }
  }
  if (exit === null) exit = cs[i + H].c * (1 - P.slip / 100);
  return (exit / entry - 1) * 100 - 2 * P.fee;
}

function newModel(NF = 12) {
  return { W: [0, 1, 2].map(() => new Array(NF).fill(0)), b: [0, 0, 0], mu: new Array(NF).fill(0), sd: new Array(NF).fill(1) };
}
function softmax(z) {
  const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e[0] + e[1] + e[2];
  return e.map(v => v / s);
}
function predict(M, x) {
  const NF = M.mu.length;
  const xs = x.map((v, j) => (v - M.mu[j]) / M.sd[j]);
  return softmax([0, 1, 2].map(k => { let s = M.b[k]; for (let j = 0; j < NF; j++) s += M.W[k][j] * xs[j]; return s; }));
}
// Treino em lote com momento; rows = [{x, y}]
function fit(M, rows, opts = {}) {
  const NF = M.mu.length;
  const { epochs = 80, lr = 0.3, l2 = 1e-3, restandardize = true } = opts, n = rows.length;
  if (!n) return M;
  if (restandardize) {
    for (let j = 0; j < NF; j++) {
      let s = 0; for (const r of rows) s += r.x[j];
      const mu = s / n; let v = 0; for (const r of rows) v += (r.x[j] - mu) ** 2;
      M.mu[j] = mu; M.sd[j] = Math.sqrt(v / n) || 1;
    }
  }
  const Xs = new Float64Array(n * NF);
  for (let r = 0; r < n; r++) for (let j = 0; j < NF; j++) Xs[r * NF + j] = (rows[r].x[j] - M.mu[j]) / M.sd[j];
  const vW = [0, 1, 2].map(() => new Array(NF).fill(0)), vb = [0, 0, 0];
  for (let e = 0; e < epochs; e++) {
    const gW = [0, 1, 2].map(() => new Array(NF).fill(0)), gb = [0, 0, 0];
    for (let r = 0; r < n; r++) {
      const o = r * NF, z = [0, 1, 2].map(k => { let s = M.b[k]; for (let j = 0; j < NF; j++) s += M.W[k][j] * Xs[o + j]; return s; });
      const p = softmax(z);
      for (let k = 0; k < 3; k++) {
        const g = p[k] - (rows[r].y === k ? 1 : 0);
        gb[k] += g;
        for (let j = 0; j < NF; j++) gW[k][j] += g * Xs[o + j];
      }
    }
    for (let k = 0; k < 3; k++) {
      vb[k] = 0.9 * vb[k] - lr * gb[k] / n; M.b[k] += vb[k];
      for (let j = 0; j < NF; j++) { vW[k][j] = 0.9 * vW[k][j] - lr * (gW[k][j] / n + l2 * M.W[k][j]); M.W[k][j] += vW[k][j]; }
    }
  }
  return M;
}
// Atualização contínua: um passo de gradiente com um exemplo novo (taxa pequena)
function sgd(M, x, y, lr = 0.003, l2 = 1e-4) {
  const NF = M.mu.length;
  const p = predict(M, x), xs = x.map((v, j) => (v - M.mu[j]) / M.sd[j]);
  for (let k = 0; k < 3; k++) {
    const g = p[k] - (y === k ? 1 : 0);
    M.b[k] -= lr * g;
    for (let j = 0; j < NF; j++) M.W[k][j] -= lr * (g * xs[j] + l2 * M.W[k][j]);
  }
}
const logLoss = (p, y) => -Math.log(Math.max(p[y], 1e-9));
const round = (v, d = 6) => Math.round(v * 10 ** d) / 10 ** d;
function pack(M) {
  return { W: M.W.map(r => r.map(v => round(v))), b: M.b.map(v => round(v)), mu: M.mu.map(v => round(v)), sd: M.sd.map(v => round(v)) };
}
module.exports = { VERSAO, MIN_ATR, H, NF, NF_EXTRA, FEATURES_EXTRA, rsiAt, adxAt, FEATURES, CLASSES, atrPctAt, btcContext, features, label, tradeNet, newModel, predict, fit, sgd, logLoss, pack, round };
