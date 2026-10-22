// forca.js: (1) cria a "força do sinal" em 5 barras e testa se as barras significam algo fora da amostra;
// (2) avalia a estratégia: comprar com R$ 50, fechar no lucro modesto e reabrir se o sinal tiver 3 barras ou mais.
const fs = require("fs");
const ML = require("./ml.js");
const { loadAll } = require("./dados.js");
const DAY = 864e5, H = 7, TOPN = 60, VOLWIN = 30;
const FEE = 0.075, SLIP = 0.1, CONV = +process.env.CONV || 0.1;
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
const NAMES = [...ML.FEATURES, "gap"];
const EXITS = { meta3: { nome: "meta 3%", tgt: 3 }, meta5: { nome: "meta 5%", tgt: 5 }, meta10: { nome: "meta 10%", tgt: 10 }, semMeta: { nome: "sem meta (só stop e prazo de 7 dias)", tgt: null } };

function buildUniverse(series) {
  const byDay = new Map();
  for (const [s, cs] of Object.entries(series)) {
    let q = 0;
    for (let i = 0; i < cs.length; i++) {
      q += cs[i].q;
      if (i >= VOLWIN) { q -= cs[i - VOLWIN].q; if (!byDay.has(cs[i].t)) byDay.set(cs[i].t, []); byDay.get(cs[i].t).push([s, q / VOLWIN]); }
    }
  }
  const top = new Map(Object.keys(series).map(s => [s, new Set()]));
  for (const [t, arr] of byDay) { arr.sort((a, b) => b[1] - a[1]); for (const [s] of arr.slice(0, TOPN)) top.get(s).add(t); }
  return top;
}
// Operação: entrada na abertura do dia seguinte; stop de 2 oscilações (entre 4% e 15%); prazo de 7 dias
function trade(cs, i, a, tgt) {
  const entry = cs[i + 1].o * (1 + SLIP / 100), stopP = entry * (1 - clamp(2 * a, 4, 15) / 100), tgtP = tgt ? entry * (1 + tgt / 100) : null;
  for (let k = i + 1; k <= i + H; k++) {
    if (cs[k].l <= stopP) return { net: (stopP * (1 - SLIP / 100) / entry - 1) * 100 - 2 * FEE, endT: cs[k].t };
    if (tgtP && cs[k].h >= tgtP) return { net: (tgtP / entry - 1) * 100 - 2 * FEE, endT: cs[k].t };
  }
  return { net: (cs[i + H].c * (1 - SLIP / 100) / entry - 1) * 100 - 2 * FEE, endT: cs[i + H].t };
}
function st(a) {
  const n = a.length;
  if (n < 2) return { n, mean: n ? a[0] : null, se: null, win: null };
  const m = mean(a), sd = Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (n - 1));
  return { n, mean: m, se: sd / Math.sqrt(n), win: a.filter(x => x > 0).length / n * 100 };
}
// Regressão ridge: score = b + sum(w_j * z_j)
function ridge(X, y, lam) {
  const n = X.length, p = X[0].length, mu = Array(p).fill(0), sd = Array(p).fill(1);
  for (let j = 0; j < p; j++) { mu[j] = mean(X.map(r => r[j])); sd[j] = Math.sqrt(mean(X.map(r => (r[j] - mu[j]) ** 2))) || 1; }
  const Z = X.map(r => r.map((v, j) => (v - mu[j]) / sd[j])), ym = mean(y);
  const A = Array.from({ length: p }, (_, a) => { const row = Array(p + 1).fill(0); for (let b = 0; b < p; b++) { let s = 0; for (let k = 0; k < n; k++) s += Z[k][a] * Z[k][b]; row[b] = s + (a === b ? lam : 0); } let s = 0; for (let k = 0; k < n; k++) s += Z[k][a] * (y[k] - ym); row[p] = s; return row; });
  for (let c = 0; c < p; c++) { // eliminação de Gauss
    let m = c; for (let r = c + 1; r < p; r++) if (Math.abs(A[r][c]) > Math.abs(A[m][c])) m = r;
    [A[c], A[m]] = [A[m], A[c]];
    for (let r = 0; r < p; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k <= p; k++) A[r][k] -= f * A[c][k]; }
  }
  return { w: A.map((r, i) => r[p] / r[i]), b: ym, mu, sd };
}
const scoreOf = (M, x) => M.b + x.reduce((s, v, j) => s + M.w[j] * (v - M.mu[j]) / M.sd[j], 0);
const quant = (arr, q) => { const s = arr.slice().sort((a, b) => a - b), i = (s.length - 1) * q, f = Math.floor(i), c = Math.ceil(i); return s[f] + (s[c] - s[f]) * (i - f); };
const barsOf = (edges, s) => 1 + edges.filter(e => s > e).length;

// Carteira de R$ 50, uma operação por vez, reabrindo no melhor sinal com barras suficientes
function sequential(sigs, minBars, rule) {
  const byDay = new Map();
  for (const s of sigs) if (s.bars >= minBars) { if (!byDay.has(s.entryT)) byDay.set(s.entryT, []); byDay.get(s.entryT).push(s); }
  let eq = 1, peak = 1, dd = 0, flat = -Infinity, n = 0, w = 0, worstRun = 0, run = 0;
  for (const d of [...byDay.keys()].sort((a, b) => a - b)) {
    if (d < flat) continue;
    const s = byDay.get(d).sort((a, b) => b.score - a.score)[0], r = s.net[rule];
    eq *= 1 + r / 100; n++; if (r > 0) { w++; run = 0; } else { run++; worstRun = Math.max(worstRun, run); }
    flat = s.endT[rule] + DAY; peak = Math.max(peak, eq); dd = Math.max(dd, 1 - eq / peak);
  }
  const final = 50 * eq, finalConv = final * (1 - CONV / 100) ** 2;
  return { minBars, regra: EXITS[rule].nome, operacoes: n, acerto: n ? w / n * 100 : null, finalSemConversao: final, finalComConversao: finalConv, retornoPct: (finalConv / 50 - 1) * 100, quedaMaxima: dd * 100, maiorSequenciaDePerdas: worstRun };
}

(async () => {
  console.log("Baixando histórico...");
  const series = await loadAll();
  const btc = series["BTCUSDT"];
  if (!btc) throw new Error("BTCUSDT não encontrado");
  const bctx = ML.btcContext(btc), top = buildUniverse(series);

  console.log("Procurando sinais de momentum...");
  const sigs = [];
  for (const [sym, cs] of Object.entries(series)) {
    const set = top.get(sym); let last = -1;
    for (let i = 56; i < cs.length - H - 1; i++) {
      if (i <= last || !set.has(cs[i].t)) continue;
      let mh = 0, sv = 0;
      for (let j = i - 20; j < i; j++) { if (cs[j].h > mh) mh = cs[j].h; sv += cs[j].v; }
      if (!(sv > 0) || !(cs[i].c > mh) || cs[i].v / (sv / 20) < 2) continue;
      const f = ML.features(cs, i, bctx);
      if (!f) continue;
      const gap = clamp((cs[i + 1].o / cs[i].c - 1) / (f.a / 100), -3, 3);
      const net = {}, endT = {};
      for (const [id, R] of Object.entries(EXITS)) { const tr = trade(cs, i, f.a, R.tgt); net[id] = tr.net; endT[id] = tr.endT; }
      const n7 = clamp((cs[i + H].c / cs[i + 1].o - 1) * 100 / f.a, -5, 8);
      sigs.push({ sym, entryT: cs[i + 1].t, x: [...f.x, gap], n7, net, endT });
      last = i + H; // sem sobreposição por moeda
    }
  }
  sigs.sort((a, b) => a.entryT - b.entryT);
  console.log(`${sigs.length} sinais.`);
  if (sigs.length < (+process.env.MIN_SIG || 300)) throw new Error("sinais insuficientes para calibrar as barras");

  // Treino na parte antiga (60%), teste na parte recente (40%)
  const cut = Math.floor(sigs.length * 0.6), train = sigs.slice(0, cut), test = sigs.slice(cut);
  const Mt = ridge(train.map(s => s.x), train.map(s => s.n7), 20);
  const trScores = train.map(s => scoreOf(Mt, s.x)), edgesT = [0.2, 0.4, 0.6, 0.8].map(q => quant(trScores, q));
  for (const s of test) { s.score = scoreOf(Mt, s.x); s.bars = barsOf(edgesT, s.score); }

  const levels = [1, 2, 3, 4, 5].map(L => {
    const g = test.filter(s => s.bars === L), o = { nivel: L, n: g.length, n7: st(g.map(s => s.n7)) };
    for (const id of Object.keys(EXITS)) o[id] = st(g.map(s => s.net[id]));
    return o;
  });
  const m7 = levels.filter(l => l.n7.mean !== null).map(l => l.n7.mean), mSem = levels.filter(l => l.semMeta.mean !== null).map(l => l.semMeta.mean);
  const increasing = a => a.every((v, i) => i === 0 || v >= a[i - 1] - 1e-9);
  const all = { n: test.length, n7: st(test.map(s => s.n7)) };
  for (const id of Object.keys(EXITS)) all[id] = st(test.map(s => s.net[id]));

  // Estratégia de carteira, só no período de teste
  const strategy = [];
  for (const rule of ["meta3", "meta5", "meta10", "semMeta"]) for (const mb of [1, 3, 4, 5]) strategy.push(sequential(test, mb, rule));

  // Modelo final (todos os sinais) para uso do robô e da página
  const Mf = ridge(sigs.map(s => s.x), sigs.map(s => s.n7), 20);
  const edgesF = [0.2, 0.4, 0.6, 0.8].map(q => quant(sigs.map(s => scoreOf(Mf, s.x)), q));
  const out = {
    generated: Date.now(), conversao: CONV, features: NAMES,
    modelo: { w: Mf.w.map(v => ML.round(v)), b: ML.round(Mf.b), mu: Mf.mu.map(v => ML.round(v)), sd: Mf.sd.map(v => ML.round(v)), edges: edgesF.map(v => ML.round(v)) },
    periodo: { sinais: sigs.length, treino: train.length, teste: test.length, de: test[0].entryT, ate: test[test.length - 1].entryT },
    calibracao: { niveis: levels, todos: all, n7Crescente: increasing(m7), semMetaCrescente: increasing(mSem) },
    estrategia: strategy,
    aviso: "As barras valem apenas se o resultado melhorar de forma consistente de 1 para 5 barras no período de teste, que não foi usado para criar as barras."
  };
  fs.mkdirSync("data", { recursive: true });
  fs.writeFileSync("data/forca.json", JSON.stringify(out));
  console.log("Níveis (teste):", levels.map(l => `${l.nivel}b n=${l.n} n7=${l.n7.mean && l.n7.mean.toFixed(2)} semMeta=${l.semMeta.mean && l.semMeta.mean.toFixed(2)}%`).join(" | "));
  console.log("Crescente (n7):", out.calibracao.n7Crescente, "| sem meta:", out.calibracao.semMetaCrescente);
  for (const s of strategy) console.log(`${s.regra} / ${s.minBars}+ barras: ${s.operacoes} ops, acerto ${s.acerto && s.acerto.toFixed(0)}%, final R$ ${s.finalComConversao.toFixed(2)}, queda máx ${s.quedaMaxima.toFixed(0)}%`);
})().catch(e => { console.error(e); process.exit(1); });
