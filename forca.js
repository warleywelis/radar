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
      sigs.push({ sym, i, entryT: cs[i + 1].t, x: [...f.x, gap], n7, net, endT, c0: cs[i].c, af0: f.a / 100, stopPct: clamp(2 * f.a, 4, 15) });
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

  // Regra dinâmica do usuário: a meta muda com as barras, 1 barra encerra, e opcionalmente troca para a moeda de melhor sinal
  const idxOf = {}; for (const [sym, cs] of Object.entries(series)) idxOf[sym] = new Map(cs.map((c, i) => [c.t, i]));
  const fcache = new Map();
  const featAt = (sym, idx) => { const k = sym + ":" + idx; if (!fcache.has(k)) fcache.set(k, ML.features(series[sym], idx, bctx)); return fcache.get(k); };
  const candBy = new Map(); for (const s of test) { if (!candBy.has(s.entryT)) candBy.set(s.entryT, []); candBy.get(s.entryT).push(s); }
  // Barras para a operação ABERTA: modelo próprio de "continuação", treinado só com o período antigo.
  // (A nota do sinal foi feita para o dia do sinal; nos dias seguintes ela cairia sempre, por isso precisa de calibração própria.)
  const makeSamples = list => {
    const out = [];
    for (const s of list) {
      const cs = series[s.sym];
      for (let e = s.i + 2; e <= s.i + H; e++) {
        const f = featAt(s.sym, e - 1);
        if (!f) continue;
        const gap = clamp((cs[e].o / s.c0 - 1) / s.af0, -3, 3), left = (s.i + H - e + 1) / H, ret = (cs[s.i + H].c / cs[e].o - 1) * 100;
        out.push({ x: [...f.x, gap, left], y: clamp(ret / (s.af0 * 100), -5, 8), ret });
      }
    }
    return out;
  };
  const trainS = makeSamples(train), testS = makeSamples(test);
  const Mc = ridge(trainS.map(r => r.x), trainS.map(r => r.y), 20);
  const edgesC = [0.2, 0.4, 0.6, 0.8].map(q => quant(trainS.map(r => scoreOf(Mc, r.x)), q));
  const contLevels = [1, 2, 3, 4, 5].map(L => {
    const g = testS.filter(r => barsOf(edgesC, scoreOf(Mc, r.x)) === L);
    return { nivel: L, n: g.length, y: st(g.map(r => r.y)), ret: st(g.map(r => r.ret)) };
  });
  const contMeans = contLevels.filter(l => l.y.mean !== null).map(l => l.y.mean);
  const contCrescente = contMeans.every((v, i) => i === 0 || v >= contMeans[i - 1] - 1e-9);
  const tStart = test[0].entryT, tEnd = test[test.length - 1].entryT + (H + 1) * DAY;
  const byScore = (a, b) => b.score - a.score;
  // Metas por barras (em % do capital) e folga do stop por barras (em oscilações diárias), ambos dinâmicos nos dois sentidos
  const MAPA = { 1: 3, 2: 4, 3: 5, 4: 8, 5: 10 }, MULT = { 1: 1, 2: 1.25, 3: 1.5, 4: 2, 5: 2.5 };
  function runDynamic(V) {
    let eq = 1, peak = 1, dd = 0, pos = null, n = 0, w = 0, run = 0, worst = 0, hold = 0, chg = 0, chgN = 0; const why = {}, nets = [];
    const close = (px, reason) => {
      const net = (px / pos.entry - 1) * 100 - 2 * FEE;
      nets.push(net); eq *= 1 + net / 100; n++; hold += pos.days; if (net > 0) { w++; run = 0; } else { run++; worst = Math.max(worst, run); }
      peak = Math.max(peak, eq); dd = Math.max(dd, 1 - eq / peak); why[reason] = (why[reason] || 0) + 1; pos = null;
    };
    const stopAt = (bars, a) => clamp(MULT[bars] * a, 3, 15);
    const intraday = (cs, idx, bars, isEntry) => {
      const o = cs[idx].o, h = cs[idx].h, l = cs[idx].l, c = cs[idx].c, tp = V.target(bars), tgtP = tp ? pos.entry * (1 + tp / 100) : null;
      if (!isEntry) {
        if (o <= pos.stopP) { close(o * (1 - SLIP / 100), "stop"); return true; }
        if (tgtP && o >= tgtP) { close(o, "meta"); return true; }
      }
      if (l <= pos.stopP) { close(pos.stopP * (1 - SLIP / 100), "stop"); return true; }
      if (tgtP && h >= tgtP) { close(tgtP, "meta"); return true; }
      pos.days++; pos.last = c;
      if (pos.days >= H) { close(c * (1 - SLIP / 100), "prazo"); return true; }
      return false;
    };
    for (let d = tStart; d <= tEnd; d += DAY) {
      let canEnter = true;
      if (pos) {
        const cs = series[pos.sym], idx = idxOf[pos.sym].get(d);
        if (idx === undefined) { close(pos.last * (1 - SLIP / 100), "sem dados"); }
        else {
          const f = featAt(pos.sym, idx - 1);
          let bars = pos.bars;
          if (f) {
            const raw = barsOf(edgesC, scoreOf(Mc, [...f.x, clamp((cs[idx].o / pos.c0 - 1) / pos.af0, -3, 3), (H - pos.days) / H]));
            bars = V.smooth ? clamp(raw, pos.bars - 1, pos.bars + 1) : raw;
            chg += Math.abs(bars - pos.bars); chgN++; pos.bars = bars;
            if (V.dynStop) pos.stopP = pos.entry * (1 - stopAt(bars, ML.atrPctAt(cs, idx - 1) || pos.af0 * 100) / 100);
          }
          let closedAtOpen = false;
          if (V.rotate && bars <= 2) {
            const alt = (candBy.get(d) || []).filter(x => x.sym !== pos.sym && x.bars >= 3 && x.bars > bars).sort(byScore)[0];
            if (alt) { close(cs[idx].o * (1 - SLIP / 100), "troca de moeda"); closedAtOpen = true; }
          }
          if (!closedAtOpen && intraday(cs, idx, bars, false)) canEnter = false;
        }
      }
      if (!pos && canEnter) {
        const c = (candBy.get(d) || []).filter(x => x.bars >= 3).sort(byScore)[0];
        if (c) {
          const cs = series[c.sym], idx = idxOf[c.sym].get(d);
          if (idx !== undefined) {
            const entry = cs[idx].o * (1 + SLIP / 100), sp = V.dynStop ? stopAt(c.bars, c.af0 * 100) : c.stopPct;
            pos = { sym: c.sym, entry, stopP: entry * (1 - sp / 100), c0: c.c0, af0: c.af0, days: 0, bars: c.bars, last: cs[idx].o };
            intraday(cs, idx, c.bars, true);
          }
        }
      }
    }
    if (pos) close(pos.last * (1 - SLIP / 100), "aberta no fim");
    const final = 50 * eq * (1 - CONV / 100) ** 2, m = n ? mean(nets) : null, se = n > 1 ? Math.sqrt(nets.reduce((a, x) => a + (x - m) ** 2, 0) / (n - 1)) / Math.sqrt(n) : null;
    return { id: V.id, nome: V.nome, operacoes: n, acerto: n ? w / n * 100 : null, mediaPorOperacao: m, seMedia: se, diasMedios: n ? hold / n : null, finalComConversao: final, retornoPct: (final / 50 - 1) * 100, quedaMaxima: dd * 100, maiorSequenciaDePerdas: worst, mudancaMediaDeBarras: chgN ? chg / chgN : null, motivos: why, passa: n >= 30 && se !== null && m - 2 * se > 0 && final > 50 };
  }
  const dinamica = [
    { id: "atual", nome: "Referência: meta fixa de 10% e stop fixo", target: () => 10 },
    { id: "sem", nome: "Referência: sem meta, stop fixo e prazo de 7 dias", target: () => null },
    { id: "barras", nome: "Metas por barras (1: 3%, 2: 4%, 3: 5%, 4: 8%, 5: 10%) e stop fixo", target: b => MAPA[b] },
    { id: "barrasStop", nome: "Metas por barras e stop dinâmico (muda com as barras e com a oscilação do dia)", target: b => MAPA[b], dynStop: true },
    { id: "suave", nome: "Anterior, com barras suavizadas (mudam no máximo 1 nível por dia, contra oscilação brusca)", target: b => MAPA[b], dynStop: true, smooth: true },
    { id: "troca", nome: "Anterior, e troca para a moeda de melhor sinal quando a atual cai para 2 barras ou menos", target: b => MAPA[b], dynStop: true, smooth: true, rotate: true }
  ].map(runDynamic);

  // Modelo final (todos os sinais) para uso do robô e da página
  const Mf = ridge(sigs.map(s => s.x), sigs.map(s => s.n7), 20);
  const edgesF = [0.2, 0.4, 0.6, 0.8].map(q => quant(sigs.map(s => scoreOf(Mf, s.x)), q));
  const out = {
    generated: Date.now(), conversao: CONV, features: NAMES,
    modelo: { w: Mf.w.map(v => ML.round(v)), b: ML.round(Mf.b), mu: Mf.mu.map(v => ML.round(v)), sd: Mf.sd.map(v => ML.round(v)), edges: edgesF.map(v => ML.round(v)) },
    periodo: { sinais: sigs.length, treino: train.length, teste: test.length, de: test[0].entryT, ate: test[test.length - 1].entryT },
    calibracao: { niveis: levels, todos: all, n7Crescente: increasing(m7), semMetaCrescente: increasing(mSem) },
    estrategia: strategy, dinamica, mapa: { metas: MAPA, folgaDoStop: MULT },
    confiabilidade: { barrasDeEntradaCrescem: increasing(m7) && increasing(mSem), barrasDeContinuacaoCrescem: contCrescente },
    continuacao: { niveis: contLevels, crescente: contCrescente, modelo: { w: Mc.w.map(v => ML.round(v)), b: ML.round(Mc.b), mu: Mc.mu.map(v => ML.round(v)), sd: Mc.sd.map(v => ML.round(v)), edges: edgesC.map(v => ML.round(v)) } },
    aviso: "As barras valem apenas se o resultado melhorar de forma consistente de 1 para 5 barras no período de teste, que não foi usado para criar as barras."
  };
  fs.mkdirSync("data", { recursive: true });
  fs.writeFileSync("data/forca.json", JSON.stringify(out));
  console.log("Níveis (teste):", levels.map(l => `${l.nivel}b n=${l.n} n7=${l.n7.mean && l.n7.mean.toFixed(2)} semMeta=${l.semMeta.mean && l.semMeta.mean.toFixed(2)}%`).join(" | "));
  console.log("Crescente (n7):", out.calibracao.n7Crescente, "| sem meta:", out.calibracao.semMetaCrescente);
  console.log("Barras da operação aberta (teste): crescente =", contCrescente, "|", contLevels.map(l => `${l.nivel}b n=${l.n} y=${l.y.mean && l.y.mean.toFixed(2)}`).join(" | "));
  for (const d of dinamica) console.log(`DINÂMICA ${d.id}: ${d.operacoes} ops, acerto ${d.acerto && d.acerto.toFixed(0)}%, média ${d.mediaPorOperacao && d.mediaPorOperacao.toFixed(2)}%, final R$ ${d.finalComConversao.toFixed(2)}, queda máx ${d.quedaMaxima.toFixed(0)}%, motivos ${JSON.stringify(d.motivos)}`);
  for (const s of strategy) console.log(`${s.regra} / ${s.minBars}+ barras: ${s.operacoes} ops, acerto ${s.acerto && s.acerto.toFixed(0)}%, final R$ ${s.finalComConversao.toFixed(2)}, queda máx ${s.quedaMaxima.toFixed(0)}%`);
})().catch(e => { console.error(e); process.exit(1); });
