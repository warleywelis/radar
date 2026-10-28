// pesquisa.js: baixa o histórico UMA vez e roda as três pesquisas em sequência: treino (modelo), saidas (regras de saída) e forca (barras e carteira).
// Substitui os antigos treino.js, saidas.js e forca.js (podem ser apagados) e o fluxo único "pesquisa" substitui os fluxos treino, saidas e forca.
const fs = require("fs");
const ML = require("./ml.js");
const { loadAll } = require("./dados.js");

// ===================== treino =====================
async function treino(series) {
  const DAY = 864e5, H = ML.H, MIN_N = +process.env.MIN_N || 40;
  const P = { tgt: 10, k: 2, maxl: 15, fee: 0.075, slip: 0.1, volWin: 30, minTrainMonths: 24 };
  const mkey = t => { const d = new Date(t); return d.getUTCFullYear() * 12 + d.getUTCMonth(); };
  const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
  const btc = series["BTCUSDT"];
  if (!btc) throw new Error("BTCUSDT não encontrado no arquivo público");
  const bctx = ML.btcContext(btc);

  // Posição diária de cada moeda no ranking de volume (média de 30 dias): rank.get(moeda).get(dia), 1 = maior volume
  const rank = new Map(Object.keys(series).map(s => [s, new Map()]));
  {
    const byDay = new Map();
    for (const [s, cs] of Object.entries(series)) {
      let q = 0;
      for (let i = 0; i < cs.length; i++) {
        q += cs[i].q;
        if (i >= P.volWin) { q -= cs[i - P.volWin].q; if (!byDay.has(cs[i].t)) byDay.set(cs[i].t, []); byDay.get(cs[i].t).push([s, q / P.volWin]); }
      }
    }
    for (const [t, arr] of byDay) { arr.sort((a, b) => b[1] - a[1]); arr.forEach(([s], k) => rank.get(s).set(t, k + 1)); }
  }
  // Operações sem sobreposição por moeda (8 dias de intervalo)
  const pick = (list, ok) => {
    const last = {}, out = [];
    for (const r of list) { if (!ok(r)) continue; if (last[r.sym] !== undefined && r.t < last[r.sym] + (H + 1) * DAY) continue; last[r.sym] = r.t; out.push(r.net); }
    return out;
  };
  const st = a => {
    const n = a.length;
    if (n < 2) return { n, mean: n ? a[0] : 0, se: 0, win: n ? (a[0] > 0 ? 100 : 0) : 0 };
    const m = mean(a), sd = Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (n - 1));
    return { n, mean: m, se: sd / Math.sqrt(n), win: a.filter(x => x > 0).length / n * 100 };
  };
  const zOf = (a, b) => a.n >= 2 && b.n >= 2 ? (a.mean - b.mean) / Math.sqrt(a.se ** 2 + b.se ** 2) : 0;

  function runVariant(V) {
    const nf = V.extra ? ML.NF_EXTRA : ML.NF, rows = [];
    for (const [sym, cs] of Object.entries(series)) {
      const rk = rank.get(sym);
      for (let i = 56; i < cs.length - H - 1; i++) {
        const r0 = rk.get(cs[i].t);
        if (!r0 || r0 > V.topN) continue;
        if (V.minDias) { let ok = true; for (let d = 1; d <= V.minDias; d++) { const r = rk.get(cs[i - d].t); if (!r || r > V.topN) { ok = false; break; } } if (!ok) continue; }
        const f = ML.features(cs, i, bctx, V.extra);
        if (!f) continue;
        rows.push({ sym, t: cs[i].t, x: f.x, y: ML.label(cs, i, f.a), net: ML.tradeNet(cs, i, f.a, P), mom: f.x[4] >= 0 && f.x[5] >= Math.log(2) });
      }
    }
    rows.sort((a, b) => a.t - b.t);
    if (rows.length < 5000) throw new Error("exemplos insuficientes");

    // Teste cronológico: treina só com o passado (com intervalo de segurança) e prevê o mês seguinte
    const months = [...new Set(rows.map(r => mkey(r.t)))].sort((a, b) => a - b);
    let M = ML.newModel(nf), started = false;
    const oos = [];
    for (const m of months.filter(m => m >= months[0] + P.minTrainMonths)) {
      const startT = Date.UTC(Math.floor(m / 12), m % 12, 1);
      const train = rows.filter(r => r.t + (H + 1) * DAY < startT), test = rows.filter(r => mkey(r.t) === m);
      if (train.length < 3000 || !test.length) continue;
      ML.fit(M, train, { epochs: started ? 40 : 120, restandardize: !started });
      started = true;
      const pr = [0, 0, 0]; for (const r of train) pr[r.y]++;
      const prior = pr.map(v => v / train.length);
      for (const r of test) oos.push({ sym: r.sym, t: r.t, y: r.y, net: r.net, x: r.x, mom: r.mom, p: ML.predict(M, r.x), prior });
    }
    if (oos.length < 2000) throw new Error("poucos exemplos fora da amostra");

    const ll = mean(oos.map(r => ML.logLoss(r.p, r.y))), llb = mean(oos.map(r => ML.logLoss(r.prior, r.y)));
    const am = r => r.p.indexOf(Math.max(...r.p)) === r.y ? 1 : 0, ab = r => r.prior.indexOf(Math.max(...r.prior)) === r.y ? 1 : 0;
    const bins = [[0, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 1.01]].map(([lo, hi]) => {
      const g = oos.filter(r => r.p[0] >= lo && r.p[0] < hi);
      return { lo, hi: Math.min(hi, 1), n: g.length, previsto: g.length ? mean(g.map(r => r.p[0])) : null, observado: g.length ? mean(g.map(r => r.y === 0 ? 1 : 0)) : null };
    });
    // Limite de sinal escolhido na parte inicial do período fora da amostra e conferido na parte final
    const tSplit = oos[Math.floor(oos.length * 0.7)].t, sel = oos.filter(r => r.t < tSplit), hold = oos.filter(r => r.t >= tSplit);
    let thr = null, best = -1e9;
    for (const c of [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]) { const s = st(pick(sel, r => r.p[0] >= c)); if (s.n >= MIN_N && s.mean > best) { best = s.mean; thr = c; } }
    const control = st(pick(hold, () => true)), momentum = st(pick(hold, r => r.x[4] >= 0 && r.x[5] >= Math.log(2)));
    const model = thr === null ? { n: 0, mean: 0, se: 0, win: 0 } : st(pick(hold, r => r.p[0] >= thr));
    const mm = thr === null ? { n: 0, mean: 0, se: 0, win: 0 } : st(pick(hold, r => r.p[0] >= thr && r.mom));
    const z = zOf(model, control), validated = thr !== null && ll < llb && model.n >= MIN_N && model.mean > 0 && z >= 2;

    // Modelo final: treinado com todo o histórico da variante
    const finalM = ML.newModel(nf), pr = [0, 0, 0]; for (const r of rows) pr[r.y]++;
    ML.fit(finalM, rows, { epochs: 150, restandardize: true });
    return {
      id: V.id, nome: V.nome, spec: { topN: V.topN, minDias: V.minDias || 0, extra: !!V.extra }, exemplos: rows.length, foraDaAmostra: oos.length,
      logLoss: ML.round(ll), logLossBase: ML.round(llb), acuracia: ML.round(mean(oos.map(am))), acuraciaBase: ML.round(mean(oos.map(ab))), calibracao: bins, limite: thr,
      teste: { modelo: model, controle: control, momentum, modeloMomentum: mm, z: ML.round(z, 2), zModeloMomentum: ML.round(zOf(mm, control), 2) },
      periodoTeste: { de: oos[0].t, ate: oos[oos.length - 1].t, cortes: tSplit }, validated, packed: ML.pack(finalM), priors: pr.map(v => ML.round(v / rows.length))
    };
  }

  // Poucas variantes de propósito: quanto mais se testa, maior a chance de uma parecer boa por sorte.
  const variantes = [
    { id: "base60", nome: "Top 60, atributos básicos (regra atual)", topN: 60, minDias: 0, extra: false },
    { id: "extra60", nome: "Top 60, com RSI, ADX e volume de 5 dias como atributos", topN: 60, minDias: 0, extra: true },
    { id: "base40", nome: "Top 40, atributos básicos", topN: 40, minDias: 0, extra: false },
    { id: "base80", nome: "Top 80, atributos básicos", topN: 80, minDias: 0, extra: false },
    { id: "base100", nome: "Top 100, atributos básicos", topN: 100, minDias: 0, extra: false },
    { id: "persist7", nome: "Top 60, só moedas no ranking há 7 dias ou mais", topN: 60, minDias: 7, extra: false }
  ];
  const res = [];
  for (const V of variantes) {
    console.log("Variante " + V.id + "...");
    try { res.push(runVariant(V)); } catch (e) { console.error("Variante " + V.id + " falhou: " + e.message); }
  }
  const base = res.find(r => r.id === "base60");
  if (!base) throw new Error("a variante base falhou");
  // Troca de variante só com prova: validada e melhor que a base com z de 3 ou mais, no mesmo período
  let esc = base;
  for (const r of res) if (r !== base && r.validated && zOf(r.teste.modelo, base.teste.modelo) >= 3 && r.teste.modelo.mean > esc.teste.modelo.mean) esc = r;

  const out = {
    version: 2, trainedAt: Date.now(), features: esc.spec.extra ? [...ML.FEATURES, ...ML.FEATURES_EXTRA] : ML.FEATURES, classes: ML.CLASSES, ...esc.packed,
    priors: esc.priors, thr: esc.limite, validated: esc.validated, demoted: false, live: { n: 0, ll: 0, llb: 0, hits: 0 }, spec: esc.spec,
    report: {
      minAtr: ML.MIN_ATR || null, exemplos: esc.exemplos, moedas: Object.keys(series).length, foraDaAmostra: esc.foraDaAmostra,
      logLoss: esc.logLoss, logLossBase: esc.logLossBase, acuracia: esc.acuracia, acuraciaBase: esc.acuraciaBase, calibracao: esc.calibracao, limite: esc.limite,
      teste: esc.teste, periodoTeste: esc.periodoTeste, variante: esc.id
    },
    variantes: res.map(r => ({
      id: r.id, nome: r.nome, spec: r.spec, exemplos: r.exemplos, logLoss: r.logLoss, logLossBase: r.logLossBase, limite: r.limite,
      modelo: r.teste.modelo, controle: r.teste.controle, momentum: r.teste.momentum, modeloMomentum: r.teste.modeloMomentum, z: r.teste.z, zModeloMomentum: r.teste.zModeloMomentum,
      validated: r.validated, escolhida: r === esc
    }))
  };
  fs.mkdirSync("data", { recursive: true });
  fs.writeFileSync("data/model.json", JSON.stringify(out));
  console.log(JSON.stringify(out.variantes.map(v => ({ id: v.id, n: v.modelo.n, media: v.modelo.mean && +v.modelo.mean.toFixed(2), z: v.z, validado: v.validated, escolhida: v.escolhida }))));
  console.log(esc.validated ? "MODELO VALIDADO (" + esc.id + ")" : "Modelo NÃO validado (o bot continua com a regra de momentum).");
}

// ===================== saidas =====================
async function saidas(series) {
  // saidas.js: compara regras de SAÍDA (meta, stop, stop móvel, prazo) sobre os mesmos sinais de momentum,
  // no histórico público da Binance (inclusive moedas que saíram), contra entradas ao acaso com a mesma saída.
  const fs = require("fs");
  const ML = require("./ml.js");
  const DAY = 864e5;
  const P = { fee: 0.075, slip: 0.1, topN: 60, volWin: 30 };
  const CONV = +process.env.CONV || 0.1; // custo de cada conversão reais<->USDT (%); uma ida e uma volta = 2 conversões
  const clamp = (v, a, b) => Math.min(Math.max(v, a), b);

  // stop e meta em % da entrada; a = oscilação diária (ATR %)
  const VARIANTS = [
    { id: "atual", nome: "Atual: meta 10%, stop de 2 oscilações (entre 4% e 15%), prazo 7 dias", days: 7, tgt: () => 10, stop: a => clamp(2 * a, 4, 15) },
    { id: "stop-curto", nome: "Stop curto: meta 10%, stop de 1 oscilação (entre 3% e 10%), prazo 7 dias", days: 7, tgt: () => 10, stop: a => clamp(a, 3, 10) },
    { id: "proporcional", nome: "Proporcional: meta de 1,5 oscilação (entre 4% e 25%), stop de 1 oscilação (entre 3% e 12%), prazo 7 dias", days: 7, tgt: a => clamp(1.5 * a, 4, 25), stop: a => clamp(a, 3, 12) },
    { id: "movel", nome: "Stop móvel: sem meta, stop de 2 oscilações (entre 4% e 15%) que sobe com a máxima, prazo 7 dias", days: 7, tgt: null, stop: a => clamp(2 * a, 4, 15), trail: true },
    { id: "sem-meta", nome: "Sem meta: stop fixo de 2 oscilações (entre 4% e 15%), sai no prazo de 7 dias", days: 7, tgt: null, stop: a => clamp(2 * a, 4, 15) },
    { id: "meta-5", nome: "Meta menor: meta 5%, stop de 2 oscilações (entre 4% e 15%), prazo 7 dias", days: 7, tgt: () => 5, stop: a => clamp(2 * a, 4, 15) },
    { id: "prazo-3", nome: "Prazo curto: meta 10%, stop de 2 oscilações (entre 4% e 15%), prazo 3 dias", days: 3, tgt: () => 10, stop: a => clamp(2 * a, 4, 15) }
  ];

  function buildUniverse(series) {
    const byDay = new Map();
    for (const [s, cs] of Object.entries(series)) {
      let q = 0;
      for (let i = 0; i < cs.length; i++) {
        q += cs[i].q;
        if (i >= P.volWin) {
          q -= cs[i - P.volWin].q;
          if (!byDay.has(cs[i].t)) byDay.set(cs[i].t, []);
          byDay.get(cs[i].t).push([s, q / P.volWin]);
        }
      }
    }
    const top = new Map(Object.keys(series).map(s => [s, new Set()]));
    for (const [t, arr] of byDay) { arr.sort((a, b) => b[1] - a[1]); for (const [s] of arr.slice(0, P.topN)) top.get(s).add(t); }
    return top;
  }
  // Simula uma operação com os candles diários. Se stop e meta forem tocados no mesmo dia, vale o stop; se abrir abaixo do stop, sai na abertura.
  function run(cs, i, a, v) {
    const entry = cs[i + 1].o * (1 + P.slip / 100);
    let stopP = entry * (1 - v.stop(a) / 100), hh = entry;
    const tgtP = v.tgt ? entry * (1 + v.tgt(a) / 100) : null;
    let exit = null, why = "prazo";
    for (let k = 1; k <= v.days; k++) {
      const x = cs[i + k];
      if (x.l <= stopP) { exit = Math.min(stopP, x.o) * (1 - P.slip / 100); why = "stop"; break; }
      if (tgtP && x.h >= tgtP) { exit = tgtP; why = "meta"; break; }
      if (v.trail) { hh = Math.max(hh, x.h); stopP = Math.max(stopP, hh * (1 - v.stop(a) / 100)); }
    }
    if (exit === null) exit = cs[i + v.days].c * (1 - P.slip / 100);
    return { net: (exit / entry - 1) * 100 - 2 * P.fee, why };
  }
  const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
  function stats(list) {
    const n = list.length;
    if (n < 2) return { n };
    const r = list.map(x => x.net), m = mean(r), sd = Math.sqrt(r.reduce((s, x) => s + (x - m) ** 2, 0) / (n - 1));
    const w = r.filter(x => x > 0), l = r.filter(x => x <= 0);
    const h1 = list.filter(x => x.pos < 0.5).map(x => x.net), h2 = list.filter(x => x.pos >= 0.5).map(x => x.net);
    const cnt = k => list.filter(x => x.why === k).length / n * 100;
    return {
      n, mean: m, margem: 2 * sd / Math.sqrt(n), win: w.length / n * 100, ganhoMedio: mean(w), perdaMedia: mean(l), pior: Math.min(...r),
      meta: cnt("meta"), stop: cnt("stop"), prazo: cnt("prazo"), metade1: h1.length ? mean(h1) : null, metade2: h2.length ? mean(h2) : null
    };
  }

  return (async () => {
    console.log("Baixando histórico...");
    // série compartilhada (baixada uma só vez pelo pesquisa.js)
    const top = buildUniverse(series);
    let t0 = Infinity, tEnd = 0;
    for (const cs of Object.values(series)) { t0 = Math.min(t0, cs[0].t); tEnd = Math.max(tEnd, cs[cs.length - 1].t); }
    const span = tEnd - t0 || 1;

    // Candidatos: dias em que a moeda estava no universo; marca quais têm sinal de momentum
    const cands = [];
    for (const [sym, cs] of Object.entries(series)) {
      const set = top.get(sym);
      for (let i = 25; i < cs.length - 9; i++) {
        if (!set.has(cs[i].t)) continue;
        const a = ML.atrPctAt(cs, i);
        if (!a || a < ML.MIN_ATR) continue; // ignora stablecoins e ativos atrelados
        let mh = 0, sv = 0;
        for (let j = i - 20; j < i; j++) { if (cs[j].h > mh) mh = cs[j].h; sv += cs[j].v; }
        cands.push({ sym, cs, i, a, t: cs[i].t, mom: sv > 0 && cs[i].c > mh && cs[i].v / (sv / 20) >= 2 });
      }
    }
    cands.sort((x, y) => x.t - y.t);
    console.log(`${cands.length} dias-moeda, ${cands.filter(c => c.mom).length} com sinal de momentum.`);

    const out = { generated: Date.now(), minAtr: ML.MIN_ATR || null, conversao: CONV, moedas: Object.keys(series).length, diasMoeda: cands.length, sinais: cands.filter(c => c.mom).length, variantes: [] };
    for (const v of VARIANTS) {
      const lastSig = {}, lastCtl = {}, sig = [], ctl = [];
      for (const c of cands) {
        if (c.i + v.days >= c.cs.length) continue;
        const gap = (v.days + 1) * DAY, pos = (c.t - t0) / span;
        if (lastCtl[c.sym] === undefined || c.t >= lastCtl[c.sym] + gap) { lastCtl[c.sym] = c.t; ctl.push({ ...run(c.cs, c.i, c.a, v), pos }); }
        if (c.mom && (lastSig[c.sym] === undefined || c.t >= lastSig[c.sym] + gap)) { lastSig[c.sym] = c.t; sig.push({ ...run(c.cs, c.i, c.a, v), pos }); }
      }
      const s = stats(sig), k = stats(ctl);
      const z = s.n >= 2 && k.n >= 2 ? (s.mean - k.mean) / Math.sqrt((s.margem / 2) ** 2 + (k.margem / 2) ** 2) : null;
      out.variantes.push({ id: v.id, nome: v.nome, sinal: s, controle: k, vantagem: s.n && k.n ? s.mean - k.mean : null, z, mediaComConversao: s.n >= 2 ? s.mean - 2 * CONV : null });
      console.log(`${v.id}: z=${z === null ? "n/d" : z.toFixed(2)} | sinal n=${s.n} média ${s.mean && s.mean.toFixed(2)}% ± ${s.margem && s.margem.toFixed(2)} | controle n=${k.n} média ${k.mean && k.mean.toFixed(2)}%`);
    }
    fs.mkdirSync("data", { recursive: true });
    fs.writeFileSync("data/saidas.json", JSON.stringify(out));
  })();

}

// ===================== forca =====================
async function forca(series) {
  // forca.js: (1) cria a "força do sinal" em 5 barras e testa se as barras significam algo fora da amostra;
  // (2) avalia a estratégia: comprar com R$ 50, fechar no lucro modesto e reabrir se o sinal tiver 3 barras ou mais.
  const fs = require("fs");
  const ML = require("./ml.js");
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

  return (async () => {
    console.log("Baixando histórico...");
    // série compartilhada (baixada uma só vez pelo pesquisa.js)
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
    // As barras separam sinais bons dos ruins? (teste mais robusto que exigir melhora perfeita nível a nível)
    const rank = arr => { const idx = arr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]), r = Array(arr.length); let i = 0; while (i < idx.length) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1; i = j + 1; } return r; };
    const spearman = (x, y) => { const rx = rank(x), ry = rank(y), mx = mean(rx), my = mean(ry); let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < rx.length; i++) { sxy += (rx[i] - mx) * (ry[i] - my); sxx += (rx[i] - mx) ** 2; syy += (ry[i] - my) ** 2; } return sxy / Math.sqrt(sxx * syy); };
    const rho7 = spearman(test.map(s => s.score), test.map(s => s.n7)), rhoNet = spearman(test.map(s => s.score), test.map(s => s.net.semMeta));
    const lowS = test.filter(s => s.bars <= 2), highS = test.filter(s => s.bars >= 3), separacao = {};
    for (const id of ["semMeta", "meta10", "meta5"]) { const a = st(highS.map(s => s.net[id])), b = st(lowS.map(s => s.net[id])); separacao[id] = { altas: a, baixas: b, diff: a.mean - b.mean, z: (a.mean - b.mean) / Math.sqrt(a.se ** 2 + b.se ** 2) }; }
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
      generated: Date.now(), minAtr: ML.MIN_ATR || null, conversao: CONV, features: NAMES,
      modelo: { w: Mf.w.map(v => ML.round(v)), b: ML.round(Mf.b), mu: Mf.mu.map(v => ML.round(v)), sd: Mf.sd.map(v => ML.round(v)), edges: edgesF.map(v => ML.round(v)) },
      periodo: { sinais: sigs.length, treino: train.length, teste: test.length, de: test[0].entryT, ate: test[test.length - 1].entryT },
      calibracao: { niveis: levels, todos: all, rho7: ML.round(rho7, 3), rhoNet: ML.round(rhoNet, 3), zRho7: ML.round(rho7 * Math.sqrt(test.length - 1), 2), zRhoNet: ML.round(rhoNet * Math.sqrt(test.length - 1), 2), separacao, n7Crescente: increasing(m7), semMetaCrescente: increasing(mSem) },
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
  })();

}

(async () => {
  if (!ML.MIN_ATR) throw new Error("ml.js desatualizado: envie o ml.js novo ao repositório antes de rodar a pesquisa");
  console.log("Baixando histórico (uma vez só)...");
  const series = await loadAll();
  const res = { generated: Date.now(), versoes: { ml: ML.VERSAO || null, minAtr: ML.MIN_ATR }, ok: {}, erros: {} };
  for (const [nome, fn] of [["treino", treino], ["saidas", saidas], ["forca", forca]]) {
    console.log("\n=== " + nome + " ===");
    try { await fn(series); res.ok[nome] = true; }
    catch (e) { console.error("Falhou em " + nome + ": " + e.message); res.ok[nome] = false; res.erros[nome] = e.message; process.exitCode = 1; }
  }
  fs.mkdirSync("data", { recursive: true });
  fs.writeFileSync("data/pesquisa.json", JSON.stringify(res));
  console.log("\nResumo:", JSON.stringify(res.ok));
})().catch(e => { console.error(e); process.exit(1); });
