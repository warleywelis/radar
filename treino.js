// treino.js: treina e VALIDA o modelo dos 3 caminhos. Só vira "validado" se bater entradas ao acaso, fora da amostra.
const fs = require("fs");
const ML = require("./ml.js");
const { loadAll } = require("./dados.js");
const DAY = 864e5, H = ML.H;
const MIN_N = +process.env.MIN_N || 40; // mínimo de operações no teste final (variável só para testes)
const P = { tgt: 10, k: 2, maxl: 15, fee: 0.075, slip: 0.1, topN: 60, volWin: 30, minTrainMonths: 24 };
const mkey = t => { const d = new Date(t); return d.getUTCFullYear() * 12 + d.getUTCMonth(); };
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;

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
  for (const [t, arr] of byDay) {
    arr.sort((a, b) => b[1] - a[1]);
    for (const [s] of arr.slice(0, P.topN)) top.get(s).add(t);
  }
  return top;
}
// Operações sem sobreposição por moeda (8 dias de intervalo)
function pick(list, ok) {
  const last = {}, out = [];
  for (const r of list) {
    if (!ok(r)) continue;
    if (last[r.sym] !== undefined && r.t < last[r.sym] + (H + 1) * DAY) continue;
    last[r.sym] = r.t; out.push(r.net);
  }
  return out;
}
function st(a) {
  const n = a.length;
  if (n < 2) return { n, mean: n ? a[0] : 0, se: 0, win: n ? (a[0] > 0 ? 100 : 0) : 0 };
  const m = mean(a), sd = Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (n - 1));
  return { n, mean: m, se: sd / Math.sqrt(n), win: a.filter(x => x > 0).length / n * 100 };
}

(async () => {
  console.log("Baixando histórico...");
  const series = await loadAll();
  const btc = series["BTCUSDT"];
  if (!btc) throw new Error("BTCUSDT não encontrado no arquivo público");
  const bctx = ML.btcContext(btc);
  const top = buildUniverse(series);

  console.log("Montando exemplos...");
  const rows = [];
  for (const [sym, cs] of Object.entries(series)) {
    const set = top.get(sym);
    for (let i = 56; i < cs.length - H - 1; i++) {
      if (!set.has(cs[i].t)) continue;
      const f = ML.features(cs, i, bctx);
      if (!f) continue;
      rows.push({ sym, t: cs[i].t, x: f.x, y: ML.label(cs, i, f.a), net: ML.tradeNet(cs, i, f.a, P) });
    }
  }
  rows.sort((a, b) => a.t - b.t);
  console.log(`${rows.length} exemplos de ${Object.keys(series).length} moedas.`);
  if (rows.length < 5000) throw new Error("exemplos insuficientes para treinar");

  // Teste cronológico: treina só com o passado (com intervalo de segurança) e prevê o mês seguinte
  const months = [...new Set(rows.map(r => mkey(r.t)))].sort((a, b) => a - b);
  const testMonths = months.filter(m => m >= months[0] + P.minTrainMonths);
  let M = ML.newModel(), started = false;
  const oos = [];
  for (const m of testMonths) {
    const startT = Date.UTC(Math.floor(m / 12), m % 12, 1);
    const train = rows.filter(r => r.t + (H + 1) * DAY < startT), test = rows.filter(r => mkey(r.t) === m);
    if (train.length < 3000 || !test.length) continue;
    ML.fit(M, train, { epochs: started ? 40 : 120, restandardize: !started });
    started = true;
    const pr = [0, 0, 0]; for (const r of train) pr[r.y]++;
    const prior = pr.map(v => v / train.length);
    for (const r of test) oos.push({ sym: r.sym, t: r.t, y: r.y, net: r.net, x: r.x, p: ML.predict(M, r.x), prior });
  }
  if (oos.length < 2000) throw new Error("poucos exemplos fora da amostra");

  const ll = mean(oos.map(r => ML.logLoss(r.p, r.y))), llb = mean(oos.map(r => ML.logLoss(r.prior, r.y)));
  const acc = mean(oos.map(r => r.p.indexOf(Math.max(...r.p)) === r.y ? 1 : 0)), accb = mean(oos.map(r => r.prior.indexOf(Math.max(...r.prior)) === r.y ? 1 : 0));
  const bins = [[0, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 1.01]].map(([lo, hi]) => {
    const g = oos.filter(r => r.p[0] >= lo && r.p[0] < hi);
    return { lo, hi: Math.min(hi, 1), n: g.length, previsto: g.length ? mean(g.map(r => r.p[0])) : null, observado: g.length ? mean(g.map(r => r.y === 0 ? 1 : 0)) : null };
  });

  // Limite de sinal: escolhido na primeira parte do período fora da amostra e conferido na última parte (nunca vista na escolha)
  const tSplit = oos[Math.floor(oos.length * 0.7)].t;
  const sel = oos.filter(r => r.t < tSplit), hold = oos.filter(r => r.t >= tSplit);
  let thr = null, best = -1e9;
  for (const c of [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]) {
    const s = st(pick(sel, r => r.p[0] >= c));
    if (s.n >= MIN_N && s.mean > best) { best = s.mean; thr = c; }
  }
  const control = st(pick(hold, () => true));
  const momentum = st(pick(hold, r => r.x[4] >= 0 && r.x[5] >= Math.log(2)));
  const model = thr === null ? { n: 0, mean: 0, se: 0, win: 0 } : st(pick(hold, r => r.p[0] >= thr));
  const z = model.n >= 2 ? (model.mean - control.mean) / Math.sqrt(model.se ** 2 + control.se ** 2) : 0;
  const validated = thr !== null && ll < llb && model.n >= MIN_N && model.mean > 0 && z >= 2;

  // Modelo final: treinado com todo o histórico
  const finalM = ML.newModel();
  const all = rows.filter(r => true), pr = [0, 0, 0]; for (const r of all) pr[r.y]++;
  ML.fit(finalM, all, { epochs: 150, restandardize: true });
  const out = {
    version: 1, trainedAt: Date.now(), features: ML.FEATURES, classes: ML.CLASSES, ...ML.pack(finalM),
    priors: pr.map(v => ML.round(v / all.length)), thr, validated, demoted: false,
    live: { n: 0, ll: 0, llb: 0, hits: 0 },
    report: {
      exemplos: rows.length, moedas: Object.keys(series).length, foraDaAmostra: oos.length,
      logLoss: ML.round(ll), logLossBase: ML.round(llb), acuracia: ML.round(acc), acuraciaBase: ML.round(accb),
      calibracao: bins, limite: thr,
      teste: { modelo: model, controle: control, momentum, z: ML.round(z, 2) },
      periodoTeste: { de: oos[0].t, ate: oos[oos.length - 1].t, cortes: tSplit }
    }
  };
  fs.mkdirSync("data", { recursive: true });
  fs.writeFileSync("data/model.json", JSON.stringify(out));
  console.log(JSON.stringify(out.report, null, 1));
  console.log(validated ? "MODELO VALIDADO" : "Modelo NÃO validado (o bot continua com a regra de momentum).");
})().catch(e => { console.error(e); process.exit(1); });
