// saidas.js: compara regras de SAÍDA (meta, stop, stop móvel, prazo) sobre os mesmos sinais de momentum,
// no histórico público da Binance (inclusive moedas que saíram), contra entradas ao acaso com a mesma saída.
const fs = require("fs");
const ML = require("./ml.js");
const { loadAll } = require("./dados.js");
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

(async () => {
  console.log("Baixando histórico...");
  const series = await loadAll();
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
      if (!a) continue;
      let mh = 0, sv = 0;
      for (let j = i - 20; j < i; j++) { if (cs[j].h > mh) mh = cs[j].h; sv += cs[j].v; }
      cands.push({ sym, cs, i, a, t: cs[i].t, mom: sv > 0 && cs[i].c > mh && cs[i].v / (sv / 20) >= 2 });
    }
  }
  cands.sort((x, y) => x.t - y.t);
  console.log(`${cands.length} dias-moeda, ${cands.filter(c => c.mom).length} com sinal de momentum.`);

  const out = { generated: Date.now(), conversao: CONV, moedas: Object.keys(series).length, diasMoeda: cands.length, sinais: cands.filter(c => c.mom).length, variantes: [] };
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
})().catch(e => { console.error(e); process.exit(1); });
