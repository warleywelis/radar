// Robô do Radar Binance: roda sozinho no GitHub Actions, sem dinheiro de verdade.
// Sinal de entrada = previsão do modelo dos 3 caminhos (alta, estabilidade, baixa) quando o modelo está VALIDADO;
// caso contrário, usa a regra de momentum (rompimento da máxima de 20 dias com volume forte). Meta 10%, prazo 7 dias.
const fs = require("fs");
const ML = require("./ml.js");
const BASE = "https://data-api.binance.vision";
const C = { cap: 50, tgt: 10, days: 7, vm: 2, k: 2, maxl: 15, minVol: 5e6, fee: 0.075, slip: 0.1, top: 60, L: 20 };
const DAY = 864e5;
const SKIP = new Set(["USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "EUR", "TRY", "BRL", "AEUR", "USD1", "XUSD", "PAXG", "USDS", "BFUSD", "RLUSD", "USDE", "USDD", "PYUSD", "USD0", "EURI", "XAUT", "U"]);
const whyTxt = { stop: "saiu no stop", meta: "saiu na meta", tempo: "saiu pelo prazo" };

async function get(path) {
  let last;
  for (let i = 0; i < 3; i++) {
    try {
      const r = await fetch(BASE + path);
      if (!r.ok) throw new Error("HTTP " + r.status + " em " + path);
      return await r.json();
    } catch (e) { last = e; await new Promise(r => setTimeout(r, 1500)); }
  }
  throw last;
}
async function tg(text) {
  const t = process.env.TG_TOKEN, c = process.env.TG_CHAT;
  if (!t || !c || !text) return;
  try {
    await fetch(`https://api.telegram.org/bot${t}/sendMessage`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ chat_id: c, text })
    });
  } catch (e) { console.log("Telegram falhou:", e.message); }
}
const money = (pct) => `R$ ${(C.cap * pct / 100).toFixed(2).replace(".", ",")} (${pct.toFixed(1).replace(".", ",")}%)`;
const loadJson = (f, d) => { try { return JSON.parse(fs.readFileSync(f, "utf8")); } catch (e) { return d; } };
const save = (f, o) => fs.writeFileSync(f, JSON.stringify(o));
const r4 = v => Math.round(v * 1e4) / 1e4;

function fin(t, px, why, at) {
  t.status = "fechada"; t.exit = px; t.why = why; t.end = at;
  t.net = (px / t.entry - 1) * 100 - 2 * t.fee;
}
async function resolve(t) {
  const k = await get(`/api/v3/klines?symbol=${t.sym}&interval=1h&startTime=${t.time}&limit=1000`);
  let last = null;
  for (const c of k) {
    const ot = +c[0];
    if (ot >= t.exp) break;
    last = +c[4];
    if (+c[3] <= t.stopP) return fin(t, t.stopP * (1 - t.slip / 100), "stop", ot);
    if (+c[2] >= t.tgtP) return fin(t, t.tgtP, "meta", ot);
  }
  if (Date.now() >= t.exp) return fin(t, (last || t.entry) * (1 - t.slip / 100), "tempo", t.exp);
  t.cur = last || t.entry;
}
function evalMomentum(cs, i = cs.length - 2) {
  if (i < 35) return null;
  let mh = 0, sv = 0;
  for (let j = i - C.L; j < i; j++) { if (cs[j].h > mh) mh = cs[j].h; sv += cs[j].v; }
  const atr = ML.atrPctAt(cs, i), vr = sv > 0 ? cs[i].v / (sv / C.L) : 0;
  if (!atr || atr < ML.MIN_ATR) return null; // ignora stablecoins e ativos atrelados
  return { sig: cs[i].c > mh && vr >= C.vm, vr, atr, t: cs[i].t };
}
async function daily(sym) {
  const k = await get(`/api/v3/klines?symbol=${sym}&interval=1d&limit=70`);
  return k.map(c => ({ t: +c[0], o: +c[1], h: +c[2], l: +c[3], c: +c[4], v: +c[5] }));
}
function newTrade(s, now, source, rule) {
  const entry = s.ask * (1 + C.slip / 100), semMeta = rule === "momentum-sem-meta"; // sem meta: só stop e prazo (tgtP enorme, pois JSON não guarda Infinity)
  return {
    id: now + Math.random(), sym: s.sym, base: s.base, time: now, exp: now + C.days * DAY, entry, cap: C.cap, fee: C.fee, slip: C.slip,
    vr: s.vr, stopPct: s.stopPct, tgt: semMeta ? null : C.tgt, stopP: entry * (1 - s.stopPct / 100), tgtP: semMeta ? 1e18 : entry * (1 + C.tgt / 100),
    status: "aberta", cur: entry, source, rule, sigT: s.sigT, p: s.p ? s.p.map(r4) : undefined, bars: s.bars
  };
}

(async () => {
  fs.mkdirSync("data", { recursive: true });
  const S = loadJson("data/trades.json", { trades: [] });
  const SH = loadJson("data/shadow.json", { trades: [] });
  const PR = loadJson("data/predictions.json", { items: [] });
  const M = loadJson("data/model.json", null);
  const FORCA = loadJson("data/forca.json", null); // pesos da força do sinal (5 barras), criados pelo fluxo "forca"
  const msgs = [];

  // 1) Conferir operações abertas (principal e sombra)
  for (const t of S.trades.filter(t => t.status === "aberta")) {
    await resolve(t);
    if (t.status === "fechada") msgs.push(`Operação simulada fechada: ${t.base}/USDT ${whyTxt[t.why]}, resultado ${money(t.net)}.`);
  }
  for (const t of SH.trades.filter(t => t.status === "aberta")) {
    try { await resolve(t); } catch (e) { console.log("Sombra falhou em " + t.sym + ": " + e.message); }
  }

  // 2) Aprendizado contínuo: previsões com 8 dias ou mais já têm resultado real
  let learned = 0;
  if (M) {
    for (const it of PR.items) {
      if (it.done || Date.now() < it.t + 8 * DAY + 36e5) continue;
      try {
        const k = await get(`/api/v3/klines?symbol=${it.sym}&interval=1d&startTime=${it.t}&limit=10`);
        if (k.length < 8) continue;
        const n = (+k[7][4] / +k[1][1] - 1) * 100 / it.a, y = n >= 1 ? 0 : n <= -1 ? 2 : 1;
        if (it.m === M.trainedAt) { // só conta no placar ao vivo as previsões feitas pela versão atual do modelo
          M.live.n++; M.live.ll += ML.logLoss(it.p, y); M.live.llb += ML.logLoss(M.priors, y);
          if (it.p.indexOf(Math.max(...it.p)) === y) M.live.hits++;
        }
        ML.sgd(M, it.x, y);
        it.done = true; learned++;
      } catch (e) { console.log("Previsão não resolvida " + it.sym + ": " + e.message); }
    }
    PR.items = PR.items.filter(it => !it.done);
    M.demoted = M.live.n >= 100 && (M.live.ll - M.live.llb) / M.live.n > 0.02;
  }

  // 3) Dados de hoje: top moedas por volume, candles diários e contexto do Bitcoin
  const ticker = await get("/api/v3/ticker/24hr");
  const cands = ticker.filter(x => {
    if (!x.symbol.endsWith("USDT")) return false;
    const b = x.symbol.slice(0, -4);
    return !SKIP.has(b) && !/(UP|DOWN|BULL|BEAR)$/.test(b) && +x.quoteVolume >= C.minVol && +x.askPrice > 0 && +x.bidPrice > 0;
  }).sort((a, b) => b.quoteVolume - a.quoteVolume).slice(0, C.top);
  const bctx = ML.btcContext(await daily("BTCUSDT"));

  const scan = [], sigs = []; // sigs = sinais de momentum nos últimos 3 candles fechados (janela de entrada)
  for (let i = 0; i < cands.length; i += 10) {
    const batch = await Promise.all(cands.slice(i, i + 10).map(x => daily(x.symbol).catch(() => null)));
    batch.forEach((cs, j) => {
      const x = cands[i + j];
      if (!cs) return;
      const row = { sym: x.symbol, base: x.symbol.slice(0, -4), ask: +x.askPrice };
      const m = evalMomentum(cs);
      if (m) { row.mom = m.sig; row.vr = m.vr; row.sigT = m.t; row.atr = m.atr; row.stopPct = Math.min(Math.max(C.k * m.atr, 4), C.maxl); }
      const f = ML.features(cs, cs.length - 2, bctx);
      if (f) {
        row.x = f.x; row.a = f.a; row.c = cs[cs.length - 2].c;
        if (FORCA) { // força dinâmica: usa a distância entre o preço de agora e o fechamento do sinal
          const gap = Math.max(-3, Math.min(3, (row.ask / row.c - 1) / (f.a / 100))), F = FORCA.modelo;
          row.score = F.b + [...f.x, gap].reduce((sum, v, j) => sum + F.w[j] * (v - F.mu[j]) / F.sd[j], 0);
          row.bars = 1 + F.edges.filter(e => row.score > e).length;
        }
        if (M) {
          row.p = ML.predict(M, f.x); row.sigT = cs[cs.length - 2].t;
          row.stopPct = Math.min(Math.max(C.k * f.a, 4), C.maxl);
          if (row.vr === undefined) row.vr = 0;
        }
      }
      scan.push(row);
      for (let off = 2; off <= 4; off++) { // hoje (0), ontem (1) e anteontem (2); a força é recalculada com o preço de agora
        const i = cs.length - off, m2 = evalMomentum(cs, i);
        if (!m2 || !m2.sig) continue;
        const f2 = ML.features(cs, i, bctx);
        const sg = { sym: x.symbol, base: x.symbol.slice(0, -4), ask: +x.askPrice, sigT: cs[i].t, vr: m2.vr, atr: m2.atr, c: cs[i].c, age: off - 2, stopPct: Math.min(Math.max(C.k * m2.atr, 4), C.maxl) };
        if (f2) {
          sg.x = f2.x; sg.a = f2.a;
          if (FORCA) {
            const gap = Math.max(-3, Math.min(3, (sg.ask / sg.c - 1) / (f2.a / 100))), F = FORCA.modelo;
            sg.score = F.b + [...f2.x, gap].reduce((sum, v, j) => sum + F.w[j] * (v - F.mu[j]) / F.sd[j], 0);
            sg.bars = 1 + F.edges.filter(e => sg.score > e).length;
          }
        }
        sigs.push(sg);
      }
    });
  }

  // 4) Registrar previsões de hoje (para aprender quando o resultado chegar)
  if (M) {
    for (const r of scan) {
      if (r.p && !PR.items.some(it => it.sym === r.sym && it.t === r.sigT))
        PR.items.push({ sym: r.sym, t: r.sigT, p: r.p.map(r4), x: r.x.map(r4), a: r4(r.a), m: M.trainedAt, done: false });
    }
  }

  // 5) Sinais
  const momSignals = sigs.filter(g => g.age === 0).sort((a, b) => b.vr - a.vr); // regra original: só o último candle fechado
  const modelSignals = M && M.thr !== null ? scan.filter(r => r.p && r.p[0] >= M.thr).sort((a, b) => b.p[0] - a.p[0]) : [];
  const goodSignals = FORCA ? sigs.filter(g => g.bars >= 3).sort((a, b) => b.score - a.score) : []; // bom sinal = 3 barras ou mais, em qualquer dia da janela
  const sep = FORCA && FORCA.calibracao && FORCA.calibracao.separacao && FORCA.calibracao.separacao.semMeta;
  const useBars = !!(sep && sep.z >= 3 && FORCA.calibracao.rho7 > 0); // só filtra por barras se o teste provou que as barras separam bons e ruins
  const useModel = !!(M && M.validated && !M.demoted);
  const mainRule = useModel ? "modelo" : useBars ? "momentum-barras" : "momentum";
  const mainSignals = useModel ? modelSignals : useBars ? goodSignals : momSignals;

  // Operações-sombra: guardam o resultado de TODOS os sinais de cada regra, separadas da operação principal
  const now0 = Date.now();
  let created = 0;
  for (const [rule, list] of [["momentum", momSignals], ["momentum-sem-meta", momSignals], ["modelo", modelSignals], ["barras3-janela", goodSignals]]) {
    for (const sg of list) {
      if (!SH.trades.some(t => t.sym === sg.sym && t.sigT === sg.sigT && (t.rule || "momentum") === rule)) {
        SH.trades.push(newTrade(sg, now0, "sombra", rule));
        if (rule === "barras3-janela") { created++; msgs.push(`Bom sinal: ${sg.base}/USDT com ${sg.bars} barras de força (${sg.age === 0 ? "sinal de hoje" : `sinal de ${sg.age} ${sg.age === 1 ? "dia" : "dias"} atrás`}).`); }
      }
    }
  }
  let entered = null;
  const fresh = mainSignals.filter(x => !S.trades.some(t => t.sym === x.sym && t.sigT === x.sigT));
  if (!S.trades.some(t => t.status === "aberta") && fresh.length) {
    const sg = fresh[0];
    S.trades.push(newTrade(sg, Date.now(), "nuvem", mainRule));
    entered = sg.sym;
    msgs.push(`Nova operação simulada (${mainRule}): ${sg.base}/USDT${sg.p ? `, chance de alta ${(sg.p[0] * 100).toFixed(0)}%` : ""}. Stop em ${money(-sg.stopPct)}, meta em ${money(C.tgt)}.`);
  }

  save("data/trades.json", S);
  save("data/shadow.json", SH);
  save("data/predictions.json", PR);
  if (M) save("data/model.json", M);
  const top = scan.filter(r => r.p).sort((a, b) => b.p[0] - a.p[0]).slice(0, 8).map(r => ({ sym: r.sym, base: r.base, p: r.p.map(r4) }));
  save("data/last-run.json", {
    time: Date.now(), analyzed: cands.length, entered, rule: mainRule,
    regraEntrada: useModel ? "previsão do modelo" : useBars ? "3 barras ou mais, em até 3 dias do sinal" : "sinal do último candle fechado (o filtro de barras ainda não foi comprovado)",
    signals: sigs.map(s => ({ sym: s.sym, base: s.base, vr: s.vr, stopPct: s.stopPct, ask: s.ask, bars: s.bars, age: s.age, x: s.x && s.x.map(r4), c: s.c, a: s.a && r4(s.a) })),
    modelSignals: modelSignals.map(s => ({ sym: s.sym, base: s.base, p: s.p.map(r4), stopPct: s.stopPct })),
    modelTop: top,
    model: M ? { validated: M.validated, demoted: M.demoted, thr: M.thr, live: M.live, trainedAt: M.trainedAt, learned } : null,
    shadowOpen: SH.trades.filter(t => t.status === "aberta").length, shadowClosed: SH.trades.filter(t => t.status === "fechada").length,
    params: C
  });
  console.log(`Analisadas ${cands.length} moedas. Regra: ${mainRule}. Momentum: ${momSignals.length} sinais, janela de 3 dias: ${sigs.length}, bons (3+ barras): ${goodSignals.length}. Modelo: ${modelSignals.length} sinais. Aprendeu com ${learned} resultados. Entrada: ${entered || "nenhuma"}.`);
  await tg(msgs.join("\n"));
})().catch(e => { console.error(e); process.exit(1); });
