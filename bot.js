// Robô do Radar Binance: roda sozinho no GitHub Actions, sem dinheiro de verdade.
// Regra: rompimento da máxima de 20 dias com volume forte (candles diários), meta 10%, prazo 7 dias.
const fs = require("fs");
const BASE = "https://data-api.binance.vision";
const C = { cap: 50, tgt: 10, days: 7, vm: 2, k: 2, maxl: 15, minVol: 5e6, fee: 0.075, slip: 0.1, top: 60, L: 20 };
const DAY = 864e5;
const SKIP = new Set(["USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "EUR", "TRY", "BRL", "AEUR", "USD1", "XUSD", "PAXG"]);
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
function evalSignal(cs) {
  const i = cs.length - 2;
  if (i < 35) return null;
  let mh = 0, sv = 0;
  for (let j = i - C.L; j < i; j++) { if (cs[j].h > mh) mh = cs[j].h; sv += cs[j].v; }
  let s = 0;
  for (let j = i - 13; j <= i; j++) {
    const pc = cs[j - 1].c;
    s += Math.max(cs[j].h - cs[j].l, Math.abs(cs[j].h - pc), Math.abs(cs[j].l - pc));
  }
  const atr = s / 14 / cs[i].c * 100, vr = sv > 0 ? cs[i].v / (sv / C.L) : 0;
  return { sig: cs[i].c > mh && vr >= C.vm, vr, atr, t: cs[i].t };
}
async function daily(sym) {
  const k = await get(`/api/v3/klines?symbol=${sym}&interval=1d&limit=60`);
  return k.map(c => ({ t: +c[0], o: +c[1], h: +c[2], l: +c[3], c: +c[4], v: +c[5] }));
}

function newTrade(s, now, source) {
  const entry = s.ask * (1 + C.slip / 100);
  return {
    id: now + Math.random(), sym: s.sym, base: s.base, time: now, exp: now + C.days * DAY, entry, cap: C.cap, fee: C.fee, slip: C.slip,
    vr: s.vr, stopPct: s.stopPct, tgt: C.tgt, stopP: entry * (1 - s.stopPct / 100), tgtP: entry * (1 + C.tgt / 100),
    status: "aberta", cur: entry, source, sigT: s.sigT
  };
}

(async () => {
  fs.mkdirSync("data", { recursive: true });
  let S = { trades: [] };
  try { S = JSON.parse(fs.readFileSync("data/trades.json", "utf8")); } catch (e) {}
  const msgs = [];

  for (const t of S.trades.filter(t => t.status === "aberta")) {
    await resolve(t);
    if (t.status === "fechada") msgs.push(`Operação simulada fechada: ${t.base}/USDT ${whyTxt[t.why]}, resultado ${money(t.net)}.`);
  }

  // Operações-sombra: guardam o resultado de TODOS os sinais, separadas da operação principal
  let SH = { trades: [] };
  try { SH = JSON.parse(fs.readFileSync("data/shadow.json", "utf8")); } catch (e) {}
  for (const t of SH.trades.filter(t => t.status === "aberta")) {
    try { await resolve(t); } catch (e) { console.log("Sombra falhou em " + t.sym + ": " + e.message); }
  }

  const ticker = await get("/api/v3/ticker/24hr");
  const cands = ticker.filter(x => {
    if (!x.symbol.endsWith("USDT")) return false;
    const b = x.symbol.slice(0, -4);
    return !SKIP.has(b) && !/(UP|DOWN|BULL|BEAR)$/.test(b) && +x.quoteVolume >= C.minVol && +x.askPrice > 0 && +x.bidPrice > 0;
  }).sort((a, b) => b.quoteVolume - a.quoteVolume).slice(0, C.top);

  const signals = [];
  for (let i = 0; i < cands.length; i += 10) {
    const batch = await Promise.all(cands.slice(i, i + 10).map(x => daily(x.symbol).catch(() => null)));
    batch.forEach((cs, j) => {
      const x = cands[i + j], e = cs ? evalSignal(cs) : null;
      if (e && e.sig) signals.push({ sym: x.symbol, base: x.symbol.slice(0, -4), ask: +x.askPrice, vr: e.vr, atr: e.atr, sigT: e.t, stopPct: Math.min(Math.max(C.k * e.atr, 4), C.maxl) });
    });
  }
  signals.sort((a, b) => b.vr - a.vr);

  let entered = null;
  const now0 = Date.now();
  for (const sg of signals) {
    if (!SH.trades.some(t => t.sym === sg.sym && t.sigT === sg.sigT)) SH.trades.push(newTrade(sg, now0, "sombra"));
  }
  const fresh = signals.filter(x => !S.trades.some(t => t.sym === x.sym && t.sigT === x.sigT));
  if (!S.trades.some(t => t.status === "aberta") && fresh.length) {
    const sg = fresh[0];
    S.trades.push(newTrade(sg, Date.now(), "nuvem"));
    entered = sg.sym;
    msgs.push(`Nova operação simulada: ${sg.base}/USDT, volume ${sg.vr.toFixed(1)}x a média. Stop em ${money(-sg.stopPct)}, meta em ${money(C.tgt)}.`);
  } else if (signals.length) {
    msgs.push(`Sinais hoje: ${signals.map(x => x.base).join(", ")}. Já há uma operação aberta.`);
  }

  fs.writeFileSync("data/trades.json", JSON.stringify(S));
  fs.writeFileSync("data/shadow.json", JSON.stringify(SH));
  fs.writeFileSync("data/last-run.json", JSON.stringify({
    time: Date.now(), analyzed: cands.length,
    signals: signals.map(s => ({ sym: s.sym, base: s.base, vr: s.vr, stopPct: s.stopPct, ask: s.ask })),
    entered, shadowOpen: SH.trades.filter(t => t.status === "aberta").length, shadowClosed: SH.trades.filter(t => t.status === "fechada").length, params: C
  }));
  console.log(`Analisadas ${cands.length} moedas, ${signals.length} com sinal. Entrada: ${entered || "nenhuma"}.`);
  await tg(msgs.join("\n"));
})().catch(e => { console.error(e); process.exit(1); });
