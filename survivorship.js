// Teste de viés de sobrevivência do momentum diário (rompimento de 20 dias + volume 2x, meta 10%, prazo 7 dias).
// Universo em cada dia = as N moedas de maior volume médio dos 30 dias anteriores, entre as que existiam na época.
// A: todas as moedas do arquivo público (inclusive as que saíram da Binance). B: só as que negociam hoje.
const fs = require("fs"), os = require("os"), path = require("path");
const { execFileSync } = require("child_process");
const S3 = process.env.S3_BASE || "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision";
const BASE = process.env.API_BASE || "https://data-api.binance.vision";
const P = { tgt: 10, days: 7, vm: 2, L: 20, k: 2, maxl: 15, fee: 0.075, slip: 0.1, topN: 25, volWin: 30 };
const DAY = 864e5;
const SKIP = new Set(["USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "EUR", "TRY", "BRL", "AEUR", "USD1", "XUSD", "PAXG", "USDS", "UST", "USTC"]);

async function fetchRetry(url, asBuffer) {
  let last;
  for (let i = 0; i < 3; i++) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error("HTTP " + r.status + " " + url);
      return asBuffer ? Buffer.from(await r.arrayBuffer()) : await r.text();
    } catch (e) { last = e; await new Promise(r => setTimeout(r, 800)); }
  }
  throw last;
}
async function list(prefix, delim) {
  let marker = "", prefixes = [], keys = [];
  for (;;) {
    const u = `${S3}?prefix=${encodeURIComponent(prefix)}${delim ? "&delimiter=/" : ""}${marker ? "&marker=" + encodeURIComponent(marker) : ""}`;
    const x = await fetchRetry(u);
    const ks = [...x.matchAll(/<Key>([^<]+)<\/Key>/g)].map(m => m[1]);
    const ps = [...x.matchAll(/<CommonPrefixes><Prefix>([^<]+)<\/Prefix><\/CommonPrefixes>/g)].map(m => m[1]);
    keys.push(...ks); prefixes.push(...ps);
    if (!/<IsTruncated>true<\/IsTruncated>/.test(x)) break;
    const nm = /<NextMarker>([^<]+)<\/NextMarker>/.exec(x);
    marker = nm ? nm[1] : (ks.length ? ks[ks.length - 1] : ps[ps.length - 1]);
  }
  return { prefixes, keys };
}
async function pool(items, n, fn) {
  let i = 0;
  await Promise.all(Array.from({ length: n }, async () => { while (i < items.length) { const k = i++; await fn(items[k], k); } }));
}
function unzipCsv(buf) {
  const tmp = path.join(os.tmpdir(), "z" + process.pid + "_" + Math.random().toString(36).slice(2) + ".zip");
  fs.writeFileSync(tmp, buf);
  try { return execFileSync("unzip", ["-p", tmp], { maxBuffer: 64 * 1024 * 1024 }).toString(); }
  finally { fs.unlinkSync(tmp); }
}
function parseCsv(txt) {
  const out = [];
  for (const line of txt.split("\n")) {
    const f = line.split(",");
    let t = +f[0];
    if (!(t > 0)) continue;
    if (t > 1e14) t = Math.floor(t / 1000); // microssegundos nos arquivos mais novos
    out.push({ t, o: +f[1], h: +f[2], l: +f[3], c: +f[4], v: +f[5], q: +f[7] });
  }
  return out;
}
function atrSeries(cs, n = 14) {
  const a = [];
  for (let i = 0; i < cs.length; i++) {
    if (i < n) { a.push(null); continue; }
    let s = 0;
    for (let j = i - n + 1; j <= i; j++) {
      const p = cs[j - 1].c;
      s += Math.max(cs[j].h - cs[j].l, Math.abs(cs[j].h - p), Math.abs(cs[j].l - p));
    }
    a.push(s / n / cs[i].c * 100);
  }
  return a;
}
function features(cs) {
  return cs.map((c, i) => {
    if (i < 31) return null;
    let mh = 0, sv = 0, q = 0;
    for (let j = i - P.L; j < i; j++) { if (cs[j].h > mh) mh = cs[j].h; sv += cs[j].v; }
    for (let j = i - P.volWin; j < i; j++) q += cs[j].q;
    return { brk: c.c > mh, vr: sv > 0 ? c.v / (sv / P.L) : 0, qv: q / P.volWin };
  });
}
// topSets: Map símbolo -> Set de dias (t) em que a moeda está entre as N de maior volume
function buildTop(series, syms) {
  const byDay = new Map();
  for (const s of syms) {
    const { fe, cs } = series[s];
    cs.forEach((c, i) => { if (fe[i]) { if (!byDay.has(c.t)) byDay.set(c.t, []); byDay.get(c.t).push([s, fe[i].qv]); } });
  }
  const top = new Map(syms.map(s => [s, new Set()]));
  for (const [t, arr] of byDay) {
    arr.sort((a, b) => b[1] - a[1]);
    arr.slice(0, P.topN).forEach(([s]) => top.get(s).add(t));
  }
  return top;
}
function sim(s, topSet, mode, t0, span) {
  const { cs, atr, fe } = s, out = [];
  let i = 31;
  while (i < cs.length - P.days - 2) {
    if (atr[i] && fe[i] && topSet.has(cs[i].t) && (mode === "ctl" || (fe[i].brk && fe[i].vr >= P.vm))) {
      const sp = Math.min(Math.max(P.k * atr[i], 4), P.maxl);
      const entry = cs[i + 1].o * (1 + P.slip / 100), stopP = entry * (1 - sp / 100), tgtP = entry * (1 + P.tgt / 100);
      let exit = null, k = i + 1;
      for (; k <= i + P.days; k++) {
        if (cs[k].l <= stopP) { exit = stopP * (1 - P.slip / 100); break; }
        if (cs[k].h >= tgtP) { exit = tgtP; break; }
      }
      if (exit === null) { k = i + P.days; exit = cs[k].c * (1 - P.slip / 100); }
      out.push({ r: (exit / entry - 1) * 100 - 2 * P.fee, pos: (cs[i].t - t0) / span });
      i = k + 1; continue;
    }
    i++;
  }
  return out;
}
const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : null;
function stats(a) {
  const n = a.length;
  if (n < 2) return { n };
  const r = a.map(x => x.r), m = mean(r), sd = Math.sqrt(r.reduce((s, x) => s + (x - m) ** 2, 0) / (n - 1));
  return {
    n, mean: m, margin: 2 * sd / Math.sqrt(n), win: r.filter(x => x > 0).length / n * 100,
    half1: mean(a.filter(x => x.pos < 0.5).map(x => x.r)), half2: mean(a.filter(x => x.pos >= 0.5).map(x => x.r))
  };
}

(async () => {
  console.log("Listando símbolos...");
  const root = "data/spot/monthly/klines/";
  const { prefixes } = await list(root, true);
  const syms = prefixes.map(p => p.slice(root.length, -1)).filter(s => {
    if (!s.endsWith("USDT")) return false;
    const b = s.slice(0, -4);
    return !SKIP.has(b) && !/(UP|DOWN|BULL|BEAR)$/.test(b);
  });
  console.log(syms.length + " pares USDT no arquivo público.");

  const info = JSON.parse(await fetchRetry(BASE + "/api/v3/exchangeInfo"));
  const trading = new Set(info.symbols.filter(x => x.status === "TRADING").map(x => x.symbol));

  const series = {};
  let done = 0;
  await pool(syms, 12, async (sym) => {
    try {
      const pre = `${root}${sym}/1d/`;
      const { keys } = await list(pre, false);
      const zips = keys.filter(k => k.endsWith(".zip"));
      let rows = [];
      await pool(zips, 8, async (k) => { rows.push(...parseCsv(unzipCsv(await fetchRetry(S3 + "/" + k, true)))); });
      rows.sort((a, b) => a.t - b.t);
      rows = rows.filter((r, i) => i === 0 || r.t !== rows[i - 1].t);
      if (rows.length >= 120) series[sym] = { cs: rows, atr: atrSeries(rows), fe: features(rows) };
    } catch (e) { console.log("Falhou " + sym + ": " + e.message); }
    if (++done % 25 === 0) console.log(`${done} de ${syms.length}`);
  });

  const all = Object.keys(series), cur = all.filter(s => trading.has(s));
  let t0 = Infinity, tEnd = 0;
  for (const s of all) { t0 = Math.min(t0, series[s].cs[0].t); tEnd = Math.max(tEnd, series[s].cs[series[s].cs.length - 1].t); }
  const span = tEnd - t0 || 1;

  const result = { generated: Date.now(), params: P, symbols: { all: all.length, current: cur.length, delisted: all.length - cur.length }, groups: {} };
  for (const [name, set] of [["A_todas_inclusive_as_que_sairam", all], ["B_so_as_que_negociam_hoje", cur]]) {
    const top = buildTop(series, set), ctl = [], mom = [];
    let delistedTrades = 0;
    for (const s of set) {
      ctl.push(...sim(series[s], top.get(s), "ctl", t0, span));
      const m = sim(series[s], top.get(s), "mom", t0, span);
      mom.push(...m);
      if (!trading.has(s)) delistedTrades += m.length;
    }
    result.groups[name] = { controle: stats(ctl), momentum: stats(mom), operacoes_em_moedas_que_sairam: delistedTrades };
  }
  fs.mkdirSync("data", { recursive: true });
  fs.writeFileSync("data/survivorship.json", JSON.stringify(result, null, 1));
  console.log(JSON.stringify(result, null, 1));
})().catch(e => { console.error(e); process.exit(1); });
