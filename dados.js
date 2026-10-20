// dados.js: baixa o histórico diário público da Binance (data.binance.vision), inclusive de moedas que já saíram.
const fs = require("fs"), os = require("os"), path = require("path");
const { execFileSync } = require("child_process");
const S3 = process.env.S3_BASE || "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision";
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
async function loadAll(log = console.log) {
  const root = "data/spot/monthly/klines/";
  const { prefixes } = await list(root, true);
  const syms = prefixes.map(p => p.slice(root.length, -1)).filter(s => {
    if (!s.endsWith("USDT")) return false;
    const b = s.slice(0, -4);
    return !SKIP.has(b) && !/(UP|DOWN|BULL|BEAR)$/.test(b);
  });
  log(syms.length + " pares USDT no arquivo público.");
  const series = {};
  let done = 0;
  await pool(syms, 12, async (sym) => {
    try {
      const { keys } = await list(`${root}${sym}/1d/`, false);
      let rows = [];
      await pool(keys.filter(k => k.endsWith(".zip")), 8, async (k) => { rows.push(...parseCsv(unzipCsv(await fetchRetry(S3 + "/" + k, true)))); });
      rows.sort((a, b) => a.t - b.t);
      rows = rows.filter((r, i) => i === 0 || r.t !== rows[i - 1].t);
      if (rows.length >= 120) series[sym] = rows;
    } catch (e) { log("Falhou " + sym + ": " + e.message); }
    if (++done % 25 === 0) log(`${done} de ${syms.length}`);
  });
  return series;
}
module.exports = { loadAll };
