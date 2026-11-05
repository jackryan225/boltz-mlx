// Result components shared by the structure-job page and the screen's ligand panel: the Mol* structure panel (pLDDT colours
// with a legend), the confidence numbers, the affinity readout, the PAE heatmap and the per-chain table.
import { downloadFrom } from "./api.js";
import { download, fmt, h, icon, replace, toast } from "./dom.js";
import { announce, errorRetry } from "./ui.js";
import { PoseViewer } from "./viewer.js";

export const PLDDT_BANDS = [["#0053d6", "> 90", "very high"], ["#65cbf3", "70–90", "confident"], ["#ffdb13", "50–70", "low"], ["#ff7d45", "< 50", "very low"]];
export const LIGAND_COLOR = "#e5006e";

export function plddtColor(v) {
  if (v === null || v === undefined) return "var(--line)";
  const x = v <= 1.0001 ? v * 100 : v;
  return x > 90 ? PLDDT_BANDS[0][0] : x > 70 ? PLDDT_BANDS[1][0] : x > 50 ? PLDDT_BANDS[2][0] : PLDDT_BANDS[3][0];
}

export function plddtLegend(withLigand = true) {
  return h("div", { class: "legend" }, h("span", { class: "lab" }, "pLDDT"),
    PLDDT_BANDS.map(([c, t]) => h("span", null, h("i", { style: { background: c } }), t)),
    withLigand ? h("span", null, h("i", { style: { background: LIGAND_COLOR } }), "ligand") : null);
}

// Camera: the whole complex, or the ligand with its pocket (padded so the contacts stay in view). A two-way selector that shows the
// framing in force: a new structure is framed on the pocket while Contacts is on (and a ligand exists), otherwise on the complex;
// seg.set(mode) follows that.
export function cameraButtons(getViewer, hasLigand, initial = hasLigand ? "pocket" : "complex") {
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Camera" });
  const btn = (mode, text, title) => h("button", { type: "button", title, "data-cam": mode, "aria-pressed": String(mode === initial),
    onclick: () => {
      getViewer()?.view(mode).then(() => { for (const b of seg.querySelectorAll("[data-cam]")) b.setAttribute("aria-pressed", String(b.dataset.cam === mode)); })
        .catch((e) => toast(e.message, "bad"));
    } }, text);
  seg.append(btn("complex", "Complex", "Frame the whole complex"));
  if (hasLigand) seg.append(btn("pocket", "Pocket", "Frame the ligand and its pocket"));
  seg.set = (mode) => { for (const b of seg.querySelectorAll("[data-cam]")) b.setAttribute("aria-pressed", String(b.dataset.cam === mode)); };
  return seg;
}

// ------------------------------------------------------------------------------------------------ structure panel
// Built once per page: the Mol* frame inside it must never be removed from the page (that reloads it). display() takes a pose that is
// already fetched; calls are queued and a newer call supersedes an older one, and the download name / text change only when the pose
// is drawn, so what the viewer shows and what the download saves are always the same model. A failure stays up with Retry.
export class MolPanel {
  constructor({ hasLigand = false, binder = null, ligandChains = [], dark = true }) {
    this.hasLigand = hasLigand;
    this.binder = binder || ligandChains[0] || null;
    this.dark = dark;
    this.viewer = null;
    this.contacts = hasLigand;
    this.initP = null;
    this.chain = Promise.resolve();
    this.gen = 0;
    this.text = null;
    this.fileName = "structure.cif";
    this.host = h("div", { class: "mol-host" });
    this.msg = h("div", { class: "msg" });
    this.box = h("div", { class: "mol-box", "data-state": "loading" }, this.host, this.msg);
    const mode = (v, t) => h("button", { type: "button", "data-mode": v, "aria-pressed": String(v === "plddt"), onclick: () => this.setColor(v) }, t);
    this.modeSeg = h("div", { class: "seg", role: "group", "aria-label": "Colour by" }, mode("plddt", "pLDDT"), mode("chain", "Chain"));
    this.legend = h("details", { class: "legend-d" }, h("summary", null, "pLDDT legend"), plddtLegend(hasLigand));
    this.el = h("div", null, this.box,
      h("div", { class: "mol-ctl" }, this.modeSeg,
        hasLigand ? h("label", { class: "check small", title: "Residues within 5 Å of the ligand as sticks" }, h("input", { type: "checkbox", checked: true, onchange: (e) => { this.contacts = e.target.checked; this.viewer?.setPocket(e.target.checked).then(() => this.camera?.set(this.contacts ? "pocket" : "complex")).catch((err) => toast(err.message, "bad")); } }), "Contacts") : null,
        ligandChains.length > 1 ? h("label", { class: "row tight small" }, "Ligand", h("select", { "aria-label": "Ligand to focus",
          onchange: (e) => { this.binder = e.target.value; this.viewer?.setBinder(e.target.value).catch((err) => toast(err.message, "bad")); } },
          ligandChains.map((c) => h("option", { value: c, selected: c === this.binder }, c)))) : null,
        h("span", { class: "grow" }),
        this.camera = cameraButtons(() => this.viewer, hasLigand)),
      this.legend);
    this.loading();
  }

  loading() { this.box.dataset.state = "loading"; this.msg.hidden = false; replace(this.msg, h("span", { class: "spinner" }), " Loading"); }

  ensure() {
    if (this.viewer) return Promise.resolve(this.viewer);
    if (!this.initP) {
      this.initP = (async () => {
        const v = new PoseViewer(this.host);
        v.dark = this.dark;
        v.pocket = this.hasLigand;
        v.binder = this.binder;
        await v.init();
        this.viewer = v;
        return v;
      })().catch((e) => { this.initP = null; throw e; });
    }
    return this.initP;
  }

  // Draws {text, format} and then (only then, and only if nothing newer was asked for) commits {fileName}. Resolves true when it is on screen.
  display({ text, format, fileName }) {
    const my = ++this.gen;
    this.loading();
    const run = this.chain.then(async () => {
      if (my !== this.gen) return false;
      try {
        const v = await this.ensure();
        if (my !== this.gen) return false;
        await v.show(text, format || "mmcif");
        if (my !== this.gen) return false;
        await v.resize();
        this.text = text; this.fileName = fileName;
        this.camera?.set(this.hasLigand && this.contacts ? "pocket" : "complex");
        this.msg.hidden = true;
        this.box.dataset.state = "ready";
        return true;
      } catch (e) {
        console.error(e);
        if (my === this.gen) {
          this.box.dataset.state = "error";
          this.msg.hidden = false;
          replace(this.msg, h("div", { class: "stack tight center" }, h("strong", null, "3D view unavailable"), h("div", { class: "small" }, e.message || String(e)),
            h("button", { type: "button", class: "btn sm", onclick: () => { this.reset(); this.display({ text, format, fileName }); } }, icon("refresh"), "Retry")));
        }
        return false;
      }
    });
    this.chain = run.catch(() => {});
    return run;
  }

  // The 3D view could not be shown: the message and Retry stay inside this panel.
  fail(message, retry) {
    this.gen++;
    this.box.dataset.state = "error";
    this.msg.hidden = false;
    replace(this.msg, h("div", { class: "stack tight center" }, h("strong", null, "3D view unavailable"), h("div", { class: "small" }, message),
      h("button", { type: "button", class: "btn sm", onclick: retry }, icon("refresh"), "Retry")));
  }

  reset() { this.viewer?.dispose(); this.viewer = null; this.initP = null; }

  async setColor(v) {
    for (const b of this.modeSeg.querySelectorAll("[data-mode]")) b.setAttribute("aria-pressed", String(b.dataset.mode === v));
    this.legend.hidden = v !== "plddt";
    try { await this.viewer?.setColor(v); } catch (e) { toast(e.message, "bad"); }
  }

  setDark(d) { this.dark = d; this.viewer?.setDark(d); }
  resize() { this.viewer?.resize(); }
  dispose() { this.gen++; this.reset(); }
}

// ------------------------------------------------------------------------------------------------ confidence
export function bandWord(v) {
  if (v === null || v === undefined) return "";
  const x = v <= 1.0001 ? v * 100 : v;
  return x > 90 ? "very high" : x > 70 ? "confident" : x > 50 ? "low" : "very low";
}

// Which interface metrics exist for a complex: ipTM needs two or more chains, ligand ipTM a ligand chain. One rule for every view.
export function metricScope(chains) {
  const list = chains || [];
  return { multi: list.length > 1, ligand: list.some((c) => c.type === "ligand") };
}

// What the overview strip does not show. Interface metrics need an interface: protein ipTM two protein chains, ipLDDT / iPDE two
// chains; where they do not apply they read "–". pLDDT-type values are 0-100, PDE-type values are Å.
export function confidenceCard(conf, chains) {
  conf = conf || {};
  const list = chains || [];
  const multi = list.length > 1 || !list.length;
  const rows = [
    ["Protein ipTM", conf.protein_iptm, list.filter((c) => c.type === "protein").length > 1 || !list.length, (v) => fmt.num(v, 2)],
    ["Complex ipLDDT", conf.complex_iplddt, multi, (v) => fmt.plddt(v)],
    ["Complex PDE (Å)", conf.complex_pde, true, (v) => fmt.num(v, 2)],
    ["Complex iPDE (Å)", conf.complex_ipde, multi, (v) => fmt.num(v, 2)],
  ].filter(([, v]) => v !== null && v !== undefined);
  return h("div", { class: "card" }, h("div", { class: "card-h" }, h("h2", null, "Confidence")),
    h("div", { class: "card-b" }, rows.length ? h("dl", { class: "kv" }, rows.flatMap(([k, v, applies, f]) => [h("dt", null, k), h("dd", { class: "num" }, applies ? f(v) : "–")]))
      : h("div", { class: "small muted" }, "No further metrics")));
}

// ------------------------------------------------------------------------------------------------ affinity
// affinity_pred_value is log10(IC50 / uM) (Boltz-2); the kcal/mol figure uses the Boltz-2 README conversion (6 - y) * 1.364.
export function ic50Text(y) {
  if (y === null || y === undefined || !Number.isFinite(y)) return ["–", ""];
  const um = Math.pow(10, y);
  const sig = (x) => { const v = Number(x.toPrecision(2)); return v >= 100 ? Math.round(v).toLocaleString("en-US") : String(v); };
  if (um < 1) return [sig(um * 1000), "nM"];
  if (um >= 1000) return [sig(um / 1000), "mM"];
  return [sig(um), "µM"];
}

// The affinity numbers the page does not already show elsewhere (skip: "ic50", "p", "y"), the verdict and the scope. pIC50, the
// ensemble members, the MW correction and upstream's kcal/mol conversion are listed with it. Everything is a model estimate.
export function affinityCard(aff, { mwCorrected, skip = [], scope = "" } = {}) {
  if (!aff) return null;
  const y = aff.affinity_pred_value, p = aff.affinity_probability_binary;
  const [v, unit] = ic50Text(y);
  const hasP = typeof p === "number" && Number.isFinite(p);   // a missing output is not evidence against binding
  const binder = hasP && p >= 0.5;
  const none = y === null || y === undefined || !Number.isFinite(y);
  const ens = [aff.affinity_pred_value1, aff.affinity_pred_value2].filter((x) => x !== undefined && x !== null);
  const rows = [
    ["ic50", "IC50 (est.)", none ? "–" : `${v} ${unit}`],
    ["p", "P(binder)", hasP ? fmt.p(p) : "unavailable"],
    ["y", "Affinity (log10 IC50, µM)", fmt.num(y, 2)],
    ens.length ? [null, "Ensemble (log10 µM)", ens.map((x) => fmt.num(x, 2)).join(" / ")] : null,
    mwCorrected !== undefined && mwCorrected !== null ? [null, "MW-corrected (log10 µM)", fmt.num(mwCorrected, 2)] : null,
  ].filter((r) => r && !skip.includes(r[0]));
  const derived = [["pIC50 = 6 − affinity", none ? "–" : (6 - y).toFixed(2)], ["Energy (6 − affinity) × 1.364 kcal/mol, Boltz-2 README convention", none ? "–" : ((6 - y) * 1.364).toFixed(2)]];
  return h("div", { class: "card" },
    h("div", { class: "card-h" }, h("h2", null, "Affinity")),
    h("div", { class: "card-b aff" },
      h("div", { class: "aff-top" },
        hasP ? h("span", { class: binder ? "verdict" : "verdict weak" }, icon(binder ? "check" : "info"), binder ? "Likely binder" : "Unlikely binder") : h("span", { class: "verdict weak" }, icon("info"), "P(binder) unavailable"),
        h("span", { class: "muted small" }, ["Model estimate", scope].filter(Boolean).join(" · "))),
      rows.length ? h("dl", { class: "kv" }, rows.flatMap(([, k, x]) => [h("dt", null, k), h("dd", { class: "num" }, x)])) : null,
      h("details", { class: "adv" }, h("summary", null, "Derived"),
        h("dl", { class: "kv", style: { marginTop: "6px" } }, derived.flatMap(([k, x]) => [h("dt", null, k), h("dd", { class: "num" }, x)])))));
}

// ------------------------------------------------------------------------------------------------ PAE heatmap
// One colour scale in both themes: 0 Å dark blue to 30 Å near white (the map has its own frame).
const PAE_STOPS = [[0, [23, 37, 84]], [0.17, [37, 99, 235]], [0.4, [110, 155, 240]], [0.7, [203, 217, 246]], [1, [250, 250, 250]]];

function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i];
      const u = (t - a) / (b - a || 1);
      return ca.map((x, k) => Math.round(x + (cb[k] - x) * u));
    }
  }
  return stops[stops.length - 1][1];
}

function niceTicks(n) {
  const step = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000].find((s) => n / s <= 6) || 5000;
  const out = [];
  for (let t = step; t < n; t += step) out.push(t);
  return [1, ...out];
}

export function paeCard(load, { name = "pae", rawUrl = null } = {}) {
  const body = h("div", { class: "card-b" }, h("span", { class: "spinner" }), " Loading");
  const expand = h("button", { type: "button", class: "btn ghost sm", "aria-pressed": "false", title: "Larger matrix", onclick: () => {
    const big = card.classList.toggle("pae-big");
    expand.setAttribute("aria-pressed", String(big));
    replace(expand, big ? "Smaller" : "Expand");
  } }, "Expand");
  const card = h("div", { class: "card" }, h("div", { class: "card-h" }, h("h2", null, "PAE"), h("span", { class: "chip" }, "Å"), h("span", { class: "grow" }), expand), body);
  let data = null;
  function draw() {
    if (!data) return;
    const n = data.side;
    const bytes = Uint8Array.from(atob(data.data), (c) => c.charCodeAt(0));
    const MISS = data.missing ?? 255;  // a cell the matrix has no value for: hatched, never drawn as 0 Å
    const miss = (r, c) => bytes[r * n + c] === MISS;
    const val = (r, c) => (miss(r, c) ? NaN : bytes[r * n + c] / data.step);
    const cv = h("canvas", { width: n, height: n });
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(n, n);
    for (let i = 0; i < n * n; i++) {
      const [r, g, b] = bytes[i] === MISS ? ((i % n + Math.floor(i / n)) % 2 ? [110, 110, 110] : [170, 170, 170]) : ramp(PAE_STOPS, bytes[i] / data.step / 30);
      img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const box = h("div", { class: "pae-box" }, cv);
    const SVG = "http://www.w3.org/2000/svg";
    if (data.chains.length > 1) {  // chain boundaries
      const svg = document.createElementNS(SVG, "svg");
      svg.setAttribute("viewBox", `0 0 ${data.n} ${data.n}`);
      svg.setAttribute("preserveAspectRatio", "none");
      Object.assign(svg.style, { position: "absolute", inset: "0", width: "100%", height: "100%", pointerEvents: "none" });
      for (const c of data.chains.slice(1)) {
        for (const [x1, y1, x2, y2] of [[c.start, 0, c.start, data.n], [0, c.start, data.n, c.start]]) {
          const l = document.createElementNS(SVG, "line");
          for (const [k, v] of Object.entries({ x1, y1, x2, y2 })) l.setAttribute(k, v);
          l.setAttribute("stroke", "#fff"); l.setAttribute("stroke-width", String(Math.max(1, data.n / 260))); l.setAttribute("stroke-dasharray", `${data.n / 80} ${data.n / 120}`); l.setAttribute("opacity", ".85");
          svg.appendChild(l);
        }
      }
      box.appendChild(svg);
    }
    const cursor = h("i", { class: "pae-cursor", hidden: true });
    box.appendChild(cursor);
    const chainOf = (t) => data.chains.find((c) => t >= c.start && t < c.end);
    const read = h("div", { class: "pae-read small num", role: "status", "aria-live": "polite" }, "\u00a0");
    let cur = null;
    function inspect(fx, fy) {
      const ci = Math.min(n - 1, Math.max(0, Math.floor(fx * n))), ri = Math.min(n - 1, Math.max(0, Math.floor(fy * n)));
      const ti = Math.min(data.n - 1, Math.floor(fy * data.n)), tj = Math.min(data.n - 1, Math.floor(fx * data.n));
      const a = chainOf(ti), b = chainOf(tj);
      cur = { fx, fy };
      cursor.hidden = false;
      cursor.style.left = fx * 100 + "%"; cursor.style.top = fy * 100 + "%";
      read.textContent = `Aligned on ${a ? a.id + " " : ""}${ti - (a?.start || 0) + 1}, scored ${b ? b.id + " " : ""}${tj - (b?.start || 0) + 1}: ${miss(ri, ci) ? "no value" : val(ri, ci).toFixed(1) + " Å"}`;
    }
    const at = (e) => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
    cv.addEventListener("mousemove", (e) => inspect(...at(e)));
    cv.addEventListener("click", (e) => inspect(...at(e)));
    cv.tabIndex = 0;
    cv.setAttribute("role", "application");
    cv.setAttribute("aria-label", `PAE matrix, ${data.n} tokens. Arrow keys move the cursor, Shift moves 10.`);
    cv.addEventListener("keydown", (e) => {
      const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
      if (!d) return;
      e.preventDefault();
      const k = (e.shiftKey ? 10 : 1) / data.n;
      const [x, y] = cur ? [cur.fx, cur.fy] : [0.5 / data.n, 0.5 / data.n];
      inspect(Math.min(1 - 0.5 / data.n, Math.max(0.5 / data.n, x + d[0] * k)), Math.min(1 - 0.5 / data.n, Math.max(0.5 / data.n, y + d[1] * k)));
    });
    // axes: sparse token ticks and the chain ids
    const xs = h("div", { class: "pae-ax x", "aria-hidden": "true" }), ys = h("div", { class: "pae-ax y", "aria-hidden": "true" });
    // a token tick that would sit on a chain id ("50" under "A") is left out: the id says more there
    const mids = data.chains.length > 1 ? data.chains.map((c) => ({ id: c.id, at: ((c.start + c.end) / 2 / data.n) * 100 })) : [];
    for (const t of niceTicks(data.n)) {
      const at = ((t - 0.5) / data.n) * 100;
      if (mids.some((m) => Math.abs(m.at - at) < 6)) continue;
      xs.appendChild(h("span", { style: { left: at + "%" } }, t));
      ys.appendChild(h("span", { style: { top: at + "%" } }, t));
    }
    for (const m of mids) { xs.appendChild(h("b", { style: { left: m.at + "%" } }, m.id)); ys.appendChild(h("b", { style: { top: m.at + "%" } }, m.id)); }
    const grad = `linear-gradient(90deg, ${[0, 0.17, 0.4, 0.7, 1].map((t) => `rgb(${ramp(PAE_STOPS, t).join(",")}) ${t * 100}%`).join(", ")})`;
    const pairs = data.pairs || [];
    const csv = () => {
      const rows = [];
      for (let r = 0; r < n; r++) rows.push(Array.from({ length: n }, (_, c) => (miss(r, c) ? "" : val(r, c).toFixed(3))).join(","));
      const head = `# display matrix: ${data.factor > 1 ? `block-averaged ${data.factor}x (${data.n} tokens shown as ${n})` : "not binned"}, quantised to ${1 / data.step} A; blank = no value (the NPZ has the original)\n`;
      download(`${name}_display${data.factor > 1 ? `_binned${data.factor}` : ""}.csv`, head + rows.join("\n") + "\n");
    };
    replace(body, h("div", { class: "pae-wrap" },
      h("div", { class: "pae-frame" }, ys, box, xs),
      read,
      h("div", { class: "pae-scale" }, h("span", null, "0 Å"), h("i", { style: { background: grad } }), h("span", null, "30 Å")),
      h("div", { class: "row tight" },
        h("div", { class: "chips grow" }, pairs.map((p) => h("span", { class: "chip info", title: `mean PAE of chain ${p.b} aligned on chain ${p.a}` }, `${p.a}→${p.b} ${p.mean} Å`))),
        h("button", { type: "button", class: "btn sm", onclick: csv, title: `Display matrix, ${n} × ${n}${data.factor > 1 ? `, binned ${data.factor}×` : ""}, quantised to ${1 / data.step} Å` }, icon("download"), "Display CSV"),
        rawUrl ? h("button", { type: "button", class: "btn sm", title: "Original full-precision matrix (.npz)", onclick: () => downloadFrom(rawUrl, `${name}.npz`).catch((e) => toast(e.message, "bad")) }, icon("download"), "NPZ") : null),
      h("p", { class: "small faint" }, "Row: aligned on; column: scored."
        + (data.factor > 1 ? ` Binned ${data.factor}× (${data.n} tokens).` : "") + (data.n_missing ? ` ${fmt.int(data.n_missing)} cells have no value (hatched).` : ""))));
  }
  const start = () => {
    replace(body, h("span", { class: "spinner" }), " Loading");
    load().then((d) => { data = d; draw(); }).catch((e) => replace(body, errorRetry(`PAE could not be loaded: ${e.message || e}`, start)));
  };
  start();
  return { el: card, setDark() {} };
}

// ------------------------------------------------------------------------------------------------ chains
function heat(v) {
  if (v === null || v === undefined) return h("span", { class: "heat" }, "–");
  const a = Math.max(0, Math.min(1, v));
  return h("span", { class: "heat", style: { background: `color-mix(in srgb, var(--accent) ${Math.round(8 + 50 * a)}%, transparent)` } }, v.toFixed(2));
}

export function chainsCard(chains, conf, pocket) {
  chains = chains || [];
  const cp = conf?.chains_ptm || {};
  const pair = conf?.pair_chains_iptm || {};
  const multi = chains.length > 1 && Object.keys(pair).length > 1;
  const desc = (c) => c.type === "ligand" ? (c.ccd ? `CCD ${c.ccd}` : `${c.n_atoms ?? "?"} atoms`) : `${c.length} ${c.type === "protein" ? "aa" : "nt"}`;
  return h("div", { class: "card" },
    h("div", { class: "card-h" }, h("h2", null, "Chains")),
    h("div", { class: "card-b stack" },
      h("div", { class: "table-wrap" }, h("table", { class: "t chain-tbl" },
        h("thead", null, h("tr", null, h("th", null, "Chain"), h("th", null, "Type"), h("th", null, "Size"), h("th", null, "pTM"),
          multi ? chains.map((c) => h("th", { title: `ipTM with chain ${c.id}` }, "ipTM " + c.id)) : null)),
        h("tbody", null, chains.map((c, i) => h("tr", null,
          h("td", null, h("span", { class: "chain-tag" }, c.id)), h("td", null, c.type + (c.cyclic ? " (cyclic)" : "")), h("td", { class: "num" }, desc(c)),
          h("td", null, heat(cp[String(i)])),
          multi ? chains.map((_, j) => h("td", null, i === j ? h("span", { class: "faint" }, "·") : heat(pair[String(i)]?.[String(j)]))) : null))))),
      pocket && pocket.pocket?.length ? h("details", { class: "adv" }, h("summary", null, `Pocket residues (${pocket.pocket_total})`),
        h("div", { class: "small muted" }, `Within ${pocket.cutoff} Å of ligand ${pocket.ligand_chain}; pLDDT in brackets.`),
        h("div", { class: "pocket-chips" }, pocket.pocket.map((x) => h("span", { class: "chip", title: `closest atom ${x.min_dist} Å` },
          `${x.resname}${x.resseq}${x.chain !== "A" ? ":" + x.chain : ""}${x.plddt != null ? " (" + Math.round(x.plddt) + ")" : ""}`)))) : null));
}
