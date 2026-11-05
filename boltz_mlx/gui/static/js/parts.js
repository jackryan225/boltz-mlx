// Pieces shared by the pages: the page header, the desktop sidebar of recent runs, and the preset selector.
import { api, enc } from "./api.js";
import { fmt, h, icon, replace } from "./dom.js";
import { clearUndo, draftStatus, drafts, go, hasUndo, saveDraft, takeUndo } from "./state.js";
import { radioSeg } from "./ui.js";

// The compact page header: the title, the run name beside it, the actions on the trailing edge.
export function pageHead(title, actions = [], name = null) {
  const warn = h("span", { class: "chip warn", "data-draft-warn": "", hidden: draftStatus.level === "ok" }, draftStatus.level === "files" ? "Files not kept" : "Draft not saved");
  return h("div", { class: "ph" }, h("h1", null, title), name, h("div", { class: "ph-act" }, warn, actions));
}

// "Inputs loaded · Undo" after Use inputs replaced a draft: Undo brings the previous draft back; dismissing keeps the new one.
export function undoNotice(key, st, redraw) {
  if (!st.undoable || !hasUndo(key)) return null;
  const dismiss = () => { st.undoable = false; clearUndo(key); saveDraft(key, st, ["val", "est", "depict", "valBusy", "valError", "starting", "busy"]); redraw(); };
  return h("div", { class: "notice info undo" }, icon("info"), h("div", { class: "grow" }, "Inputs loaded"),
    h("button", { type: "button", class: "btn sm", onclick: () => {
      const u = takeUndo(key);
      st.discarded = true;
      if (u.draft) { saveDraft(key, u.draft); drafts[key] = u.mem ? u.draft : null; }
      else { drafts[key] = null; try { localStorage.removeItem("boltz-draft-" + key); } catch (e) { /* ignore */ } }
      go(location.hash);
    } }, "Undo"),
    h("button", { type: "button", class: "btn ghost icon-only sm", "aria-label": "Keep", title: "Keep", onclick: dismiss }, icon("x")));
}

// ------------------------------------------------------------------------------------------------ sidebar
const side = { el: null, active: null, runs: [], loaded: 0 };

// Engine ids ("screen", "strict", "a3m:file") stay in Settings; headers and cards use the preset names and the MSA file.
export const presetLabel = (n) => (!n ? "" : /^screen/i.test(n) ? "Fast" : /^strict/i.test(n) ? "Full" : String(n));
export function msaLabel(src) {
  const t = String(src || "");
  if (!t || t === "empty") return "No MSA";
  const m = /^(?:a3m|csv):(.+)$/.exec(t);
  return m ? `MSA: ${m[1]}` : t.startsWith("mixed") ? "MSA: mixed" : `MSA: ${t}`;
}

export const DOT = { complete: "var(--ok)", running: "var(--accent)", stopping: "var(--warn)", stopped: "var(--warn)", interrupted: "var(--warn)", failed: "var(--bad)", new: "var(--faint)" };

function runSub(r) {
  const m = r.metrics || {};
  if (r.kind === "predict") {
    if (r.state !== "complete") return r.state;
    return [m.complex_plddt != null ? `pLDDT ${fmt.plddt(m.complex_plddt)}` : null,
      m.affinity_probability_binary != null ? `P ${fmt.p(m.affinity_probability_binary)}` : null].filter(Boolean).join(" · ");
  }
  return r.state === "complete" ? fmt.plural(r.n_ligands ?? r.n_done ?? 0, "ligand") : `${r.state} · ${fmt.int(r.n_done ?? 0)}/${fmt.int(r.n_ligands ?? r.n_done ?? 0)}`;
}

export async function renderSide(active, refresh = false) {
  side.el = side.el || document.getElementById("side");
  if (!side.el) return;
  side.active = active;
  if (refresh || Date.now() - side.loaded > 15000) {
    try {
      side.runs = (await api.get("/api/runs")).runs.slice(0, 8);
      side.loaded = Date.now();
    } catch (e) { /* the views report connection problems */ }
  }
  replace(side.el, h("div", { class: "side-in" },
    h("div", { class: "eyebrow" }, "Recent"),
    side.runs.length ? side.runs.map((r) => h("a", { href: "#/run/" + enc(r.id), class: "run", "aria-current": active === "run:" + r.id ? "page" : undefined },
      h("i", { class: "sdot", style: { background: DOT[r.state] || "var(--faint)" } }), h("span", null, r.name, h("small", null, runSub(r))))) : h("span", { class: "small faint side-none" }, "No runs"),
    h("a", { href: "#/runs", class: "all", "aria-current": active === "runs" ? "page" : undefined }, "All runs")));
}

export function refreshSide() { if (side.active !== null) renderSide(side.active, true); }
export { go };

// ---------------------------------------------------------------------------------------------- GUI presets
// The two presets (boltz_mlx/gui/presets.py): Fast (CLI "screen" on M5-class GPUs: f16, 20 + 25 steps; strict on Macs without NAX)
// and Full (CLI "strict": upstream defaults at fp32). The keys stay "standard" / "reference"; the page shows Fast / Full.
export function guiKeyOf(name, gui, fallback) {
  if (gui.some((g) => g.key === name && g.enabled)) return name;
  const byCli = gui.find((g) => g.cli && g.cli === name && g.enabled);
  return byCli ? byCli.key : fallback;
}

export const precisionOf = (cli) => (cli && cli.name === "strict" ? "fp32" : cli ? "f16" : "");

// "~6.8 s/ligand" from a preset's measured per-ligand time (null when this chip has none).
export function perLigand(g) {
  const m = g?.measured;
  if (!m) return null;
  const x = m.pooled ?? (m.lo + m.hi) / 2;
  return `~${x < 10 ? x.toFixed(1) : Math.round(x)} s/ligand`;
}

// The numbers the preset brings, as the strings the fields hold. The fields are real inputs holding these values: editing one is a
// custom run (the chip turns to "Custom · reset"), picking a preset or Reset puts the preset's numbers back.
export function presetNumbers(cli, keys) {
  const all = { steps: cli?.steps, recycling: cli?.recycling, affinity_steps: cli?.affinity_steps, affinity_samples: cli?.affinity_samples ?? 5 };
  return Object.fromEntries(keys.filter((k) => all[k] !== undefined && all[k] !== null).map((k) => [k, String(all[k])]));
}

// Preset selector: Fast | Full, the precision, a Custom chip that is also the Reset button (back to the preset's numbers) as soon as a
// number leaves the preset, one plain line about the preset, and an info button with its details (and the measured time). The step and recycle fields sit below it, filled with the
// preset's numbers; refresh() follows the form without rebuilding anything.
// o: {preset, steps, recycling, affinity_steps, affinity_samples} (strings); cli(): the CLI preset record; affinity(): bool;
// onPick(key): the page applies the preset (puts its numbers into the fields); onReset(): the same without changing the preset.
export function presetBlock({ gui, o, cli, affinity, onPick, onReset, showTime = false }) {
  const chip = h("span", { class: "chip" });
  const reset = h("button", { type: "button", class: "chip warn chip-btn", hidden: true, title: "A number differs from the preset: restore the preset's numbers", onclick: () => onReset?.() }, "Custom · reset");
  const line = h("span", { class: "num muted preset-line" });
  const seg = radioSeg({ label: "Preset", items: gui.map((g) => [g.key, g.label, !g.enabled, g.tooltip || g.summary]), value: o.preset,
    onChange: (k) => { onPick(k); refresh(); } });
  const sum = h("p", { class: "small muted preset-sum" });
  const det = h("div", { class: "prov small muted", hidden: true });
  const info = h("button", { type: "button", class: "btn ghost icon-only sm", "aria-label": "Preset info", "aria-expanded": "false", title: "Preset info", onclick: () => {
    det.hidden = !det.hidden; info.setAttribute("aria-expanded", String(!det.hidden)); } }, icon("info"));
  const num = (v, d) => (v === "" || v === null || v === undefined ? d : Number(v));
  let custom = false;
  function refresh() {
    const g = gui.find((x) => x.key === o.preset), c = cli();
    const eff = { steps: num(o.steps, c?.steps), rec: num(o.recycling, c?.recycling), aff: num(o.affinity_steps, c?.affinity_steps), affN: num(o.affinity_samples, c?.affinity_samples ?? 5) };
    custom = !!c && (eff.steps !== c.steps || eff.rec !== c.recycling || (affinity() && (eff.aff !== c.affinity_steps || eff.affN !== (c.affinity_samples ?? 5))));
    const status = g?.status || "";
    chip.className = "chip " + (status === "provisional" ? "warn" : "");
    chip.textContent = status === "provisional" ? "Provisional" : "Experimental";
    chip.hidden = custom || status === "validated";
    reset.hidden = !custom;
    line.textContent = c ? `${precisionOf(c)} precision` : "";
    const t = showTime ? perLigand(g) : null;
    sum.textContent = g?.line || g?.summary || "";
    replace(det, g?.note ? h("p", null, g.note) : g?.summary ? h("p", null, g.summary) : null,
      t ? h("p", null, `Measured on this Mac: ${t} in a full screen.`) : null);
  }
  refresh();
  return { el: h("div", { class: "preset-block" }, h("div", { class: "row tight" }, seg, chip, reset, line, info), sum, det), refresh, seg, isCustom: () => custom };
}
