// One screen run: live progress (bar, counts, s/ligand, ETA, Stop / Resume) and its ranked, sortable, filterable results with a
// ligand panel (Mol* 3D, affinity, PAE, 2D depiction, details). Late replies for a ligand that is no longer selected are dropped
// (every asynchronous update checks the selection it was started for).
import { api, downloadFrom, enc } from "./api.js";
import { debounce, download, fmt, h, icon, replace, store, toast } from "./dom.js";
import { stateChip, stateLine } from "./runs.js";
import { msaLabel, presetLabel, refreshSide } from "./parts.js";
import { drafts, go, saveDraft, stashDraft } from "./state.js";
import { PoseViewer } from "./viewer.js";
import { affinityCard, cameraButtons, paeCard, plddtLegend } from "./results.js";
import * as jobView from "./job.js";
import { announce, confirmDialog, copyText, errorRetry, exportMenu, logPanel, radioSeg, tabset, trapTab } from "./ui.js";

const PAGE = 100;
const LONG = 14;   // more rows than this scroll inside the table (sticky header, fade at the edge); fewer use the page
const HIGH = new Set(["affinity_probability_binary", "iptm", "ligand_iptm", "complex_plddt"]);  // sort descending first
const COLS = [
  { key: "rank", label: "Rank", align: "r", cls: "pin-rank", on: true, num: true, f: (r) => (r.rank ?? "–") },
  { key: "ligand_id", label: "ID", cls: "id pin-id", on: true, f: (r) => r.ligand_id },
  { key: "affinity_probability_binary", label: "P(binder)", title: "affinity_probability_binary: chance the ligand is a binder; higher is better", align: "r", on: true, num: true, bar: true, f: (r) => fmt.p(r.affinity_probability_binary) },
  { key: "affinity_pred_value", label: "Affinity (log10 µM)", title: "affinity_pred_value = log10(IC50 in µM); lower is stronger", align: "r", on: true, num: true, f: (r) => fmt.num(r.affinity_pred_value) },
  { key: "iptm", label: "Complex ipTM", title: "ipTM of the complex", align: "r", on: true, num: true, f: (r) => fmt.num(r.iptm), lo: true },
  { key: "ligand_iptm", label: "Ligand ipTM", title: "ipTM of the protein-ligand interface", align: "r", on: true, num: true, f: (r) => fmt.num(r.ligand_iptm), lo: true },
  { key: "complex_plddt", label: "pLDDT", title: "mean pLDDT of the complex (0-100)", align: "r", on: true, num: true, f: (r) => fmt.plddt(r.complex_plddt), lo: true },
  { key: "status", label: "Status", on: true, chip: true },
  { key: "smiles", label: "SMILES", cls: "smiles", on: true, f: (r) => r.smiles || "–", lo: true },
  { key: "value_mw_corrected", label: "y, MW-corrected", title: "affinity value with the paper's molecular-weight correction", align: "r", num: true, f: (r) => fmt.num(r.value_mw_corrected), lo: true },
  { key: "ligand_plddt", label: "Ligand pLDDT", align: "r", num: true, f: (r) => fmt.plddt(r.ligand_plddt), lo: true },
  { key: "ptm", label: "pTM", align: "r", num: true, f: (r) => fmt.num(r.ptm), lo: true },
  { key: "n_atoms", label: "Atoms", align: "r", num: true, f: (r) => r.n_atoms ?? "–", lo: true },
  { key: "mw", label: "MW", align: "r", num: true, f: (r) => fmt.num(r.mw, 1), lo: true },
  { key: "seconds", label: "Time (s)", align: "r", num: true, f: (r) => fmt.num(r.seconds, 1), lo: true },
];
const csvCell = (v) => { const s = v === null || v === undefined ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };

export async function mount(root, app, runId, selRecord) {
  const base = "/api/runs/" + enc(runId);
  try {  // structure jobs have their own page
    const first = await api.get(base);
    if (first.kind === "predict") return jobView.mount(root, app, runId);
  } catch (e) { /* the screen page below shows the error */ }
  let alive = true;
  let timer = null;
  let run = null;
  let rows = [];
  let version = null;
  let selected = selRecord || null;
  let detail = null, detailRec = null;   // the loaded detail and the record it belongs to
  let gen = 0;                           // bumped on every selection: asynchronous results carry the value they started with
  let viewer = null;
  let svgUrl = null;
  let busy = false;
  const ticked = new Set();  // ligand ids ticked for Compare / Re-check
  let cmp = null;            // compare payload of a re-check run
  const ui = { rankBy: null, sort: { key: "rank", dir: 1 }, q: "", status: "all", minProb: "", maxValue: "", minIptm: "", page: 0, tab: "3d", cols: null };
  const phone = window.matchMedia?.("(max-width: 720px)").matches;
  const PHONE_COLS = ["rank", "ligand_id", "affinity_probability_binary", "affinity_pred_value", "status"];
  ui.cols = new Set(store("cols") || (phone ? PHONE_COLS : COLS.filter((c) => c.on).map((c) => c.key)));

  // ------------------------------------------------------------------------------------------ loading
  async function loadRun() {
    try {
      run = await api.get(base);
    } catch (e) {
      if (!alive) return false;
      if (run) timer = setTimeout(tick, 4000);  // keep showing the run; the connection indicator says it is stale
      else replace(root, h("div", { class: "notice bad" }, icon("alert"), h("div", null, /no such run/i.test(e.message) ? "Run not found" : e.message), h("a", { class: "btn sm", href: "#/runs" }, "Back to runs")));
      return false;
    }
    if (ui.rankBy === null) ui.rankBy = run.meta?.options?.rank_by || (run.config?.rank_by) || "prob";
    if (run.recheck_of) {
      try { cmp = await api.get(`${base}/compare`); } catch (e) { cmp = { error: e.message }; }
    }
    return true;
  }

  // Every request carries the ranking it asked for and a number. A reply is used only if that ranking is still the selected one and no
  // later request has been used already, so a slow answer for P(binder) never replaces the rows of y (or older rows newer ones),
  // whether it came from a switch or from polling. Resolves true (used), false (superseded / unchanged) or null (failed).
  let rankGen = 0, applied = 0;
  async function loadResults(force = false) {
    const my = ++rankGen, rb = ui.rankBy;
    try {
      const r = await api.get(`${base}/results?rank_by=${rb}` + (force || !version ? "" : `&v=${enc(version)}`));
      if (!alive || my < applied || rb !== ui.rankBy || r.unchanged) return false;
      applied = my;
      rows = r.rows;
      version = r.version;
      return true;
    } catch (e) {
      if (alive && rb === ui.rankBy) toast(e.message, "bad");
      return null;
    }
  }

  async function tick() {
    clearTimeout(timer);
    if (!alive) return;
    const prevState = run?.state, prevVer = run?.results_version;
    if (!(await loadRun()) || !alive) return;
    renderHead();
    renderProgress();
    renderCompare();
    if (prevState !== undefined && run.state !== prevState) refreshSide();   // the sidebar follows state changes seen here
    if (run.results_version !== prevVer || run.state !== prevState || !version) {
      if (await loadResults()) {
        renderResults();
        renderHead();
      }
    }
    const live = run.state === "running" || run.state === "stopping";
    timer = setTimeout(tick, live ? 1500 : 8000);
  }

  // ------------------------------------------------------------------------------------------ actions
  async function act(fn, ok) {
    if (busy) return;
    busy = true;
    renderHead();
    try {
      await fn();
      if (ok) toast(ok);
    } catch (e) {
      toast(e.message, "bad");
    }
    busy = false;
    await tick();
    refreshSide();
  }
  const stop = (force) => act(() => api.post(`${base}/stop`, { force }), force ? "Stopping now" : "Stopping after this ligand");
  const resume = (retry) => act(() => api.post(`${base}/resume`, { retry_errors: retry }), retry ? "Retrying failed" : "Resumed");
  const save = async (path, name) => { try { await downloadFrom(path, name); } catch (e) { toast(e.message, "bad"); } };
  async function recheck(ids = [...ticked]) {
    if (!ids.length) return;
    let ref = null;
    try { ref = (await api.get("/api/presets")).gui?.find((g) => g.key === "reference"); } catch (e) { /* the rows below omit what is unknown */ }
    const per = ref?.measured ? (ref.measured.pooled ?? (ref.measured.lo + ref.measured.hi) / 2) : null;
    const ok = await confirmDialog({ title: "Re-check with Reference", ok: "Run re-check",
      rows: [["Ligands", fmt.int(ids.length)], ["Preset", "Reference · fp32"], ["Steps", "200 + 200 · 3 recycles"], per ? ["Estimate", fmt.roughDur(per * ids.length)] : null].filter(Boolean) });
    if (!ok) return;
    try {
      const r = await api.post(`${base}/recheck`, { ligand_ids: ids });
      location.hash = "#/run/" + enc(r.id);
    } catch (e) {
      toast(e.message, "bad");
    }
  }
  async function useInputs() {
    try {
      const d = await api.get(`${base}/inputs`);
      const keep = (f) => (f && f.content != null ? { kind: "file", filename: f.filename || "file", content: f.content, size: f.content.length } : null);
      const o = d.options || {};
      let saved = null;
      if (d.target.saved) { try { saved = await api.get(`/api/targets/${enc(d.target.saved)}`); } catch (e) { /* deleted: this run's own copy is used */ } }
      // A deleted saved target (or an uploaded file) comes back as this run's staged target with its MSA and template files.
      const bundle = { kind: "run", id: runId, filename: d.target.filename || "target", size: d.target.content?.length || 0 };
      const draft = { target: saved ? { tab: "saved", file: null, text: "", saved } : { tab: "file", file: d.target.bundle ? bundle : keep(d.target), text: "", saved: null },
        msa: { mode: d.msa.content != null ? "file" : d.msa.mode === "empty" ? "empty" : "default", file: keep(d.msa) },
        ligands: { tab: "file", file: keep(d.ligands), text: "" },
        o: { preset: o.preset || "", rank_by: o.rank_by || "prob", chunk: String(o.chunk ?? 50), saveAll: o.save_structures === "all", saveK: o.save_structures === "all" ? "10" : String(o.save_structures ?? 10),
          seed: String(o.seed ?? 0), maxMinutes: o.max_minutes ? String(o.max_minutes) : "", pocket: (o.pocket || []).join(", "), ligandChain: o.ligand_chain || "", idCol: o.id_col || "", smilesCol: o.smiles_col || "",
          steps: String(o.steps ?? ""), recycling: String(o.recycling ?? ""), affinity_steps: String(o.affinity_steps ?? ""), keepWork: !!o.keep_work },
        name: `${d.name || runId} (copy)`, filter: "all", tform: { open: false, f: null }, undoable: true };
      stashDraft("screen", ["val", "valBusy", "valError", "valGen", "est", "estBusy", "page", "selRow", "starting", "depict", "discarded"]);   // the draft being replaced, for Undo
      drafts.screen = draft;
      saveDraft("screen", draft);
      go("#/screen");
    } catch (e) { toast(e.message, "bad"); }
  }
  const dl = (name, text) => download(name, text);
  async function csvAll() {
    try { dl(`${runId}_results_${ui.rankBy}.csv`, await api.text(`${base}/results.csv?rank_by=${ui.rankBy}`)); } catch (e) { toast(e.message, "bad"); }
  }
  // Some rows only (the ticked or the filtered ones): the server writes them with the same columns and ranks as the whole table.
  async function csvOnly(list, name) {
    try { dl(name, await api.postText(`${base}/results.csv`, { rank_by: ui.rankBy, ids: list.map((r) => r.record_id) })); } catch (e) { toast(e.message, "bad"); }
  }
  const exportItems = () => {
    const sel = rows.filter((r) => ticked.has(r.ligand_id));
    return [
      { label: "All CSV", title: "The ranked table, every row", onClick: csvAll },
      sel.length ? { label: `Selected CSV (${sel.length})`, onClick: () => csvOnly(sel, `${runId}_selected.csv`) } : null,
      filterActive() ? { label: `Filtered CSV (${filtered().length})`, onClick: () => csvOnly(filtered(), `${runId}_filtered.csv`) } : null,
      { sep: "Bundle" },
      { label: "mmCIF poses", title: "Inputs, settings, results.csv, raw arrays kept and the saved poses as mmCIF", onClick: () => save(`${base}/export.zip?rank_by=${ui.rankBy}`, `${runId}_export.zip`) },
      { label: "PDB poses", title: "The same bundle with the poses as PDB", onClick: () => save(`${base}/export.zip?rank_by=${ui.rankBy}&format=pdb`, `${runId}_export.zip`) },
    ].filter(Boolean);
  };

  // ------------------------------------------------------------------------------------------ header
  const elHead = h("div");
  let headKey = null;
  function resolved() {  // the immutable parameters this run resolved to
    const o = run.meta?.options || {}, c = run.config || {};
    const name = c.preset || o.preset;
    const kv = [["Run", runId], ["Preset", `${run.preset_label || presetLabel(name) || name || "–"}${name ? ` (${name})` : ""}`], ["Precision", c.env?.BOLTZ_MLX_PRECISION || (name === "strict" ? "fp32" : name ? "f16" : "–")],
      ["Structure steps", c.steps ?? o.steps ?? "–"], ["Recycles", c.recycling ?? o.recycling ?? "–"], ["Affinity steps", c.affinity_steps ?? o.affinity_steps ?? "–"], ["Affinity recycles", c.affinity_recycling ?? "–"],
      ["Seed", c.seed ?? o.seed ?? "–"], ["Ranked by", ui.rankBy === "prob" ? "P(binder)" : "Affinity (log10 µM)"],
      ["Poses kept", o.save_structures ?? "–"], ["PAE kept", o.keep_work ? "yes" : "no"], ["Chunk", o.chunk ?? "–"]];
    if (run.ligand_chain) kv.push(["Ligand chain", run.ligand_chain]);
    return kv;
  }
  function renderHead() {
    if (!run) return;
    // rebuilt only when something shown changed: a button replaced between mouse-down and mouse-up would lose the click
    const key = JSON.stringify([run.state, run.name, run.can_stop, run.can_force_stop, run.can_resume, run.can_retry, busy, rows.length > 0, run.progress.done > 0, run.path, run.rechecks, run.recheck_of, ui.rankBy, run.progress.ok, run.id]);
    if (key === headKey) return;
    headKey = key;
    const o = run.meta?.options || {};
    const t = run.meta?.summary?.target;
    const done = run.progress.ok + run.progress.error;
    const btns = [];
    if (run.can_stop) btns.push(h("button", { type: "button", class: "btn danger", disabled: busy, onclick: () => stop(false) }, icon("stop"), "Stop"));
    if (run.can_force_stop) btns.push(h("button", { type: "button", class: "btn danger", disabled: busy, onclick: () => stop(true), title: "Interrupt now; finished ligands are kept" }, icon("stop"), "Stop now"));
    if (run.can_resume) btns.push(h("button", { type: "button", class: "btn primary", disabled: busy, onclick: () => resume(false) }, icon("play"), run.progress.done ? "Resume remaining" : "Start"));
    if (run.can_retry) btns.push(h("button", { type: "button", class: "btn", disabled: busy, onclick: () => resume(true) }, icon("refresh"), "Retry failed"));
    if (rows.length) btns.push(exportMenu(exportItems));
    if (!run.external) btns.push(h("button", { type: "button", class: "btn", title: "Open these inputs and settings as a new draft", onclick: useInputs }, icon("copy"), "Use inputs"));
    const cfgOpen = elHead.querySelector("details")?.open;
    replace(elHead,
      h("div", { class: "run-h" },
        h("div", { class: "titles" },
          h("h1", null, h("a", { class: "crumb", href: "#/runs", title: "All runs" }, icon("back"), "Runs"), h("span", { class: "rname", title: run.name }, run.name), stateChip(run.state), run.external ? h("span", { class: "chip", title: "Created by the command line" }, "CLI") : null),
          h("div", { class: "meta-line" }, h("span", null, "Screen"),
            run.created ? h("span", null, fmt.when(run.created)) : null,
            o.preset || run.config?.preset ? h("span", null, run.preset_label || presetLabel(o.preset || run.config.preset)) : null,
            t ? h("span", null, `${fmt.int(t.n_residues)} aa`, " · ", msaLabel(t.msa_source)) : null,
            ["stopped", "interrupted"].includes(run.state) ? h("span", { class: "strong" }, stateLine({ kind: "screen", state: run.state, n_done: done, n_ligands: run.progress.valid || run.progress.total })) : null)),
        h("div", { class: "actions" }, btns)),
      h("details", { class: "adv run-det", open: cfgOpen || undefined }, h("summary", null, "Run details"),
        h("div", { class: "stack tight" }, h("dl", { class: "kv kv-wide" }, resolved().flatMap(([k, v]) => [h("dt", null, k), h("dd", { class: "num" }, String(v))])),
          h("div", { class: "row tight small" }, h("span", { class: "mono path" }, run.path), h("button", { type: "button", class: "btn ghost sm", onclick: () => copyText(run.path, "Path copied") }, icon("copy"), "Copy")))),
      run.recheck_of ? h("div", { class: "notice info" }, icon("info"), h("div", null, "Recheck of ", h("a", { href: "#/run/" + enc(run.recheck_of) }, cmp?.original_name || run.recheck_of), " with Reference")) : null,
      run.rechecks?.length ? h("div", { class: "notice info" }, icon("info"), h("div", null, "Rechecked: ", run.rechecks.map((id, i) => [i ? ", " : "", h("a", { href: "#/run/" + enc(id) }, id)]))) : null);
  }

  // ------------------------------------------------------------------------------------------ re-check comparison
  const elCmp = h("div");
  function renderCompare() {
    if (!run?.recheck_of || !cmp) return replace(elCmp);
    if (cmp.error) return replace(elCmp, h("div", { class: "notice bad" }, icon("alert"), h("div", null, cmp.error)));
    const lab = /^(Standard|Reference)\b/.test(cmp.original_label || "") ? cmp.original_label.split(/ [(,]/)[0] : "Original";
    const d = (x, digits = 2) => (x === null || x === undefined ? "–" : (x > 0 ? "+" : "") + x.toFixed(digits));
    const rowsC = cmp.rows;
    replace(elCmp, h("div", { class: "card" },
      h("div", { class: "card-h" }, h("h2", null, `${lab} vs Reference`), h("span", { class: "chip" }, fmt.plural(rowsC.length, "ligand"))),
      h("div", { class: "card-b stack" },
        h("div", { class: "table-wrap", style: { maxHeight: "50vh" } }, h("table", { class: "t" },
          h("thead", null, h("tr", null, ["Ligand", `${lab} rank / ${cmp.n_original ?? "?"}`, `Ref. rank / ${cmp.n_recheck ?? "?"}`, `P ${lab}`, "P Ref.", "ΔP", `y ${lab}`, "y Ref.", "Δy"].map((x, i) => h("th", { class: i ? "r" : "" }, x)))),
          h("tbody", null, rowsC.map((r) => h("tr", { class: "clickable", onclick: () => r.record_id && select(r.record_id) },
            h("td", { class: "id" }, r.ligand_id), h("td", { class: "r num" }, r.orig_rank ?? "–"), h("td", { class: "r num" }, r.rank ?? (r.status === "ok" ? "–" : r.status)),
            h("td", { class: "r num" }, fmt.num(r.orig_affinity_probability_binary, 2)), h("td", { class: "r num" }, fmt.num(r.affinity_probability_binary, 2)), h("td", { class: "r num" }, d(r.delta_affinity_probability_binary)),
            h("td", { class: "r num" }, fmt.num(r.orig_affinity_pred_value, 2)), h("td", { class: "r num" }, fmt.num(r.affinity_pred_value, 2)), h("td", { class: "r num" }, d(r.delta_affinity_pred_value))))))))));
  }

  // ------------------------------------------------------------------------------------------ progress
  // The log is created once and fed in place (ui.js logPanel): a refresh does not move what is being read.
  const log = logPanel({ fetchFull: () => api.text(`${base}/log?lines=300`) });
  const elProg = h("div", { class: "card" }), elProgTop = h("div"), elProgLog = h("div", { class: "log-slot" }, log.el);
  elProg.append(elProgTop, elProgLog);
  function renderProgress() {
    if (!run) return;
    log.set((run.log_tail || []).join("\n"));
    if ((run.state === "failed" || run.state === "interrupted") && !log.el.dataset.seen) { log.el.open = true; log.el.dataset.seen = "1"; }   // routine logs stay closed; a failure opens it
    const p = run.progress;
    const live = run.state === "running" || run.state === "stopping";
    const total = p.valid || p.total;
    const donePct = total ? (100 * (p.ok + p.error)) / total : 100;
    const partPct = total ? (100 * (p.ok + p.error + p.structures_pending * 0.4)) / total : 100;
    const tile = (k, v, s) => h("div", { class: "tile" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), s ? h("div", { class: "s" }, s) : null);
    const cur = p.current;
    const phase = { starting: "Starting", structure: "Structure pass", affinity: "Affinity pass", idle: "", finished: "" }[p.phase] || p.phase;
    const tiles = live
      ? [tile("Done", `${fmt.int(p.ok + p.error)} / ${fmt.int(total)}`),
         p.s_per_ligand ? tile("Rate", `${p.s_per_ligand.toFixed(1)} s/ligand`) : null,
         p.eta_s !== null ? tile("ETA", fmt.dur(p.eta_s), p.eta_source === "estimate" ? "from the estimate" : null) : null,
         tile("Elapsed", fmt.dur(p.elapsed_s))].filter(Boolean)
      : [tile("OK", fmt.int(p.ok)), p.remaining_valid ? tile("Left", fmt.int(p.remaining_valid)) : null,
         run.status?.wall_s ? tile("Last session", fmt.dur(run.status.wall_s), run.status.s_per_ligand_this_run ? `${run.status.s_per_ligand_this_run} s/ligand` : null) : null].filter(Boolean);
    const notes = [];
    if (run.state === "failed" || run.state === "interrupted") notes.push(h("div", { class: "notice bad" }, icon("alert"), h("div", null, run.state === "failed" ? "Failed" : "Interrupted", " · finished ligands are kept", run.run_error ? h("pre", { class: "logbox", style: { marginTop: "8px" } }, run.run_error) : null)));
    if (run.target_warnings?.length) notes.push(h("div", { class: "notice warn" }, icon("alert"), h("ul", null, run.target_warnings.map((w) => h("li", null, w)))));
    const open = [...elProgTop.querySelectorAll("details:not(.finished)")].map((d) => d.open);
    const finished = run.state === "complete";
    const fin = elProgTop.querySelector("details.finished");
    const nDone = p.ok + p.error;
    const title = live ? "Progress" : run.state === "stopped" ? `Stopped · ${fmt.int(nDone)}/${fmt.int(total)} complete · results saved` : "Progress";
    const body = h("div", { class: finished ? "acc-b stack" : "card-b stack" },
      finished ? null : h("div", { class: "row" },
        h("h2", { class: "grow" }, title),
        p.error ? h("span", { class: "chip bad" }, `${fmt.int(p.error)} failed`) : null, p.invalid ? h("span", { class: "chip" }, `${fmt.int(p.invalid)} invalid`) : null,
        h("span", { class: "num strong" }, `${Math.floor(donePct)}%`)),
      finished ? null :
      h("div", { class: "progress" + (live ? " stripe" : run.state === "complete" ? " ok" : run.state === "failed" ? " bad" : ""), role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.floor(donePct), "aria-label": "Ligands screened" },
        h("div", { class: "part", style: { width: Math.min(100, partPct) + "%" } }), h("div", { class: "done", style: { width: Math.min(100, donePct) + "%" } })),
      live ? h("div", { class: "now" }, h("span", { class: "chip info live" }, h("span", { class: "dot" }), run.state === "stopping" ? "Stopping after this ligand" : phase),
        cur ? [h("span", null, h("b", { class: "num" }, `${cur.k}/${cur.n}`), " in chunk"), h("code", null, cur.ligand_id)] : null,
        p.chunk ? h("span", { class: "muted" }, `chunk ${p.chunk.index}/${p.chunk.of}`) : null) : null,
      tiles.length ? h("div", { class: "tiles" }, tiles) : null, notes,
      p.errors_total ? h("details", { class: "acc", open: open[0] }, h("summary", null, `Failed and invalid (${p.errors_total})`),
        h("div", { class: "acc-b" }, h("div", { class: "table-wrap err-list" }, h("table", { class: "t" },
          h("thead", null, h("tr", null, ["Ligand", "Status", "Phase", "Reason"].map((x) => h("th", null, x)))),
          h("tbody", null, p.errors.map((e) => h("tr", null, h("td", { class: "id" }, e.ligand_id), h("td", null, h("span", { class: "chip " + (e.status === "error" ? "bad" : "warn") }, e.status)), h("td", { class: "muted" }, e.phase || ""),
            h("td", { style: { whiteSpace: "normal", minWidth: "240px" } }, e.error || ""))))))),
          p.errors_total > p.errors.length ? h("div", { class: "small muted", style: { marginTop: "6px" } }, `First ${p.errors.length} shown; results.csv has all.`) : null) : null,
      finished ? log.el : null);   // a finished run keeps its log inside the status disclosure
    if (finished) {
      const bits = ["Complete", `${fmt.int(p.ok)} ok`, p.error ? `${fmt.int(p.error)} failed` : null, p.invalid ? `${fmt.int(p.invalid)} invalid` : null, run.status?.wall_s ? fmt.dur(run.status.wall_s) : null].filter(Boolean);
      replace(elProgTop, h("details", { class: "acc finished", open: (fin ? fin.open : p.errors_total > 0) || undefined }, h("summary", null, bits.join(" · ")), body));
    } else replace(elProgTop, body);
    if (!finished) elProgLog.replaceChildren(log.el);
    elProgLog.hidden = finished;
  }
  // ------------------------------------------------------------------------------------------ results
  const elRes = h("div", { class: "results-grid no-sel" });
  const elTable = h("div"), elTool = h("div"), elPager = h("div"), elViewer = h("div", { class: "viewer-wrap" });
  let toolBuilt = false;

  // A number filter that is not a number is left off (and marked); a row with no value for the metric never passes an active threshold.
  const num = (s) => { const t = String(s).trim(); return t !== "" && Number.isFinite(Number(t)) ? Number(t) : null; };
  const ok = (x) => typeof x === "number" && Number.isFinite(x);
  function filtered() {
    const q = ui.q.trim().toLowerCase();
    const mp = num(ui.minProb), xv = num(ui.maxValue), mi = num(ui.minIptm);
    let out = rows.filter((r) => {
      if (ui.status !== "all" && r.status !== ui.status) return false;
      if (q && !(String(r.ligand_id).toLowerCase().includes(q) || String(r.smiles || "").toLowerCase().includes(q))) return false;
      if (mp !== null && !(ok(r.affinity_probability_binary) && r.affinity_probability_binary >= mp)) return false;
      if (xv !== null && !(ok(r.affinity_pred_value) && r.affinity_pred_value <= xv)) return false;
      if (mi !== null && !(ok(r.iptm) && r.iptm >= mi)) return false;
      return true;
    });
    const { key, dir } = ui.sort;
    const col = COLS.find((c) => c.key === key);
    const val = (r) => (key === "status" ? r.status : r[key]);
    out = out.slice().sort((a, b) => {
      const x = val(a), y = val(b);
      const xn = x === null || x === undefined || (typeof x === "number" && Number.isNaN(x)), yn = y === null || y === undefined || (typeof y === "number" && Number.isNaN(y));
      if (xn !== yn) return xn ? 1 : -1; // empty values last, whatever the direction
      if (xn && yn) return (a.index ?? 0) - (b.index ?? 0);
      const c = col?.num ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true });
      return c * dir || (a.index ?? 0) - (b.index ?? 0);
    });
    return out;
  }
  const filterActive = () => ui.q.trim() !== "" || ui.status !== "all" || num(ui.minProb) !== null || num(ui.maxValue) !== null || num(ui.minIptm) !== null;

  // The toolbar is built once: polling re-renders only the table, so typing in a filter is never interrupted.
  const filterInputs = [];
  let statusSel = null;
  const elClear = h("button", { type: "button", class: "btn ghost sm", hidden: true, onclick: () => {
    Object.assign(ui, { q: "", status: "all", minProb: "", maxValue: "", minIptm: "", page: 0 });
    for (const i of elTool.querySelectorAll("input[data-f]")) { i.value = ""; i.classList.remove("on"); i.removeAttribute("aria-invalid"); }
    if (statusSel) { statusSel.value = "all"; statusSel.classList.remove("on"); }
    renderTable();
  } }, "Clear filters");
  const elCount = h("span", { class: "small muted num", role: "status" });
  const elFilterN = h("span", { class: "chip accent", hidden: true });
  function buildToolbar() {
    toolBuilt = true;
    const set = (k, v) => { ui[k] = v; ui.page = 0; renderTable(); };
    const numIn = (k, label) => h("label", { class: "field" }, h("span", null, label),
      h("input", { type: "text", inputmode: "decimal", "data-f": k, placeholder: "Any", value: ui[k], class: ui[k] ? "on" : "", "aria-label": label,
        oninput: debounce((e) => { const t = e.target.value.trim(); e.target.classList.toggle("on", t !== ""); e.target.setAttribute("aria-invalid", String(t !== "" && num(t) === null)); set(k, e.target.value); }, 200) }));
    statusSel = h("select", { "aria-label": "Status", class: ui.status !== "all" ? "on" : "", onchange: (e) => { e.target.classList.toggle("on", e.target.value !== "all"); set("status", e.target.value); } },
      ["all", "ok", "error", "invalid"].map((v) => h("option", { value: v, selected: ui.status === v }, v === "all" ? "All" : v)));
    const rankSeg = radioSeg({ label: "Rank by", value: ui.rankBy, items: [["prob", "P(binder)", false, "affinity_probability_binary: higher is better"], ["value", "Affinity (log10 µM)", false, "affinity_pred_value = log10(IC50 in µM), lower is better"]],
      onChange: async (v) => {
        const prev = { rankBy: ui.rankBy, sort: ui.sort };
        ui.rankBy = v; ui.sort = { key: "rank", dir: 1 };
        const ok = await loadResults(true);
        if (ok) { renderResults(); renderHead(); }
        else if (ok === null && alive && ui.rankBy === v) { ui.rankBy = prev.rankBy; ui.sort = prev.sort; rankSeg.set(prev.rankBy); }   // the new ranking did not arrive: the selector goes back
      } });
    replace(elTool,
      h("div", { class: "toolbar" },
        h("div", { class: "row tight" }, h("span", { class: "small muted" }, "Rank by"), rankSeg),
        h("input", { type: "search", class: "grow-f", placeholder: "Search ID or SMILES", "aria-label": "Search ID or SMILES", "data-f": "q", value: ui.q,
          oninput: debounce((e) => { e.target.classList.toggle("on", e.target.value.trim() !== ""); set("q", e.target.value); }, 150) }),
        h("details", { class: "adv filters-menu" }, h("summary", { class: "btn sm" }, "Filters", elFilterN),
          h("div", { class: "card filters-pop" },
            h("div", { class: "filters-grid" }, h("label", { class: "field" }, h("span", null, "Status"), statusSel),
              numIn("minProb", "P(binder) ≥"), numIn("maxValue", "y ≤"), numIn("minIptm", "Complex ipTM ≥")),
            h("div", { class: "eyebrow" }, "Columns"),
            h("div", { class: "cols-grid" }, COLS.filter((c) => !["rank", "ligand_id"].includes(c.key)).map((c) => h("label", { class: "check small" },
              h("input", { type: "checkbox", checked: ui.cols.has(c.key), onchange: (e) => { e.target.checked ? ui.cols.add(c.key) : ui.cols.delete(c.key); store("cols", [...ui.cols]); renderTable(); } }), c.label))))),
        elClear));
    const q = elTool.querySelector('input[data-f="q"]');
    if (q && ui.q) q.classList.add("on");
  }

  function renderTable() {
    const list = filtered();
    const pages = Math.max(1, Math.ceil(list.length / PAGE));
    ui.page = Math.min(ui.page, pages - 1);
    const slice = list.slice(ui.page * PAGE, (ui.page + 1) * PAGE);
    let cols = COLS.filter((c) => ui.cols.has(c.key) || c.key === "rank" || c.key === "ligand_id");
    if (phone) cols = [...["ligand_id", "status"].map((k) => cols.find((c) => c.key === k)).filter(Boolean), ...cols.filter((c) => !["ligand_id", "status"].includes(c.key))];   // ID, then its status; rank scrolls with the numbers
    const maxProb = Math.max(0.0001, ...rows.map((r) => r.affinity_probability_binary || 0));
    const wrap = elTable.querySelector(".table-wrap");
    const top = wrap?.scrollTop || 0, left = wrap?.scrollLeft || 0;
    const focused = elTable.contains(document.activeElement) ? document.activeElement.closest("tr")?.dataset.rec : null;
    const th = (c) => h("th", { class: (c.align === "r" ? "r " : "") + (c.lo ? "lo " : "") + (c.cls || "").split(" ").filter((x) => x.startsWith("pin-")).join(" "), title: c.title, "aria-sort": ui.sort.key === c.key ? (ui.sort.dir > 0 ? "ascending" : "descending") : "none" },
      h("button", { type: "button", class: "thbtn", onclick: () => { ui.sort = ui.sort.key === c.key ? { key: c.key, dir: -ui.sort.dir } : { key: c.key, dir: HIGH.has(c.key) ? -1 : 1 }; renderTable(); } },
        c.label, h("span", { class: "arrow" }, ui.sort.key === c.key ? (ui.sort.dir > 0 ? "▲" : "▼") : "")));
    const tickCell = (r) => h("td", { class: "tick pin-tick", onclick: (e) => e.stopPropagation() },
      r.status === "ok" ? h("input", { type: "checkbox", checked: ticked.has(r.ligand_id), "aria-label": `Select ${r.ligand_id}`,
        onchange: (e) => { e.target.checked ? ticked.add(r.ligand_id) : ticked.delete(r.ligand_id); renderShortlist(); } }) : null);
    const body = slice.map((r) => {
      const tr = h("tr", { class: "clickable" + (selected === r.record_id ? " sel" : ""), tabindex: "0", "data-rec": r.record_id, "aria-selected": String(selected === r.record_id),
        onclick: () => select(r.record_id), onkeydown: (e) => keyNav(e, tr, r) },
        tickCell(r),
        cols.map((c) => c.chip ? h("td", null, h("span", { class: "chip " + (r.status === "ok" ? "ok" : r.status === "error" ? "bad" : "warn"), title: r.error || (r.warnings || []).join("; ") || undefined }, r.status))
          : h("td", { class: (c.cls || "") + (c.align === "r" ? " r num" : "") + (c.lo ? " lo" : ""), title: c.key === "smiles" ? r.smiles : c.key === "ligand_id" ? r.ligand_id : undefined },
            c.bar && r.affinity_probability_binary != null ? h("span", { class: "cell-bar", style: { "--w": (100 * r.affinity_probability_binary) / maxProb + "%" } }, c.f(r)) : c.f(r))));
      return tr;
    });
    replace(elTable,
      list.length ? h("div", { class: "table-wrap" + (slice.length > LONG ? " scrolls" : ""), style: slice.length > LONG ? { maxHeight: "70vh" } : null }, h("table", { class: "t res" }, h("thead", null, h("tr", null, h("th", { class: "tick pin-tick" }, ""), cols.map(th))), h("tbody", null, body)))
        : h("div", { class: "empty-s" }, rows.length ? "No ligands match these filters." : run.state === "running" || run.state === "stopping" ? "No results yet." : "No results in this run."));
    const nw = elTable.querySelector(".table-wrap");
    if (nw) { nw.scrollTop = top; nw.scrollLeft = left; }
    if (focused) elTable.querySelector(`tr[data-rec="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
    replace(elPager, h("div", { class: "pager" },
      h("span", null, `${list.length ? ui.page * PAGE + 1 : 0}–${Math.min(list.length, (ui.page + 1) * PAGE)} of ${fmt.int(list.length)}`),
      h("button", { type: "button", class: "btn sm", disabled: ui.page <= 0, onclick: () => { ui.page--; renderTable(); } }, "Prev"),
      h("button", { type: "button", class: "btn sm", disabled: ui.page >= pages - 1, onclick: () => { ui.page++; renderTable(); } }, "Next")));
    const act = filterActive();
    elClear.hidden = !act;
    const nf = (ui.status !== "all" ? 1 : 0) + [ui.minProb, ui.maxValue, ui.minIptm].filter((x) => num(x) !== null).length;
    elFilterN.hidden = !nf; elFilterN.textContent = String(nf);
    elCount.textContent = act ? `${fmt.int(list.length)} of ${fmt.plural(rows.length, "row")} shown` : fmt.plural(rows.length, "row");
    if (act) announce(`${list.length} of ${rows.length} shown`);
    renderShortlist();
  }

  const elShort = h("div", { class: "row tight shortlist" });
  function renderShortlist() {
    if (!run) return replace(elShort);
    const canRecheck = !run.recheck_of && !run.external;
    const recs = () => rows.filter((r) => ticked.has(r.ligand_id) && r.status === "ok").map((r) => r.record_id);
    const okRows = () => filtered().filter((r) => r.status === "ok");
    const firstN = (n) => { ticked.clear(); for (const r of okRows().slice(0, n)) ticked.add(r.ligand_id); renderTable(); };
    const sel = rows.filter((r) => ticked.has(r.ligand_id));
    const n = ticked.size;
    replace(elShort,
      elCount,
      h("span", { class: "grow" }),
      filterActive() ? h("button", { type: "button", class: "btn ghost sm", onclick: () => csvRows(filtered(), `${runId}_filtered.csv`), title: "The filtered rows" }, icon("download"), "Filtered CSV") : null,
      n ? [h("span", { class: "small num strong" }, `${n} selected`),
        h("button", { type: "button", class: "btn ghost sm", onclick: () => { ticked.clear(); renderTable(); } }, "Clear"),
        h("button", { type: "button", class: "btn sm", onclick: () => csvRows(sel, `${runId}_selected.csv`), title: "The selected rows" }, icon("download"), "CSV"),
        h("button", { type: "button", class: "btn sm", disabled: n < 2 || n > 30, onclick: () => { location.hash = `#/leadopt/${enc(runId)}/${recs().map(enc).join(",")}`; },
          title: "Compare 2 to 30 selected analogues against a reference, poses superposed" }, icon("grid"), `Compare${n > 30 ? " (max 30)" : ""}`),
        canRecheck ? h("button", { type: "button", class: "btn sm primary", disabled: run.state === "running" || run.state === "stopping" || !!app.info?.active_run, onclick: () => recheck(),
          title: "Re-run the selected ligands with Reference and compare" }, icon("refresh"), "Re-check") : null]
        : h("button", { type: "button", class: "btn sm", onclick: () => firstN(10), disabled: !rows.length, title: "Select the first 10 rows as sorted and filtered" }, "Select first 10"));
  }

  function keyNav(e, tr, r) {
    if (e.target !== tr) return;   // keys from the tick box (Space, Enter, arrows) belong to the box
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(r.record_id); }
    else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const n = e.key === "ArrowDown" ? tr.nextElementSibling : tr.previousElementSibling;
      if (n) { n.focus(); select(n.dataset.rec, false); }
    }
  }

  function renderResults() {
    if (!elRes.firstChild) {
      const left = h("div", { class: "res-left" }, h("div", { class: "row" }, h("h2", { class: "grow" }, "Results")), elTool, elShort, elTable, elPager);
      replace(elRes, left, elViewer);
    }
    if (!toolBuilt) buildToolbar();
    renderTable();
    elRes.classList.toggle("no-sel", !selected);
    if (selected && !detail && !cardBuilt) select(selected, false);
  }

  // ------------------------------------------------------------------------------------------ ligand panel
  // The card is built once and then only updated: the Mol* frame inside it must never be removed from the page (that reloads it).
  let mol = null;   // { box, host, msg, rec }
  const elVTitle = h("h3", { class: "vtitle" }), elVMetrics = h("div", { class: "metric-strip" }), elVNotice = h("div");
  const pane3d = h("div"), pane2d = h("div", { hidden: true }), paneInfo = h("div", { hidden: true });
  const paneAff = h("div", { hidden: true }), panePae = h("div", { hidden: true });
  let paeView = null;
  let tabs = null;
  let cardBuilt = false;
  const rowOf = (rec) => rows.find((r) => r.record_id === rec);
  const rowNow = () => (detailRec === selected && detail?.row) || rowOf(selected) || {};
  const detailFor = (rec) => (detailRec === rec ? detail : null);  // the detail only for the ligand it was loaded for

  function metric(k, v) { return h("div", null, h("div", { class: "k" }, k), h("div", { class: "v" }, v)); }

  function build3d() {
    const host = h("div", { class: "mol-host" }), msg = h("div", { class: "msg" });
    mol = { host, msg, box: h("div", { class: "mol-box", "data-state": "loading" }, host, msg), rec: null, contacts: true };
    const modeBtn = (v, t) => h("button", { type: "button", "data-mode": v, "aria-pressed": String(v === "plddt"), onclick: async () => {
      if (!viewer) return;
      for (const b of pane3d.querySelectorAll("[data-mode]")) b.setAttribute("aria-pressed", String(b.dataset.mode === v));
      mol.legend.hidden = v !== "plddt";
      try { await viewer.setColor(v); } catch (e) { toast(e.message, "bad"); }
    } }, t);
    mol.legend = h("details", { class: "legend-d" }, h("summary", null, "pLDDT legend"), plddtLegend(true));
    replace(pane3d, mol.box,
      h("div", { class: "viewer-ctl" },
        h("div", { class: "seg", role: "group", "aria-label": "Colour by" }, modeBtn("plddt", "pLDDT"), modeBtn("chain", "Chain")),
        h("label", { class: "check small", title: "Residues within 5 Å of the ligand as sticks" }, h("input", { type: "checkbox", checked: true, onchange: async (e) => { try { mol.contacts = e.target.checked; await viewer?.setPocket(e.target.checked); mol.camera?.set(mol.contacts ? "pocket" : "complex"); } catch (err) { toast(err.message, "bad"); } } }), "Contacts"),
        mol.camera = cameraButtons(() => viewer, true),
        h("span", { class: "grow" }),
        h("button", { type: "button", class: "btn sm pose-dl", title: "This pose as mmCIF", onclick: () => selected && save(`${base}/ligand/${enc(selected)}/pose?format=mmcif`, `${selected}.cif`) }, icon("download"), "mmCIF"),
        h("button", { type: "button", class: "btn sm pose-dl", title: "This pose as PDB", onclick: () => selected && save(`${base}/ligand/${enc(selected)}/pose?format=pdb`, `${selected}.pdb`) }, icon("download"), "PDB")),
      mol.legend);
  }

  function neighbour(step) {
    const list = filtered();
    const i = list.findIndex((r) => r.record_id === selected);
    const n = list[i + step];
    if (n) select(n.record_id, false);
  }

  function buildCard() {
    cardBuilt = true;
    build3d();
    tabs = tabset({ label: "Ligand view", items: [["3d", "3D"], ["aff", "Affinity"], ["pae", "PAE"], ["2d", "2D"], ["info", "Details"]], value: ui.tab, cls: "seg vtabs",
      onChange: (v) => { ui.tab = v; updateCard(); loadPane(); } });
    tabs.panel.append(pane3d, paneAff, panePae, pane2d, paneInfo);
    replace(elViewer, h("div", { class: "card viewer-card", role: "region", "aria-label": "Ligand" },
      h("div", { class: "viewer-h" }, elVTitle,
        h("button", { type: "button", class: "btn ghost icon-only", "aria-label": "Previous ligand", onclick: () => neighbour(-1) }, icon("back")),
        h("button", { type: "button", class: "btn ghost icon-only", "aria-label": "Next ligand", onclick: () => neighbour(1) }, icon("arrow")),
        h("button", { type: "button", class: "btn ghost icon-only", "aria-label": "Close", onclick: closeViewer }, icon("x"))),
      h("div", { class: "viewer-tabs" }, tabs.bar),
      elVMetrics, elVNotice, tabs.panel));
  }

  // On a narrow screen the ligand panel is a full-screen sheet: a dialog, with the page behind it inert, Escape to close, and focus
  // returned to the row it came from. On a wide one it is a panel beside the table.
  const sheetMQ = window.matchMedia ? window.matchMedia("(max-width: 1100px)") : { matches: false, addEventListener() {}, removeEventListener() {} };
  let wasModal = false, lastRec = null;
  const behind = () => [document.getElementById("nav"), document.getElementById("side"), document.querySelector(".foot"), elHead, elProg, elCmp, elRes.firstChild].filter(Boolean);
  function syncModal() {
    const modal = !!selected && sheetMQ.matches;
    const card = elViewer.firstChild;
    if (card) { card.setAttribute("role", modal ? "dialog" : "region"); if (modal) card.setAttribute("aria-modal", "true"); else card.removeAttribute("aria-modal"); }
    for (const el of behind()) el.inert = modal;
    if (modal && !wasModal) elViewer.querySelector('[aria-label="Close"]')?.focus();
    if (!modal && wasModal && !selected && lastRec) elTable.querySelector(`tr[data-rec="${CSS.escape(lastRec)}"]`)?.focus({ preventScroll: true });
    wasModal = modal;
  }
  const onKey = (e) => {
    if (e.key === "Escape" && wasModal && !document.querySelector("dialog[open]")) { e.preventDefault(); closeViewer(); }
    else if (e.key === "Tab" && wasModal && !document.querySelector("dialog[open]")) trapTab(e, elViewer.firstChild);   // the sheet keeps focus until it is closed
  };
  document.addEventListener("keydown", onKey);
  sheetMQ.addEventListener?.("change", () => alive && syncModal());

  function updateCard() {
    if (!cardBuilt) buildCard();
    elViewer.hidden = !selected;
    document.body.classList.toggle("sheet-open", !!selected);
    if (selected) lastRec = selected;
    syncModal();
    if (!selected) return;
    const r = rowOf(selected), row = rowNow();
    const ok = (r?.status || row.status) === "ok";
    replace(elVTitle, row.ligand_id || selected, r?.rank ? h("span", { class: "muted", style: { fontWeight: 500 } }, `  ·  rank ${r.rank}`) : null);
    replace(elVMetrics,
      metric("P(binder)", fmt.p(row.affinity_probability_binary)), metric("Affinity (log10 µM)", fmt.num(row.affinity_pred_value)), metric("Complex ipTM", fmt.num(row.iptm)),
      metric("Ligand ipTM", fmt.num(row.ligand_iptm)), metric("pLDDT", fmt.plddt(row.complex_plddt)), metric("Ligand pLDDT", fmt.plddt(row.ligand_plddt)));
    tabs.set(ui.tab);
    const known = Boolean(r || detailFor(selected));
    const derr = detailFor(selected)?.error;   // the detail request failed: not the same as "no pose" or "no PAE"
    replace(elVNotice, derr ? h("div", { class: "pane" }, errorRetry(`Details could not be loaded: ${derr}`, () => select(selected, false)))
      : !ok && known ? h("div", { class: "pane" }, h("div", { class: "notice bad" }, icon("alert"), h("div", null, row.error || "This ligand has no result."))) : null);
    const noPose = ok && (r?.pose === false || (detailRec === selected && detail && detail.pose === null));
    for (const b of pane3d.querySelectorAll(".pose-dl")) b.disabled = noPose;
    pane3d.hidden = !(ok && ui.tab === "3d");
    pane2d.hidden = !(ok && ui.tab === "2d");
    paneInfo.hidden = !(ok && ui.tab === "info");
    paneAff.hidden = !(ok && ui.tab === "aff");
    panePae.hidden = !(ok && ui.tab === "pae");
    if (ok && ui.tab === "3d" && viewer) viewer.resize();
  }

  async function select(rec, scroll = true) {
    selected = rec;
    const my = ++gen;
    history.replaceState(null, "", `#/run/${enc(runId)}/${enc(rec)}`);
    elRes.classList.remove("no-sel");
    for (const tr of elTable.querySelectorAll("tr.sel")) { tr.classList.remove("sel"); tr.setAttribute("aria-selected", "false"); }
    const trSel = elTable.querySelector(`tr[data-rec="${CSS.escape(rec)}"]`);
    if (trSel) { trSel.classList.add("sel"); trSel.setAttribute("aria-selected", "true"); }
    detail = null; detailRec = null;
    updateCard();
    if (ui.tab !== "3d") loadPane();   // the 3D load waits for the detail (it names the pose format)
    let d;
    try {
      d = await api.get(`${base}/ligand/${enc(rec)}`);
    } catch (e) {
      d = { error: e.message };
    }
    if (!alive || my !== gen || selected !== rec) return;  // a newer selection owns the panel: this reply is dropped
    detail = d; detailRec = rec;
    updateCard();
    loadPane();
  }

  function closeViewer() {
    selected = null; detail = null; detailRec = null; gen++;
    history.replaceState(null, "", `#/run/${enc(runId)}`);
    updateCard();
    elRes.classList.add("no-sel");
    for (const tr of elTable.querySelectorAll("tr.sel")) tr.classList.remove("sel");
  }

  async function loadPane() {
    const rec = selected, my = gen;
    if (!rec || rowNow().status !== "ok") return;
    const live = () => alive && selected === rec && gen === my;
    const dd = detailFor(rec);
    if (ui.tab === "2d") {
      replace(pane2d, h("div", { class: "pane svg2d" }, h("span", { class: "spinner" })));
      try {
        const url = await api.blobUrl(`${base}/ligand/${enc(rec)}/2d.svg?theme=${app.theme}`);
        if (!live() || ui.tab !== "2d") { URL.revokeObjectURL(url); return; }
        if (svgUrl) URL.revokeObjectURL(svgUrl);
        svgUrl = url;
        // the drawing is of the SMILES the model ran (smiles_used); that is what is shown and copied, the input stays one click away
        const rw = detailFor(rec)?.row || rowOf(rec) || {};
        const used = rw.smiles_used || rw.smiles || "", input = rw.smiles || "";
        replace(pane2d, h("div", { class: "pane" }, h("div", { class: "svg2d" }, h("img", { src: url, alt: "2D structure of " + rec, width: 320 })),
          h("div", { class: "row", style: { marginTop: "6px" } }, h("code", { class: "small grow", style: { overflowWrap: "anywhere" } }, used),
            h("button", { type: "button", class: "btn sm", onclick: () => copyText(used, "SMILES copied") }, icon("copy"), "Copy")),
          used !== input ? h("details", { class: "adv" }, h("summary", null, "Input SMILES"),
            h("div", { class: "small muted" }, "The model ran the largest fragment with salts removed."),
            h("code", { class: "small", style: { overflowWrap: "anywhere" } }, input),
            (rw.warnings || []).length ? h("div", { class: "small" }, rw.warnings.join("; ")) : null) : null));
      } catch (e) { if (live()) replace(pane2d, h("div", { class: "pane muted" }, e.message)); }
    } else if (ui.tab === "aff") {
      const d = dd?.row || rowOf(rec) || {};
      replace(paneAff, h("div", { class: "pane flat" }, affinityCard(d, { mwCorrected: d.value_mw_corrected, skip: ["p", "y"] })));
    } else if (ui.tab === "pae") {
      if (!dd) return replace(panePae, h("div", { class: "pane" }, h("span", { class: "spinner" })));
      if (dd.error) return replace(panePae, h("div", { class: "pane" }, errorRetry("PAE availability is unknown: the ligand details did not load", () => select(selected, false))));
      if (!dd.has_pae) {
        paeView = null;
        return replace(panePae, h("div", { class: "pane" }, h("div", { class: "notice info" }, icon("info"), h("div", null, "PAE was not kept for this run. Start a screen with Keep PAE on."))));
      }
      paeView = paeCard(() => api.get(`${base}/ligand/${enc(rec)}/pae`).then((x) => (live() ? x : Promise.reject(new Error("superseded")))), { name: `${rec}_pae`, rawUrl: `${base}/ligand/${enc(rec)}/pae.npz` });
      replace(panePae, h("div", { class: "pane flat" }, paeView.el));
    } else if (ui.tab === "info") {
      if (!dd) return replace(paneInfo, h("div", { class: "pane" }, h("span", { class: "spinner" })));
      if (dd.error) return replace(paneInfo, h("div", { class: "pane" }, errorRetry(`Details could not be loaded: ${dd.error}`, () => select(selected, false))));
      const d = dd.row || {}, pk = dd.pocket;
      const kv = [["Ensemble (log10 µM)", [d.affinity_pred_value1, d.affinity_pred_value2].map((x) => fmt.num(x)).join(" / ")], ["pTM", fmt.num(d.ptm)],
        ["Atoms / heavy / MW", `${d.n_atoms ?? "–"} / ${d.n_heavy ?? "–"} / ${fmt.num(d.mw, 1)}`], ["Model time", d.seconds ? fmt.num(d.seconds, 1) + " s" : "–"], ["Engine", d.mode || "–"], ["MSA", msaLabel(d.msa_source)], ["Seed", d.seed ?? "–"]];
      if (d.value_mw_corrected != null) kv.unshift(["y, MW-corrected", fmt.num(d.value_mw_corrected, 3)]);
      if (d.fp16_fallbacks) kv.push(["fp16 fallbacks", d.fp16_fallbacks]);
      replace(paneInfo, h("div", { class: "pane stack" },
        d.warnings?.length ? h("div", { class: "notice warn" }, icon("alert"), h("ul", null, d.warnings.map((w) => h("li", null, w)))) : null,
        h("dl", { class: "kv" }, kv.flatMap(([k, v]) => [h("dt", null, k), h("dd", { class: "num" }, String(v))])),
        pk && pk.pocket?.length ? h("details", { class: "adv" }, h("summary", null, `Pocket residues (${pk.pocket_total})`),
          h("div", { class: "small muted" }, `Within ${pk.cutoff} Å of the ligand; pLDDT in brackets.`),
          h("div", { class: "pocket-chips" }, pk.pocket.map((x) => h("span", { class: "chip", title: `closest atom ${x.min_dist} Å` }, `${x.resname}${x.resseq}${x.chain !== "A" ? ":" + x.chain : ""}${x.plddt != null ? " (" + Math.round(x.plddt) + ")" : ""}`)))) : null));
    } else {
      load3d(rec);
    }
  }

  // One viewer for the whole panel, created once even if two loads ask at the same time.
  let viewerP = null;
  function ensureViewer() {
    if (viewer) return Promise.resolve(viewer);
    if (!viewerP) {
      viewerP = (async () => {
        const v = new PoseViewer(mol.host);
        v.dark = app.theme === "dark";
        try { await v.init(); } catch (e) { v.dispose(); throw e; }
        if (!alive) { v.dispose(); throw new Error("viewer closed"); }
        viewer = v;
        return v;
      })().catch((e) => { viewerP = null; throw e; });
    }
    return viewerP;
  }

  async function load3d(rec) {
    if (!mol) build3d();
    const my = gen;
    const current = () => selected === rec && gen === my && ui.tab === "3d" && alive;
    const setMsg = (state, ...c) => { mol.box.dataset.state = state; mol.msg.hidden = false; replace(mol.msg, ...c); };
    const row = rowOf(rec) || {};
    if (row.pose === false) {   // the screen kept structures for its top ligands only: say so and offer to generate this one
      mol.rec = null;
      const can = !run.recheck_of && !run.external && !["running", "stopping"].includes(run.state) && !app.info?.active_run;
      return setMsg("empty", h("div", { class: "stack tight center" }, h("strong", null, "Pose not retained"),
        h("div", { class: "small" }, "Scores are kept; the structure was not saved for this ligand."),
        can ? h("button", { type: "button", class: "btn sm", onclick: () => recheck([row.ligand_id]) }, icon("refresh"), "Re-check to generate it") : null));
    }
    if (viewer && mol.rec === rec) { viewer.resize(); return; }
    setMsg("loading", h("span", { class: "spinner" }), " Loading");
    try {
      const text = await api.text(`${base}/ligand/${enc(rec)}/pose`);
      if (!current()) return;
      const v = await ensureViewer();
      if (!current()) return;
      // drawn through the viewer's queue: only the newest request reaches Mol*, so the structure in the frame is the current ligand's
      const shown = await v.display(text, detailFor(rec)?.pose?.format || "mmcif", current);
      if (!shown || !current()) return;   // superseded: the newer load owns the loading mask and draws its own structure
      mol.rec = rec;
      await v.resize();
      if (current()) { mol.msg.hidden = true; mol.box.dataset.state = "ready"; mol.camera?.set(mol.contacts ? "pocket" : "complex"); }
    } catch (e) {
      console.error(e);
      if (viewer && /closed|timeout/.test(e.message || "")) { viewer.dispose(); viewer = null; viewerP = null; mol.rec = null; }
      if (current()) setMsg("error", h("div", { class: "stack tight center" }, h("strong", null, "3D view unavailable"), h("div", { class: "small" }, e.message || String(e)),
        h("button", { type: "button", class: "btn sm", onclick: () => { mol.rec = null; load3d(rec); } }, icon("refresh"), "Retry")));
    }
  }

  const onTheme = (e) => { if (viewer) viewer.setDark(e.detail === "dark"); if (ui.tab === "2d") loadPane(); };
  window.addEventListener("themechange", onTheme);

  // ------------------------------------------------------------------------------------------ mount
  replace(root, elHead, h("div", { class: "stack" }, elProg, elCmp, elRes));
  if (selected) elRes.classList.remove("no-sel");
  await tick();
  return () => {
    alive = false;
    clearTimeout(timer);
    document.body.classList.remove("sheet-open");
    for (const el of behind()) el.inert = false;
    document.removeEventListener("keydown", onKey);
    window.removeEventListener("themechange", onTheme);
    if (viewer) viewer.dispose();
    if (svgUrl) URL.revokeObjectURL(svgUrl);
  };
}
