// Screen: target (with its MSA), ligands (validated as they arrive), settings and retention, and a sticky Run panel.
// The form is a draft kept in memory across route changes (state.js) until the screen starts or Clear is pressed.
import { api } from "./api.js";
import { debounce, fmt, h, icon, replace, store, toast } from "./dom.js";
import { guiKeyOf, pageHead, presetBlock, presetNumbers, undoNotice } from "./parts.js";
import { targetLibrary } from "./targets.js";
import { drafts, dropDraft, go, loadDraft, lostFile, saveDraft } from "./state.js";
import { announce, dropzone, fileButton, fileSlots, intCheck, numField, radioSeg, readFileText, reserveBar, tabset } from "./ui.js";

const PAGE = 50;
const TRANSIENT = ["val", "valBusy", "valError", "valGen", "est", "estBusy", "page", "selRow", "starting", "depict", "discarded"];
const LONG = 14;   // more rows than this scroll inside the table (sticky header, fade at the edge); fewer use the page
const unitOf = (type) => (type === "protein" ? "aa" : "nt");

export async function mount(root, app) {
  const info = app.info || {};
  const limits = info.limits || {};
  let presets = [];
  let gui = [];
  let overhead = 1.5;
  try {
    const r = await api.get("/api/presets");
    presets = r.presets;
    gui = r.gui || [];
    overhead = r.overhead_s ?? 1.5;
  } catch (e) {
    toast(e.message, "bad");
  }
  const saved = store("options") || {};
  const defKey = info.default_screen_preset || "standard";   // Fast on M5-class Macs, Full on M1-M4 (from the server)
  const fresh = () => ({
    target: { tab: "saved", file: null, text: "", saved: null, files: [], needs: [] },
    msa: { mode: "default", file: null },
    ligands: { tab: "file", file: null, text: "" },
    o: Object.assign({ preset: defKey, rank_by: "prob", chunk: "50", saveAll: false, saveK: "10", seed: "0", maxMinutes: "", pocket: "", ligandChain: "", idCol: "", smilesCol: "",
      steps: "", recycling: "", affinity_steps: "", keepWork: false },
      Object.fromEntries(["preset", "rank_by", "seed", "saveAll", "saveK"].filter((k) => saved[k] !== undefined && saved[k] !== null && (k !== "preset" || saved.v === 2)).map((k) => [k, saved[k]]))),
    name: "", filter: "all", tform: { open: false, f: null },
  });
  function restore() {  // the draft of the last session; an uploaded file too big to keep comes back as "lost" (its name, to attach again)
    const d = loadDraft("screen");
    if (!d || !d.target || !d.ligands) return null;
    const base = fresh();
    d.target.file = lostFile(d.target.file); d.ligands.file = lostFile(d.ligands.file);
    d.target.files = (d.target.files || []).map(lostFile); d.target.needs = d.target.needs || [];
    d.msa = { ...(d.msa || {}), file: lostFile(d.msa?.file) };
    if (d.tform?.f) { d.tform.f.file = lostFile(d.tform.f.file); d.tform.f.msaFile = lostFile(d.tform.f.msaFile); }
    return { ...base, ...d, o: { ...base.o, ...(d.o || {}) } };
  }
  const st = drafts.screen || (drafts.screen = restore() || fresh());
  if (!st.tform) st.tform = { open: false, f: null };
  if (!st.target.files) st.target.files = [];
  if (!st.target.needs) st.target.needs = [];
  Object.assign(st, { val: null, valBusy: false, valError: null, valGen: -1, est: null, estBusy: false, page: 0, selRow: st.selRow ?? null, starting: false, depict: null });
  st.o.preset = guiKeyOf(st.o.preset, gui, defKey);
  const curGui = () => gui.find((g) => g.key === st.o.preset);
  const curCli = () => presets.find((p) => p.name === curGui()?.cli);
  let gen = 0, depictSeq = 0;
  let depictUrl = null;
  let alive = true;

  // ------------------------------------------------------------------------------------------ options
  const SPEC = { steps: { min: 1, max: 1000 }, recycling: { min: 0, max: 20 }, affinity_steps: { min: 1, max: 1000 },
    seed: { min: -2147483648, max: 2147483647 }, chunk: { min: 1, max: 10000 }, saveK: { min: 0, max: 1000000 }, maxMinutes: { min: 1, max: 1000000, optional: true } };
  const checks = () => Object.fromEntries(Object.entries(SPEC).map(([k, s]) => [k, intCheck(st.o[k], { ...s, optional: s.optional || (k === "saveK" && st.o.saveAll) })]));
  const settingsOk = () => Object.values(checks()).every((c) => c.ok);
  const persistOptions = () => store("options", { v: 2, preset: st.o.preset, rank_by: st.o.rank_by, seed: st.o.seed, saveAll: st.o.saveAll, saveK: st.o.saveK });

  const usable = (f) => (f && !f.lost ? f : null);
  const targetSrc = () => (st.target.tab === "saved" ? (st.target.saved ? { kind: "saved", id: st.target.saved.id } : null)
    : st.target.tab === "file" ? usable(st.target.file) : st.target.text.trim() ? { kind: "text", filename: "pasted", content: st.target.text } : null);
  const ligandSrc = () => (st.ligands.tab === "file" ? usable(st.ligands.file) : st.ligands.text.trim() ? { kind: "text", filename: "pasted", content: st.ligands.text } : null);
  const lostNames = () => [st.target.tab === "file" && st.target.file?.lost ? st.target.file : null, st.ligands.tab === "file" && st.ligands.file?.lost ? st.ligands.file : null,
    st.target.tab !== "saved" && st.msa.mode === "file" && st.msa.file?.lost ? st.msa.file : null,
    ...(st.target.tab !== "saved" ? st.target.files.filter((f) => f.lost) : [])].filter(Boolean).map((f) => f.filename);
  const ownTarget = () => targetSrc()?.kind === "run";   // the target (with its MSA and templates) copied from the run this draft came from
  const options = () => {
    const c = checks(), v = (k) => (c[k].ok ? c[k].value : null);
    const own = (k) => { const x = v(k); return x !== null && String(x) === presetNumbers(curCli(), [k])[k] ? null : x; };
    return {
      preset: st.o.preset, rank_by: st.o.rank_by, chunk: v("chunk") ?? 50,
      save_structures: st.o.saveAll ? "all" : String(v("saveK") ?? 10),
      seed: v("seed") ?? 0, max_minutes: v("maxMinutes"),
      pocket: st.o.pocket.trim() ? st.o.pocket.trim().split(/[\s;,]+/).filter(Boolean) : [], ligand_chain: st.o.ligandChain.trim() || null,
      id_col: st.o.idCol.trim() || null, smiles_col: st.o.smilesCol.trim() || null,
      // a number left at the preset's value goes as "the preset's" (null), so Fast's 200 steps for a ligand with < 10 heavy atoms apply
      steps: own("steps"), recycling: own("recycling"), affinity_steps: own("affinity_steps"), keep_work: !!st.o.keepWork,
    };
  };
  const payload = () => {
    const t = targetSrc(), l = ligandSrc();
    return {
      name: st.name.trim() || undefined,
      target: t ? (t.kind === "saved" || t.kind === "run" ? { kind: t.kind, id: t.id }
        : { kind: t.kind, filename: t.filename, content: t.content, ...(st.target.needs.length ? { files: st.target.files.filter((f) => !f.lost).map((f) => ({ filename: f.filename, content: f.content })) } : {}) }) : undefined,
      msa: t?.kind === "saved" || t?.kind === "run" ? { mode: "default" } : st.msa.mode === "file" && usable(st.msa.file) ? { mode: "file", filename: st.msa.file.filename, content: st.msa.file.content } : { mode: st.msa.mode === "file" ? "default" : st.msa.mode },
      ligands: l ? { kind: l.kind, filename: l.filename, content: l.content } : undefined,
      options: options(),
    };
  };

  // ------------------------------------------------------------------------------------------ validation
  // Every edit invalidates the last check at once (gen moves on); only a reply to the newest payload counts, and Start needs it.
  const persistDraft = debounce(() => saveDraft("screen", st, TRANSIENT), 400);
  const stale = () => st.valGen !== gen;
  const validate = debounce(async () => {
    const p = payload();
    const my = gen;
    if (!p.target && !p.ligands) { st.val = null; st.valBusy = false; st.valError = null; st.valGen = my; renderAll(); return; }
    st.valBusy = true; st.valError = null;
    renderPlan(); renderChips();
    let res = null, err = null;
    try {
      res = await api.post("/api/validate", p);
    } catch (e) {
      err = e.message;
    }
    if (my !== gen || !alive) return;
    st.valBusy = false; st.valGen = my;
    if (err) { st.val = null; st.valError = err; } else {
      st.val = res;
      st.page = 0;
      if (st.selRow !== null && !(res.ligands?.rows || []).some((r) => r.index === st.selRow)) { st.selRow = null; st.depict = null; }
    }
    renderAll();
    const v = st.val;
    announce(st.valError || (v?.ligands?.ok ? `${v.ligands.counts.valid} valid ligands` : v?.target?.error || ""));
  }, 350);
  const changed = () => { gen++; st.est = null; validate(); renderPlan(); preset.refresh(); persistDraft(); };

  const chip = (text, kind = "", title) => h("span", { class: "chip " + kind, title }, text);
  const filePill = (f, onClear) => h("div", { class: "file-pill" }, icon("file"), h("span", { class: "name" }, f.filename),
    h("span", { class: "muted small" }, f.kind === "run" ? "target and MSA of the source run" : fmt.bytes(f.size)), h("span", { class: "grow" }),
    h("button", { type: "button", class: "btn ghost sm", onclick: onClear }, icon("x"), "Remove"));
  // A file too big to keep with the draft: its name stays, and the page asks for it again.
  const lostHint = (f) => (f?.lost ? `${f.filename} was not kept with the draft: choose it again` : null);
  const keepScroll = (host, fn) => { const w = host.querySelector(".table-wrap"); const top = w?.scrollTop || 0; fn(); const n = host.querySelector(".table-wrap"); if (n) n.scrollTop = top; };

  // ------------------------------------------------------------------------------------------ target
  const elTargetBody = h("div", { class: "card-b stack" });
  const elTResult = h("div");
  let wasEmpty = null;
  const lib = targetLibrary({ draft: st.tform, onDraft: () => persistDraft(), selected: () => st.target.saved?.id, onSelect: (t) => { st.target.saved = t; lib.render(); renderTarget(); changed(); },
    onList: () => { if (lib.isEmpty() !== wasEmpty) { wasEmpty = lib.isEmpty(); renderTarget(); } } });
  let tTabs = null;
  function renderTarget() {
    const t = st.target;
    tTabs = tabset({ label: "Target source", items: [["saved", "Saved"], ["file", "File"], ["text", "Paste"]], value: t.tab, onChange: (v) => { t.tab = v; renderTarget(); refreshNeeds(); changed(); } });
    if (t.tab === "saved" && lib.isEmpty()) { replace(elTargetBody, lib.el, elTResult); return renderTargetResult(); }  // no library yet: just the entry form
    const pane = t.tab === "saved" ? lib.el : t.tab === "file"
      ? (t.file && !t.file.lost ? filePill(t.file, () => { t.file = null; t.needs = []; renderTarget(); changed(); })
        : dropzone({ accept: ".yaml,.yml,.fasta,.fa,.fas,.faa", label: t.file?.lost ? "Attach again" : "Choose file", hint: lostHint(t.file) || "Boltz YAML or FASTA",
            onFile: async (f) => { const r = await readFileText(f, limits.target, "The target file"); if (r) { t.file = r; t.files = []; renderTarget(); refreshNeeds(); changed(); } } }))
      : h("label", { class: "field" }, h("span", { class: "sr" }, "Target sequence, FASTA or Boltz YAML"),
          h("textarea", { rows: 5, spellcheck: "false", placeholder: "Sequence, FASTA or Boltz YAML", value: t.text, oninput: (e) => { t.text = e.target.value; needsSoon(); changed(); } }, t.text));
    tTabs.panel.append(pane);
    const slots = t.tab !== "saved" && t.needs.length ? fileSlots({ needs: t.needs, files: t.files, limit: limits.msa, onChange: (files) => { t.files = files; renderTarget(); changed(); } }) : null;
    replace(elTargetBody, tTabs.bar, tTabs.panel, slots, t.tab === "saved" || ownTarget() || t.needs.length ? null : msaRow(), elTResult);
    renderTargetResult();
  }
  // A YAML target names MSA / template files: one attach slot for each (checked on the server like Predict's YAML input).
  let needsSeq = 0;
  async function refreshNeeds() {
    const my = ++needsSeq, t = st.target;
    const src = t.tab === "file" ? usable(t.file) : t.tab === "text" && t.text.trim() ? { content: t.text } : null;
    let needs = [];
    if (src && /^\s*sequences\s*:/m.test(src.content) && !src.content.trimStart().startsWith(">")) {
      try { needs = (await api.post("/api/predict/import", { content: src.content })).needs || []; } catch (e) { needs = []; }
    }
    if (my !== needsSeq || !alive) return;
    if (JSON.stringify(needs) !== JSON.stringify(t.needs)) { t.needs = needs; t.files = t.files.filter((f) => needs.includes(f.filename)); renderTarget(); changed(); }
  }
  const needsSoon = debounce(refreshNeeds, 400);
  function msaRow() {
    const m = st.msa;
    const seg = radioSeg({ label: "MSA", items: [["default", "As in the target YAML"], ["empty", "None (faster, less accurate)"], ["file", "File (.a3m / .csv)"]], value: m.mode, onChange: (v) => { m.mode = v; renderTarget(); changed(); } });
    return h("div", { class: "row tight msa-row" }, h("span", { class: "small muted" }, "MSA"), seg,
      m.mode === "file" ? (m.file && !m.file.lost ? h("span", { class: "chip info" }, icon("file"), m.file.filename, " ", h("button", { type: "button", class: "link", onclick: () => { m.file = null; renderTarget(); changed(); } }, "remove"))
        : [m.file ? h("span", { class: "chip warn" }, icon("alert"), `${m.file.filename} not restored`) : null, fileButton({ label: m.file ? "Attach again" : "Choose file", accept: ".a3m,.csv", onFile: async (f) => { const r = await readFileText(f, limits.msa, "The MSA file"); if (r) { m.file = r; renderTarget(); changed(); } } })]) : null);
  }
  function renderTargetResult() {
    const v = st.val?.target;
    if (st.valBusy && !v) return replace(elTResult, h("span", { class: "muted small" }, h("span", { class: "spinner" }), " Checking"));
    if (!v) return replace(elTResult);
    if (!v.ok) return replace(elTResult, h("div", { class: "ferr", role: "alert" }, v.error));
    const chains = v.chains.map((c) => c.type === "ligand"
      ? chip(`${c.ids.join(",")} · ligand`, "info")
      : chip(`${c.ids.join(",")} · ${c.length ?? "?"} ${unitOf(c.type)}${c.type === "protein" ? ` · MSA ${c.msa && c.msa !== "empty" ? msaRows(c) : "none"}` : ""}`,
        c.type === "protein" && (!c.msa || c.msa === "empty") ? "warn" : "", c.msa && c.msa !== "empty" ? c.msa : undefined));
    replace(elTResult, h("div", { class: "stack tight" },
      h("div", { class: "chips" }, chains, chip(`screen ${v.ligand_chain}`, "accent", "Chain id given to each screened ligand")),
      v.pocket?.length ? h("div", { class: "small muted" }, "Pocket: ", v.pocket.map((p) => p.contacts.map((c) => c.join(":")).join(", ")).join(" | ")) : null,
      v.warnings.length ? h("div", { class: "notice warn" }, icon("alert"), h("ul", null, v.warnings.map((w) => h("li", null, w)))) : null));
  }
  // "25 rows" for a saved target (its record knows), otherwise the file name
  function msaRows(c) { const t = st.target.saved; return st.target.tab === "saved" && t?.msa?.rows ? `${fmt.int(t.msa.rows)} rows` : "file"; }

  // ------------------------------------------------------------------------------------------ ligands
  const elLigBody = h("div", { class: "card-b stack" });
  const elLResult = h("div", { class: "stack tight" });
  const elDepictHost = h("div");
  function renderLigands() {
    const l = st.ligands;
    const tb = tabset({ label: "Ligand source", items: [["file", "File"], ["text", "Paste"]], value: l.tab, onChange: (v) => { l.tab = v; renderLigands(); changed(); } });
    tb.panel.append(l.tab === "file"
      ? (l.file && !l.file.lost ? filePill(l.file, () => { l.file = null; renderLigands(); changed(); })
        : dropzone({ accept: ".csv,.tsv,.smi,.smiles,.txt,.sdf", label: l.file?.lost ? "Attach again" : "Choose file", hint: lostHint(l.file) || "CSV / TSV, SMI or SDF",
            onFile: async (f) => { const r = await readFileText(f, limits.ligands, "The ligand file"); if (r) { l.file = r; renderLigands(); changed(); } } }))
      : h("label", { class: "field" }, h("span", { class: "sr" }, "SMILES, one per line, optionally followed by a name"),
          h("textarea", { rows: 6, spellcheck: "false", placeholder: "SMILES [name], one per line", value: l.text, oninput: (e) => { l.text = e.target.value; changed(); } }, l.text)));
    replace(elLigBody, tb.bar, tb.panel, elLResult);
    renderLigandResult();
  }
  function ligRows() {
    const rows = st.val?.ligands?.rows || [];
    if (st.filter === "issues") return rows.filter((r) => r.status !== "ok");
    if (st.filter === "valid") return rows.filter((r) => r.status !== "invalid");
    return rows;
  }
  function renderLigandResult() {
    const v = st.val?.ligands;
    if (st.valBusy && !v) return replace(elLResult, h("span", { class: "muted small" }, h("span", { class: "spinner" }), " Checking"));
    if (!v) return replace(elLResult);
    if (!v.ok) return replace(elLResult, h("div", { class: "ferr", role: "alert" }, v.error));
    keepScroll(elLResult, () => {
      const c = v.counts;
      const rows = ligRows();
      const pages = Math.max(1, Math.ceil(rows.length / PAGE));
      st.page = Math.min(st.page, pages - 1);
      const slice = rows.slice(st.page * PAGE, (st.page + 1) * PAGE);
      const cols = [["#", "num"], ["ID", ""], ["Status", ""], ["SMILES", "c-smiles"], ["Atoms", "r c-atoms"], ["MW", "r c-mw"], ["Note", "c-note"]];
      const thead = h("thead", null, h("tr", null, cols.map(([x, cl]) => h("th", { class: cl }, x))));
      const tbody = h("tbody", null, slice.map((r) => {
        const note = r.error || r.warnings.join("; ");
        const kind = r.status === "invalid" ? "bad" : r.status === "warn" ? "warn" : "ok";
        return h("tr", { class: "clickable" + (st.selRow === r.index ? " sel" : ""), tabindex: "0", "aria-selected": String(st.selRow === r.index), onclick: () => selectRow(r),
          onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectRow(r); } } },
          h("td", { class: "num muted" }, r.index + 1), h("td", { class: "id", title: r.id }, r.id),
          h("td", null, chip(r.status === "invalid" ? "Invalid" : r.status === "warn" ? (r.size === "large" ? "Large" : "Warning") : "OK", kind, note || undefined)),
          h("td", { class: "smiles c-smiles", title: r.smiles }, r.smiles || "–"),
          h("td", { class: "r num c-atoms" }, r.n_atoms ?? "–"), h("td", { class: "r num c-mw" }, r.mw ?? "–"),
          h("td", { class: "smiles c-note sans", title: note }, note || ""));
      }));
      const filt = radioSeg({ label: "Show", items: [["all", "All"], ["issues", "Issues"], ["valid", "Valid"]], value: st.filter, onChange: (x) => { st.filter = x; st.page = 0; renderLigandResult(); } });
      replace(elLResult,
        h("div", { class: "row" },
          h("div", { class: "chips grow" },
            chip(fmt.plural(c.total, "ligand")), chip(`${fmt.int(c.valid)} valid`, c.valid ? "ok" : "bad"),
            c.invalid ? chip(`${fmt.int(c.invalid)} invalid`, "bad", c.duplicates ? `${c.duplicates} duplicate ids` : undefined) : null,
            c.large ? chip(`${fmt.int(c.large)} > 56 atoms`, "warn", "Above the training maximum: runs with a warning") : null,
            c.warn_other ? chip(`${fmt.int(c.warn_other)} with warnings`, "warn") : null,
            c.median_atoms ? chip(`median ${c.median_atoms} atoms`) : null),
          filt),
        c.truncated ? h("div", { class: "small muted" }, `Showing the first ${fmt.int(c.shown)} of ${fmt.int(c.total)}; all are screened.`) : null,
        h("div", { class: st.selRow === null ? "" : "layout-2" },
          h("div", null,
            rows.length ? h("div", { class: "table-wrap" + (slice.length > LONG ? " scrolls" : ""), style: slice.length > LONG ? { maxHeight: "420px" } : null, tabindex: "-1" }, h("table", { class: "t" }, thead, tbody)) : h("div", { class: "muted small" }, "No ligands match."),
            h("div", { class: "pager" }, h("span", null, `${rows.length ? st.page * PAGE + 1 : 0}–${Math.min(rows.length, (st.page + 1) * PAGE)} of ${fmt.int(rows.length)}`),
              h("button", { type: "button", class: "btn sm", disabled: st.page <= 0, onclick: () => { st.page--; renderLigandResult(); } }, "Prev"),
              h("button", { type: "button", class: "btn sm", disabled: st.page >= pages - 1, onclick: () => { st.page++; renderLigandResult(); } }, "Next"))),
          elDepictHost));
    });
    renderDepict();
  }

  // The depiction follows the selected row; a late reply for a row that is no longer selected is dropped.
  async function drawDepict(r) {
    const my = ++depictSeq;
    st.depict = { row: r, loading: true };
    renderDepict();
    try {
      const url = await api.postBlobUrl("/api/depict", { smiles: r.smiles, dark: app.theme === "dark" });
      if (!alive || my !== depictSeq || st.selRow !== r.index) { URL.revokeObjectURL(url); return; }
      if (depictUrl) URL.revokeObjectURL(depictUrl);
      depictUrl = url;
      st.depict = { row: r, url };
    } catch (e) {
      if (my !== depictSeq) return;
      st.depict = { row: r, error: e.message };
    }
    renderDepict();
  }
  function selectRow(r) {
    if (st.selRow === r.index) { st.selRow = null; st.depict = null; depictSeq++; renderLigandResult(); return; }
    st.selRow = r.index;
    renderLigandResult();
    drawDepict(r);
  }
  function renderDepict() {
    const d = st.depict;
    if (!d || st.selRow === null) return replace(elDepictHost);
    replace(elDepictHost, h("div", { class: "depict" },
      d.loading ? h("div", { class: "muted small" }, h("span", { class: "spinner" }), " Drawing") : d.error ? h("div", { class: "muted small" }, d.error) : h("img", { src: d.url, alt: "2D structure of " + d.row.id }),
      h("div", { style: { fontWeight: "600" } }, d.row.id), h("div", { class: "smi" }, d.row.smiles_used || d.row.smiles),
      d.row.smiles_used && d.row.smiles_used !== d.row.smiles ? h("details", { class: "adv" }, h("summary", null, "Input SMILES"), h("div", { class: "smi" }, d.row.smiles)) : null,
      d.row.error ? h("div", { class: "small", style: { color: "var(--bad)" } }, d.row.error) : null,
      ...d.row.warnings.map((w) => h("div", { class: "small", style: { color: "var(--warn)" } }, w))));
  }
  const onTheme = () => { if (st.depict?.row && st.selRow !== null) drawDepict(st.depict.row); };  // redraw for the new theme; the selection stays
  window.addEventListener("themechange", onTheme);

  // ------------------------------------------------------------------------------------------ settings
  const elSetBody = h("div", { class: "card-b stack" });
  const F = {};
  const nf = (k, label, extra = {}) => (F[k] = numField({ label, value: st.o[k], ...SPEC[k], onInput: (v) => { st.o[k] = v; if (k === "seed" || k === "saveK") persistOptions(); changed(); }, ...extra }));
  // the fields hold the preset's numbers (real, editable values); picking a preset or Reset puts them back
  const OVERRIDES = ["steps", "recycling", "affinity_steps"];
  const fillNumbers = (force) => { const n = presetNumbers(curCli(), OVERRIDES); for (const k of OVERRIDES) if (n[k] !== undefined && (force || st.o[k] === "")) { st.o[k] = n[k]; F[k]?.set(n[k]); } };
  fillNumbers(false);
  nf("steps", "Structure steps"); nf("recycling", "Recycles"); nf("affinity_steps", "Affinity steps"); nf("seed", "Seed");
  nf("saveK", "Top N", { disabled: st.o.saveAll });
  nf("chunk", "Ligands per chunk", { title: "Results are ranked and saved after every chunk" });
  nf("maxMinutes", "Stop after (min)", { placeholder: "No limit" });
  const preset = presetBlock({ gui, o: st.o, cli: curCli, affinity: () => true, showTime: true,
    onPick: (k) => { st.o.preset = k; fillNumbers(true); persistOptions(); changed(); },   // a preset brings its own numbers
    onReset: () => { fillNumbers(true); changed(); } });
  const elStorage = h("span", { class: "small muted num" });
  function storageText() {
    const n = st.val?.ligands?.ok ? st.val.ligands.counts.valid : 0;
    return st.o.keepWork ? (n ? `PAE kept: ≈ ${fmt.int(Math.round(n * 2))} MB (1–3 MB per ligand)` : "PAE kept: 1–3 MB per ligand") : "PAE discarded";
  }
  const txt = (key, label, ph, extra = {}) => h("label", { class: "field" }, h("span", null, label),
    h("input", { type: "text", value: st.o[key], autocomplete: "off", spellcheck: "false", placeholder: ph, ...extra, oninput: (e) => { st.o[key] = e.target.value; changed(); } }));
  const rank = radioSeg({ label: "Rank by", items: [["prob", "P(binder)", false, "affinity_probability_binary: higher is better"], ["value", "Affinity (log10 µM)", false, "affinity_pred_value = log10(IC50 in µM), lower is better"]], value: st.o.rank_by,
    onChange: (v) => { st.o.rank_by = v; persistOptions(); changed(); } });
  const posesSeg = radioSeg({ label: "Poses to keep", items: [["all", "All"], ["top", "Top N"]], value: st.o.saveAll ? "all" : "top",
    onChange: (v) => { st.o.saveAll = v === "all"; F.saveK.input.disabled = st.o.saveAll; persistOptions(); changed(); } });
  const pae = h("input", { type: "checkbox", checked: st.o.keepWork, onchange: (e) => { st.o.keepWork = e.target.checked; elStorage.textContent = storageText(); changed(); } });
  function renderSettings() {
    replace(elSetBody, preset.el,
      h("div", { class: "num-grid" }, F.steps.el, F.recycling.el, F.affinity_steps.el, F.seed.el),
      h("div", { class: "row set-row" }, h("span", { class: "setlbl" }, "Rank by"), rank),
      h("div", { class: "row set-row" }, h("span", { class: "setlbl" }, "Poses"), posesSeg, F.saveK.el,
        h("label", { class: "check" }, pae, "Keep PAE"), elStorage),
      h("details", { class: "adv" }, h("summary", null, "Advanced"),
        h("div", { class: "adv-grid" }, F.chunk.el, F.maxMinutes.el,
          txt("pocket", "Pocket residues", "CHAIN:RESIDUE, ..."), txt("ligandChain", "Ligand chain", "L", { maxlength: 4 }), txt("idCol", "ID column", "Auto"), txt("smilesCol", "SMILES column", "Auto"))));
    elStorage.textContent = storageText();
  }

  // ------------------------------------------------------------------------------------------ run panel
  const nameIn = h("input", { type: "text", class: "ph-name", maxlength: 80, placeholder: "Run name", "aria-label": "Run name", value: st.name, oninput: (ev) => { st.name = ev.target.value; persistDraft(); } });
  const elSummary = h("dl", { class: "kv" });
  const elEst = h("div");
  const elMsgs = h("div", { class: "stack tight" });
  const elStatus = h("span", { class: "ready small", role: "status" });
  const elEstBtn = h("button", { type: "button", class: "btn", onclick: () => doEstimate() });
  const elStart = h("button", { type: "button", class: "btn primary lg", onclick: () => doStart() });
  const elPlan = h("div", { class: "plan-col" },
    h("aside", { class: "card plan", "aria-label": "Run" }, h("div", { class: "card-b stack" }, elSummary, elEst, elMsgs)),
    h("div", { class: "startbar" }, elStatus, elEstBtn, elStart));
  function blockers() {
    const out = [];
    const v = st.val;
    const probs = app.info?.problems_by?.screen || [];
    if (probs.length) out.push(probs[0].replace(/ (missing).*$/, " $1"));
    if (!targetSrc()) out.push("Choose a target");
    else if (v?.target && !v.target.ok && !stale()) out.push("Fix the target");
    if (!ligandSrc()) out.push("Add ligands");
    else if (v?.ligands && !v.ligands.ok && !stale()) out.push("Fix the ligand list");
    else if (v?.ligands?.ok && v.ligands.counts.valid === 0 && !stale()) out.push("No valid ligands");
    if (!settingsOk()) out.push("Fix settings");
    if (lostNames().length) out.push(`Attach ${lostNames()[0]} again`);
    if (st.target.tab !== "saved" && !ownTarget() && st.msa.mode === "file" && !st.msa.file) out.push("Choose the MSA file");
    if (st.valError && !stale()) out.push("Check failed");
    else if (stale() || st.valBusy || !v) out.push("Checking…");
    if (app.info?.active_run) out.push("Job running");
    return out;
  }
  function renderChips() {
    const v = st.val;
    const tOk = v?.target?.ok, lOk = v?.ligands?.ok;
    setChip(elTChip, st.valBusy && !v ? "…" : tOk ? "Ready" : v?.target && !v.target.ok ? "Fix" : "", tOk ? "ok" : "bad");
    setChip(elLChip, lOk ? `${fmt.int(v.ligands.counts.valid)} valid` : v?.ligands && !lOk ? "Fix" : "", lOk && v.ligands.counts.valid ? "ok" : "bad");
  }
  const setChip = (el, text, kind) => { el.hidden = !text; el.textContent = text; el.className = "chip " + kind; };
  function renderPlan() {
    const v = st.val, g = curGui();
    const why = blockers();
    const active = app.info?.active_run;
    const e = st.est?.estimate;
    const ready = why.length === 0 && !st.starting;
    const custom = preset.isCustom();
    const tgt = v?.target?.ok ? v.target.chains.filter((c) => c.type !== "ligand").map((c) => `${c.ids.join(",")} ${c.length ?? "?"} ${unitOf(c.type)}`).join(" · ") : "";
    replace(elSummary,
      h("dt", null, "Target"), h("dd", null, tgt ? `${st.target.saved && st.target.tab === "saved" ? st.target.saved.name + " · " : ""}${tgt}` : "–"),
      h("dt", null, "Ligands"), h("dd", { class: "num" }, v?.ligands?.ok ? `${fmt.int(v.ligands.counts.valid)} of ${fmt.int(v.ligands.counts.total)}` : "–"),
      h("dt", null, "Preset"), h("dd", null, (g?.label || "–").replace(/ \(.*\)$/, "") + (custom ? " · Custom" : "")),
      h("dt", null, "Rank by"), h("dd", null, st.o.rank_by === "prob" ? "P(binder)" : "Affinity (log10 µM)"));
    const lines = st.est?.lines;
    replace(elEst, e && e.n_to_run ? (e.available === false
      ? h("div", { class: "small muted" }, e.custom ? "No estimate for custom settings" : e.measuring ? "Measuring" : "No estimate for this chip")
      : h("div", { class: "est" }, h("div", { class: "eyebrow" }, "Estimate"), h("div", { class: "big" }, e.measured ? fmt.roughSpan(e.total_lo_s, e.total_hi_s) : fmt.roughDur(e.total_s)),
          h("div", { class: "small muted", title: e.caveat }, e.measured ? `${e.per_ligand_min_s}–${e.per_ligand_max_s} s/ligand` : `~${Math.round(e.per_ligand_median_s)} s/ligand · ${fmt.int(e.n_chunks)} chunk${e.n_chunks === 1 ? "" : "s"}`),
          e.steps_note ? h("div", { class: "small muted" }, e.steps_note) : null,
          lines?.length ? h("details", { class: "adv" }, h("summary", null, "Plan"), h("pre", { class: "logbox" }, lines.join("\n"))) : null)) : null);
    replace(elMsgs, st.valError ? h("div", { class: "ferr", role: "alert" }, st.valError) : null);
    const nWarn = (v?.target?.warnings?.length || 0) + (v?.ligands?.ok ? v.ligands.counts.warn : 0);   // a large ligand is one warned ligand, counted once
    const lc = v?.ligands?.ok ? v.ligands.counts : null;
    replace(elStatus, ready ? "Ready" + (lc ? ` · ${fmt.int(lc.valid)} valid${lc.invalid ? ` · ${fmt.int(lc.invalid)} skipped` : ""}` : "") + (nWarn ? ` · ${fmt.plural(nWarn, "warning")}` : "")
      : why[0] === "Job running" ? ["Job running · ", h("a", { href: "#/run/" + encodeURIComponent(active) }, "Open")] : why[0]);
    elStatus.className = "ready small" + (ready ? " ok" : "");
    replace(elEstBtn, st.estBusy ? h("span", { class: "spinner" }) : icon("clock"), "Estimate");
    elEstBtn.disabled = why.length > 0 || st.estBusy;
    replace(elStart, st.starting ? h("span", { class: "spinner" }) : icon("play"), "Start");
    elStart.disabled = !ready;
    elStart.title = ready ? "" : why[0] || "";
  }
  async function doEstimate() {
    const my = gen;
    st.estBusy = true; renderPlan();
    try { const r = await api.post("/api/estimate", payload()); if (my === gen) st.est = r; } catch (e) { toast(e.message, "bad"); }
    st.estBusy = false; if (alive) renderPlan();
  }
  async function doStart() {
    if (blockers().length) return;
    st.starting = true; renderPlan();
    try {
      const r = await api.post("/api/runs", payload());
      persistDraft.cancel();
      dropDraft("screen");
      go("#/run/" + encodeURIComponent(r.id));
    } catch (e) {
      toast(e.message, "bad");
      st.starting = false; if (alive) renderPlan();
    }
  }

  // ------------------------------------------------------------------------------------------ assemble
  const elTChip = h("span", { class: "chip", hidden: true }), elLChip = h("span", { class: "chip", hidden: true });
  const panel = (title, chipEl, body) => h("section", { class: "card" }, h("div", { class: "card-h" }, h("h2", null, title), chipEl), body);
  function renderAll() { renderTargetResult(); renderLigandResult(); renderChips(); elStorage.textContent = storageText(); renderPlan(); preset.refresh(); }

  const elUndo = h("div");
  const drawUndo = () => replace(elUndo, undoNotice("screen", st, drawUndo));
  replace(root,
    pageHead("Screen", [h("button", { type: "button", class: "btn ghost sm", onclick: () => { persistDraft.cancel(); dropDraft("screen"); go(location.hash); } }, "Clear")], nameIn),
    elUndo,
    h("div", { class: "work" }, h("div", { class: "form-col" }, panel("Target", elTChip, elTargetBody), panel("Ligands", elLChip, elLigBody), panel("Settings", null, elSetBody)), elPlan));
  drawUndo();
  renderTarget(); renderLigands(); renderSettings(); renderPlan(); renderChips(); preset.refresh();
  if (st.target.tab !== "saved") refreshNeeds();
  if (st.selRow !== null && st.val === null) { st.selRow = null; st.depict = null; }
  if (targetSrc() || ligandSrc()) { gen++; validate(); renderPlan(); }

  const onInfo = () => { if (alive) renderPlan(); };  // a job started / ended elsewhere: Start follows
  window.addEventListener("infochange", onInfo);
  const freeBar = reserveBar(elPlan.querySelector(".startbar"));
  // Leaving the page keeps the draft only if it is still the live one: Clear and a started screen have discarded it for good.
  return () => { alive = false; freeBar(); needsSoon.cancel(); validate.cancel(); persistDraft.cancel(); if (!st.discarded && drafts.screen === st) saveDraft("screen", st, TRANSIENT); window.removeEventListener("themechange", onTheme); window.removeEventListener("infochange", onInfo); if (depictUrl) URL.revokeObjectURL(depictUrl); };
}
