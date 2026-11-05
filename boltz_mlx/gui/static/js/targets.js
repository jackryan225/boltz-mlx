// The target library: proteins prepared once (sequence checked, MSA uploaded or none, a PDB / mmCIF kept as a template) and
// reused by every screen. Stored on this computer under <runs>/.targets; each run copies the files it uses.
import { api, enc } from "./api.js";
import { fmt, h, icon, replace, toast } from "./dom.js";
import { debounce } from "./dom.js";
import { describe, searchBox } from "./search.js";
import { errorRetry, fileButton, fileSlots, radioSeg, readFileText, tabset } from "./ui.js";

const STRUCT = /\.(pdb|ent|cif|mmcif)$/i;

// One line about a target: length, MSA, template. Shown once, where the target is selected.
export function targetSummary(t) {
  const chips = [h("span", { class: "chip" }, `${fmt.int(t.n_residues)} aa`)];
  if ((t.chains || []).length > 1) chips.push(h("span", { class: "chip" }, `${t.chains.length} chains`));
  chips.push(t.msa?.mode === "file"
    ? h("span", { class: "chip ok", title: t.msa.filename }, t.msa.rows == null ? "MSA files" : `MSA ${fmt.int(t.msa.rows)} rows`)
    : h("span", { class: "chip warn" }, "No MSA"));
  if (t.template) chips.push(h("span", { class: "chip info", title: `from ${t.template.from}` }, `Template ${t.template.chain}`));
  return chips;
}

// selected(): the id in use; onSelect(target | null); readFile(file, limit, what) -> {kind, filename, content, size} | null
// draft: {open, f} kept by the page with its own draft, so an unfinished target (name, sequence, uploads) survives leaving the page;
// it is cleared only by a saved target or Cancel. onDraft(): called after each edit so the page can store it.
export function targetLibrary({ selected, onSelect, onList, draft, onDraft }) {
  const el = h("div", { class: "stack tight" });
  const emptyForm = () => ({ name: "", tab: "file", file: null, text: "", chain: "", template: true, msa: "none", msaFile: null, files: [], needs: [], q: "", found: "", autoName: false });
  const ts = draft || { open: false, f: null };
  if (!ts.f) ts.f = emptyForm();
  if (!ts.f.files) ts.f.files = [];
  if (!ts.f.needs) ts.f.needs = [];
  const touch = () => onDraft?.();
  const st = { list: null, error: null, busy: false, ferr: null, editing: null };

  // A YAML source names MSA / template files: one slot each, attached here and stored with the target.
  let needsSeq = 0;
  async function refreshNeeds() {
    const my = ++needsSeq, f = ts.f;
    const text = f.tab === "file" ? (f.file && !f.file.lost ? f.file.content : "") : f.text;
    let needs = [];
    if (text && /^\s*sequences\s*:/m.test(text) && !text.trimStart().startsWith(">")) {
      try { needs = (await api.post("/api/predict/import", { content: text })).needs || []; } catch (e) { needs = []; }
    }
    if (my !== needsSeq || ts.f !== f) return;
    if (JSON.stringify(needs) !== JSON.stringify(f.needs)) { f.needs = needs; f.files = f.files.filter((x) => needs.includes(x.filename)); touch(); render(); }
  }
  const needsSoon = debounce(refreshNeeds, 400);
  // Save target stays off until the source is there and everything it names is attached.
  const ready = () => {
    const f = ts.f;
    const src = f.tab === "file" ? f.file && !f.file.lost : f.text.trim();
    if (!src) return false;
    if (f.msa === "file" && (!f.msaFile || f.msaFile.lost) && !f.needs.length) return false;
    return f.needs.every((n) => f.files.some((x) => x.filename === n && !x.lost));
  };

  async function refresh() {
    try {
      st.list = (await api.get("/api/targets")).targets;
      st.error = null;
    } catch (e) {
      st.error = e.message;
    }
    if (st.list && !st.list.length) ts.open = true;
    render();
    onList?.(st.list);
  }

  async function remove(t) {
    if (!window.confirm(`Delete "${t.name}"? Runs that used it keep their copy.`)) return;
    try {
      await api.post(`/api/targets/${enc(t.id)}/delete`, {});
      if (selected() === t.id) onSelect(null);
      await refresh();
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  async function rename(t, name) {
    try {
      const r = await api.post(`/api/targets/${enc(t.id)}/rename`, { name });
      st.editing = null;
      if (selected() === t.id) onSelect(r);
      await refresh();
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  // A search result -> the Paste box (and the name, unless the user typed one). UniProt results carry the sequence; a PDB chain or an
  // isoform is fetched by id first.
  async function pickFound(r) {
    const f = ts.f;
    let seq = r.sequence, about = describe(r);
    if (!seq) {
      st.finding = true; st.ferr = null; render();
      try {
        const x = await api.post("/api/lookup", { query: r.id, type: "protein" });
        seq = x.sequence; about = `${x.source === "RCSB" ? "PDB" : x.source} ${x.id} · ${x.title} · ${x.length} aa`;
      } catch (e) {
        st.finding = false; st.ferr = e.message; return render();
      }
      st.finding = false;
      if (ts.f !== f) return;
    }
    f.tab = "text"; f.text = seq; f.found = about; f.q = r.id; f.files = []; f.needs = [];
    if (!f.name.trim() || f.autoName) { f.name = `${r.gene || r.name} (${r.id})`.slice(0, 80); f.autoName = true; }
    touch(); render();
  }

  async function prepare() {
    const f = ts.f;
    const src = f.tab === "file" ? f.file : f.text.trim() ? { kind: "text", filename: null, content: f.text } : null;
    if (f.tab === "file" && f.file?.lost) { st.ferr = `Attach ${f.file.filename} again`; return render(); }
    if (f.msa === "file" && f.msaFile?.lost) { st.ferr = `Attach ${f.msaFile.filename} again`; return render(); }
    if (!src) { st.ferr = "Add a structure or sequence file, or paste a sequence"; return render(); }
    if (f.msa === "file" && !f.msaFile && !f.needs.length) { st.ferr = "Choose the MSA file, or use MSA: None"; return render(); }
    st.busy = true; st.ferr = null; render();
    try {
      const t = await api.post("/api/targets", {
        name: f.name.trim() || undefined, source: { kind: src.kind, filename: src.filename, content: src.content },
        chain: f.chain.trim() || undefined, template: f.template,
        msa: f.msa === "file" && !f.needs.length ? { mode: "file", filename: f.msaFile.filename, content: f.msaFile.content } : { mode: "none" },
        ...(f.needs.length ? { files: f.files.filter((x) => !x.lost).map((x) => ({ filename: x.filename, content: x.content })) } : {}) });
      ts.open = false;
      ts.f = emptyForm();
      touch();
      await refresh();
      onSelect(t);
      toast(`Saved ${t.name}`);
    } catch (e) {
      st.ferr = e.message;
    }
    st.busy = false;
    render();
  }

  function form() {
    const f = ts.f;
    const isStruct = f.tab === "file" && f.file && STRUCT.test(f.file.filename);
    const src = tabset({ label: "New target source", items: [["file", "File"], ["text", "Paste"]], value: f.tab, onChange: (v) => { f.tab = v; touch(); render(); refreshNeeds(); } });
    src.panel.append(f.tab === "file"
      ? h("div", { class: "row" }, fileButton({ label: f.file ? (f.file.lost ? "Attach again" : "Change file") : "Choose file", accept: ".pdb,.ent,.cif,.mmcif,.fasta,.fa,.fas,.faa,.yaml,.yml",
          onFile: async (file) => { const r = await readFileText(file, 32 * 1024 * 1024, "The target file"); if (r) { f.file = r; f.files = []; if (!f.name) f.name = file.name.replace(/\.[^.]+$/, ""); touch(); render(); refreshNeeds(); } } }),
        f.file ? h("span", { class: "small" }, h("b", null, f.file.filename), " ", f.file.lost ? h("span", { class: "bad-t" }, "not restored") : h("span", { class: "muted" }, fmt.bytes(f.file.size))) : h("span", { class: "small muted" }, "PDB, mmCIF, FASTA or Boltz YAML"))
      : h("textarea", { rows: 4, spellcheck: "false", "aria-label": "Amino-acid sequence or FASTA", placeholder: "Sequence or FASTA", value: f.text, oninput: (e) => { f.text = e.target.value; f.found = ""; touch(); needsSoon(); render2(); } }, f.text));
    const msa = radioSeg({ label: "MSA", items: [["none", "None (faster, less accurate)"], ["file", "File (.a3m / .csv)"]], value: f.msa, onChange: (v) => { f.msa = v; touch(); render(); } });
    const bare = !st.list?.length;  // no library yet: the form is the whole panel, not a box inside one
    return h("div", { class: bare ? "stack" : "prep stack" },
      bare ? null : h("div", { class: "row" }, h("h3", { class: "grow" }, "New target"),
        h("button", { type: "button", class: "btn ghost sm", onclick: () => { ts.open = false; ts.f = emptyForm(); st.ferr = null; touch(); render(); } }, "Cancel")),
      h("div", { class: "field" }, h("span", null, "Find a protein"),
        h("div", { class: "row tight" }, searchBox({ type: () => "protein", value: f.q, placeholder: "Gene, protein name or ID: EGFR, thrombin, BRD4, P00533, 8CPK", label: "Find a protein",
          onInput: (v) => { f.q = v; touch(); }, onPick: pickFound }).el, st.finding ? h("span", { class: "spinner", "aria-label": "Loading" }) : null),
        f.found && f.tab === "text" ? h("div", { class: "small muted look-ok" }, icon("check"), " ", f.found) : h("div", { class: "small muted" }, "Or choose a file or paste a sequence below.")),
      h("label", { class: "field" }, h("span", null, "Name"), h("input", { type: "text", maxlength: 80, placeholder: "Target name", value: f.name, oninput: (e) => { f.name = e.target.value; f.autoName = false; touch(); } })),
      src.bar, src.panel,
      f.needs.length ? fileSlots({ needs: f.needs, files: f.files, limit: 64 * 1024 * 1024, onChange: (files) => { f.files = files; touch(); render(); } }) : null,
      isStruct ? h("div", { class: "row" },
        h("label", { class: "field", style: { width: "110px" } }, h("span", null, "Chain"), h("input", { type: "text", maxlength: 4, placeholder: "First", value: f.chain, oninput: (e) => { f.chain = e.target.value; touch(); } })),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: f.template, onchange: (e) => { f.template = e.target.checked; touch(); } }), "Use as template")) : null,
      f.needs.length ? null : h("div", { class: "row" }, msa, f.msa === "file" ? [fileButton({ label: f.msaFile ? (f.msaFile.lost ? "Attach again" : "Change") : "Choose file", accept: ".a3m,.csv",
        onFile: async (file) => { const r = await readFileText(file, 64 * 1024 * 1024, "The MSA file"); if (r) { f.msaFile = r; touch(); render(); } } }),
        f.msaFile ? h("span", { class: "small" }, h("b", null, f.msaFile.filename), " ", f.msaFile.lost ? h("span", { class: "bad-t" }, "not restored") : h("span", { class: "muted" }, fmt.bytes(f.msaFile.size))) : null] : null),
      st.ferr ? h("div", { class: "ferr", role: "alert" }, st.ferr) : null,
      h("div", { class: "row" }, h("button", { type: "button", class: "btn primary save-target", disabled: st.busy || !ready(), onclick: prepare }, st.busy ? h("span", { class: "spinner" }) : icon("check"), "Save target")));
  }

  function row(t) {
    const sel = selected() === t.id;
    if (st.editing === t.id) {
      const input = h("input", { type: "text", maxlength: 80, value: t.name, "aria-label": "Target name", onkeydown: (e) => { if (e.key === "Enter") rename(t, input.value); if (e.key === "Escape") { st.editing = null; render(); } } });
      setTimeout(() => input.focus(), 0);
      return h("div", { class: "trow" + (sel ? " sel" : "") }, h("div", { class: "grow" }, input),
        h("button", { type: "button", class: "btn sm", onclick: () => rename(t, input.value) }, "Save"),
        h("button", { type: "button", class: "btn ghost sm", onclick: () => { st.editing = null; render(); } }, "Cancel"));
    }
    return h("div", { class: "trow" + (sel ? " sel" : "") },
      h("label", { class: "tpick" }, h("input", { type: "radio", name: "saved-target", checked: sel, onchange: () => onSelect(t) }),
        h("span", { class: "grow" }, h("span", { class: "tname" }, t.name), h("span", { class: "chips" }, targetSummary(t)))),
      h("button", { type: "button", class: "btn ghost icon-only sm", "aria-label": `Rename ${t.name}`, title: "Rename", onclick: () => { st.editing = t.id; render(); } }, icon("edit")),
      h("button", { type: "button", class: "btn ghost icon-only sm", "aria-label": `Delete ${t.name}`, title: "Delete", onclick: () => remove(t) }, icon("trash")));
  }

  // typing in the paste box re-evaluates the Save button without rebuilding the box (it would lose focus)
  function render2() { const b = el.querySelector(".save-target"); if (b) b.disabled = st.busy || !ready(); }
  function render() {
    if (!st.list && !st.error) return replace(el, h("span", { class: "muted small" }, h("span", { class: "spinner" }), " Loading"));
    replace(el,
      st.error ? errorRetry(`Saved targets could not be loaded: ${st.error}`, refresh) : null,   // the form below keeps working
      st.list?.length ? h("div", { class: "tlist", role: "group", "aria-label": "Saved targets" }, st.list.map(row)) : null,
      ts.open || st.error ? form() : h("button", { type: "button", class: "btn sm", onclick: () => { ts.open = true; render(); } }, icon("plus"), "New target"));
  }

  render();
  refresh();
  return { el, refresh, render, isEmpty: () => Array.isArray(st.list) && st.list.length === 0 };
}
