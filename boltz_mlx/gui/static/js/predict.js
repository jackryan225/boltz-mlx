// Predict: one complex (proteins, DNA / RNA, ligands, optional affinity) as chains in rows, settings, and a sticky Run panel.
// The job runs as `boltz-mlx predict` (gui/predict.py). The form is a draft kept in memory across route changes (state.js).
import { api } from "./api.js";
import { cxFrom, cxPanel, emptyCx } from "./constraints.js";
import { debounce, fmt, h, icon, replace, store, toast } from "./dom.js";
import { guiKeyOf, pageHead, presetBlock, presetNumbers, undoNotice } from "./parts.js";
import { drafts, dropDraft, go, loadDraft, lostFile, saveDraft } from "./state.js";
import { describe, searchBox } from "./search.js";
import { announce, confirmDialog, fileButton, fileSlots, intCheck, nextId, numField, readFileText, reserveBar } from "./ui.js";

const TYPES = [["protein", "Protein"], ["dna", "DNA"], ["rna", "RNA"], ["ligand", "Ligand"]];
const PLACEHOLDER = { protein: "Sequence or FASTA", dna: "DNA sequence", rna: "RNA sequence" };
const LOOKUP_PH = { protein: "Find a protein: gene, name or ID (EGFR, thrombin, P00533, 8CPK)", dna: "Find by PDB id (8CPK_B)", rna: "Find by PDB id (8CPK_B)" };
const ID_RE = /^[A-Za-z0-9]{1,4}$/;
// MSA (multiple sequence alignment: related sequences that tell the model which residues co-evolve). Without one the model sees
// the sequence alone: faster, less accurate. Online asks the public ColabFold server (api.colabfold.com) for it, which means the
// sequence leaves this Mac; a file is an alignment made elsewhere.
export const MSA_CHOICES = [["empty", "None (faster, less accurate)"], ["server", "Online: ColabFold (sends the sequence)"], ["file", "File (.a3m / .csv)"]];
const MSA_HELP = "MSA: related sequences that improve accuracy. None: this sequence alone. Online: fetched from the public ColabFold server, which receives the sequence. File: an alignment you already have.";
const SEED = { min: -2147483648, max: 2147483647 };
const MAX_CHAINS = 26;
const TRANSIENT = ["val", "busy", "starting", "valSeq", "yamlOpen", "discarded", "revert"];
const CX_ERR = /^(pocket|contact|bond|template) \d+:|^(chain \S+: )?(modification \d+:|residue \d+ is modified)|^(constraints|templates|each constraint)/i;

function fastaRecords(text) {
  const out = [];
  let head = null, buf = [];
  for (const line of String(text).split(/\r?\n/)) {
    if (line.startsWith(">")) { if (head !== null) out.push({ head, seq: buf.join("") }); head = line.slice(1).trim(); buf = []; }
    else if (head !== null) buf.push(line.trim());
  }
  if (head !== null) out.push({ head, seq: buf.join("") });
  return out.filter((r) => r.seq.replace(/\s/g, ""));
}

export async function mount(root, app, arg) {
  const info = app.info || {};
  const key = "predict";
  let presets = [];
  let gui = [];
  try {
    const r = await api.get("/api/presets");
    presets = r.presets;
    gui = r.gui || [];
  } catch (e) {
    toast(e.message, "bad");
  }
  const saved = store("predict-options") || {};
  const freshChain = (type, id) => ({ key: nextId("ch"), type, id, copies: "1", sequence: "", cyclic: false, msa: { mode: "empty", file: null }, ligMode: "smiles", smiles: "", ccd: "", lookQ: "", look: null });
  const fresh = () => ({ name: "", chains: [freshChain("protein", "A")], affinity: false, binder: "", note: "", unsupported: [], ack: false, yamlOpen: false, adv: null, useYaml: false, cx: emptyCx(),
    o: { preset: (saved.v === 2 && saved.preset) || info.default_predict_preset || "standard", steps: "", recycling: "", affinity_steps: "", affinity_samples: "", samples: saved.samples ?? "1", seed: saved.seed ?? "0" } });
  function restore() {  // the draft of the last session: fresh element keys; an uploaded file that was too big to keep comes back as "lost" (its name, to attach again)
    const d = loadDraft(key);
    if (!d || !Array.isArray(d.chains) || !d.chains.length) return null;
    const base = fresh();
    d.chains = d.chains.map((c) => ({ ...freshChain(c.type || "protein", c.id || ""), ...c, key: nextId("ch"), look: null, copies: String(c.copies ?? 1), msa: c.msa?.file ? { mode: "file", file: lostFile(c.msa.file) } : { mode: ["file", "server"].includes(c.msa?.mode) ? c.msa.mode : "empty", file: null } }));
    if (d.adv) d.adv = { ...d.adv, files: (d.adv.files || []).map(lostFile) };
    if (d.cxIn) { d.cx = cxFrom(d.cxIn); delete d.cxIn; }   // "Use inputs" of a run: its constraints, modifications and templates
    d.cx = d.cx && Array.isArray(d.cx.items) ? { ...d.cx, items: d.cx.items.map((it) => ({ ...it, key: nextId("cx"), file: lostFile(it.file) })) } : emptyCx();
    d.o = { ...base.o, ...(d.o || {}) };
    return { ...base, ...d };
  }
  const st = drafts[key] || (drafts[key] = restore() || fresh());
  if (st.cxIn) { st.cx = cxFrom(st.cxIn); delete st.cxIn; }   // "Use inputs" of a run, in this page load
  if (!st.cx?.items) st.cx = emptyCx();
  Object.assign(st, { val: null, busy: false, starting: false, valSeq: -1 });
  st.o.preset = guiKeyOf(st.o.preset, gui, info.default_predict_preset || "standard");
  const curGui = () => gui.find((g) => g.key === st.o.preset);
  const curCli = () => presets.find((p) => p.name === curGui()?.cli);
  let alive = true;
  let seq = 0;
  const ligands = () => st.chains.filter((c) => c.type === "ligand");
  const copiesOf = (c) => { const n = Number(String(c.copies ?? "1").trim() || "1"); return Number.isInteger(n) && n >= 1 && n <= MAX_CHAINS ? n : 1; };
  const copiesBad = (c) => !/^\d{1,2}$/.test(String(c.copies ?? "1").trim() || "1") || Number(c.copies || 1) < 1 || Number(c.copies || 1) > MAX_CHAINS;
  // One line per problem with a chain id: 1-4 letters or digits, and unique (the server compares case-insensitively).
  const idProblem = (c) => {
    const id = c.id.trim();
    if (!id) return "Chain ID needed";
    if (!ID_RE.test(id)) return "Chain ID: 1–4 letters or digits";
    return st.chains.some((x) => x !== c && x.id.trim().toLowerCase() === id.toLowerCase()) ? `Chain ID ${id} is used twice` : "";
  };
  const chainProblem = (c) => idProblem(c) || (copiesBad(c) ? `Copies: a whole number from 1 to ${MAX_CHAINS}` : "");
  const nextId$ = () => {
    const used = new Set(st.chains.map((c) => c.id.toLowerCase()));
    for (let k = 65; k <= 90; k++) if (!used.has(String.fromCharCode(k).toLowerCase())) return String.fromCharCode(k);
    return "Z" + st.chains.length;
  };
  if (arg === "affinity" && !st.affinity) {  // the #/predict/affinity link: the same draft with the affinity pass switched on
    if (!ligands().length) st.chains.push(freshChain("ligand", nextId$()));
    st.affinity = true;
  }

  // ------------------------------------------------------------------------------------------ settings and payload
  const AFF = ["affinity_steps", "affinity_samples"];
  const SPEC = { steps: { min: 1, max: 1000 }, recycling: { min: 0, max: 20 }, affinity_steps: { min: 1, max: 1000 },
    affinity_samples: { min: 1, max: 25 }, samples: { min: 1, max: 25 }, seed: SEED };
  const checks = () => Object.fromEntries(Object.entries(SPEC).filter(([k]) => st.affinity || !AFF.includes(k)).map(([k, s]) => [k, intCheck(st.o[k], s)]));
  const settingsOk = () => Object.values(checks()).every((c) => c.ok);
  const persist = () => store("predict-options", { v: 2, preset: st.o.preset, samples: st.o.samples, seed: st.o.seed });
  function payload() {
    const c = checks();
    const val = (k) => (c[k]?.ok ? c[k].value : null);  // invalid entries are never replaced by defaults in a started job (Start stays off)
    // A number the user left at the preset's value is sent as "the preset's" (null): only then do the preset's own rules apply
    // (Fast runs 200 structure steps for a ligand with < 10 heavy atoms); a changed number is explicit and always wins.
    const own = (k) => { const v = val(k); return v !== null && String(v) === presetNumbers(curCli(), [k])[k] ? null : v; };
    const options = { preset: st.o.preset, steps: own("steps"), recycling: own("recycling"), affinity_steps: st.affinity ? own("affinity_steps") : null,
      affinity_samples: st.affinity ? own("affinity_samples") : null, samples: val("samples") ?? 1, seed: val("seed") ?? 0 };
    if (st.useYaml && st.adv) {   // advanced input: the YAML as supplied, with its companion files
      return { kind: "predict", name: st.name.trim() || undefined, options,
        yaml: { content: st.adv.content, files: st.adv.files.filter((f) => !f.lost).map((f) => ({ filename: f.filename, content: f.content })) } };
    }
    const x = cx.payload();
    return {
      kind: "predict", name: st.name.trim() || autoName() || undefined,
      chains: st.chains.map((ch) => {
        const d = { type: ch.type, id: ch.id.trim() };
        if (ch.type !== "ligand" && x.mods[ch.id.trim()]) d.modifications = x.mods[ch.id.trim()];
        if (copiesOf(ch) > 1) d.copies = copiesOf(ch);
        if (ch.type === "ligand") { if (ch.ligMode === "ccd") d.ccd = ch.ccd.trim(); else d.smiles = ch.smiles.trim(); } else d.sequence = ch.sequence;
        if (ch.type === "protein") {
          d.cyclic = ch.cyclic;
          d.msa = ch.msa.mode === "file" && ch.msa.file && !ch.msa.file.lost ? { mode: "file", filename: ch.msa.file.filename, content: ch.msa.file.content } : { mode: ch.msa.mode === "server" ? "server" : ch.msa.mode === "file" ? "file" : "empty" };
        }
        return d;
      }),
      affinity: st.affinity ? { binder: st.binder || ligands()[0]?.id || "" } : null,
      constraints: x.constraints.length ? x.constraints : undefined,
      templates: x.templates.length ? x.templates : undefined,
      options,
    };
  }
  const empty = () => !st.useYaml && st.chains.every((c) => (c.type === "ligand" ? !(c.smiles.trim() || c.ccd.trim()) : !c.sequence.trim()));
  // Every edit invalidates the last check at once (seq moves on); only a reply to the newest payload counts, and Start needs it.
  const persistDraft = debounce(() => saveDraft(key, st, TRANSIENT), 400);
  const lostFiles = () => (st.useYaml && st.adv ? st.adv.files : [...st.chains.filter((c) => c.type === "protein" && c.msa.mode === "file").map((c) => c.msa.file), ...cx.lost()]).filter((f) => f?.lost);
  const validate = debounce(async () => {
    if (empty() || st.chains.some(chainProblem)) { st.val = null; st.busy = false; st.valSeq = seq; apply(); return; }
    const my = seq;
    st.busy = true; renderPlan();
    let r;
    try {
      r = await api.post("/api/predict/validate", payload());
    } catch (e) {
      r = { ok: false, error: e.message };
    }
    if (my !== seq || !alive) return;
    st.val = r; st.busy = false; st.valSeq = my;
    apply();
    announce(st.val.ok ? `Valid, ${st.val.n_tokens} tokens` : st.val.error);
  }, 300);
  const stale = () => st.valSeq !== seq;
  const changed = () => { seq++; cx.check(); validate(); renderPlan(); preset.refresh(); persistDraft(); nameIn.placeholder = autoName() || "Run name"; };

  // Which chain a validation error belongs to: "chain A: ..." (by id, else by 1-based index).
  function errorChain() {
    const e = st.val && !st.val.ok ? st.val.error || "" : "";
    if (CX_ERR.test(e)) return null;   // shown in the Constraints & templates panel
    const m = /^chain (\S+?):\s*(.*)$/i.exec(e);
    if (!m) { const t = /^chain id (\S+) is used twice/i.exec(e); return t ? { chain: st.chains.find((c) => c.id.toLowerCase() === t[1].toLowerCase()), msg: e } : null; }
    const c = st.chains.find((x) => x.id.toLowerCase() === m[1].toLowerCase()) || st.chains[Number(m[1]) - 1];
    return c ? { chain: c, msg: `Chain ${c.id}: ${m[2]}` } : null;
  }

  // ------------------------------------------------------------------------------------------ chain rows
  const elChains = h("div", { class: "chain-list" });
  const rowRefs = new Map();  // chain -> {meta, err, field}
  const elChainCount = h("span", { class: "chip" });
  const elNote = h("div");

  // Several FASTA records pasted into one row become one chain each, all of that row's type (protein, DNA or RNA).
  function splitFasta(c, ta) {
    const text = ta.value;
    if (!st.chains.includes(c)) return false;
    const recs = fastaRecords(text);
    if (recs.length < 2) return false;
    if (st.chains.length - 1 + recs.length > MAX_CHAINS) { toast(`${recs.length} records do not fit: at most ${MAX_CHAINS} chains`, "bad"); return false; }
    const taken = () => new Set(st.chains.map((x) => x.id.toLowerCase()));
    const idOf = (head) => { const t = (head.split(/[|\s]/)[0] || ""); return /^[A-Za-z0-9]$/.test(t) && !taken().has(t.toLowerCase()) ? t : nextId$(); };
    c.sequence = recs[0].seq;
    ta.value = c.sequence;  // the row is about to be replaced; a late change event must not split the old text again
    const extra = [];
    recs.slice(1).forEach((r, i) => { const n = freshChain(c.type, idOf(r.head)); n.sequence = r.seq; st.chains.splice(st.chains.indexOf(c) + 1 + i, 0, n); extra.push(n); });
    const unit = c.type === "protein" ? "aa" : "nt";
    const len = (x) => x.sequence.replace(/[^A-Za-z]/g, "").length;
    st.note = `FASTA split into ${recs.length} ${c.type === "protein" ? "" : c.type.toUpperCase() + " "}chains: ${[c, ...extra].map((x) => `${x.id} (${len(x)} ${unit})`).join(", ")}`;
    renderChains(); renderNote(); changed();
    announce(st.note);
    return true;
  }

  // Find a sequence: the type-ahead search (search.js) lists UniProt / PDB matches as the user types; a UniProt result carries its
  // sequence, a PDB chain (or an isoform) is fetched by id (/api/lookup), which also offers the entry's bound small molecules.
  async function doLookup(c, q) {
    if (!q || c.look?.busy) return;
    c.look = { busy: true };
    renderChains();
    try {
      const r = await api.post("/api/lookup", { query: q, type: c.type });
      if (!st.chains.includes(c) || !alive) return;
      c.sequence = r.sequence;
      named(c, `${r.title} (${r.id})`);
      const pdb = r.source === "RCSB";
      c.look = { ok: `${pdb ? "PDB" : r.source} ${r.id} · ${r.title}${r.note ? ` · ${r.note}` : ""} · ${r.length} ${c.type === "protein" ? "aa" : "nt"}${pdb && !(r.ligands || []).length ? " · no bound ligands" : ""}`,
        ligands: r.ligands || [], copies: r.copies > 1 ? Math.min(r.copies, MAX_CHAINS) : 0, entry: r.id };
      announce(`Sequence of ${r.id} loaded`);
      renderChains(); changed();
    } catch (e) {
      if (!st.chains.includes(c) || !alive) return;
      c.look = { err: e.message };
      renderChains();
    }
  }
  function pickResult(c, r) {
    c.lookQ = r.id;
    if (!r.sequence) return doLookup(c, r.id);
    c.sequence = r.sequence;
    named(c, `${r.gene || r.name} (${r.id})`);
    c.look = { ok: describe(r) };
    announce(`Sequence of ${r.id} loaded`);
    renderChains(); changed();
  }
  function addLigandFrom(c, l) {
    const n = freshChain("ligand", nextId$());
    n.smiles = l.smiles;
    named(n, l.id);
    st.chains.push(n);
    c.look = { ...c.look, ligands: c.look.ligands.filter((x) => x !== l) };
    renderChains(); renderAffinity(); changed();
    announce(`Ligand ${l.id} added as chain ${n.id}`);
  }
  // The run's default name: what was looked up ("CA2 (P00918) + ligand"), while those chains still hold what was found.
  const named = (c, name) => { c.lookName = name.slice(0, 60); c.lookSeq = c.type === "ligand" ? c.smiles : c.sequence; };
  function autoName() {
    const found = (c) => c.lookName && (c.type === "ligand" ? c.ligMode !== "ccd" && c.smiles === c.lookSeq : c.sequence === c.lookSeq);
    const names = st.chains.filter((c) => c.type !== "ligand" && found(c)).map((c) => c.lookName);
    if (!names.length) return "";
    const ligs = ligands(), known = ligs.filter(found).map((c) => c.lookName);
    const rest = ligs.length - known.length;
    return [...names, ...known, ...(rest ? [rest === 1 ? "ligand" : `${rest} ligands`] : [])].join(" + ").slice(0, 80);
  }
  function setCopies(c, n) {
    c.copies = String(n);
    renderChains(); changed();
    announce(`Chain ${c.id}: ${n} copies`);
  }
  function lookupRow(c) {
    const sb = searchBox({ type: () => c.type, value: c.lookQ || "", placeholder: LOOKUP_PH[c.type], label: `Chain ${c.id}: find a sequence`,
      onInput: (v) => { c.lookQ = v; persistDraft(); }, onPick: (r) => pickResult(c, r) });
    const l = c.look || {};
    return h("div", { class: "lookup" },
      h("div", { class: "row tight" }, sb.el, l.busy ? h("span", { class: "spinner", "aria-label": "Loading" }) : null),
      l.err ? h("div", { class: "ferr", role: "alert" }, l.err) : l.ok ? h("div", { class: "small muted look-ok" }, icon("check"), " ", l.ok) : null,
      // A homo-oligomer (HIV protease, hemoglobin ...): the entry holds several copies of this chain; one click puts them all in
      l.copies && copiesOf(c) !== l.copies ? h("div", { class: "row tight small look-lig" }, `PDB ${l.entry} has ${l.copies} copies of this chain`,
        h("button", { type: "button", class: "chip chip-btn", title: "Model the complex as in the entry (a binding site can sit between the copies)", onclick: () => setCopies(c, l.copies) }, icon("plus"), `Use ${l.copies} copies`)) : null,
      l.ligands?.length ? h("div", { class: "row tight small look-lig" }, "Add ligand", l.ligands.map((x) => h("button", { type: "button", class: "chip chip-btn", title: `${x.name}\n${x.smiles}`, onclick: () => addLigandFrom(c, x) }, icon("plus"), x.id))) : null);
  }

  function copiesRow(c) {
    const binder = st.affinity && ligands().length && c.id === st.binder && c.type === "ligand";
    const inp = h("input", { type: "text", inputmode: "numeric", class: "copies", value: binder ? "1" : String(c.copies ?? "1"), disabled: binder || undefined, "aria-label": `Chain ${c.id} copies`,
      title: binder ? "The affinity ligand is a single copy" : "Identical copies of this chain (a homo-oligomer): one YAML entity with several ids",
      oninput: (e) => { c.copies = e.target.value.trim(); checkIds(); changed(); } });
    return h("label", { class: "row tight small copies-l" }, "Copies", inp);
  }

  function chainRow(c, i) {
    const errId = c.key + "-err";
    const err = h("div", { class: "ferr", id: errId, role: "alert", hidden: true });
    const idErr = h("div", { class: "ferr", role: "alert", hidden: true });
    const meta = h("div", { class: "meta" });
    const idIn = h("input", { type: "text", class: "cid", value: c.id, maxlength: 4, "aria-label": `Chain ${i + 1} ID`, spellcheck: "false", autocomplete: "off", title: "Chain ID: 1–4 letters or digits, unique",
      oninput: (e) => { c.id = e.target.value.trim(); checkIds(); renderAffinity(); cx.render(); changed(); } });
    const typeSel = h("select", { class: "ctype", "aria-label": `Chain ${i + 1} type`, onchange: (e) => { c.type = e.target.value; c.look = null; renderChains(); renderAffinity(); changed(); } },
      TYPES.map(([v, t]) => h("option", { value: v, selected: c.type === v }, t)));
    const rm = h("button", { type: "button", class: "btn ghost icon-only rm", "aria-label": `Remove chain ${c.id}`, title: "Remove", disabled: st.chains.length < 2,
      onclick: () => { st.chains.splice(st.chains.indexOf(c), 1); renderChains(); renderAffinity(); changed(); } }, icon("trash"));
    let field, body;
    if (c.type === "ligand") {
      const ph = c.ligMode === "ccd" ? "CCD code" : "SMILES";
      field = h("input", { type: "text", class: "mono", id: c.key + "-f", value: c.ligMode === "ccd" ? c.ccd : c.smiles, placeholder: ph, maxlength: c.ligMode === "ccd" ? 5 : undefined, spellcheck: "false",
        "aria-label": `Chain ${c.id} ${ph}`, "aria-describedby": errId, oninput: (e) => { if (c.ligMode === "ccd") c.ccd = e.target.value; else c.smiles = e.target.value; changed(); } });
      body = h("div", { class: "cbody" }, h("div", { class: "lig" },
        h("select", { "aria-label": "Ligand given as", onchange: (e) => { c.ligMode = e.target.value; renderChains(); changed(); } },
          h("option", { value: "smiles", selected: c.ligMode !== "ccd" }, "SMILES"), h("option", { value: "ccd", selected: c.ligMode === "ccd" }, "CCD")), field), idErr, err, h("div", { class: "row copts" }, copiesRow(c)));
    } else {
      field = h("textarea", { rows: 2, id: c.key + "-f", spellcheck: "false", placeholder: PLACEHOLDER[c.type], "aria-label": `Chain ${c.id} ${c.type} sequence`, "aria-describedby": errId,
        oninput: (e) => { c.sequence = e.target.value; changed(); }, onpaste: (e) => { const ta = e.target; setTimeout(() => splitFasta(c, ta), 0); }, onchange: (e) => splitFasta(c, e.target) });
      field.value = c.sequence;
      body = h("div", { class: "cbody" }, lookupRow(c), field, idErr, err,
        h("div", { class: "row copts" },
          c.type === "protein" ? [h("label", { class: "row tight small", title: MSA_HELP }, "MSA",
            h("select", { "aria-label": "MSA", title: MSA_HELP, onchange: (e) => { c.msa.mode = e.target.value; renderChains(); changed(); } },
              MSA_CHOICES.map(([v, t]) => h("option", { value: v, selected: (["file", "server"].includes(c.msa.mode) ? c.msa.mode : "empty") === v }, t)))),
          c.msa.mode === "file" ? (c.msa.file && !c.msa.file.lost
            ? h("span", { class: "chip info" }, icon("file"), c.msa.file.filename, " ", h("button", { type: "button", class: "link", onclick: () => { c.msa.file = null; renderChains(); changed(); } }, "remove"))
            : [c.msa.file ? h("span", { class: "chip warn", title: "Too large to keep with the draft" }, icon("alert"), `${c.msa.file.filename} not restored`) : null,
               fileButton({ label: c.msa.file ? "Attach again" : "Choose file", accept: ".a3m,.csv", onFile: async (f) => { const r = await readFileText(f, info.limits?.msa, "The MSA file"); if (r) { c.msa.file = r; renderChains(); changed(); } } })]) : null] : null,
          h("span", { class: "grow" }), copiesRow(c),
          c.type === "protein" ? h("label", { class: "check small" }, h("input", { type: "checkbox", checked: c.cyclic, onchange: (e) => { c.cyclic = e.target.checked; changed(); } }), "Cyclic") : null));
    }
    rowRefs.set(c, { meta, err, idErr, idIn, field, msa: body.querySelector('select[aria-label="MSA"]') });
    return h("div", { class: "chainrow" }, idIn, typeSel, body, meta, rm);
  }
  // Chain id / copies problems, shown beside the id (every row: one id's edit changes the others' "used twice").
  function checkIds() {
    for (const c of st.chains) {
      const r = rowRefs.get(c), msg = chainProblem(c);
      if (!r) continue;
      r.idErr.hidden = !msg;
      r.idErr.textContent = msg;
      r.idIn.setAttribute("aria-invalid", String(!!idProblem(c)));
    }
  }
  function renderChains() {
    rowRefs.clear();
    replace(elChains, h("div", { class: "chainhead", "aria-hidden": "true" }, h("span", null, "Chain ID"), h("span", null, "Type"), h("span", null, "Sequence or ligand")), st.chains.map(chainRow),
      h("div", { class: "add-row" }, TYPES.map(([v, t]) => h("button", { type: "button", class: "btn sm", disabled: st.chains.length >= 26,
        onclick: () => { st.chains.push(freshChain(v, nextId$())); renderChains(); renderAffinity(); changed(); document.getElementById(st.chains.at(-1).key + "-f")?.focus(); } }, icon("plus"), t))));
    checkIds();
    cx.render();
    apply(false);
  }
  // Updates sizes and error text in place (the inputs are not touched).
  function apply(plan = true) {
    const current = !stale();
    const vc = st.val?.ok && current ? st.val.chains : null;
    const ec = errorChain();
    st.chains.forEach((c, i) => {
      const r = rowRefs.get(c);
      if (!r) return;
      const v = vc?.[i];
      replace(r.meta, v
        ? [(v.type === "ligand" ? (v.ccd ? h("b", null, v.ccd) : h("b", null, `${v.n_atoms} atoms${v.mw ? ` · MW ${Math.round(v.mw)}` : ""}`))
          : h("b", null, `${v.length} ${v.type === "protein" ? "aa" : "nt"}`)), v.ids ? h("span", { title: `Chain IDs ${v.ids.join(", ")}` }, `× ${v.copies}: ${v.ids.join(", ")}`) : null]
        : c.type === "ligand" ? null : c.sequence ? h("span", null, `${c.sequence.replace(/[^A-Za-z]/g, "").length}`) : null);
      const bad = ec?.chain === c;
      r.err.hidden = !bad;
      r.err.textContent = bad ? ec.msg : "";
      r.field?.setAttribute("aria-invalid", String(!!bad));
    });
    cx.showError();
    elChainCount.textContent = st.val?.ok && current ? `${st.val.chains.length} chain${st.val.chains.length === 1 ? "" : "s"} · ${fmt.int(st.val.n_tokens)} tokens` : `${st.chains.length} chain${st.chains.length === 1 ? "" : "s"}`;
    if (plan) renderPlan();
  }
  function renderNote() {
    const un = st.unsupported?.length && !st.ack && !st.useYaml ? st.unsupported : null;
    replace(elNote,
      undoNotice(key, st, renderNote),
      st.note ? h("div", { class: "notice info" }, h("div", { class: "grow" }, st.note), st.revert ? h("button", { type: "button", class: "btn sm", onclick: undoEdit }, "Undo") : null,
        h("button", { type: "button", class: "btn ghost icon-only sm", "aria-label": "Dismiss", onclick: () => { st.note = ""; st.revert = null; renderNote(); } }, icon("x"))) : null,
      un ? h("div", { class: "notice warn", role: "alert" }, icon("alert"), h("div", null, `Not in the form: ${un.join(", ")}. `,
        st.adv ? [h("button", { type: "button", class: "link", onclick: () => { st.useYaml = true; renderNote(); renderChains(); renderAffinity(); changed(); } }, "Run the YAML as written"), " · "] : null,
        h("button", { type: "button", class: "link", onclick: () => { st.ack = true; st.adv = null; renderNote(); changed(); } }, "Run without them"))) : null);
    renderAdv();
  }

  // Back to the form: constraints, modifications and templates come along (a template takes the file attached to the YAML); what the
  // form cannot hold (MSA paths, steering-potential settings ...) is dropped; Undo restores the YAML.
  async function editInForm() {
    const lost = st.unsupported || [];
    if (lost.length && !(await confirmDialog({ title: "Edit in form", rows: [["Dropped", lost.join(", ")]], ok: "Drop and edit" }))) return;
    st.revert = { adv: st.adv, unsupported: lost, ack: st.ack, cx: JSON.parse(JSON.stringify(st.cx)) };
    for (const it of st.cx.items) {
      const f = it.kind === "template" && !it.file && it.want ? (st.adv?.files || []).find((x) => x.filename === it.want && !x.lost) : null;
      if (f) { it.file = { ...f }; it.want = ""; }
    }
    if (lost.length) st.note = `Dropped ${lost.join(", ")}`;
    st.useYaml = false; st.ack = true; st.adv = null;
    renderNote(); renderChains(); renderAffinity(); changed();
  }
  const undoEdit = () => {
    const r = st.revert;
    if (!r) return;
    st.revert = null; st.useYaml = true; st.adv = r.adv; st.unsupported = r.unsupported; st.ack = r.ack;
    if (r.cx) st.cx = r.cx;
    renderNote(); renderChains(); renderAffinity(); changed();
  };

  // Constraints & templates (constraints.js): written into the job's YAML by the server; extra copy ids come from the last check.
  const cx = cxPanel({ st, chains: () => st.chains,
    extraIds: () => Object.fromEntries((st.val?.ok ? st.val.chains : []).flatMap((c) => (c.ids || []).map((i) => [i, c.id]))),
    onChange: () => changed(),
    error: () => { const e = st.val && !st.val.ok && !stale() ? st.val.error || "" : ""; return CX_ERR.test(e) ? e : null; } });

  // Advanced input: the imported YAML runs as written (templates, constraints, modifications, MSA paths) with the companion files it names.
  const elAdv = h("div");
  function renderAdv() {
    const a = st.useYaml ? st.adv : null;
    elChains.hidden = elAff.hidden = cx.el.hidden = !!a;
    if (!a) return replace(elAdv);
    replace(elAdv, h("div", { class: "stack tight adv-in" },
      h("div", { class: "row tight" }, icon("file"), h("b", null, a.filename), h("span", { class: "chip info" }, "Run as written"), h("span", { class: "grow" }),
        h("button", { type: "button", class: "btn ghost sm", onclick: editInForm }, "Edit in form")),
      st.val?.ok ? h("div", { class: "chips" }, st.val.chains.map((c) => h("span", { class: "chip" }, `${c.id} · ${c.type === "ligand" ? (c.ccd || "ligand") : `${c.length} ${c.type === "protein" ? "aa" : "nt"}`}`))) : null,
      (a.needs || []).length ? fileSlots({ needs: a.needs, files: a.files, limit: info.limits?.msa, onChange: (files) => { a.files = files; renderAdv(); changed(); } }) : null));
  }

  // ------------------------------------------------------------------------------------------ affinity row
  const elAff = h("div", { class: "aff-row" });
  function renderAffinity() {
    const ligs = ligands();
    if (st.binder && !ligs.some((c) => c.id === st.binder)) st.binder = ligs[0]?.id || "";
    if (!ligs.length) st.affinity = false;
    if (st.affinity && !st.binder) st.binder = ligs[0]?.id || "";
    toggleAffFields();   // the affinity fields follow the checkbox however it changed (last ligand removed, type changed ...)
    replace(elAff,
      h("label", { class: "check", title: ligs.length ? "" : "Add a ligand chain first" },
        h("input", { type: "checkbox", checked: st.affinity, disabled: !ligs.length, onchange: (e) => { st.affinity = e.target.checked; if (st.affinity && !st.binder) st.binder = ligs[0]?.id || ""; renderAffinity(); renderChains(); changed(); } }), "Affinity"),
      st.affinity ? (ligs.length === 1 ? h("span", { class: "small" }, `Affinity for ${ligs[0].id}`)   // one ligand: nothing to choose
        : h("label", { class: "row tight small" }, "Binder",
          h("select", { "aria-label": "Affinity binder", onchange: (e) => { st.binder = e.target.value; renderAffinity(); renderChains(); changed(); } }, ligs.map((c) => h("option", { value: c.id, selected: c.id === st.binder }, c.id))))) : null,
      st.affinity ? h("span", { class: "small muted" }, "Value: log10(IC50 in µM), lower is stronger") : null);
  }

  // ------------------------------------------------------------------------------------------ settings
  // The fields hold the preset's numbers (real values, editable): picking a preset or Reset puts them back.
  const OVERRIDES = ["steps", "recycling", "affinity_steps", "affinity_samples"];
  const F = {};
  const fillNumbers = (force) => { const n = presetNumbers(curCli(), OVERRIDES); for (const k of OVERRIDES) if (n[k] !== undefined && (force || st.o[k] === "")) { st.o[k] = n[k]; F[k]?.set(n[k]); } };
  const preset = presetBlock({ gui, o: st.o, cli: curCli, affinity: () => st.affinity,
    onPick: (k) => { st.o.preset = k; fillNumbers(true); persist(); renderAffinity(); changed(); },   // a preset brings its own numbers
    onReset: () => { fillNumbers(true); changed(); } });
  fillNumbers(false);
  const nf = (k, label, extra = {}) => (F[k] = numField({ label, value: st.o[k], ...SPEC[k], onInput: (v) => { st.o[k] = v; if (k === "samples" || k === "seed") persist(); changed(); }, ...extra }));
  nf("steps", "Structure steps", { title: "Diffusion sampling steps" });
  nf("recycling", "Recycles", { title: "Trunk recycling passes" });
  nf("affinity_steps", "Affinity steps", { title: "Sampling steps of the affinity pass" });
  nf("samples", "Structure samples", { title: "Diffusion samples; model 0 is the best ranked" });
  nf("affinity_samples", "Affinity samples", { title: "Diffusion samples of the affinity pass" });
  nf("seed", "Seed", { title: "Same seed and inputs: same structure" });
  function toggleAffFields() { for (const k of AFF) if (F[k]) F[k].el.hidden = !st.affinity; }

  // ------------------------------------------------------------------------------------------ run panel
  const nameIn = h("input", { type: "text", class: "ph-name", maxlength: 80, placeholder: autoName() || "Run name", "aria-label": "Run name", value: st.name, oninput: (ev) => { st.name = ev.target.value; persistDraft(); } });
  const elSummary = h("dl", { class: "kv" });
  const elMsgs = h("div", { class: "stack tight" });
  const elStatus = h("span", { class: "ready small", role: "status" });
  const elStart = h("button", { type: "button", class: "btn primary lg", onclick: () => start() });
  const elYaml = h("div");
  const elPlan = h("div", { class: "plan-col" },
    h("aside", { class: "card plan", "aria-label": "Run" }, h("div", { class: "card-b stack" }, elSummary, elMsgs, elYaml)),
    h("div", { class: "startbar" }, elStatus, elStart));
  const goField = () => {
    const e = errorChain();
    if (!e) return;
    const row = rowRefs.get(e.chain);
    const msa = /MSA/.test(e.msg) ? row?.msa : null;   // an alignment error belongs to the MSA control, not the sequence
    (msa || document.getElementById(e.chain.key + "-f"))?.focus();
  };
  const shortWarn = (w) => w.replace(/^chain (\S+): /i, "$1 · ").replace(/no MSA \(single sequence\)/, "no MSA: less accurate (an MSA helps: see the MSA menu)");
  function blockers() {
    const out = [];
    const probs = app.info?.problems_by?.[st.affinity ? "affinity" : "predict"] || [];
    if (probs.length) out.push(probs[0].replace(/ (missing).*$/, " $1"));
    if (!settingsOk()) out.push("Fix settings");
    if (!st.useYaml && st.chains.some(chainProblem)) out.push("Fix chain IDs");
    if (!st.useYaml && cx.problems().length) out.push("Fix constraints");
    if (st.useYaml && st.adv) {
      const miss = (st.adv.needs || []).find((n) => !st.adv.files.some((f) => f.filename === n && !f.lost));
      if (miss) out.push(`Attach ${miss}`);
    } else if (st.unsupported?.length && !st.ack) out.push("Imported fields not supported");
    const lost = lostFiles()[0];
    if (lost) out.push(`Attach ${lost.filename} again`);
    if (empty()) out.push("Add a molecule");
    else if (st.val && !st.val.ok && !stale()) out.push("Input error");
    else if (stale() || st.busy || !st.val) out.push("Checking…");
    if (app.info?.active_run) out.push("Job running");
    return out;
  }
  function renderPlan() {
    const v = st.val?.ok && !stale() ? st.val : null;
    const e = v?.estimate;
    const m = v ? e?.memory : null;
    const tight = !!(m?.ram_gb && m.peak_gb > 0.7 * m.ram_gb);
    const why = blockers();
    const active = app.info?.active_run;
    const ready = !why.length && !st.starting;
    const g = curGui();
    const o = v?.options;
    const custom = preset.isCustom();
    replace(elSummary,
      h("dt", null, "Chains"), h("dd", null, v ? v.chains.map((x) => x.id).join(" · ") : st.chains.length),
      h("dt", null, "Tokens"), h("dd", { class: "num" }, v ? fmt.int(v.n_tokens) : "–"),
      h("dt", null, "Preset"), h("dd", null, (g?.label || st.o.preset) + (custom ? " · Custom" : "")),
      h("dt", null, "Structure"), h("dd", { class: "num" }, o ? `${o.steps} steps · ${o.recycling} recycles · ${o.samples} sample${o.samples === 1 ? "" : "s"}` : "–"),
      o?.steps_reason ? [h("dt", null, "Steps"), h("dd", { class: "small", title: e?.steps_note }, `Fast uses ${o.steps} structure steps: ${o.steps_reason}`)] : null,
      v?.affinity ? [h("dt", null, "Affinity"), h("dd", { class: "num" }, `${o.affinity_steps} steps × ${o.affinity_samples} samples`)] : null,
      v ? [h("dt", null, "Time (estimate)"), h("dd", { class: "num", title: e?.caveat }, e?.available ? fmt.roughDur(e.total_s).replace("about ", "~") : "–")] : null,
      m ? [h("dt", null, "Memory (estimate)"), h("dd", { class: "num" + (tight ? " warn-t" : ""), title: `${e.caveat}\n${m.basis}` }, `~${Math.round(m.peak_gb)} GB${m.ram_gb ? ` of ${m.ram_gb} GB` : ""}`)] : null);
    // The complete error sits beside its field; the summary only counts it. An error that belongs to no field is shown here in full.
    const bad = st.val && !st.val.ok && !stale() ? st.val.error : null;
    const chainErr = bad ? errorChain() : null;
    const warns = v?.warnings || [];
    replace(elMsgs,
      warns.length ? h("div", { class: "notice warn" }, icon("alert"), h("ul", null, warns.map((w) => h("li", null, shortWarn(w))))) : null,
      bad && !chainErr ? h("div", { class: "notice bad", role: "alert" }, icon("alert"), h("div", null, bad)) : null);
    const first = why[0];
    const nW = warns.length ? ` · ${warns.length} warning${warns.length === 1 ? "" : "s"}` : "";
    replace(elStatus,
      ready ? (e?.available ? `Ready · ${fmt.roughDur(e.total_s).replace("about ", "~")}` : "Ready") + nW
        : first === "Job running" ? ["Job running · ", h("a", { href: "#/run/" + encodeURIComponent(active) }, "Open")]
        : first === "Input error" && chainErr ? h("button", { type: "button", class: "link bad-t", onclick: goField }, first)
        : first);
    elStatus.className = "ready small" + (ready ? " ok" : bad ? " bad" : "");
    replace(elStart, st.starting ? h("span", { class: "spinner" }) : icon("play"), "Start");
    elStart.disabled = !ready;
    const det = elYaml.querySelector("details");
    if (det) st.yamlOpen = det.open;
    replace(elYaml, v ? h("details", { class: "adv", open: st.yamlOpen || undefined, ontoggle: (ev) => { st.yamlOpen = ev.target.open; } }, h("summary", null, "Input YAML"), h("pre", { class: "logbox yaml" }, v.yaml)) : null);
  }

  async function start() {
    if (st.starting || blockers().length) return;
    st.starting = true; renderPlan();
    try {
      const r = await api.post("/api/runs", payload());
      persistDraft.cancel();
      dropDraft(key);
      go("#/run/" + encodeURIComponent(r.id));
    } catch (e) {
      toast(e.message, "bad");
      st.starting = false; if (alive) renderPlan();
    }
  }

  // ------------------------------------------------------------------------------------------ import
  async function importFile(file) {
    const f = await readFileText(file, 2 * 1024 * 1024, "The file");
    if (!f) return;
    try {
      const r = await api.post("/api/predict/import", { content: f.content });
      st.chains = r.chains.map((c) => ({ ...freshChain(c.type, c.id), sequence: c.sequence || "", cyclic: !!c.cyclic, smiles: c.smiles || "", ccd: c.ccd || "", ligMode: c.ccd ? "ccd" : "smiles" }));
      st.cx = cxFrom({ constraints: r.constraints, templates: r.templates, mods: r.chains.flatMap((c) => (c.modifications || []).map((m) => ({ ...m, chain: c.id }))) });
      st.affinity = !!r.binder; st.binder = r.binder || "";
      st.unsupported = r.skipped || []; st.ack = false; st.useYaml = false;
      st.adv = st.unsupported.length && !f.content.trimStart().startsWith(">") ? { filename: file.name, content: f.content, files: [], needs: r.needs || [] } : null;
      st.note = `Imported ${file.name}: ${st.chains.length} chain${st.chains.length === 1 ? "" : "s"}`;
      renderChains(); renderAffinity(); toggleAffFields(); renderNote(); changed();
    } catch (e) {
      toast(e.message, "bad");
    }
  }

  // ------------------------------------------------------------------------------------------ assemble
  const panel = (title, body, ...right) => h("section", { class: "card" }, h("div", { class: "card-h" }, h("h2", null, title), right), body);
  const elForm = h("div", { class: "form-col" },
    panel("Molecules", h("div", { class: "card-b stack" }, elChains, elNote, elAdv, elAff, cx.el), elChainCount),
    panel("Settings", h("div", { class: "card-b stack" }, preset.el, h("div", { class: "num-grid" }, F.steps.el, F.recycling.el, F.samples.el, F.seed.el, F.affinity_steps.el, F.affinity_samples.el))));
  replace(root,
    pageHead("Predict", [
      fileButton({ label: "Import YAML/FASTA", accept: ".yaml,.yml,.fasta,.fa,.faa,.txt", onFile: importFile, cls: "btn sm", ic: "upload", title: "Fill this form from a Boltz YAML or a FASTA file" }),
      h("button", { type: "button", class: "btn ghost sm", title: "Empty the form", onclick: () => { persistDraft.cancel(); dropDraft(key); go(location.hash); } }, "Clear")], nameIn),
    h("div", { class: "work" }, elForm, elPlan));
  renderChains(); renderAffinity(); toggleAffFields(); renderNote(); renderPlan(); preset.refresh();
  if (!empty()) { seq++; validate(); renderPlan(); }
  const onInfo = () => { if (alive) renderPlan(); };
  window.addEventListener("infochange", onInfo);
  // Leaving the page keeps the draft only if it is still the live one: Clear and a started job have discarded it for good.
  const freeBar = reserveBar(elPlan.querySelector(".startbar"));
  return () => { alive = false; freeBar(); validate.cancel(); persistDraft.cancel(); if (!st.discarded && drafts[key] === st) saveDraft(key, st, TRANSIENT); window.removeEventListener("infochange", onInfo); };
}
