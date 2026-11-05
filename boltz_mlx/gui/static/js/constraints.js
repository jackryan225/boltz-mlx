// Predict's "Constraints & templates" panel: pocket, contact and bond constraints, residue modifications and templates, written by
// the server into the same Boltz-2 YAML a hand-written file would hold (gui/constraints.py checks every chain, residue and atom).
// State (kept with the Predict draft): st.cx = {open, items: [{key, kind, ...text fields}]}; the payload sends only complete,
// valid rows, and problems() keeps Start off while a row is wrong.
import { h, icon, replace } from "./dom.js";
import { fileButton, nextId, readFileText } from "./ui.js";

export const KINDS = [["pocket", "Pocket"], ["contact", "Contact"], ["bond", "Bond"], ["mod", "Modification"], ["template", "Template"]];
const TEMPLATE_LIMIT = 32 * 1024 * 1024;
const HELP = [
  ["Pocket", "the binder (usually a ligand) sits within the distance of these residues."],
  ["Contact", "two residues on different chains come within the distance."],
  ["Bond", "a covalent bond between two atoms, e.g. a cysteine SG and a ligand atom (A:25:SG and L:C1)."],
  ["Modification", "replaces one residue by a modified one: SEP (phospho-Ser), TPO (phospho-Thr), PTR (phospho-Tyr), MSE (seleno-Met)."],
  ["Template", "a known structure (PDB or mmCIF) of a protein chain for the model to start from; one chain of the file is used."],
];

export const emptyCx = () => ({ open: false, items: [] });
const fresh = (kind) => ({ key: nextId("cx"), kind, binder: "", res: "", a: "", b: "", max: "", chain: "", pos: "", ccd: "", tchain: "", file: null, want: "" });

// Boltz YAML structures (an imported YAML, a run's inputs) -> panel rows. constraints: [{pocket|contact|bond: {...}}];
// templates: [{file | filename, content?, chain_id, template_id}]; mods: [{chain, position, ccd}].
export function cxFrom({ constraints = [], templates = [], mods = [] }) {
  const tok = (t) => (Array.isArray(t) ? t.join(":") : "");
  const items = [];
  for (const c of constraints || []) {
    const [kind, b] = Object.entries(c || {})[0] || [];
    if (!b) continue;
    const md = b.max_distance != null ? String(b.max_distance) : "";
    if (kind === "pocket") items.push({ ...fresh("pocket"), binder: String(b.binder ?? ""), res: (b.contacts || []).map(tok).join(", "), max: md });
    else if (kind === "contact") items.push({ ...fresh("contact"), a: tok(b.token1), b: tok(b.token2), max: md });
    else if (kind === "bond") items.push({ ...fresh("bond"), a: tok(b.atom1), b: tok(b.atom2) });
  }
  for (const m of mods || []) items.push({ ...fresh("mod"), chain: String(m.chain), pos: String(m.position ?? ""), ccd: String(m.ccd ?? "") });
  for (const t of templates || []) {
    const name = t.filename || t.file || "";
    const file = t.content != null ? { kind: "file", filename: name, content: t.content, size: t.content.length } : null;
    items.push({ ...fresh("template"), chain: String(t.chain_id ?? ""), tchain: t.template_id ? String(t.template_id) : "", file, want: file ? "" : name });
  }
  return { open: items.length > 0, items };
}

// chains(): the form's chains [{id, type, sequence, smiles, ccd}]; extraIds(): {id: form chain id} for copies the server named;
// onChange(): an edit that changes the payload; error(): the server's message when it is about this panel.
export function cxPanel({ st, chains, extraIds = () => ({}), onChange, error = () => null }) {
  if (!st.cx || !Array.isArray(st.cx.items)) st.cx = emptyCx();
  const refs = new Map();   // item -> error element
  const elBody = h("div", { class: "stack tight cx-body" });
  const elCount = h("span", { class: "chip", hidden: true });
  const elErr = h("div", { class: "notice bad", role: "alert", hidden: true });
  const help = h("div", { class: "prov small muted", hidden: true }, HELP.map(([k, v]) => h("p", null, h("b", null, k), ": ", v)),
    h("p", null, "They guide the model; they are not forced during sampling (boltz-mlx does not run upstream's steering potentials)."));
  const infoBtn = h("button", { type: "button", class: "btn ghost icon-only sm", "aria-label": "About constraints and templates", "aria-expanded": "false", title: "What these do",
    onclick: (e) => { e.preventDefault(); help.hidden = !help.hidden; infoBtn.setAttribute("aria-expanded", String(!help.hidden)); } }, icon("info"));
  const det = h("details", { class: "adv cx", open: st.cx.open || undefined, ontoggle: (e) => { st.cx.open = e.target.open; } },
    h("summary", null, "Constraints & templates", elCount, infoBtn), help, elErr, elBody);

  // ---------------------------------------------------------------------------------------- parsing (the server checks again)
  const seqLen = (c) => String(c.sequence || "").replace(/^>.*$/gm, "").replace(/[^A-Za-z]/g, "").length;
  function chainOf(id) {
    const all = chains();
    const c = all.find((x) => x.id.trim() === id) || all.find((x) => x.id.trim() === extraIds()[id]);
    return c || null;
  }
  // "A:45" (polymer) or "L:C1" (ligand atom) -> [chain, residue | atom]
  function token(text) {
    const m = /^([A-Za-z0-9]{1,4})\s*:\s*([A-Za-z0-9']{1,4})$/.exec(text.trim());
    if (!m) return { err: `"${text.trim()}": write chain:residue, e.g. A:45` };
    const c = chainOf(m[1]);
    if (!c) return { err: `${text.trim()}: there is no chain ${m[1]}` };
    if (c.type === "ligand") return /^\d+$/.test(m[2]) ? { err: `${text.trim()}: chain ${m[1]} is a ligand; name an atom (${m[1]}:C1)` } : { v: [m[1], m[2].toUpperCase()] };
    if (!/^\d+$/.test(m[2])) return { err: `${text.trim()}: the residue is a number` };
    const n = Number(m[2]), len = seqLen(c);
    if (n < 1 || (len && n > len)) return { err: `${m[1]}:${n} is outside chain ${m[1]} (1–${len})` };
    return { v: [m[1], n] };
  }
  // "A:45:SG", "L:1:C1" or "L:C1" -> [chain, residue, atom]
  function atom(text) {
    const t = text.trim();
    let m = /^([A-Za-z0-9]{1,4})\s*:\s*(\d+)\s*:\s*([A-Za-z0-9']{1,4})$/.exec(t);
    if (!m) {
      const l = /^([A-Za-z0-9]{1,4})\s*:\s*([A-Za-z][A-Za-z0-9']{0,3})$/.exec(t);
      if (l && chainOf(l[1])?.type === "ligand") m = [t, l[1], "1", l[2]];
    }
    if (!m) return { err: `"${t}": write chain:residue:atom, e.g. A:25:SG (a ligand: L:C1)` };
    const c = chainOf(m[1]);
    if (!c) return { err: `${t}: there is no chain ${m[1]}` };
    const n = Number(m[2]);
    if (c.type === "ligand" && n !== 1) return { err: `${t}: a ligand is residue 1` };
    const len = c.type === "ligand" ? 1 : seqLen(c);
    if (n < 1 || (len && n > len)) return { err: `${m[1]}:${n} is outside chain ${m[1]} (1–${len})` };
    return { v: [m[1], n, m[3].toUpperCase()] };
  }
  const dist = (s) => {
    if (!String(s).trim()) return { v: null };
    const d = Number(s);
    return Number.isFinite(d) && d >= 4 && d <= 20 ? { v: d } : { err: "Distance: 4–20 Å" };
  };
  const blank = (it) => (it.kind === "pocket" ? !it.res.trim() : it.kind === "contact" || it.kind === "bond" ? !it.a.trim() && !it.b.trim()
    : it.kind === "mod" ? !it.pos.trim() && !it.ccd.trim() : !it.file && !it.want);
  // {v: payload part} | {err} | {skip} for one row
  function parse(it) {
    if (blank(it)) return { skip: true };
    if (it.kind === "pocket") {
      if (!it.binder || !chainOf(it.binder)) return { err: "Choose the binder" };
      const toks = it.res.split(/[,;\s]+/).filter(Boolean).map(token);
      const bad = toks.find((x) => x.err);
      if (bad) return bad;
      if (toks.some((x) => x.v[0] === it.binder)) return { err: `Pocket residues are on the other chains, not on ${it.binder}` };
      const d = dist(it.max);
      if (d.err) return d;
      return { v: { pocket: { binder: it.binder, contacts: toks.map((x) => x.v), ...(d.v != null ? { max_distance: d.v } : {}) } } };
    }
    if (it.kind === "contact") {
      const a = token(it.a), b = token(it.b);
      if (a.err || b.err) return a.err ? a : b;
      if (a.v[0] === b.v[0]) return { err: "The two residues must be on different chains" };
      const d = dist(it.max);
      if (d.err) return d;
      return { v: { contact: { token1: a.v, token2: b.v, ...(d.v != null ? { max_distance: d.v } : {}) } } };
    }
    if (it.kind === "bond") {
      const a = atom(it.a), b = atom(it.b);
      if (a.err || b.err) return a.err ? a : b;
      return { v: { bond: { atom1: a.v, atom2: b.v } } };
    }
    if (it.kind === "mod") {
      const c = chainOf(it.chain);
      if (!c || c.type === "ligand") return { err: "Choose a protein, DNA or RNA chain" };
      const n = Number(it.pos), len = seqLen(c);
      if (!/^\d+$/.test(it.pos.trim()) || n < 1 || (len && n > len)) return { err: `Position: a residue number from 1 to ${len || "the chain length"}` };
      if (!/^[A-Za-z0-9]{1,5}$/.test(it.ccd.trim())) return { err: "CCD code: 1–5 letters or digits (SEP, TPO, MSE …)" };
      return { v: { chain: c.id.trim(), position: n, ccd: it.ccd.trim().toUpperCase() } };
    }
    const c = chainOf(it.chain);
    if (!c || c.type !== "protein") return { err: "Choose the protein chain it is a template for" };
    if (!it.file || it.file.lost) return { err: `Attach ${it.file?.filename || it.want || "the structure file"}${it.file?.lost ? " again" : ""}` };
    return { v: { filename: it.file.filename, content: it.file.content, chain_id: c.id.trim(), template_id: it.tchain.trim() || null } };
  }

  // ---------------------------------------------------------------------------------------- rows
  const edited = (it) => { const r = refs.get(it); if (r) { const p = parse(it); r.hidden = !p.err; r.textContent = p.err || ""; } count(); onChange(); };
  const text = (it, k, label, ph, extra = {}) => h("input", { type: "text", value: it[k], placeholder: ph, "aria-label": label, spellcheck: "false", autocomplete: "off",
    oninput: (e) => { it[k] = e.target.value; edited(it); }, ...extra });
  const chainSel = (it, k, label, ok, first = "Chain") => {
    const opts = chains().filter(ok);
    if (it[k] && !opts.some((c) => c.id.trim() === it[k])) it[k] = "";
    if (!it[k] && opts.length === 1) it[k] = opts[0].id.trim();
    return h("select", { "aria-label": label, onchange: (e) => { it[k] = e.target.value; edited(it); } },
      h("option", { value: "", selected: !it[k] }, first), opts.map((c) => h("option", { value: c.id.trim(), selected: c.id.trim() === it[k] }, `${c.id.trim()} · ${c.type === "ligand" ? "ligand" : c.type}`)));
  };
  function row(it) {
    const n = st.cx.items.filter((x) => x.kind === it.kind).indexOf(it) + 1;
    const name = KINDS.find(([k]) => k === it.kind)[1];
    const tag = `${name} ${n}`;
    const err = h("div", { class: "ferr", role: "alert", hidden: true });
    refs.set(it, err);
    const rm = h("button", { type: "button", class: "btn ghost icon-only sm", "aria-label": `Remove ${tag.toLowerCase()}`, title: "Remove",
      onclick: () => { st.cx.items.splice(st.cx.items.indexOf(it), 1); render(); onChange(); } }, icon("trash"));
    const dField = () => h("label", { class: "row tight small cx-d" }, "≤", text(it, "max", `${tag} max distance`, "6", { inputmode: "decimal", class: "cx-num" }), "Å");
    let fields;
    if (it.kind === "pocket") {
      if (!it.binder) it.binder = chains().find((c) => c.type === "ligand")?.id.trim() || "";
      fields = [chainSel(it, "binder", `${tag} binder`, () => true, "Binder"), text(it, "res", `${tag} residues`, "Pocket residues: A:45, A:67, A:102", { class: "grow mono" }), dField()];
    } else if (it.kind === "contact") {
      fields = [text(it, "a", `${tag} first residue`, "A:45", { class: "mono cx-tok" }), h("span", { class: "muted small" }, "to"), text(it, "b", `${tag} second residue`, "B:12", { class: "mono cx-tok" }), dField()];
    } else if (it.kind === "bond") {
      fields = [text(it, "a", `${tag} first atom`, "A:25:SG", { class: "mono cx-tok" }), h("span", { class: "muted small" }, "to"), text(it, "b", `${tag} second atom`, "L:C1", { class: "mono cx-tok" })];
    } else if (it.kind === "mod") {
      fields = [chainSel(it, "chain", `${tag} chain`, (c) => c.type !== "ligand"), h("label", { class: "row tight small" }, "Residue", text(it, "pos", `${tag} position`, "45", { inputmode: "numeric", class: "cx-num" })),
        h("label", { class: "row tight small" }, "becomes", text(it, "ccd", `${tag} CCD code`, "SEP", { class: "mono cx-ccd", maxlength: 5 }))];
    } else {
      const f = it.file;
      fields = [chainSel(it, "chain", `${tag} chain`, (c) => c.type === "protein", "Protein chain"),
        f && !f.lost ? h("span", { class: "chip info" }, icon("file"), f.filename) : f?.lost || it.want ? h("span", { class: "chip warn" }, icon("alert"), `${f?.filename || it.want} needed`) : null,
        fileButton({ label: f && !f.lost ? "Change" : "Choose PDB / mmCIF", accept: ".pdb,.ent,.cif,.mmcif", onFile: async (file) => {
          const r = await readFileText(file, TEMPLATE_LIMIT, "The structure file");
          if (r) { it.file = r; it.want = ""; render(); onChange(); } } }),
        h("label", { class: "row tight small", title: "Which chain of the file to use (default: its first protein chain)" }, "File chain", text(it, "tchain", `${tag} chain in the file`, "first", { class: "mono cx-ccd", maxlength: 4 }))];
    }
    const el = h("div", { class: "cx-row" }, h("span", { class: "cx-tag small" }, tag), h("div", { class: "row tight cx-f" }, fields), rm, err);
    const p = parse(it);
    err.hidden = !p.err; err.textContent = p.err || "";
    return el;
  }
  function count() {
    const n = st.cx.items.filter((it) => !blank(it)).length;
    elCount.hidden = !n; elCount.textContent = String(n);
  }
  function render() {
    refs.clear();
    replace(elBody, st.cx.items.map(row),
      h("div", { class: "add-row" }, KINDS.map(([k, t]) => h("button", { type: "button", class: "btn sm", disabled: st.cx.items.length >= 40 || undefined,
        onclick: () => { st.cx.items.push(fresh(k)); render(); onChange(); const rows = elBody.querySelectorAll(".cx-row"); rows[rows.length - 1]?.querySelector("select, input")?.focus(); } }, icon("plus"), t))),
      h("p", { class: "small muted" }, "Anything else (MSA file paths, steering potentials …): Import a Boltz YAML; it runs as written."));
    count(); showError();
  }
  // re-checks every row in place (a chain's sequence or id changed) without rebuilding the inputs
  function check() { for (const it of st.cx.items) { const r = refs.get(it); if (r) { const p = parse(it); r.hidden = !p.err; r.textContent = p.err || ""; } } count(); }
  function showError() { const e = error(); elErr.hidden = !e; replace(elErr, e ? [icon("alert"), h("div", null, e)] : []); if (e && !det.open) { det.open = true; st.cx.open = true; } }

  const parsed = () => st.cx.items.map((it) => [it, parse(it)]);
  return {
    el: det, render, check, showError,
    payload() {
      const out = { constraints: [], templates: [], mods: {} };
      for (const [it, p] of parsed()) {
        if (!p.v) continue;
        if (it.kind === "template") out.templates.push(p.v);
        else if (it.kind === "mod") (out.mods[p.v.chain] ||= []).push({ position: p.v.position, ccd: p.v.ccd });
        else out.constraints.push(p.v);
      }
      return out;
    },
    problems: () => parsed().filter(([, p]) => p.err).map(([it, p]) => `${KINDS.find(([k]) => k === it.kind)[1]}: ${p.err}`),
    lost: () => st.cx.items.filter((it) => it.kind === "template" && (it.file?.lost || (!it.file && it.want))).map((it) => it.file || { filename: it.want, lost: true }),
  };
}
