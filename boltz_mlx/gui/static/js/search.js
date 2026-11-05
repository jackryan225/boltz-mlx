// Type-ahead protein search (POST /api/search, gui/lookup.py): a gene or protein name, a keyword, a UniProt accession or a PDB id
// -> a dropdown of matches, human Swiss-Prot entries first. Used by the Predict chain rows and the Screen "New target" form.
// The server contacts UniProt / RCSB only for what is typed here (>= 2 characters, 300 ms after the last key).
import { api } from "./api.js";
import { debounce, h, icon, replace } from "./dom.js";
import { nextId } from "./ui.js";

export const OFFLINE = "Search unavailable offline — paste a sequence or choose a file";
const NONE = "No matches. Try a gene (EGFR), a protein name (thrombin), a UniProt accession (P00533) or a PDB id (8CPK).";
const memo = new Map();   // "query|type" -> reply, for this page load

const unit = (r) => (r.type && r.type !== "protein" ? "nt" : "aa");
// One line about a chosen result, for under the box: "UniProt P00533 · Epidermal growth factor receptor · EGFR · Homo sapiens · 1210 aa".
export const describe = (r) => [r.source === "RCSB" ? `PDB ${r.id}` : `UniProt ${r.id}`, r.name, r.gene, r.organism, r.length ? `${r.length} ${unit(r)}` : null]
  .filter(Boolean).join(" · ");

function option(r, i, id, on, pick) {
  const sub = r.source === "RCSB"
    ? [`PDB ${r.entry}`, r.chains?.length ? `chain${r.chains.length > 1 ? "s" : ""} ${r.chains.join(", ")}` : null, r.organism]
    : [r.gene, r.organism, r.id, r.reviewed ? null : "unreviewed"];
  return h("div", { class: "search-opt" + (on ? " on" : ""), role: "option", id: `${id}-${i}`, "aria-selected": String(on),
    onmousedown: (e) => e.preventDefault(), onclick: () => pick(i) },
    h("div", { class: "so-1" }, h("b", { class: "so-name" }, r.name), h("span", { class: "grow" }), r.length ? h("span", { class: "num muted" }, `${r.length} ${unit(r)}`) : null),
    h("div", { class: "so-2 small muted" }, sub.filter(Boolean).join(" · ")));
}

// type(): "protein" | "dna" | "rna"; onPick(result): the page fills in the sequence (a result without one: ask /api/lookup);
// onInput(text): every keystroke, so the page can keep the text with its draft.
export function searchBox({ type = () => "protein", value = "", placeholder = "", label, onPick, onInput }) {
  const id = nextId("sr");
  const input = h("input", { type: "text", class: "search-in", value, placeholder, spellcheck: "false", autocomplete: "off", maxlength: 60,
    role: "combobox", "aria-autocomplete": "list", "aria-expanded": "false", "aria-controls": id, "aria-label": label });
  const pop = h("div", { class: "search-pop", id, role: "listbox", "aria-label": label, hidden: true });
  let items = [], active = -1, gen = 0;

  const close = () => { pop.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); };
  const show = (...kids) => { replace(pop, kids); pop.hidden = false; input.setAttribute("aria-expanded", "true"); };
  const note = (text, cls = "") => { items = []; active = -1; input.removeAttribute("aria-activedescendant"); show(h("div", { class: "search-note small " + cls, role: "status" }, text)); };
  function render() {
    show(items.map((r, i) => option(r, i, id, i === active, pick)));
    if (active >= 0) input.setAttribute("aria-activedescendant", `${id}-${active}`);
    pop.querySelector(".on")?.scrollIntoView?.({ block: "nearest" });
  }
  async function run() {
    const q = input.value.trim().replace(/\s+/g, " ");
    if (q.length < 2) return close();
    const key = `${q.toLowerCase()}|${type()}`;
    const my = ++gen;
    let r = memo.get(key);
    if (!r) {
      note([h("span", { class: "spinner" }), " Searching…"]);
      try {
        r = await api.post("/api/search", { query: q, type: type() });
        if (!r.offline) memo.set(key, r);
      } catch (e) {
        if (my === gen && document.activeElement === input) note(e.status === 0 ? OFFLINE : e.message, e.status === 0 ? "warn-t" : "bad-t");
        return;
      }
    }
    if (my !== gen || document.activeElement !== input) return;
    if (r.offline) return note(OFFLINE, "warn-t");
    items = r.results || [];
    if (!items.length) return note(r.message && r.message !== "no matches" ? r.message : NONE);
    active = 0;
    render();
  }
  const soon = debounce(run, 300);
  async function pick(i) {
    const r = items[i];
    if (!r) return;
    gen++; soon.cancel(); close();
    await onPick?.(r);
  }
  input.addEventListener("input", () => {
    onInput?.(input.value);
    gen++;
    if (input.value.trim().length < 2) { soon.cancel(); close(); } else soon();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (pop.hidden || !items.length) { soon.cancel(); run(); return; }
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      render();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (!pop.hidden && items.length) pick(active >= 0 ? active : 0);
      else { soon.cancel(); run(); }
    } else if (e.key === "Escape" && !pop.hidden) {
      e.preventDefault(); e.stopPropagation(); gen++; close();
    }
  });
  input.addEventListener("blur", () => setTimeout(() => { if (document.activeElement !== input) { gen++; close(); } }, 150));
  const el = h("div", { class: "search" }, icon("search", "icon search-ic"), input, pop);
  return { el, input, close, set(v) { input.value = v ?? ""; } };
}
