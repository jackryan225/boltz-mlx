// Form widgets with their keyboard and screen-reader behaviour: radio and tab groups (arrow keys, roving tab stop, linked panel),
// numeric fields validated in place (the input is never replaced while it has focus), upload buttons and drop zones, copy.
import { fmt, h, icon, toast } from "./dom.js";

let uid = 0;
export const nextId = (p = "u") => `${p}${++uid}`;

// A polite live region for status changes (validation results, filter counts) that must not move focus.
export function announce(text) {
  const el = document.getElementById("status");
  if (!el) return;
  el.textContent = "";
  setTimeout(() => { el.textContent = text; }, 30);
}

function arrowKeys(e, btns, pick) {
  const k = e.key;
  if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(k)) return;
  const live = btns.filter((b) => !b.disabled);
  const i = live.indexOf(document.activeElement);
  if (i < 0) return;
  e.preventDefault();
  const n = k === "Home" ? 0 : k === "End" ? live.length - 1 : (i + (k === "ArrowLeft" || k === "ArrowUp" ? -1 : 1) + live.length) % live.length;
  live[n].focus();
  pick(live[n]);
}

// items: [[value, label, disabled?, title?]]. A single-choice group: role=radiogroup, arrow keys move and select.
const slug = (t) => String(t).toLowerCase().replace(/[^a-z0-9]+/g, "-");

export function radioSeg({ label, items, value, onChange, cls = "" }) {
  const wrap = h("div", { class: "seg " + cls, role: "radiogroup", "aria-label": label, "data-seg": slug(label) });
  const btns = items.map(([v, t, dis, title]) => h("button", { type: "button", role: "radio", "data-v": v, disabled: dis || undefined, title, onclick: () => pick(v) }, t));
  function set(v) {
    value = v;
    for (const b of btns) { const on = b.dataset.v === String(v); b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; }
    if (!btns.some((b) => b.tabIndex === 0) && btns[0]) btns.find((b) => !b.disabled).tabIndex = 0;
  }
  function pick(v) {
    if (v === value) return;
    const had = wrap.contains(document.activeElement);
    set(v); onChange(v);
    if (had && !wrap.isConnected) setTimeout(() => document.querySelector(`[data-seg="${slug(label)}"] [data-v="${v}"]`)?.focus(), 0);  // the page rebuilt the group
  }
  wrap.append(...btns);
  wrap.addEventListener("keydown", (e) => arrowKeys(e, btns, (b) => pick(b.dataset.v)));
  set(value);
  wrap.set = set;
  return wrap;
}

// Tabs with one visible panel: role=tablist / tab / tabpanel, aria-controls, arrow keys. items: [[value, label]].
export function tabset({ label, items, value, onChange, cls = "seg" }) {
  const id = "tp-" + slug(label);
  const bar = h("div", { class: cls, role: "tablist", "aria-label": label });
  const panel = h("div", { role: "tabpanel", id: id + "-panel", tabindex: "-1" });
  const btns = items.map(([v, t]) => h("button", { type: "button", role: "tab", id: `${id}-${v}`, "aria-controls": id + "-panel", "data-v": v, onclick: () => pick(v) }, t));
  function set(v) {
    value = v;
    for (const b of btns) { const on = b.dataset.v === v; b.setAttribute("aria-selected", String(on)); b.tabIndex = on ? 0 : -1; }
    panel.setAttribute("aria-labelledby", `${id}-${v}`);
  }
  function pick(v) {
    if (v === value) return;
    const had = bar.contains(document.activeElement);
    set(v); onChange(v);
    if (had && !bar.isConnected) setTimeout(() => document.getElementById(`${id}-${v}`)?.focus(), 0);  // the page rebuilt the tabs
  }
  bar.append(...btns);
  bar.addEventListener("keydown", (e) => arrowKeys(e, btns, (b) => pick(b.dataset.v)));
  set(value);
  return { bar, panel, set };
}

// ------------------------------------------------------------------------------------------------ numbers
// Whole-number check that never substitutes a default: blank is allowed only when `optional` (then the preset value applies).
export function intCheck(raw, { min = -Infinity, max = Infinity, optional = false } = {}) {
  const s = String(raw ?? "").trim();
  if (s === "") return optional ? { ok: true, value: null } : { ok: false, error: "Required" };
  if (!/^[+-]?\d+$/.test(s)) return { ok: false, error: "Whole number" };
  const v = Number(s);
  if (v < min || v > max) return { ok: false, error: min === -Infinity ? `≤ ${max}` : max === Infinity ? `≥ ${min}` : `${min}–${max}` };
  return { ok: true, value: v };
}

// A labelled number field. The input is created once; validation only toggles aria-invalid and the message text.
// opts: label, value, min, max, optional, placeholder, title, disabled, onInput(rawString, check).
export function numField(o) {
  const id = nextId("nf");
  const err = h("span", { class: "ferr", id: id + "-e", role: "alert", hidden: true });
  const input = h("input", { type: "text", inputmode: "numeric", autocomplete: "off", id, value: o.value ?? "", placeholder: o.placeholder, title: o.title, disabled: o.disabled || undefined,
    "aria-describedby": id + "-e",
    oninput: (e) => { const c = check(); o.onInput?.(e.target.value, c); } });
  function check() {
    const c = intCheck(input.value, o);
    input.setAttribute("aria-invalid", String(!c.ok));
    err.hidden = c.ok;
    err.textContent = c.ok ? "" : c.error;
    return c;
  }
  const el = h("label", { class: "field num-field", for: id }, h("span", null, o.label), input, err);
  if (String(o.value ?? "") !== "") check();
  return { el, input, check, set(v) { input.value = v ?? ""; check(); }, setPlaceholder(p) { input.placeholder = p; } };
}

// ------------------------------------------------------------------------------------------------ files
// A real button (keyboard and screen readers) plus a hidden input. onFile(File).
export function fileButton({ label, accept, onFile, cls = "btn sm", ic = "upload", title }) {
  const input = h("input", { type: "file", accept, hidden: true, tabindex: "-1" });
  input.addEventListener("change", () => { if (input.files[0]) onFile(input.files[0]); input.value = ""; });
  const btn = h("button", { type: "button", class: cls, title, onclick: () => input.click() }, ic ? icon(ic) : null, label);
  return h("span", { class: "filebtn" }, btn, input);
}

export function dropzone({ accept, label, hint, onFile }) {
  const box = h("div", { class: "drop", role: "group", "aria-label": label },
    fileButton({ label, accept, onFile }), h("span", { class: "muted" }, hint || "or drop it here"));
  for (const ev of ["dragenter", "dragover"]) box.addEventListener(ev, (e) => { e.preventDefault(); box.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) box.addEventListener(ev, (e) => { e.preventDefault(); box.classList.remove("over"); });
  box.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); });
  return box;
}

export async function copyText(text, what = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
    toast(what);
  } catch (e) {
    toast("Copy failed: the browser blocked clipboard access", "bad");
  }
}

// Reads a File as text, enforcing a size limit; null (and a toast) when it cannot be used.
export async function readFileText(file, limit, what) {
  if (limit && file.size > limit) {
    toast(`${what} is too large (${(file.size / 1048576).toFixed(1)} MB)`, "bad");
    return null;
  }
  try {
    return { kind: "file", filename: file.name, content: await file.text(), size: file.size };
  } catch (e) {
    toast(`Could not read ${file.name}`, "bad");
    return null;
  }
}

// A compact confirmation dialog: a title, key/value rows, Cancel and a confirm button. Resolves true / false.
export function confirmDialog({ title, rows = [], ok = "OK" }) {
  return new Promise((resolve) => {
    const dlg = h("dialog", { class: "confirm", "aria-labelledby": "cf-title" });
    const done = (v) => { dlg.close(); dlg.remove(); resolve(v); };
    dlg.append(
      h("div", { class: "dlg-h" }, h("h2", { id: "cf-title" }, title)),
      h("div", { class: "dlg-b" }, h("dl", { class: "kv" }, rows.flatMap(([k, v]) => [h("dt", null, k), h("dd", { class: "num" }, v)]))),
      h("div", { class: "dlg-f" }, h("button", { type: "button", class: "btn", onclick: () => done(false) }, "Cancel"), h("button", { type: "button", class: "btn primary", onclick: () => done(true) }, ok)));
    dlg.addEventListener("cancel", (e) => { e.preventDefault(); done(false); });
    document.body.appendChild(dlg);
    dlg.showModal();
    dlg.querySelector(".btn.primary").focus();
  });
}

// ------------------------------------------------------------------------------------------------ log
// A log disclosure that is created once and then only fed: new lines are appended in place (a selection and the scroll position
// stay), the view follows the end only while it is already at the end. `fetchFull()` loads the longer tail when it is opened.
export function logPanel({ open = false, fetchFull = null } = {}) {
  const pre = h("pre", { class: "logbox", id: "logbox" }, "(empty)");
  let text = "";
  const det = h("details", { class: "acc log", open: open || undefined }, h("summary", null, "Log"), h("div", { class: "acc-b" }, pre));
  const atEnd = () => pre.scrollHeight - pre.scrollTop - pre.clientHeight < 6;
  function set(next) {
    next = String(next || "");
    if (next === text) return;
    const follow = atEnd();
    if (text && next.startsWith(text)) pre.append(document.createTextNode(next.slice(text.length)));
    else pre.textContent = next || "(empty)";
    text = next;
    if (follow) pre.scrollTop = pre.scrollHeight;
  }
  det.addEventListener("toggle", async () => {
    if (!det.open || !fetchFull) return;
    try { set(await fetchFull()); pre.scrollTop = pre.scrollHeight; } catch (e) { /* the connection indicator shows problems */ }
  });
  return { el: det, set };
}


// ------------------------------------------------------------------------------------------------ export menu
// One menu per result header: [{label, onClick, disabled?, title?}] or {sep: "Group"} headings. Closes after a pick, on Escape and
// when anything else is clicked.
export function exportMenu(items, { label = "Export" } = {}) {
  const pop = h("div", { class: "menu-pop", role: "menu" });
  const det = h("details", { class: "menu" }, h("summary", { class: "btn sm" }, icon("download"), label), pop);
  const fill = () => {
    pop.replaceChildren(...items().map((it) => (it.sep ? h("div", { class: "eyebrow" }, it.sep)
      : h("button", { type: "button", role: "menuitem", class: "menu-item", disabled: it.disabled || undefined, title: it.title,
        onclick: () => { det.open = false; it.onClick(); } }, it.label))));
  };
  det.addEventListener("toggle", () => {
    if (!det.open) return;
    fill();
    const off = (e) => { if (!det.contains(e.target)) { det.open = false; } };
    const key = (e) => { if (e.key === "Escape") { det.open = false; det.querySelector("summary").focus(); } };
    document.addEventListener("click", off, true);
    document.addEventListener("keydown", key);
    det.addEventListener("toggle", () => { if (!det.open) { document.removeEventListener("click", off, true); document.removeEventListener("keydown", key); } }, { once: true });
  });
  return det;
}

// ------------------------------------------------------------------------------------------------ attachments
// One row per file a YAML names: state mark, size, and an Attach / Replace button. files: [{filename, content, size, lost?}];
// onChange(newFiles) after a pick.
export function fileSlots({ needs, files, limit, onChange }) {
  const row = (name) => {
    const f = files.find((x) => x.filename === name);
    const ok = f && !f.lost;
    return h("div", { class: "row tight small slot" }, ok ? icon("check") : icon("alert"), h("code", null, name),
      ok ? h("span", { class: "muted" }, fmt.bytes(f.size)) : f ? h("span", { class: "bad-t" }, "not restored") : h("span", { class: "muted" }, "needed"),
      h("span", { class: "grow" }),
      fileButton({ label: f ? "Replace" : "Attach", accept: /\.(a3m|csv)$/i.test(name) ? ".a3m,.csv" : ".cif,.mmcif,.pdb,.ent", cls: "btn sm", ic: null,
        onFile: async (file) => {
          const r = await readFileText(file, limit, `The file for ${name}`);
          if (!r) return;
          r.filename = name;
          onChange([...files.filter((x) => x.filename !== name), r]);
        } }));
  };
  return h("div", { class: "stack tight" }, needs.map(row));
}

// ------------------------------------------------------------------------------------------------ layout
// A page with a bottom action bar (Start): the bar is fixed on narrow screens, so the document keeps room for it. The bar's height
// is measured and published as --bar-h (padding at the end of the page and scroll-padding for focused fields).
export function reserveBar(bar) {
  document.body.classList.add("has-bar");
  const set = () => document.documentElement.style.setProperty("--bar-h", bar.offsetHeight + "px");
  const ro = window.ResizeObserver ? new ResizeObserver(set) : null;
  ro?.observe(bar);
  set();
  return () => { ro?.disconnect(); document.body.classList.remove("has-bar"); document.documentElement.style.removeProperty("--bar-h"); };
}

// Keeps Tab inside `box` while it is a modal sheet.
export function trapTab(e, box) {
  if (e.key !== "Tab" || !box) return;
  const f = [...box.querySelectorAll('a[href], button:not([disabled]), input:not([disabled]), select, textarea, summary, [tabindex]:not([tabindex="-1"])')].filter((x) => x.offsetParent !== null);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && (document.activeElement === first || !box.contains(document.activeElement))) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && (document.activeElement === last || !box.contains(document.activeElement))) { e.preventDefault(); first.focus(); }
}

// A compact error with Retry, for a read that failed inside a view that otherwise works.
export function errorRetry(message, retry) {
  return h("div", { class: "notice bad" }, icon("alert"), h("div", { class: "grow" }, message), retry ? h("button", { type: "button", class: "btn sm", onclick: retry }, icon("refresh"), "Retry") : null);
}
