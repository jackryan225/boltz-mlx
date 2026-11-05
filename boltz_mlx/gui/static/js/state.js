// Shared application state and navigation (kept out of main.js so views can import it without an import cycle).
export const app = { info: null, unmount: null, theme: "light", keepScroll: false };

// Drafts of the forms, kept across route changes (looking at a run must not wipe the next experiment) and across reloads
// (localStorage). An uploaded file over 400 kB is not stored: its draft keeps the name with `lost: true` and the page asks for the
// file again (Start stays off). Clear and a started job discard the draft for good: a discarded draft is never written back.
export const drafts = {};
const DKEY = (k) => "boltz-draft-" + k;
const BIG = 400000;
export function loadDraft(key) {
  try { const t = localStorage.getItem(DKEY(key)); return t ? JSON.parse(t) : null; } catch (e) { return null; }
}
// Persistence status: the page header shows "Draft not saved" (or "Files not kept") while a write fails.
export const draftStatus = { level: "ok" };   // ok | files | none
function setStatus(level) {
  draftStatus.level = level;
  for (const el of document.querySelectorAll("[data-draft-warn]")) {
    el.hidden = level === "ok";
    el.textContent = level === "files" ? "Files not kept" : "Draft not saved";
    el.title = level === "files" ? "Browser storage is full: uploaded files will be asked for again after a reload" : "Browser storage is full or blocked: a reload loses this form";
  }
}
function replacer(skip, strip) {
  return (k, v) => (skip.includes(k) || k === "discarded" ? undefined
    : v && v.kind === "file" && typeof v.content === "string" && (strip || v.content.length > BIG) ? { ...v, content: null, lost: true } : v);
}
export function saveDraft(key, st, skip = []) {
  if (!st || st.discarded) return true;
  for (const strip of [false, true]) {
    try {
      localStorage.setItem(DKEY(key), JSON.stringify(st, replacer(skip, strip)));
      setStatus(strip ? "files" : "ok");
      return true;
    } catch (e) { /* storage full or blocked: try again without the files */ }
  }
  try { localStorage.removeItem(DKEY(key)); } catch (e) { /* ignore */ }   // never leave an older draft that a reload would restore
  setStatus("none");
  return false;
}
// "Use inputs" replaces the draft: the previous one is kept for one Undo (also across a reload) until the new one is dismissed or started.
const UKEY = (k) => "boltz-draft-undo-" + k;
const undoSnap = {};
export function stashDraft(key, skip = []) {
  let snap = null;
  try { snap = drafts[key] ? JSON.parse(JSON.stringify(drafts[key], replacer(skip, false))) : loadDraft(key); } catch (e) { snap = null; }
  undoSnap[key] = { draft: snap, mem: true };   // mem: made in this page load, so its element keys are still unique
  try { localStorage.setItem(UKEY(key), JSON.stringify({ draft: snap })); } catch (e) { /* the in-memory copy still works until a reload */ }
}
export function hasUndo(key) {
  if (undoSnap[key]) return true;
  try { const t = localStorage.getItem(UKEY(key)); if (t) { undoSnap[key] = JSON.parse(t); return true; } } catch (e) { /* ignore */ }
  return false;
}
export function takeUndo(key) {
  const u = hasUndo(key) ? undoSnap[key] : null;
  clearUndo(key);
  return u ? { draft: u.draft, mem: !!u.mem } : { draft: null, mem: false };
}
export function clearUndo(key) {
  delete undoSnap[key];
  try { localStorage.removeItem(UKEY(key)); } catch (e) { /* ignore */ }
}
export function dropDraft(key) {
  if (drafts[key]) drafts[key].discarded = true;
  drafts[key] = null;
  clearUndo(key);
  try { localStorage.removeItem(DKEY(key)); } catch (e) { /* ignore */ }
  setStatus("ok");
}
// A stored file whose content was too big to keep (or is gone): {kind, filename, size, lost}; anything else comes back as it was.
export const lostFile = (f) => (f && f.kind === "file" && f.content == null ? { kind: "file", filename: f.filename, size: f.size, lost: true } : f || null);

// Connection state: views report poll failures here; main.js shows "Disconnected · last updated ..." until a call works again.
export const conn = { ok: true, last: Date.now(), listeners: new Set(),
  set(ok) {
    if (ok) this.last = Date.now();
    if (ok !== this.ok) { this.ok = ok; for (const f of this.listeners) f(ok); }
  } };

export function go(hash) {
  if (location.hash === hash) window.dispatchEvent(new HashChangeEvent("hashchange"));
  else location.hash = hash;
}
