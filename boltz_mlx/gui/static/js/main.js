// App shell: hash router, theme, chip badge, About / privacy box, server status banner.
import { api } from "./api.js";
import { h, icon, replace, store } from "./dom.js";
import { copyText } from "./ui.js";
import * as predictView from "./predict.js";
import * as newScreen from "./new.js";
import * as runsView from "./runs.js";
import * as runView from "./run.js";
import * as leadoptView from "./leadopt.js";
import { app, conn } from "./state.js";
import { refreshSide, renderSide } from "./parts.js";

// ---------------------------------------------------------------------------------------------- theme
// Light or dark: the OS setting (prefers-color-scheme) until the app-bar toggle picks one; the pick is remembered.
export function applyTheme(t) {
  app.theme = t;
  document.documentElement.setAttribute("data-theme", t);
  const b = document.getElementById("btn-theme");
  if (b) {
    replace(b, icon(t === "dark" ? "sun" : "moon"));
    b.title = b.ariaLabel = t === "dark" ? "Switch to light mode" : "Switch to dark mode";
  }
  window.dispatchEvent(new CustomEvent("themechange", { detail: t }));
}

function initTheme() {
  const mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  const os = () => (mq && mq.matches ? "dark" : "light");
  const saved = store("theme");
  applyTheme(saved === "dark" || saved === "light" ? saved : os());
  mq?.addEventListener?.("change", () => { if (!store("theme")) applyTheme(os()); });
  document.getElementById("btn-theme").addEventListener("click", () => {
    const t = app.theme === "dark" ? "light" : "dark";
    store("theme", t);
    applyTheme(t);
  });
}

// ---------------------------------------------------------------------------------------------- about
function aboutDialog() {
  const info = app.info || {};
  const dlg = document.getElementById("about");
  const ck = info.checks || {};
  const mol = info.molstar || {};
  const chip = info.chip || {};
  const sOk = !!(ck.weights?.conf && ck.weights?.conf_cfg), aOk = !!(ck.weights?.aff && ck.weights?.aff_cfg), mOk = !!ck.cache?.mols;
  const status = (ok) => h("span", { class: "chip " + (ok ? "ok" : "bad") }, ok ? "found" : "missing");
  const path = (label, p) => h("div", { class: "row tight small" }, h("span", { class: "muted" }, label), h("code", { class: "path" }, p || ""),
    p ? h("button", { type: "button", class: "btn ghost sm", onclick: () => copyText(p, "Copied") }, icon("copy"), "Copy") : null);
  replace(
    dlg,
    h("div", { class: "dlg-h" }, h("h2", { id: "about-title" }, "Boltz-2 MLX ", h("span", { class: "muted num" }, info.version || "dev")),
      h("button", { type: "button", class: "btn ghost icon-only", "aria-label": "Close", onclick: () => dlg.close() }, icon("x"))),
    h("div", { class: "dlg-b" },
      h("dl", { class: "kv" },
        h("dt", null, "Chip"), h("dd", null, chip.name || "unknown"),
        h("dt", null, "Structure weights"), h("dd", null, status(sOk)),
        h("dt", null, "Affinity weights"), h("dd", null, status(aOk)),
        h("dt", null, "Molecule library"), h("dd", null, status(mOk))),
      h("details", { class: "adv" }, h("summary", null, "Paths"),
        h("div", { class: "stack tight paths" }, path("Weights", ck.weights?.dir), path("Molecules", ck.cache?.dir), path("Runs", info.runs_dir))),
      h("details", { class: "adv" }, h("summary", null, "Privacy"), h("p", { class: "small" }, info.privacy_gui || "")),
      h("details", { class: "adv" }, h("summary", null, "Presets"), h("div", { class: "stack tight small prov" },
        (info.gui_presets || []).filter((g) => g.enabled).map((g) => h("div", null, h("b", null, g.label), g.note ? h("p", { class: "muted" }, g.note) : null)))),
      h("details", { class: "adv" }, h("summary", null, "Licences"),
        h("p", { class: "small" }, `3D views: ${mol.name || "Mol*"} ${mol.version || ""} (MIT), bundled. `, mol.citation || ""),
        h("p", { class: "small" }, "2D depictions: RDKit. Model: Boltz-2 (MIT), run with Apple MLX."))),
  );
  dlg.showModal();
}

// ---------------------------------------------------------------------------------------------- banner, pip, chip badge
// One banner for the model setup: what is missing, by name, with the paths and commands in a disclosure (Start says the same, short).
export function setBanner(problems) {
  const b = document.getElementById("banner");
  if (!problems || !problems.length) {
    b.hidden = true;
    return;
  }
  const name = (p) => p.split(/ in |: /)[0];
  replace(b, h("details", null, h("summary", null, problems.map(name).join(" · "), h("span", { class: "link" }, " Setup")),
    h("div", { class: "stack tight" }, problems.map((p) => h("p", { class: "small" }, p)))));
  b.hidden = false;
}

// "Disconnected · last updated 14:02" while calls fail; cleared by the next call that works.
function showConn() {
  const el = document.getElementById("conn");
  if (conn.ok) { el.hidden = true; return; }
  const t = new Date(conn.last).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  el.textContent = `Disconnected · last updated ${t} · retrying`;
  el.hidden = false;
}
conn.listeners.add(showConn);

function renderChip() {
  const c = app.info?.chip;
  const el = document.getElementById("chip-badge");
  if (!c || !el) return;
  el.hidden = false;
  el.className = "chip-badge";
  el.title = c.note || c.name;
  replace(el, icon("chip"), h("span", null, c.name.replace(/^Apple /, "")));
}

async function loadInfo() {
  try {
    app.info = await api.get("/api/info");
  } catch (e) {
    app.info = app.info || {};
    return;
  }
  syncInfo(app.info);
}

// The banner, the active pip and the chip badge follow the server: only a value that changed is redrawn.
let shownSetup = null;
function syncInfo(i) {
  const key = JSON.stringify([i.problems || [], i.problems_by || {}, i.chip?.name || ""]);
  if (key !== shownSetup) {
    shownSetup = key;
    setBanner(i.problems || []);
    renderChip();
  }
  document.getElementById("active-pip").hidden = !i.active_run;
  document.getElementById("foot-version").textContent = i.version || "";
}

setInterval(async () => {
  if (document.hidden) return;
  try {
    const i = await api.get("/api/info");
    const prev = app.info;
    const changed = (prev?.active_run || null) !== (i.active_run || null);
    const setup = JSON.stringify(prev?.problems_by || []) !== JSON.stringify(i.problems_by || []);
    app.info = i;
    syncInfo(i);
    if (changed || setup) { window.dispatchEvent(new CustomEvent("infochange")); if (changed) refreshSide(); }
  } catch (e) {
    showConn();
  }
}, 5000);

// ---------------------------------------------------------------------------------------------- router
function parseHash() {
  const parts = (location.hash || "#/predict").replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  return { name: parts[0] || "predict", args: parts.slice(1) };
}

// Each navigation gets a number and its own container. A mount that finishes after a newer navigation started is disposed on the
// spot and never touches the page or app.unmount.
let routeGen = 0;
async function route() {
  const gen = ++routeGen;
  const r = parseHash();
  const view = document.getElementById("view");
  if (app.unmount) {
    try {
      app.unmount();
    } catch (e) {
      console.error(e);
    }
    app.unmount = null;
  }
  const holder = h("div", { class: "route" });
  replace(view, holder);
  const tab = r.name === "run" || r.name === "leadopt" ? "runs" : r.name === "new" ? "screen" : ["runs", "screen"].includes(r.name) ? r.name : "predict";
  const active = r.name === "run" || r.name === "leadopt" ? "run:" + r.args[0] : tab === "predict" ? (r.args[0] === "affinity" ? "affinity" : "predict") : tab;
  renderSide(active);
  for (const a of document.querySelectorAll(".tabs a")) {
    if (a.dataset.nav === tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  let off = null;
  try {
    if (r.name === "runs") off = await runsView.mount(holder, app);
    else if (r.name === "run" && r.args[0]) off = await runView.mount(holder, app, r.args[0], r.args[1]);
    else if (r.name === "leadopt" && r.args[0]) off = await leadoptView.mount(holder, app, r.args[0], (r.args[1] || "").split(",").filter(Boolean));
    else if (r.name === "screen" || r.name === "new") off = await newScreen.mount(holder, app);
    else off = await predictView.mount(holder, app, r.args[0]);
  } catch (e) {
    console.error(e);
    if (gen === routeGen) replace(holder, h("div", { class: "notice bad" }, h("div", null, "Something went wrong: ", e.message || String(e))));
  }
  if (gen !== routeGen) {   // a newer navigation owns the page: this mount is released at once
    try { if (typeof off === "function") off(); } catch (e) { console.error(e); }
    return;
  }
  app.unmount = typeof off === "function" ? off : null;
  if (app.keepScroll) app.keepScroll = false;   // the view restored its own position (Runs, coming back from a run)
  else window.scrollTo(0, 0);
  if (gen > 1) view.focus({ preventScroll: true });   // a screen-reader user lands on the new page, not on the old link
}

window.addEventListener("hashchange", route);
initTheme();
replace(document.getElementById("btn-about"), icon("info"));
document.getElementById("btn-about").addEventListener("click", aboutDialog);
document.getElementById("btn-about2").addEventListener("click", aboutDialog);
document.getElementById("about").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) e.currentTarget.close();
});
await loadInfo();
await route();
