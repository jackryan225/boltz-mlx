// Past runs: every job in the runs folder (structure predictions and screens, also CLI screens), newest first.
import { api, enc } from "./api.js";
import { fmt, h, icon, replace } from "./dom.js";
import { presetLabel } from "./parts.js";
import { app } from "./state.js";
import { radioSeg } from "./ui.js";

// Filters and scroll position survive opening a run and coming back.
const view = { kind: "all", state: "all", q: "", dense: false, y: 0 };
const STATE_GROUP = { running: ["running", "stopping"], complete: ["complete"], stopped: ["stopped", "interrupted"], failed: ["failed"] };

export const STATE = {
  new: ["New", "", "Not started"],
  running: ["Running", "info live", "Running"],
  stopping: ["Stopping", "warn live", "Stopping"],
  complete: ["Complete", "ok", "Complete"],
  stopped: ["Stopped", "warn", "Stopped"],
  interrupted: ["Interrupted", "warn", "Interrupted"],
  failed: ["Failed", "bad", "Failed"],
};

export function stateChip(state) {
  const [label, kind] = STATE[state] || [state, ""];
  return h("span", { class: "chip " + kind }, h("span", { class: "dot" }), label);
}

// "Stopped · 3/10 complete" for a screen, "Stopped · restart required" for a prediction (it has no partial results).
export function stateLine(r) {
  const [label] = STATE[r.state] || [r.state];
  if (r.kind === "predict") return ["stopped", "interrupted"].includes(r.state) ? `${label} · restart required` : label;
  const done = r.n_done ?? 0, tot = r.n_ligands ?? done;
  return ["stopped", "interrupted", "running", "stopping"].includes(r.state) ? `${label} · ${fmt.int(done)}/${fmt.int(tot)} complete` : label;
}

export async function mount(root) {
  let timer = null;
  let alive = true;
  let all = [];
  const list = h("div", { class: "runs-list" + (view.dense ? " dense" : "") });
  const segHost = h("div");
  const count = h("span", { class: "small muted num", role: "status" });

  const kindOf = (r) => r.kind || "screen";
  function matches(r, ignoreKind) {
    if (!ignoreKind && view.kind !== "all" && kindOf(r) !== view.kind) return false;
    if (view.state !== "all" && !(STATE_GROUP[view.state] || []).includes(r.state)) return false;
    const q = view.q.trim().toLowerCase();
    return !q || [r.name, r.target, r.chains, r.id].some((x) => String(x || "").toLowerCase().includes(q));
  }
  function renderSeg() {
    const n = (k) => all.filter((r) => matches(r, true) && (k === "all" || kindOf(r) === k)).length;
    replace(segHost, radioSeg({ label: "Show", value: view.kind, items: [["all", `All ${n("all")}`], ["predict", `Predictions ${n("predict")}`], ["screen", `Screens ${n("screen")}`]],
      onChange: (k) => { view.kind = k; renderList(); } }));
  }
  function renderList() {
    const rows = all.filter((r) => matches(r));
    renderSeg();
    count.textContent = all.length && rows.length !== all.length ? `${rows.length} of ${all.length}` : "";
    list.classList.toggle("dense", view.dense);
    if (!all.length) {
      replace(list, h("div", { class: "empty" }, h("h2", null, "No runs"),
        h("div", { class: "row center" }, h("a", { class: "btn primary", href: "#/predict" }, icon("helix"), "New prediction"), h("a", { class: "btn", href: "#/screen" }, icon("rows"), "New screen"))));
    } else if (!rows.length) {
      replace(list, h("div", { class: "empty" }, h("p", null, "No runs match")));
    } else replace(list, rows.map(card));
  }

  async function load() {
    try {
      const r = await api.get("/api/runs");
      if (!alive) return;
      all = r.runs;
      renderList();
      if (view.y) { window.scrollTo(0, view.y); view.y = 0; app.keepScroll = true; }
      clearTimeout(timer);
      if (r.active_run) timer = setTimeout(load, 3000);
    } catch (e) {
      if (alive) replace(list, h("div", { class: "notice bad" }, icon("alert"), h("div", null, e.message)));
    }
  }

  const met = (k, v) => h("div", null, h("small", null, k), h("b", null, v));
  function card(r) {
    const predict = r.kind === "predict";
    const kind = h("span", { class: "kind" }, icon(predict ? (r.affinity ? "hex" : "helix") : "rows"), predict ? (r.affinity ? "Affinity" : "Structure") : "Screen");
    let body;
    if (predict) {
      const m = r.metrics || {};
      const multi = (r.n_chains ?? 0) > 1;
      body = r.state === "complete" && Object.keys(m).length
        ? h("div", { class: "mets" }, met("pLDDT", fmt.plddt(m.complex_plddt)), met("pTM", fmt.num(m.ptm, 2)),
            multi && m.iptm !== undefined ? met("ipTM", fmt.num(m.iptm, 2)) : null,
            m.affinity_probability_binary !== undefined ? met("P(binder)", fmt.p(m.affinity_probability_binary)) : null)
        : h("div", { class: "small muted" }, stateLine(r));
    } else {
      const done = r.n_done ?? 0, tot = r.n_ligands ?? done;
      const pct = tot ? Math.min(100, (100 * done) / tot) : 0;
      body = h("div", null,
        h("div", { class: "small num muted", style: { marginBottom: "6px" } }, `${fmt.int(done)} / ${fmt.int(tot)}`,
          r.n_error ? h("span", { class: "bad-t" }, ` · ${r.n_error} failed`) : null, r.n_invalid ? h("span", null, ` · ${r.n_invalid} invalid`) : null),
        h("div", { class: "mini-bar" }, h("i", { style: { width: pct + "%" } })));
    }
    const sub = predict ? [r.chains, r.n_tokens ? `${fmt.int(r.n_tokens)} tokens` : null] : [r.target, r.rank_by ? (r.rank_by === "prob" ? "P(binder)" : "Affinity (log10 µM)") : null];
    return h("a", { class: "card run-card", href: "#/run/" + enc(r.id) },
      h("div", { class: "top" }, kind, h("span", { class: "grow" }), stateChip(r.state), r.external ? h("span", { class: "chip", title: "Created by the command line" }, "CLI") : null),
      h("div", { class: "rc-main" }, h("div", { class: "title", title: r.name }, r.name),
        h("div", { class: "sub", title: [...sub, presetLabel(r.preset)].filter(Boolean).join(" · ") }, [...sub, presetLabel(r.preset)].filter(Boolean).join(" · "))),
      body,
      h("div", { class: "foot-l" }, h("span", null, fmt.when(r.created)), r.wall_s >= 1 ? h("span", { title: "Wall time of the last session" }, fmt.dur(r.wall_s)) : null));
  }

  const search = h("input", { type: "search", class: "runs-q", placeholder: "Name or target", "aria-label": "Search runs", value: view.q, oninput: (e) => { view.q = e.target.value; renderList(); } });
  const stateSel = h("select", { "aria-label": "State", onchange: (e) => { view.state = e.target.value; renderList(); } },
    [["all", "Any state"], ["running", "Running"], ["complete", "Complete"], ["stopped", "Stopped"], ["failed", "Failed"]].map(([v, t]) => h("option", { value: v, selected: view.state === v }, t)));
  const denseBtn = h("button", { type: "button", class: "btn ghost sm", "aria-pressed": String(view.dense), title: "Dense list", onclick: () => { view.dense = !view.dense; denseBtn.setAttribute("aria-pressed", String(view.dense)); renderList(); } }, icon("list"));
  replace(root,
    h("div", { class: "ph" }, h("h1", null, "Runs"), h("div", { class: "ph-act" }, h("button", { type: "button", class: "btn sm", onclick: load }, icon("refresh"), "Refresh"))),
    h("div", { class: "runs-bar" }, segHost, stateSel, search, count, denseBtn),
    list);
  await load();
  return () => { alive = false; view.y = window.scrollY; clearTimeout(timer); };
}
