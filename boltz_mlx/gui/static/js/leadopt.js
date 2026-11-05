// Lead optimisation: the analogues ticked on a screen's results, side by side against one reference compound.
// The series table: each analogue drawn in the reference's 2D layout with what it changes highlighted, its affinity value
// and P(binder) and the differences from the reference. Up to four poses in 3D next to each other, every one superposed on
// the reference compound's protein (C-alpha), with the pocket residues each analogue gains or loses relative to it.
// All numbers are model estimates.
import { api, downloadFrom, enc } from "./api.js";
import { download, fmt, h, icon, replace, toast } from "./dom.js";
import { cameraButtons, ic50Text } from "./results.js";
import { PoseViewer } from "./viewer.js";
import { announce } from "./ui.js";

const MAX_3D = 4;

export async function mount(root, app, runId, recs) {
  const base = "/api/runs/" + enc(runId);
  let alive = true;
  let run, rows;
  try {
    [run, rows] = await Promise.all([api.get(base), api.get(`${base}/results?rank_by=value`).then((r) => r.rows)]);
  } catch (e) {
    replace(root, h("div", { class: "notice bad" }, icon("alert"), h("div", null, e.message, " ", h("a", { href: "#/runs" }, "Back to runs"))));
    return () => {};
  }
  const byRec = new Map(rows.map((r) => [r.record_id, r]));
  const series = recs.map((r) => byRec.get(r)).filter((r) => r && r.status === "ok");
  if (series.length < 2) {
    replace(root, h("div", { class: "notice warn" }, icon("alert"), h("div", null, "Tick at least two finished ligands on the results table to compare them. ",
      h("a", { href: "#/run/" + enc(runId) }, "Back to the run"))));
    return () => {};
  }
  const hasPose = (r) => r.pose !== false;   // false: the score is kept but the structure was not (Top N retention)
  const mq = window.matchMedia ? window.matchMedia("(max-width: 720px)") : { matches: false, addEventListener() {}, removeEventListener() {} };
  const st = { refGen: 0, ref: series[0].record_id, shown: series.filter(hasPose).slice(0, Math.min(3, series.length)).map((r) => r.record_id), sort: { key: "affinity_pred_value", dir: 1 }, posesOpen: !mq.matches };
  const urls = new Set();
  const panes = new Map();  // record_id -> { el, viewer, host, msg, pocket, info }
  const details = new Map();

  const y = (r) => r.affinity_pred_value, p = (r) => r.affinity_probability_binary;
  const refRow = () => byRec.get(st.ref);
  const fin = (x) => typeof x === "number" && Number.isFinite(x);
  // A difference exists only when both numbers do: a missing value is never read as 0 (no invented ΔpIC50 / ΔP, no green or red).
  const dPic = (r) => (fin(y(r)) && fin(y(refRow())) ? -(y(r) - y(refRow())) : null);
  const dP = (r) => (fin(p(r)) && fin(p(refRow())) ? p(r) - p(refRow()) : null);
  const aligned = () => hasPose(refRow());   // poses are superposed on the reference's protein only if the reference has a pose
  const dlt = (x, digits = 2) => {
    if (x === null || x === undefined || Number.isNaN(x)) return "–";
    const r = Number(x.toFixed(digits));  // so a tiny difference reads 0.00, not −0.00
    return (r > 0 ? "+" : r < 0 ? "−" : "") + Math.abs(r).toFixed(digits);
  };

  // ------------------------------------------------------------------------------------------ series table
  const elTable = h("div");
  async function thumb(img, rec) {
    const my = st.refGen, ref = st.ref;
    try {
      const { url, incomplete } = await api.depiction(`${base}/ligand/${enc(rec)}/2d.svg?theme=${app.theme}&ref=${enc(ref)}`);
      urls.add(url);
      if (alive && my === st.refGen) {   // drawn against a reference that is no longer the reference: dropped
        img.src = url;
        img.classList.toggle("approx", incomplete);
        img.title = incomplete ? "Approximate: the common-substructure search ran out of time, so highlights may over-mark" : "";
      }
    } catch (e) { if (my === st.refGen) img.alt = e.message; }
  }
  function sorted() {
    const ref = refRow();
    return series.slice().sort((a, b) => {
      const k = st.sort.key, va = k === "d_pic50" ? dPic(a) : a[k], vb = k === "d_pic50" ? dPic(b) : b[k];
      const ma = !fin(va), mb = !fin(vb);
      if (ma || mb) return ma === mb ? 0 : ma ? 1 : -1;   // missing values last, whatever the direction
      return (va - vb) * st.sort.dir;
    });
  }
  const SORTS = [["affinity_pred_value", "Affinity (log10 µM)", 1], ["d_pic50", "ΔpIC50", -1], ["affinity_probability_binary", "P(binder)", -1], ["ligand_iptm", "Ligand ipTM", -1]];
  const setSort = (key) => { st.sort = { key, dir: st.sort.key === key ? -st.sort.dir : (SORTS.find((x) => x[0] === key)?.[2] ?? -1) }; renderTable(); };
  const tdRef = (r) => h("input", { type: "radio", name: "lo-ref", checked: r.record_id === st.ref, "aria-label": `Use ${r.ligand_id} as the reference`, onchange: () => setRef(r.record_id) });
  const td3d = (r) => hasPose(r)
    ? h("input", { type: "checkbox", checked: st.shown.includes(r.record_id), "aria-label": `Show ${r.ligand_id} in 3D`, disabled: !st.shown.includes(r.record_id) && st.shown.length >= MAX_3D, onchange: (e) => toggle3d(r.record_id, e.target.checked) })
    : h("span", { class: "faint small", title: "Pose not retained: the screen kept structures for its top ligands only" }, "no pose");
  function renderTable() {
    const ref = refRow();
    const list = sorted();
    if (mq.matches) return renderRows(list, ref);
    const th = (label, key, title, lo) => h("th", { class: "r" + (lo ? " lo" : ""), title, "aria-sort": key && st.sort.key === key ? (st.sort.dir > 0 ? "ascending" : "descending") : undefined },
      key ? h("button", { type: "button", class: "thbtn", onclick: () => setSort(key) }, label, h("span", { class: "arrow" }, st.sort.key === key ? (st.sort.dir > 0 ? "▲" : "▼") : "")) : label);
    replace(elTable, h("div", { class: "table-wrap lo-wrap" }, h("table", { class: "t lo" },
      h("thead", null, h("tr", null,
        h("th", { title: "Reference compound: the others are compared with it" }, "Ref."), h("th", { title: `Show in 3D (up to ${MAX_3D})` }, "3D"), h("th", null, "Structure"), h("th", null, "Ligand"),
        th("Affinity (log10 µM)", "affinity_pred_value", "affinity_pred_value = log10(IC50 in µM), model estimate; lower = stronger", true),
        th("IC50 (est.)", null, null, true), th("ΔpIC50", "d_pic50", "pIC50 of this analogue minus the reference's: −(y − y_ref). Positive = predicted more potent."),
        th("P(binder)", "affinity_probability_binary", null, true), th("ΔP", null, null, true), th("Ligand ipTM", "ligand_iptm", null, true), th("Ligand pLDDT", null, null, true),
        )),
      h("tbody", null, list.map((r) => {
        const isRef = r.record_id === st.ref;
        const img = h("img", { class: "lo-2d", alt: "2D structure of " + r.ligand_id, width: 150, height: 112 });
        thumb(img, r.record_id);
        const [v, unit] = ic50Text(y(r));
        const dp = isRef ? null : dPic(r);
        return h("tr", { class: isRef ? "sel" : "" },
          h("td", null, tdRef(r)), h("td", null, td3d(r)),
          h("td", { class: "lo-2d-cell" }, img),
          h("td", { class: "id" }, r.ligand_id, isRef ? h("span", { class: "chip accent", style: { marginLeft: "6px" } }, "ref.") : null, h("div", { class: "small faint", title: "Rank within the whole run by affinity y (this page always ranks by y)" }, `y rank ${r.rank ?? "–"}`)),
          h("td", { class: "r num lo" }, fmt.num(y(r), 2)), h("td", { class: "r num lo" }, `${v} ${unit}`),
          h("td", { class: "r num" + (dp >= 0.005 ? " up" : dp <= -0.005 ? " down" : "") }, isRef ? "–" : dlt(dp)),
          h("td", { class: "r num lo" }, fmt.p(p(r))), h("td", { class: "r num lo" }, isRef ? "–" : dlt(dP(r))),
          h("td", { class: "r num lo" }, fmt.num(r.ligand_iptm, 2)), h("td", { class: "r num lo" }, fmt.plddt(r.ligand_plddt)));
      })))));
  }
  // Phone: one compact row per ligand with the numbers that decide (y, ΔpIC50, P(binder)); the rest opens under "More".
  function renderRows(list, ref) {
    replace(elTable,
      h("div", { class: "row tight lo-sort" }, h("span", { class: "small muted" }, "Sort"),
        h("select", { "aria-label": "Sort by", onchange: (e) => { st.sort = { key: e.target.value, dir: SORTS.find((x) => x[0] === e.target.value)[2] }; renderTable(); } },
          SORTS.map(([k, t]) => h("option", { value: k, selected: st.sort.key === k }, t)))),
      h("div", { class: "lo-rows" }, list.map((r) => {
        const isRef = r.record_id === st.ref;
        const img = h("img", { class: "lo-2d", alt: "2D structure of " + r.ligand_id, width: 112, height: 84 });
        thumb(img, r.record_id);
        const [v, unit] = ic50Text(y(r));
        const dp = isRef ? null : dPic(r);
        return h("div", { class: "lo-row" + (isRef ? " sel" : "") },
          h("div", { class: "lo-main" }, img,
            h("div", { class: "lo-body" },
              h("div", { class: "lo-r1" }, h("b", { class: "lo-id" }, r.ligand_id), isRef ? h("span", { class: "chip accent" }, "ref.") : null,
                h("label", { class: "check small", title: "Reference" }, tdRef(r), "Ref."), h("label", { class: "check small" }, td3d(r), hasPose(r) ? "3D" : null)),
              h("div", { class: "lo-r2 num" }, h("span", null, h("i", null, "y"), fmt.num(y(r), 2)),
                h("span", { class: dp >= 0.005 ? "up" : dp <= -0.005 ? "down" : "" }, h("i", null, "ΔpIC50"), isRef ? "–" : dlt(dp)), h("span", null, h("i", null, "P"), fmt.p(p(r)))))),
          h("details", { class: "adv lo-more" }, h("summary", null, "More"),
            h("div", { class: "lo-more-b" }, h("dl", { class: "kv" },
              h("dt", null, "IC50 (est.)"), h("dd", { class: "num" }, `${v} ${unit}`), h("dt", null, "ΔP"), h("dd", { class: "num" }, isRef ? "–" : dlt(dP(r))),
              h("dt", null, "Ligand ipTM"), h("dd", { class: "num" }, fmt.num(r.ligand_iptm, 2)), h("dt", null, "Ligand pLDDT"), h("dd", { class: "num" }, fmt.plddt(r.ligand_plddt)),
              h("dt", null, "y rank in run"), h("dd", { class: "num" }, r.rank ?? "–")))));
      })));
  }

  // ------------------------------------------------------------------------------------------ 3D panes
  const elGrid = h("div", { class: "lo-grid" });
  async function detail(rec) {
    if (!details.has(rec)) details.set(rec, api.get(`${base}/ligand/${enc(rec)}`).catch(() => ({})));
    return details.get(rec);
  }
  const resKey = (x) => `${x.chain}:${x.resseq}`;
  async function renderPocket(rec) {
    const pn = panes.get(rec);
    if (!pn) return;
    const my = st.refGen, ref = st.ref;
    const [d, dr] = await Promise.all([detail(rec), detail(ref)]);
    if (!alive || !panes.has(rec) || my !== st.refGen) return;   // a newer reference owns the annotations
    const mine = d.pocket?.pocket || [], theirs = new Set((dr.pocket?.pocket || []).map(resKey));
    const mineSet = new Set(mine.map(resKey));
    const lost = (dr.pocket?.pocket || []).filter((x) => !mineSet.has(resKey(x)));
    const isRef = rec === ref;
    replace(pn.pocket, mine.length
      ? [h("div", { class: "small muted" }, isRef ? `Pocket: ${mine.length} residues within ${d.pocket.cutoff} Å.` : `Pocket: ${mine.length} residues; highlighted = not in the reference's pocket.`),
         h("div", { class: "pocket-chips" }, mine.map((x) => h("span", { class: "chip" + (!isRef && !theirs.has(resKey(x)) ? " hl" : ""), title: `closest atom ${x.min_dist} Å` }, `${x.resname}${x.resseq}`))),
         !isRef && lost.length ? h("div", { class: "small muted", style: { marginTop: "4px" } }, "Lost vs the reference: ", lost.map((x) => `${x.resname}${x.resseq}`).join(", ")) : null]
      : h("div", { class: "small muted" }, "Pocket residues not available for this pose."));
  }
  async function loadPose(rec) {
    const pn = panes.get(rec);
    if (!pn) return;
    const my = st.refGen, ref = st.ref, mine = ++pn.gen;
    const current = () => alive && panes.get(rec) === pn && my === st.refGen && mine === pn.gen;
    pn.msg.hidden = false;
    replace(pn.msg, h("span", { class: "spinner" }), " Loading the pose…");
    try {
      const q = rec === ref || !aligned() ? "" : `?align_to=${enc(ref)}`;   // a reference without a structure cannot be superposed on
      const pose = await api.pose(`${base}/ligand/${enc(rec)}/pose${q}`);
      if (!current()) return;
      if (!pn.viewer) {
        const v = new PoseViewer(pn.host);
        v.dark = app.theme === "dark";
        try {
          await v.init();
        } catch (e) { v.dispose(); throw e; }
        if (panes.get(rec) !== pn) { v.dispose(); return; }
        pn.viewer = v;
        if (!current()) return;
      }
      // through the viewer's queue (only the newest request reaches Mol*), in the format the server says it is in (mmCIF or PDB)
      if (!(await pn.viewer.display(pose.text, pose.format, current)) || !current()) return;
      await pn.viewer.resize();
      if (!current()) return;
      pn.align = rec === ref || pose.rmsd == null ? null : { rmsd: pose.rmsd, n: pose.n };
      renderHead(rec);
      pn.msg.hidden = true;
    } catch (e) {
      if (/closed|timeout/.test(e.message || "") && pn.viewer) { pn.viewer.dispose(); pn.viewer = null; pn.host.replaceChildren(); }
      if (current()) replace(pn.msg, h("div", { class: "stack tight center" }, h("strong", null, "3D view unavailable"), h("div", { class: "small" }, e.message || String(e)),
        h("button", { type: "button", class: "btn sm", onclick: () => loadPose(rec) }, icon("refresh"), "Retry")));
    }
  }
  function paneFor(rec) {
    const r = byRec.get(rec);
    const host = h("div", { class: "mol-host" }), msg = h("div", { class: "msg" });
    const pocket = h("div", { class: "lo-pocket" });
    const head = h("div", { class: "lo-h" });
    const pn = { el: null, host, msg, pocket, head, viewer: null, gen: 0, align: null };
    const el = pn.el = h("div", { class: "card lo-pane" }, head, h("div", { class: "mol-box" }, host, msg), h("div", { class: "lo-cam" }, cameraButtons(() => pn.viewer, true)), pocket);
    panes.set(rec, pn);
    renderHead(rec);
    loadPose(rec);
    renderPocket(rec);
    return pn;
  }
  function renderHead(rec) {
    const pn = panes.get(rec), r = byRec.get(rec), ref = refRow(), isRef = rec === st.ref;
    const [v, unit] = ic50Text(y(r));
    replace(pn.head,
      h("div", { class: "row" }, h("b", { class: "grow" }, r.ligand_id), isRef ? h("span", { class: "chip accent" }, "reference") : null,
        h("button", { class: "btn ghost sm", title: aligned() && !isRef ? "Download this pose (superposed as shown)" : "Download this pose", onclick: () => downloadFrom(`${base}/ligand/${enc(rec)}/pose${isRef || !aligned() ? "?format=mmcif" : `?align_to=${enc(st.ref)}`}`, `${r.ligand_id}${isRef || !aligned() ? "" : "_aligned"}.cif`).catch((e) => toast(e.message, "bad")) }, icon("download")),
        h("button", { class: "btn ghost sm", title: "Remove from 3D", "aria-label": "Remove from 3D", onclick: () => toggle3d(rec, false) }, icon("x"))),
      h("div", { class: "lo-mets" },
        h("span", null, "Affinity (log10 µM) ", h("b", null, fmt.num(y(r), 2))), h("span", { title: "Model estimate from y, not a measurement" }, "IC50 (est.) ", h("b", null, `${v} ${unit}`)),
        isRef ? null : h("span", null, "ΔpIC50 ", h("b", null, dlt(dPic(r)))),
        h("span", null, "P ", h("b", null, fmt.p(p(r)))), h("span", null, "lig. pLDDT ", h("b", null, fmt.plddt(r.ligand_plddt))),
        pn.align && pn.align.rmsd !== null ? h("span", { title: "C-alpha superposition onto the reference's protein" }, "Cα RMSD ", h("b", null, `${pn.align.rmsd.toFixed(2)} Å`), h("span", { class: "faint" }, ` over ${pn.align.n} Cα`)) : null));
  }
  // Panes are added and removed one by one; a surviving pane keeps its place in the page (moving a frame reloads it).
  function renderGrid() {
    for (const [rec, pn] of [...panes]) {
      if (!st.shown.includes(rec)) { pn.viewer?.dispose(); pn.el.remove(); panes.delete(rec); }
    }
    elGrid.style.setProperty("--n", String(Math.max(1, st.shown.length)));
    if (!st.posesOpen) return;
    let prev = null;
    for (const rec of st.shown) {
      const pn = panes.get(rec) || paneFor(rec);
      if (pn.el.parentNode !== elGrid) elGrid.insertBefore(pn.el, prev ? prev.nextSibling : elGrid.firstChild);
      prev = pn.el;
    }
    for (const pn of panes.values()) pn.viewer?.resize();
  }
  function toggle3d(rec, on) {
    st.shown = on ? [...new Set([...st.shown, rec])].slice(0, MAX_3D) : st.shown.filter((x) => x !== rec);
    if (on && !st.posesOpen) st.posesOpen = true, syncPoses();
    renderGrid();
    renderTable();
  }
  function setRef(rec) {
    st.ref = rec;
    st.refGen++;
    renderTopline();
    if (!st.shown.includes(rec) && hasPose(byRec.get(rec))) st.shown = [rec, ...st.shown].slice(0, MAX_3D);
    syncPoses();
    renderTable();
    const had = new Set(panes.keys());
    renderGrid();
    for (const r of [...panes.keys()].filter((x) => had.has(x))) { renderHead(r); loadPose(r); renderPocket(r); }
  }

  async function matchViews() {
    const src = panes.get(st.ref)?.viewer;
    if (!src) return toast("The reference pose is not drawn yet", "bad");
    try {
      const snap = await src.getCamera();
      await Promise.all([...panes.values()].filter((pn) => pn.viewer && pn.viewer !== src).map((pn) => pn.viewer.setCamera(snap)));
    } catch (e) { toast(e.message || "Could not match the views", "bad"); }
  }

  // ------------------------------------------------------------------------------------------ assemble
  const elTop = h("div");
  const elMatch = h("button", { type: "button", class: "btn sm", title: "Give every pose the reference pane's camera (they share one coordinate frame)", onclick: matchViews }, icon("grid"), "Match views");
  const elUnaligned = h("span", { class: "chip warn", hidden: true, title: "The reference has no retained pose: the other poses are drawn as saved, not superposed" }, "Reference pose unavailable · unaligned");
  const elPosesBtn = h("button", { type: "button", class: "btn sm", "aria-expanded": String(st.posesOpen), onclick: () => { st.posesOpen = !st.posesOpen; syncPoses(); renderGrid(); } });
  function syncPoses() {
    replace(elPosesBtn, st.posesOpen ? "Hide" : `Show${st.shown.length ? ` (${st.shown.length})` : ""}`);
    elPosesBtn.setAttribute("aria-expanded", String(st.posesOpen));
    elMatch.hidden = !st.posesOpen;
    elGrid.hidden = !st.posesOpen;
    elMatch.disabled = !aligned();
    elUnaligned.hidden = aligned();
  }
  function renderTopline() {   // every reference label comes from st.ref
    replace(elTop, h("div", { class: "run-h" }, h("div", { class: "titles" },
      h("h1", null, h("a", { class: "crumb", href: "#/run/" + enc(runId), title: "Back to the run" }, icon("back"), h("span", { class: "rname" }, run.name)), "Compare"),
      h("div", { class: "meta-line" }, h("span", null, fmt.plural(series.length, "ligand")), h("span", null, "Reference ", h("b", null, byRec.get(st.ref)?.ligand_id)), h("span", null, "ranked by y"))),
      h("div", { class: "actions" }, h("button", { type: "button", class: "btn", onclick: exportCsv }, icon("download"), "CSV"))));
    announce(`Reference ${byRec.get(st.ref)?.ligand_id}`);
  }
  // The comparison as shown: reference, source run, ranks and deltas against the reference.
  function exportCsv() {
    const ref = refRow();
    const order = series.slice().sort((a, b) => (fin(y(a)) ? y(a) : Infinity) - (fin(y(b)) ? y(b) : Infinity));
    const head = ["reference_id", "source_run", "ligand_id", "record_id", "y_rank_in_run", "rank_in_subset_by_y", "affinity_pred_value", "ic50_um_est", "delta_pic50_vs_ref", "p_binder", "delta_p_vs_ref", "ligand_iptm", "ligand_plddt"];
    const cell = (v) => { const t = v === null || v === undefined ? "" : String(v); return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t; };
    const lines = order.map((r, i) => [ref.ligand_id, runId, r.ligand_id, r.record_id, r.rank ?? "", i + 1, y(r), y(r) == null ? "" : Math.pow(10, y(r)).toPrecision(3),
      r.record_id === st.ref ? 0 : dPic(r) === null ? "" : dPic(r).toFixed(3), p(r), r.record_id === st.ref ? 0 : dP(r) === null ? "" : dP(r).toFixed(3), r.ligand_iptm, r.ligand_plddt].map(cell).join(","));
    download(`${runId}_compare_ref_${ref.ligand_id}.csv`, "# model estimates, not measurements; ranks are by affinity y\n" + [head.join(","), ...lines].join("\n") + "\n");
  }
  replace(root, elTop,
    h("div", { class: "card" }, h("div", { class: "card-h" }, h("h2", null, "Series"), h("span", { class: "small muted" }, "Reference · 3D (max 4)")), h("div", { class: "card-b" }, elTable)),
    h("div", { class: "card-h", style: { border: 0, paddingLeft: 0 } }, h("h2", { class: "grow" }, "Poses"),
      elUnaligned, elMatch, elPosesBtn), elGrid);
  renderTopline();
  renderTable();
  syncPoses();
  renderGrid();
  const onTheme = (e) => { for (const pn of panes.values()) pn.viewer?.setDark(e.detail === "dark"); renderTable(); };
  const onMq = () => { if (alive) renderTable(); };
  window.addEventListener("themechange", onTheme);
  mq.addEventListener?.("change", onMq);
  return () => {
    alive = false;
    window.removeEventListener("themechange", onTheme);
    mq.removeEventListener?.("change", onMq);
    for (const pn of panes.values()) pn.viewer?.dispose();
    for (const u of urls) URL.revokeObjectURL(u);
  };
}
