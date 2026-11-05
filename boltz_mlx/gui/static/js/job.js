// One structure job (kind "predict"): progress while it runs and, once the files are there, the results: a strip of key numbers,
// Mol* beside a tabbed panel (affinity, confidence, run), then the PAE heatmap and the chains table. A job with several diffusion
// samples has a model selector; structure, metrics, PAE and downloads follow the selected model and change together: the model
// switch fetches the numbers and the pose first and commits both only if no newer switch was made meanwhile.
import { api, downloadFrom, enc } from "./api.js";
import { download, fmt, h, icon, replace, toast } from "./dom.js";
import { affinityCard, bandWord, chainsCard, confidenceCard, ic50Text, metricScope, MolPanel, paeCard, plddtColor } from "./results.js";
import { presetLabel, refreshSide } from "./parts.js";
import { stateChip } from "./runs.js";
import { drafts, go, saveDraft, stashDraft } from "./state.js";
import { copyText, exportMenu, logPanel, radioSeg, tabset } from "./ui.js";

const PHASE = { new: "Not started", starting: "Loading", structure: "Structure", affinity: "Affinity", finishing: "Writing results", done: "Done" };
const STEP_LABEL = { starting: "Load", structure: "Structure", affinity: "Affinity", done: "Done" };

// The key numbers (only the ones that apply to this complex).
function metricStrip(conf, aff, chains) {
  const { multi, ligand } = metricScope(chains);
  const m = (k, v, sub, bar) => h("div", { class: "metric" }, h("span", { class: "k" }, k), h("b", null, v),
    bar != null ? h("div", { class: "bar" }, h("i", { style: { width: Math.max(0, Math.min(100, bar * 100)) + "%" } })) : sub ? h("span", { class: "s" }, sub) : null);
  const pl = conf.complex_plddt;
  const items = [
    m("pLDDT", pl == null ? "–" : h("span", null, h("i", { class: "pl-dot", style: { background: plddtColor(pl) } }), fmt.plddt(pl)), bandWord(pl)),
    m("pTM", fmt.num(conf.ptm, 2), null, conf.ptm),
    multi && conf.iptm != null ? m("ipTM", fmt.num(conf.iptm, 2), null, conf.iptm) : null,
    ligand && conf.ligand_iptm != null ? m("Ligand ipTM", fmt.num(conf.ligand_iptm, 2), null, conf.ligand_iptm) : null,
  ];
  if (aff) {
    const y = aff.affinity_pred_value, p = aff.affinity_probability_binary;
    const [v, unit] = ic50Text(y);
    items.push(m("IC50 (est.)", h("span", null, v, h("small", null, unit))), m("P(binder)", fmt.p(p)));
  } else if (conf.confidence_score != null) items.push(m("Score", fmt.num(conf.confidence_score, 2), null, conf.confidence_score));
  return h("div", { class: "mstrip" }, items.filter(Boolean));
}

// What a run used besides sequences and settings, for the Run tab: each protein's MSA, constraints, modifications, templates.
const MSA_WORD = { empty: "none", server: "online (ColabFold)" };
function inputRows(sm) {
  const prot = (sm.chains || []).filter((c) => c.type === "protein");
  const tok = (t) => (Array.isArray(t) ? t.join(":") : String(t));
  const cx = (sm.constraints || []).map((c) => {
    const [k, b] = Object.entries(c)[0] || [];
    return k === "pocket" ? `pocket ${b.binder} → ${(b.contacts || []).map(tok).join(", ")}` : k === "contact" ? `contact ${tok(b.token1)}–${tok(b.token2)}`
      : k === "bond" ? `bond ${tok(b.atom1)}–${tok(b.atom2)}` : k;
  });
  const mods = (sm.chains || []).flatMap((c) => (c.modifications || []).map((m) => `${c.id}:${m.position} ${m.ccd}`));
  return [
    prot.length ? ["MSA", prot.map((c) => `${prot.length > 1 ? c.id + ": " : ""}${MSA_WORD[c.msa] || c.msa || "none"}`).join(" · ")] : null,
    cx.length ? ["Constraints", cx.join(" · ")] : null,
    mods.length ? ["Modifications", mods.join(" · ")] : null,
    (sm.templates || []).length ? ["Templates", sm.templates.map((t) => `${t.chain_id}: ${t.from || t.name}`).join(" · ")] : null,
  ].filter(Boolean);
}

// A card with a tab row, built from cards (their bodies become the tab panels).
let lastTab = null;
function tabsCard(list) {
  const bodies = list.map(([key, , card]) => ({ key, b: card.querySelector(":scope > .card-b") || card }));
  let cur = list.some(([k]) => k === lastTab) ? lastTab : list[0][0];
  const ts = tabset({ label: "Result panel", items: list.map(([k, label]) => [k, label]), value: cur, cls: "tabsrow-in", onChange: (k) => show(k) });
  function show(key) {
    cur = lastTab = key;
    replace(ts.panel, bodies.find((x) => x.key === cur).b);
  }
  show(cur);
  return h("section", { class: "card tabs-card" }, h("div", { class: "tabsrow" }, ts.bar), ts.panel);
}

export async function mount(root, app, runId) {
  const base = "/api/runs/" + enc(runId);
  let alive = true, timer = null, busy = false, run = null;
  let shownVersion = null, mol = null, pae = null, model = null, shownModel = null, modelsShown = "", modelsSeg = null;
  let selGen = 0;   // moves on with every model switch and result load: a reply is used only if it carries the current value
  let lastState = null;

  // ------------------------------------------------------------------------------------------ loading
  async function tick() {
    clearTimeout(timer);
    if (!alive) return;
    try {
      run = await api.get(base);
    } catch (e) {
      if (!alive) return;
      if (!run) { replace(root, h("div", { class: "notice bad" }, icon("alert"), h("div", null, /no such run/i.test(e.message) ? "Run not found" : e.message), h("a", { class: "btn sm", href: "#/runs" }, "Back to runs"))); return; }
      timer = setTimeout(tick, 4000);
      return;
    }
    if (lastState !== null && run.state !== lastState) refreshSide();   // the sidebar follows state changes seen here
    lastState = run.state;
    renderHead();
    renderProgress();
    if (run.has_result && (run.results_version !== shownVersion) && (run.state === "complete" || !["running", "stopping"].includes(run.state))) {
      const v = run.results_version;
      if (await renderResults()) shownVersion = v;   // committed only after the results are on the page: a failed fetch is retried
    }
    const live = run.state === "running" || run.state === "stopping";
    timer = setTimeout(tick, live ? 1500 : 10000);
  }

  async function act(fn, ok) {
    if (busy) return;
    busy = true; renderHead();
    try { await fn(); if (ok) toast(ok); } catch (e) { toast(e.message, "bad"); }
    busy = false;
    await tick();
    refreshSide();
  }

  async function useInputs() {
    try {
      const d = await api.get(`${base}/inputs`);
      const keep = (f) => (f && f.content != null ? { kind: "file", filename: f.filename, content: f.content, size: f.content.length } : null);
      const chains = d.chains.map((c, i) => ({ key: "ch" + (i + 1), type: c.type, id: c.id, copies: String(c.copies ?? 1), sequence: c.sequence || "", cyclic: !!c.cyclic, smiles: c.smiles || "", ccd: c.ccd || "",
        ligMode: c.ccd ? "ccd" : "smiles", msa: c.msa?.content != null ? { mode: "file", file: keep(c.msa) } : { mode: c.msa?.mode === "server" ? "server" : "empty", file: null } }));
      const o = d.options || {};
      const cxIn = { constraints: d.constraints || [], templates: d.templates || [], mods: d.chains.flatMap((c) => (c.modifications || []).map((m) => ({ ...m, chain: c.id }))) };
      const draft = { name: `${d.name || runId} (copy)`, chains, affinity: !!d.binder, binder: d.binder || "", note: "", unsupported: [], ack: false, yamlOpen: false, undoable: true, cxIn,
        useYaml: !!d.yaml, adv: d.yaml ? { filename: d.yaml.filename, content: d.yaml.content, files: d.yaml.files.map((f) => ({ kind: "file", filename: f.filename, content: f.content, size: f.content.length })), needs: d.yaml.needs || d.yaml.files.map((f) => f.filename) } : null,
        o: { preset: o.preset || "", steps: o.steps_auto ? "" : String(o.steps ?? ""), recycling: String(o.recycling ?? ""), affinity_steps: d.binder ? String(o.affinity_steps ?? "") : "", affinity_samples: d.binder ? String(o.affinity_samples ?? "") : "", samples: String(o.samples ?? 1), seed: String(o.seed ?? 0) } };
      stashDraft("predict", ["val", "busy", "starting", "valSeq", "yamlOpen", "discarded", "revert"]);   // the draft being replaced, for Undo
      drafts.predict = draft;
      saveDraft("predict", draft);
      go("#/predict");
    } catch (e) { toast(e.message, "bad"); }
  }

  // ------------------------------------------------------------------------------------------ header
  const elHead = h("div");
  let headKey = null, cur = null;   // cur: the model on show {d, fileBase}, for the Export menu
  const save = (path, name) => downloadFrom(path, name).catch((e) => toast(e.message, "bad"));
  const exportItems = () => {
    const d = cur?.d, fb = cur?.fileBase;
    return [
      d ? { sep: `Model ${d.model}` } : null,
      d ? { label: "mmCIF", onClick: () => save(`${base}/prediction/pose?model=${d.model}&format=mmcif`, `${fb}.cif`) } : null,
      d ? { label: "PDB", onClick: () => save(`${base}/prediction/pose?model=${d.model}&format=pdb`, `${fb}.pdb`) } : null,
      d?.confidence ? { label: "Confidence JSON", onClick: () => download(`${fb}_confidence.json`, JSON.stringify(d.confidence, null, 2), "application/json") } : null,
      d?.affinity ? { label: "Affinity JSON", title: "Run-level: computed once, on model 0", onClick: () => download(`${runId}_affinity.json`, JSON.stringify(d.affinity, null, 2), "application/json") } : null,
      { sep: "Run" },
      { label: "CSV, all models", onClick: () => save(`${base}/results.csv`, `${runId}_results.csv`) },
      { label: "Bundle (ZIP)", title: "Inputs, settings, every model, raw PAE / pLDDT / PDE arrays, scores, rerun command", onClick: () => save(`${base}/export.zip`, `${runId}_export.zip`) },
    ].filter(Boolean);
  };
  function renderHead() {
    const key = JSON.stringify([run.state, run.name, run.can_stop, run.can_force_stop, run.can_resume, busy, run.has_result, run.path, run.external, run.meta?.options?.changed]);
    if (key === headKey) return;
    headKey = key;
    const sm = run.meta?.summary || {}, o = run.meta?.options || {};
    const btns = [];
    if (run.can_stop) btns.push(h("button", { type: "button", class: "btn danger", disabled: busy, onclick: () => act(() => api.post(`${base}/stop`, { force: false }), "Stopping") }, icon("stop"), "Stop"));
    if (run.can_force_stop) btns.push(h("button", { type: "button", class: "btn danger", disabled: busy, onclick: () => act(() => api.post(`${base}/stop`, { force: true }), "Stopping now") }, icon("stop"), "Stop now"));
    if (run.can_resume) btns.push(h("button", { type: "button", class: "btn primary", disabled: busy, onclick: () => act(() => api.post(`${base}/resume`, {}), "Started") }, icon("play"), run.launches ? "Restart" : "Start"));
    if (run.has_result) btns.push(exportMenu(exportItems));
    if (run.meta) btns.push(h("button", { type: "button", class: "btn", title: "Open these inputs and settings as a new draft", onclick: useInputs }, icon("copy"), "Use inputs"));
    replace(elHead,
      h("div", { class: "run-h" },
        h("div", { class: "titles" },
          h("h1", null, h("a", { class: "crumb", href: "#/runs", title: "All runs" }, icon("back"), "Runs"), h("span", { class: "rname", title: run.name }, run.name), stateChip(run.state)),
          h("div", { class: "meta-line" }, h("span", null, sm.affinity ? "Structure + affinity" : "Structure"),
            run.created ? h("span", null, fmt.when(run.created)) : null,
            sm.chains_short ? h("span", { class: "trunc" }, sm.chains_short) : null,
            sm.n_tokens ? h("span", null, fmt.int(sm.n_tokens), " tokens") : null,
            o.preset ? h("span", null, presetLabel(o.preset) + ((o.changed || []).length ? " · custom" : ""), ` · ${o.steps} steps · ${o.recycling} rec${sm.affinity ? ` · aff ${o.affinity_steps} × ${o.affinity_samples ?? 5}` : ""}`) : null)),
        h("div", { class: "actions" }, btns)));
  }

  // ------------------------------------------------------------------------------------------ progress
  // An active run gets the full card (stepper, timings). A finished one collapses to one line that opens the same details. The log
  // is created once and fed in place (see ui.js logPanel), so reading it is not interrupted by a refresh.
  const log = logPanel({ fetchFull: () => api.text(`${base}/log?lines=400`) });
  const elProg = h("div", { class: "card" });
  const elProgTop = h("div"), elProgLog = h("div", { class: "log-slot" }, log.el);
  elProg.append(elProgTop, elProgLog);
  let progKey = null;
  function renderProgress() {
    const p = run.progress || {};
    const live = run.state === "running" || run.state === "stopping";
    const done = run.state === "complete";
    log.set((run.log_tail || []).join("\n"));
    if ((run.state === "failed" || run.state === "interrupted") && !log.el.dataset.seen) { log.el.open = true; log.el.dataset.seen = "1"; }   // routine logs stay closed; a failure opens it
    const t = run.timings || {};
    const warns = run.meta?.summary?.warnings || [];
    const key = JSON.stringify([run.state, p.phase, p.fraction, p.elapsed_s, p.eta_s, p.over_estimate, t, warns, run.run_error, p.steps]);
    if (key !== progKey) {
      progKey = key;
      const steps = p.steps || ["starting", "structure", "done"];
      const at = steps.indexOf(p.phase === "finishing" ? "done" : p.phase);
      const est = run.meta?.summary?.estimate || {};
      const tile = (k, v, s) => h("div", { class: "tile" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), s ? h("div", { class: "s" }, s) : null);
      const tiles = [
        tile("Elapsed", fmt.dur(done && t.total_wall_s != null ? t.total_wall_s : p.elapsed_s)),
        est.available ? tile("Estimate", fmt.roughDur(est.total_s).replace("about ", "~")) : null,
        live && p.eta_s !== null && p.eta_s !== undefined && est.available ? tile("ETA", p.over_estimate ? "Any moment" : fmt.dur(p.eta_s)) : null,
        done && t.predict_job_s !== undefined ? tile("Structure", fmt.num(t.predict_job_s, 1) + " s") : null,
        done && t.affinity_job_s !== undefined ? tile("Affinity", fmt.num(t.affinity_job_s, 1) + " s") : null,
      ].filter(Boolean);
      const notes = [];
      if (run.state === "failed" || run.state === "interrupted") notes.push(h("div", { class: "notice bad" }, icon("alert"), h("div", null,
        run.state === "failed" ? "Failed" : "Interrupted", run.run_error ? h("pre", { class: "logbox", style: { marginTop: "8px" } }, run.run_error) : null)));
      if (!done) for (const w of warns) notes.push(h("div", { class: "notice warn" }, icon("alert"), h("div", null, w)));
      if (done) {
        const outer = elProgTop.querySelector("details.finished");
        replace(elProgTop, h("details", { class: "acc finished", open: outer?.open || undefined },
          h("summary", null, ["Complete", fmt.dur(t.total_wall_s != null ? t.total_wall_s : p.elapsed_s), warns.length ? `${warns.length} warning${warns.length === 1 ? "" : "s"}` : null].filter(Boolean).join(" · ")),
          h("div", { class: "acc-b stack" }, tiles.length ? h("div", { class: "tiles" }, tiles) : null,
            warns.map((w) => h("div", { class: "notice warn" }, icon("alert"), h("div", null, w))), log.el)));
      } else {
        replace(elProgTop, h("div", { class: "card-b stack" },
          h("div", { class: "row" },
            h("h2", { class: "grow" }, live ? PHASE[p.phase] || "Running" : run.state === "stopped" ? "Stopped · restart required" : "Progress"),
            live && p.fraction !== null && p.fraction !== undefined ? h("span", { class: "num strong" }, `${Math.floor(p.fraction * 100)}%`) : null),
          h("div", { class: "stepper", role: "list" }, steps.map((s, i) => h("div", {
            class: "st" + (i < at ? " done" : i === at && live ? " on" : ""), role: "listitem", "aria-current": i === at && live ? "step" : undefined },
            h("i"), h("b", null, STEP_LABEL[s] || s)))),
          tiles.length ? h("div", { class: "tiles" }, tiles) : null, notes));
      }
    }
    if (!done) elProgLog.replaceChildren(log.el);   // a finished run keeps its log inside the status disclosure
    elProgLog.hidden = done;
    elProg.hidden = false;
  }

  // ------------------------------------------------------------------------------------------ results
  // The skeleton is built once; the structure card keeps its Mol* frame (moving an iframe reloads it).
  const elRes = h("div", { class: "stack" });
  const elResErr = h("div");
  const elStrip = h("div"), elModels = h("div"), elTabs = h("div"), elPae = h("div"), elChains = h("div");
  const elStruct = h("section", { class: "card", style: { overflow: "hidden" } });
  let built = false;

  const fetchPrediction = (k) => api.get(`${base}/prediction${k != null ? `?model=${k}` : ""}`);
  // Downloads of the pose are off while a different model is being fetched and drawn.
  const setBusy = (b) => { elRes.setAttribute("aria-busy", String(b)); };

  // Loads one model (null: the job's own choice): numbers and pose are fetched first, then committed together. Resolves true when
  // this call's result is what the page now shows; false when it failed or a newer call superseded it.
  async function renderResults(k = model) {
    const my = ++selGen;
    if (built) setBusy(true);
    let d, pose = null, poseErr = null;
    try {
      d = await fetchPrediction(k);
    } catch (e) {
      if (!alive || my !== selGen) return false;
      if (built) { setBusy(false); modelsSeg?.set(String(shownModel)); model = shownModel; }
      replace(elResErr, h("div", { class: "notice bad" }, icon("alert"), h("div", { class: "grow" }, `Results could not be loaded: ${e.message}`),
        h("button", { type: "button", class: "btn sm", onclick: () => { replace(elResErr); renderResults(k); } }, icon("refresh"), "Retry")));
      return false;
    }
    try {
      pose = await api.pose(`${base}/prediction/pose?model=${d.model}`);
    } catch (e) {
      poseErr = e;   // the numbers, PAE and downloads below do not depend on the 3D view
    }
    if (!alive || my !== selGen) return false;
    const chains = d.chains || [];
    const ligandChains = chains.filter((c) => c.type === "ligand").map((c) => c.id);
    const fileBase = `${runId}_model_${d.model}`;
    if (!built) {
      mol = new MolPanel({ hasLigand: ligandChains.length > 0, binder: d.binder, ligandChains, dark: app.theme === "dark" });
      replace(elStruct, h("div", { class: "card-h" }, h("h2", null, "Structure"), elModels), mol.el);
      replace(elRes, elStrip, h("div", { class: "res-a" }, elStruct, elTabs), h("div", { class: "res-b" }, elPae, elChains));
      built = true;
    }
    replace(elResErr);
    model = shownModel = d.model;
    renderModel(d, chains, fileBase);
    setBusy(false);
    if (pose) await mol.display({ text: pose.text, format: pose.format, fileName: `${fileBase}.${pose.format === "pdb" ? "pdb" : "cif"}` });   // a failure stays in the Structure panel, with Retry
    else mol.fail(poseErr?.message || "The structure could not be loaded", () => renderResults(k));
    return true;
  }

  function renderModel(d, chains, fileBase) {
    const o = d.options || {};
    const t = run.timings || {};
    const key = d.models.map((m) => `${m.index}:${m.confidence_score}`).join(",") + ":" + d.model;
    if (key !== modelsShown) {
      modelsShown = key;
      modelsSeg = d.models.length > 1 ? radioSeg({ label: "Model", value: String(d.model),
        items: d.models.map((m) => [String(m.index), `${m.index}${m.confidence_score != null ? " · " + fmt.num(m.confidence_score, 2) : ""}`, false, `Model ${m.index}, rank ${m.index + 1} of ${d.models.length}: confidence score`]),
        onChange: (v) => { model = Number(v); renderResults(model); } }) : null;
      replace(elModels, modelsSeg);
    }
    cur = { d, fileBase };
    const kv = [["Run", runId], ["Preset", o.preset ? `${presetLabel(o.preset)}${(o.changed || []).length ? " · custom" : ""} (${o.preset})` : "–"], ["Structure steps", o.steps_reason ? `${o.steps} (${o.steps_reason})` : o.steps], ["Recycles", o.recycling], ["Structure samples", o.samples], ["Seed", o.seed]];
    if (d.affinity) kv.splice(3, 0, ["Affinity steps", o.affinity_steps], ["Affinity samples", o.affinity_samples ?? 5], ["Affinity recycles", 5]);
    kv.push(...inputRows(run.meta?.summary || {}));
    const tk = [["Load", t.load_conf_s], ["Structure", t.predict_job_s], ["Load affinity", t.load_aff_s], ["Affinity", t.affinity_job_s], ["Total", t.total_wall_s]].filter(([, v]) => v !== undefined && v !== null);
    const runCard = h("div", { class: "card" }, h("div", { class: "card-h" }, h("h2", null, "Run")),
      h("div", { class: "card-b stack" },
        h("dl", { class: "kv" }, kv.flatMap(([k, v]) => [h("dt", null, k), h("dd", { class: "num" }, v ?? "–")])),
        tk.length ? h("dl", { class: "kv" }, tk.flatMap(([k, v]) => [h("dt", null, k), h("dd", { class: "num" }, `${Number(v).toFixed(1)} s`)])) : null,
        h("div", { class: "row tight small" }, h("span", { class: "mono path" }, run.path), h("button", { type: "button", class: "btn ghost sm", onclick: () => copyText(run.path, "Path copied") }, icon("copy"), "Copy"))));
    const conf = d.confidence || {};
    // The scores count the chains of the structure: a form chain with copies is several of them (A and C), in the YAML's order
    chains = chains.flatMap((c) => (c.ids?.length ? c.ids.map((id) => ({ ...c, id })) : [c]));
    replace(elStrip, metricStrip(conf, d.affinity, chains));
    replace(elTabs, tabsCard([
      d.affinity ? ["affinity", "Affinity", affinityCard(d.affinity, { skip: ["ic50", "p"], scope: `run-level, computed on model 0${d.model === 0 ? "" : ` (showing model ${d.model})`}` })] : null,
      ["confidence", "Confidence", confidenceCard(conf, chains)],
      ["run", "Run", runCard],
    ].filter(Boolean)));
    pae = d.has_pae ? paeCard(() => api.get(`${base}/prediction/pae?model=${d.model}`), { name: `${fileBase}_pae`, rawUrl: `${base}/prediction/pae.npz?model=${d.model}` }) : null;
    replace(elPae, pae ? pae.el : h("div", { class: "card" }, h("div", { class: "card-h" }, h("h2", null, "PAE")), h("div", { class: "card-b small muted" }, "Not in this run's output")));
    replace(elChains, chainsCard(chains, conf, d.pocket));
  }

  const onTheme = (e) => { mol?.setDark(e.detail === "dark"); };
  window.addEventListener("themechange", onTheme);

  replace(root, elHead, h("div", { class: "stack" }, elProg, elResErr, elRes));
  await tick();
  return () => { alive = false; selGen++; clearTimeout(timer); window.removeEventListener("themechange", onTheme); mol?.dispose(); };
}
