// Runs inside the viewer frame (static/viewer.html, served from the other loopback host name, so it is a different origin than the
// app page): Mol* showing one protein-ligand pose, coloured by pLDDT, with the ligand highlighted and its pocket shown (Mol*'s own
// "focus": ligand ball-and-stick, residues within 5 A as sticks, contacts; the same 5 A as the pocket lists next to the view).
// Messages from the page: {cmd: "hello"} and {id, cmd: "show"|"color"|"pocket"|"reset"|"dark"|"resize", ...};
// replies: {type: "ready", ok} and {type: "done", id, ok, error}. Only the parent window, from one of this server's own
// origins, is listened to; replies go to that origin only.
(function () {
  "use strict";
  const LIGAND_CARBON = 0xe5006e; // magenta: stands out against pLDDT blues / oranges and element colours
  const LIGAND_THEME = { color: "element-symbol", colorParams: { carbonColor: { name: "uniform", params: { value: LIGAND_CARBON } }, lightness: 0 } };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const ALLOWED = new Set([`${location.protocol}//127.0.0.1:${location.port}`, `${location.protocol}//localhost:${location.port}`]);
  let parentOrigin = null;
  let ready = null;      // {ok, error} once Mol* has started (or failed to)
  const post = (m) => { if (parentOrigin) window.parent.postMessage(m, parentOrigin); };
  let viewer = null;
  let color = "plddt";
  let pocket = true;
  let binder = null;     // chain id of the ligand to focus (the affinity binder); null: every ligand
  let dark = false;
  let token = 0;
  let focusLoci = null;  // the ligand of the current pose, for the Pocket camera
  const POCKET_PAD = 9;  // Å of padding around the ligand so its contacts stay in view

  const managers = () => viewer.plugin.managers;
  const structure = () => managers().structure.hierarchy.current.structures[0];

  function setDark(on) {
    dark = on;
    document.documentElement.classList.toggle("dark", on);  // viewer.html: light icons on dark controls (Mol*'s CSS is the light skin)
    try {
      viewer.plugin.canvas3d.setProps({ renderer: { backgroundColor: on ? 0x0b0b0c : 0xffffff } });
    } catch (e) { /* property names differ between versions: keep the default background */ }
  }

  async function show(text, format) {
    const my = ++token;
    focusLoci = null;
    await viewer.plugin.clear();
    await viewer.loadStructureFromData(text, format === "pdb" ? "pdb" : "mmcif", { dataLabel: "pose" });
    if (my !== token) return;
    await applyStyle();
  }

  async function applyStyle() {
    const my = ++token;
    const S = managers().structure;
    const st = structure();
    if (!st) return;
    const theme = color === "plddt" ? { color: "plddt-confidence" } : { color: "chain-id" };
    const isLigand = (c) => /ligand|focus-target/.test(c.key);
    try {
      await S.component.updateRepresentationsTheme(st.components.filter((c) => !isLigand(c)), theme);
      await S.component.updateRepresentationsTheme(st.components.filter(isLigand), LIGAND_THEME);
    } catch (e) {
      console.warn("theme update failed", e);
    }
    if (pocket) await focusLigand(my);
    else {
      S.focus.clear();
      managers().camera.reset();
    }
  }

  async function focusLigand(my) {
    const p = viewer.plugin;
    const S = managers().structure;
    const st = structure();
    if (!st) return;
    const query = p.query.structure.registry.list.find((q) => q.label === "Ligand");
    if (!query) return;
    S.selection.clear();
    S.selection.fromSelectionQuery("set", query, false);
    await sleep(60);
    let loci = S.selection.getLoci(st.cell.obj.data);
    S.selection.clear();
    if (binder && loci && loci.elements) {  // keep only the binder's chain (a complex can hold cofactors or several ligands)
      const keep = loci.elements.filter((e) => {
        try {
          const u = e.unit, ch = u.model.atomicHierarchy.chains, ci = u.chainIndex[u.elements[0]];
          return String(ch.auth_asym_id.value(ci)) === binder || String(ch.label_asym_id.value(ci)) === binder;
        } catch (err) { return false; }
      });
      if (keep.length) loci = { ...loci, elements: keep };
    }
    if (!loci || !loci.elements || !loci.elements.length) {
      managers().camera.reset();
      return;
    }
    S.focus.setFromLoci(loci);
    try { p.managers.interactivity.lociSelects.deselectAll(); } catch (e) { /* no selection overlay to remove */ }
    focusLoci = loci;
    managers().camera.focusLoci(loci, { durationMs: 250, extraRadius: POCKET_PAD });
    // the focus components appear asynchronously: restyle them once they exist
    for (let i = 0; i < 25; i++) {
      await sleep(100);
      if (my !== token) return;
      const cur = structure();
      const target = cur && cur.components.find((c) => c.key === "structure-focus-target-sel");
      const surr = cur && cur.components.find((c) => c.key === "structure-focus-surr-sel");
      if (target && target.representations.length && surr && surr.representations.length) {
        try {
          await S.component.updateRepresentationsTheme([target], LIGAND_THEME);
          await S.component.updateRepresentationsTheme([surr], color === "plddt" ? { color: "plddt-confidence" } : { color: "chain-id" });
        } catch (e) {
          console.warn("focus theme update failed", e);
        }
        return;
      }
    }
  }

  const handlers = {
    show: (m) => { color = m.color || color; pocket = m.pocket !== undefined ? !!m.pocket : pocket; binder = m.binder || null; return show(m.text, m.format); },
    binder: async (m) => { binder = m.chain || null; if (structure() && pocket) await applyStyle(); },
    color: async (m) => { color = m.mode; if (structure()) await applyStyle(); },
    pocket: async (m) => { pocket = !!m.on; if (structure()) await applyStyle(); },
    reset: async () => { managers().camera.reset(); },
    view: async (m) => {  // "complex": the whole structure; "pocket": the ligand with padding
      if (m.mode === "pocket" && focusLoci) managers().camera.focusLoci(focusLoci, { durationMs: 250, extraRadius: POCKET_PAD });
      else managers().camera.reset();
    },
    camera: async (m) => {  // no snapshot: read the camera; with one: apply it (poses of a comparison share one coordinate frame)
      const cam = viewer.plugin.canvas3d.camera;
      if (!m.snapshot) return JSON.parse(JSON.stringify(cam.getSnapshot(), (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : v)));
      cam.setState(m.snapshot, 0);
      try { viewer.plugin.canvas3d.requestDraw(true); } catch (e) { /* older Mol* */ }
      return null;
    },
    dark: async (m) => { setDark(!!m.dark); },
    resize: async () => { viewer.handleResize(); try { viewer.plugin.canvas3d.requestDraw(true); } catch (e) { /* older Mol*: no explicit redraw */ } },
  };

  window.addEventListener("message", async (e) => {
    if (e.source !== window.parent || !ALLOWED.has(e.origin)) return;
    parentOrigin = e.origin;
    const m = e.data;
    if (!m || typeof m !== "object") return;
    if (m.cmd === "hello") {
      if (ready) post({ type: "ready", ...ready });
      return;
    }
    if (typeof m.id !== "number" || !handlers[m.cmd] || !viewer) return;
    try {
      const result = await handlers[m.cmd](m);
      post({ type: "done", id: m.id, ok: true, result: m.cmd === "camera" ? result ?? undefined : undefined });
    } catch (err) {
      console.error(err);
      post({ type: "done", id: m.id, ok: false, error: String((err && err.message) || err) });
    }
  });

  async function start() {
    if (!window.molstar || !window.molstar.Viewer) {
      ready = { ok: false, error: "the bundled Mol* script did not load" };
      post({ type: "ready", ...ready });
      return;
    }
    try {
      viewer = await window.molstar.Viewer.create("app", {
        layoutIsExpanded: false, layoutShowControls: false, layoutShowRemoteState: false, layoutShowSequence: false,
        layoutShowLog: false, layoutShowLeftPanel: false, viewportShowExpand: false, viewportShowSelectionMode: false,
        viewportShowAnimation: false, viewportShowSettings: true,
        // nothing is ever requested from these servers (poses are posted to this frame); the page policy would block it anyway
        pdbProvider: "pdbe", emdbProvider: "pdbe",
        // Volseg lists its entries at start-up, from http://localhost:9000 when the frame's host is "localhost": blocked by the
        // page policy, it left an uncaught "Failed to fetch" in the console of every structure view
        disabledExtensions: ["volseg"],
      });
      window.addEventListener("resize", () => { try { viewer.handleResize(); } catch (e) { /* ignore */ } });
      ready = { ok: true };
    } catch (err) {
      console.error(err);
      ready = { ok: false, error: String((err && err.message) || err) };
    }
    post({ type: "ready", ...ready });
  }
  start();
})();
