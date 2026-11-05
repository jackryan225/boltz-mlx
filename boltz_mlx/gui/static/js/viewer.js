// Page side of the pose viewer. Mol* (bundled in static/vendor/molstar) runs in a frame (static/viewer.html) served from the OTHER
// loopback host name (localhost <-> 127.0.0.1), i.e. a different origin: its build evaluates generated code, which this page's
// policy forbids, and in its own origin it cannot reach this page or the API token. Poses go to it by postMessage; see
// viewer-frame.js. If the other host name does not resolve, the frame is loaded from this page's own origin instead.

export class PoseViewer {
  constructor(host) {
    this.host = host;
    this.frame = null;
    this.origin = null;
    this.color = "plddt";   // plddt | chain
    this.pocket = true;
    this.binder = null;     // chain id of the ligand to focus
    this.dark = false;
    this.seq = 0;
    this.dispGen = 0;       // display(): the newest request
    this.dispChain = Promise.resolve();
    this.pending = new Map();
    this.readyResolve = null;
    this.onMessage = (e) => {
      if (!this.frame || e.source !== this.frame.contentWindow || e.origin !== this.origin) return;
      const m = e.data;
      if (!m || typeof m !== "object") return;
      if (m.type === "ready" && this.readyResolve) {
        const r = this.readyResolve;
        this.readyResolve = null;
        r(m);
      } else if (m.type === "done" && this.pending.has(m.id)) {
        const { resolve, reject } = this.pending.get(m.id);
        this.pending.delete(m.id);
        if (m.ok) resolve(m.result);
        else reject(new Error(m.error || "viewer error"));
      }
    };
  }

  async init() {
    window.addEventListener("message", this.onMessage);
    const alias = location.hostname === "localhost" ? "127.0.0.1" : "localhost";
    const origins = [`${location.protocol}//${alias}:${location.port}`, location.origin];
    let last = new Error("the viewer did not start");
    for (const origin of origins) {
      try {
        await this.tryOrigin(origin, origin === origins[0] ? 12000 : 30000);
        await this.send({ cmd: "dark", dark: this.dark });
        return;
      } catch (e) {
        last = e;
        this.removeFrame();
      }
    }
    throw last;
  }

  tryOrigin(origin, ms) {
    return new Promise((resolve, reject) => {
      const frame = document.createElement("iframe");
      frame.setAttribute("title", "3D pose viewer");
      frame.setAttribute("referrerpolicy", "no-referrer");
      Object.assign(frame.style, { border: "0", width: "100%", height: "100%", display: "block", background: "transparent" });
      this.origin = origin;
      this.frame = frame;
      let timer = null;
      const hello = setInterval(() => { try { frame.contentWindow.postMessage({ cmd: "hello" }, origin); } catch (e) { /* not loaded yet */ } }, 250);
      const done = (fn, v) => { clearInterval(hello); clearTimeout(timer); this.readyResolve = null; fn(v); };
      this.readyResolve = (m) => (m.ok ? done(resolve) : done(reject, new Error(m.error || "the viewer did not start")));
      timer = setTimeout(() => done(reject, new Error("the viewer did not start (timeout)")), ms);
      frame.src = origin + "/static/viewer.html";
      this.host.appendChild(frame);
    });
  }

  removeFrame() {
    if (this.frame) this.frame.remove();
    this.frame = null;
  }

  // A command the frame never answers (a stuck or crashed Mol*) fails after `ms` so the page can offer Retry.
  send(msg, ms = msg.cmd === "show" ? 90000 : 20000) {
    if (!this.frame) return Promise.reject(new Error("viewer closed"));
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error("viewer timeout: no answer from the 3D view")); }, ms);
      this.pending.set(id, { resolve: (v) => { clearTimeout(timer); resolve(v); }, reject: (e) => { clearTimeout(timer); reject(e); } });
      try {
        this.frame.contentWindow.postMessage({ ...msg, id }, this.origin);
      } catch (e) {
        this.pending.delete(id);
        clearTimeout(timer);
        reject(new Error("viewer closed"));
      }
    });
  }

  // show() calls one after the other, newest wins: a request that is no longer the newest (or whose `current()` has turned false) is
  // skipped before it touches Mol*, and a draw already running finishes first, so a slow old structure can never land after a newer
  // one. Resolves true when this request is what the frame now shows, false when it was superseded or failed to start.
  display(text, format, current = () => true) {
    const my = ++this.dispGen;
    const run = this.dispChain.then(async () => {
      if (my !== this.dispGen || !current()) return false;
      await this.show(text, format);
      return my === this.dispGen;
    });
    this.dispChain = run.catch(() => {});
    return run;
  }
  show(text, format = "mmcif") { return this.send({ cmd: "show", text, format, color: this.color, pocket: this.pocket, binder: this.binder }); }
  async setBinder(chain) { this.binder = chain; await this.send({ cmd: "binder", chain }); }
  setDark(dark) { this.dark = dark; return this.frame ? this.send({ cmd: "dark", dark }) : Promise.resolve(); }
  async setColor(mode) { this.color = mode; await this.send({ cmd: "color", mode }); }
  async setPocket(on) { this.pocket = on; await this.send({ cmd: "pocket", on }); }
  getCamera() { return this.send({ cmd: "camera" }); }
  setCamera(snapshot) { return this.send({ cmd: "camera", snapshot }); }
  view(mode) { return this.send({ cmd: "view", mode }); }   // "complex" | "pocket"
  resetView() { return this.send({ cmd: "reset" }).catch(() => {}); }
  resize() { return this.frame ? this.send({ cmd: "resize" }).catch(() => {}) : Promise.resolve(); }

  dispose() {
    window.removeEventListener("message", this.onMessage);
    for (const { reject } of this.pending.values()) reject(new Error("viewer closed"));
    this.pending.clear();
    this.removeFrame();
  }
}
