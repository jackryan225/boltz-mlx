// Small DOM and formatting helpers. Text is always inserted with textContent / createTextNode, never as HTML,
// so ligand ids, SMILES and error messages from files cannot inject markup.

const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (k === "text") el.textContent = v;
      else if (v === true) el.setAttribute(k, "");
      else el.setAttribute(k, v);
    }
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function replace(el, ...children) {
  clear(el);
  return append(el, children);
}

const ICONS = {
  info: "M12 8h.01M11 12h1v5h1 M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10z M12 1v2 M12 21v2 M4.2 4.2l1.4 1.4 M18.4 18.4l1.4 1.4 M1 12h2 M21 12h2 M4.2 19.8l1.4-1.4 M18.4 5.6l1.4-1.4",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  upload: "M12 16V4 M7 9l5-5 5 5 M4 20h16",
  play: "M6 4l14 8-14 8z",
  stop: "M6 6h12v12H6z",
  refresh: "M20 11a8 8 0 0 0-14.9-3 M4 4v4h4 M4 13a8 8 0 0 0 14.9 3 M20 20v-4h-4",
  download: "M12 3v12 M7 10l5 5 5-5 M4 21h16",
  check: "M5 13l4 4L19 7",
  x: "M6 6l12 12 M18 6L6 18",
  alert: "M12 9v4 M12 17h.01 M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  clock: "M12 7v5l3 2 M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
  trash: "M4 7h16 M9 7V4h6v3 M6 7l1 13h10l1-13",
  arrow: "M5 12h14 M13 6l6 6-6 6",
  back: "M19 12H5 M11 6l-6 6 6 6",
  cube: "M12 2l9 5v10l-9 5-9-5V7z M12 12l9-5 M12 12v10 M12 12L3 7",
  flask: "M9 3h6 M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2.2h12.4a1.5 1.5 0 0 0 1.3-2.2L14 9V3",
  file: "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z M14 3v5h5",
  list: "M8 6h13 M8 12h13 M8 18h13 M3 6h.01 M3 12h.01 M3 18h.01",
  copy: "M9 9h11v11H9z M5 15V5a1 1 0 0 1 1-1h10",
  plus: "M12 5v14 M5 12h14",
  menu: "M4 7h16 M4 12h16 M4 17h16",
  chip: "M7 7h10v10H7z M10 3v4 M14 3v4 M10 17v4 M14 17v4 M3 10h4 M3 14h4 M17 10h4 M17 14h4",
  helix: "M7 3c0 6 10 6 10 12s-10 3-10 6 M17 3c0 6-10 6-10 12 M9 7h6 M9 17h6",
  hex: "M12 3l7.8 4.5v9L12 21l-7.8-4.5v-9z M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7z",
  rows: "M4 6h16 M4 12h10 M4 18h6 M18 14v6 M15 17l3 3 3-3",
  lock: "M6 11h12v10H6z M9 11V8a3 3 0 0 1 6 0v3",
  grid: "M4 4h7v7H4z M13 4h7v7h-7z M4 13h7v7H4z M13 13h7v7h-7z",
  target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z M12 12h.01",
  spark: "M12 3v4 M12 17v4 M3 12h4 M17 12h4 M6 6l2.5 2.5 M15.5 15.5L18 18 M6 18l2.5-2.5 M15.5 8.5L18 6",
  edit: "M4 20h4L19 9l-4-4L4 16z M14 6l4 4",
  external: "M14 4h6v6 M20 4l-9 9 M18 14v6H4V6h6",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14z M20 20l-4-4",
};

export function icon(name, cls = "icon") {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", cls);
  for (const d of (ICONS[name] || "").split(" M").map((s, i) => (i ? "M" + s : s))) {
    if (!d) continue;
    const p = document.createElementNS(SVG_NS, "path");
    p.setAttribute("d", d);
    svg.appendChild(p);
  }
  return svg;
}

// ------------------------------------------------------------------------------------------------ formatting
export const fmt = {
  num(v, d = 3) {
    return v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d);
  },
  int(v) {
    return v === null || v === undefined ? "–" : Number(v).toLocaleString("en-US");
  },
  plural(n, word) {   // "1 ligand", "3 ligands"
    return `${fmt.int(n)} ${word}${Number(n) === 1 ? "" : "s"}`;
  },
  p(v) {  // P(binder): a probability, two decimals everywhere
    return v === null || v === undefined || !Number.isFinite(v) ? "–" : Number(v).toFixed(2);
  },
  pct(v, d = 0) {
    return v === null || v === undefined ? "–" : (100 * v).toFixed(d) + "%";
  },
  plddt(v) {
    return v === null || v === undefined ? "–" : (v <= 1.0001 ? v * 100 : v).toFixed(1);
  },
  dur(s) {
    if (s === null || s === undefined || !Number.isFinite(s)) return "–";
    s = Math.max(0, Math.round(s));
    if (s < 60) return s + " s";
    const m = Math.floor(s / 60);
    if (s < 3600) return m + " min " + String(s % 60).padStart(2, "0") + " s";
    const hh = Math.floor(s / 3600);
    if (s < 86400) return hh + " h " + String(Math.floor((s % 3600) / 60)).padStart(2, "0") + " min";
    return Math.floor(s / 86400) + " d " + String(Math.floor((s % 86400) / 3600)) + " h";
  },
  roughDur(s) {
    if (s === null || s === undefined || !Number.isFinite(s)) return "–";
    if (s < 90) return "about " + Math.max(1, Math.round(s)) + " s";
    if (s < 5400) return "about " + Math.round(s / 60) + " min";
    if (s < 172800) return "about " + (s / 3600).toFixed(1) + " h";
    return "about " + (s / 86400).toFixed(1) + " days";
  },
  roughSpan(lo, hi) {  // a measured total range in one unit (from the larger end, as the plan line): "about 27–29 min"; "about 2 min" when both ends round alike
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return "–";
    const [d, u, f] = hi < 120 ? [1, "s", 0] : hi < 7200 ? [60, "min", 0] : hi < 172800 ? [3600, "h", 1] : [86400, "days", 1];
    const a = (Math.max(lo / d, d === 1 ? 1 : 0)).toFixed(f), b = (hi / d).toFixed(f);
    return `about ${a === b ? a : `${a}–${b}`} ${u}`;
  },
  when(iso) {
    if (!iso) return "–";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    const now = new Date();
    const same = d.toDateString() === now.toDateString();
    const t = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return same ? "today " + t : d.toLocaleDateString([], { year: "numeric", month: "short", day: "numeric" }) + " " + t;
  },
  bytes(n) {
    if (n === null || n === undefined) return "";
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / 1024 / 1024).toFixed(1) + " MB";
  },
};

export function el$(sel, root = document) {
  return root.querySelector(sel);
}

// ------------------------------------------------------------------------------------------------ toasts, misc
export function toast(message, kind = "") {
  const box = document.getElementById("toasts");
  if (!box) return;
  const t = h("div", { class: "toast " + kind, role: kind === "bad" ? "alert" : "status" }, message);
  box.appendChild(t);
  setTimeout(() => t.remove(), kind === "bad" ? 7000 : 3500);
}

export function debounce(fn, ms) {
  let t = null;
  const d = (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
  d.cancel = () => clearTimeout(t);
  return d;
}

export function download(name, text, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem("boltz-gui:" + key) || "null");
    localStorage.setItem("boltz-gui:" + key, JSON.stringify(value));
  } catch (e) {
    /* storage can be unavailable (private windows): the page works without it */
  }
  return null;
}

// Text with `code` segments (the privacy statement has one) as DOM nodes.
export function withCode(text) {
  const parts = String(text).split("`");
  return parts.map((p, i) => (i % 2 ? h("code", null, p) : p));
}
