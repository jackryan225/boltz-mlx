// Thin fetch wrapper: every call carries the per-server token that only this page (served by the same server) contains.

import { conn } from "./state.js";

const token = () => document.querySelector('meta[name="boltz-token"]')?.content || "";

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function raw(path, { method = "GET", body, signal } = {}) {
  const headers = { "X-Boltz-Token": token() };
  let payload;
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, { method, headers, body: payload, signal, cache: "no-store" });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    conn.set(false);
    throw new ApiError(0, "Cannot reach the server");
  }
  conn.set(true);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).error || msg;
    } catch (e) {
      /* not JSON */
    }
    if (res.status === 403) msg += " (the server restarted: reload the page)";
    throw new ApiError(res.status, msg);
  }
  return res;
}

// Fetch a file the server sends as an attachment (with the token) and save it under its server-given name.
export async function downloadFrom(path, fallbackName) {
  const res = await raw(path);
  const cd = res.headers.get("Content-Disposition") || "";
  const m = /filename="([^"]+)"/.exec(cd);
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = (m && m[1]) || fallbackName || "download";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export const api = {
  async get(path, opts) {
    return (await raw(path, opts)).json();
  },
  async post(path, body, opts) {
    return (await raw(path, { ...opts, method: "POST", body: body ?? {} })).json();
  },
  async text(path, opts) {
    return (await raw(path, opts)).text();
  },
  async postText(path, body, opts) {
    return (await raw(path, { ...opts, method: "POST", body: body ?? {} })).text();
  },
  // A picture with the lead-optimisation headers: {url, incomplete} (incomplete: the common-substructure search ran out of time).
  async depiction(path, opts) {
    const res = await raw(path, opts);
    return { url: URL.createObjectURL(await res.blob()), incomplete: res.headers.get("X-MCS-Incomplete") === "1" };
  },
  // A pose with the format the server says it is in (mmcif or pdb) and, for an aligned pose, the alignment RMSD and atom count.
  async pose(path, opts) {
    const res = await raw(path, opts);
    const n = (k) => (res.headers.get(k) === null ? null : Number(res.headers.get(k)));
    return { text: await res.text(), format: res.headers.get("X-Pose-Format") || "mmcif", rmsd: n("X-Align-RMSD"), n: n("X-Align-N") };
  },
  async blobUrl(path, opts) {
    return URL.createObjectURL(await (await raw(path, opts)).blob());
  },
  async postBlobUrl(path, body, opts) {
    return URL.createObjectURL(await (await raw(path, { ...opts, method: "POST", body })).blob());
  },
};

export const enc = encodeURIComponent;
