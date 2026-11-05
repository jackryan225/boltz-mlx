"""HEAD-check the download sources of `boltz-mlx setup` (boltz_mlx/assets.py) without downloading anything.

  python tools/check_asset_urls.py        exit 0 when every URL answers and the size / sha256 the server reports match the pins

Follows redirects by hand (HEAD only) and reads Hugging Face's x-linked-size / x-linked-etag (the LFS sha256) on the way.
"""

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boltz_mlx.assets import ASSETS, USER_AGENT  # noqa: E402


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def head(url: str, hops: int = 6) -> dict:
    opener = urllib.request.build_opener(_NoRedirect)
    seen = {"chain": []}
    for _ in range(hops):
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
        try:
            r = opener.open(req, timeout=30)
            status, h = r.status, r.headers
        except urllib.error.HTTPError as ex:
            status, h = ex.code, ex.headers
        seen["chain"].append(f"{status} {url.split('?')[0]}")
        for k in ("x-linked-size", "x-linked-etag", "content-length", "accept-ranges"):
            if h.get(k):
                seen[k] = h.get(k).strip('"')
        if status in (301, 302, 303, 307, 308) and h.get("location"):
            url = urllib.parse.urljoin(url, h["location"])
            continue
        seen["status"] = status
        return seen
    seen["status"] = "too many redirects"
    return seen


def main() -> int:
    ok = True
    for a in ASSETS.values():
        for url in a.urls:
            r = head(url)
            size = int(r.get("x-linked-size") or r.get("content-length") or -1)
            sha = r.get("x-linked-etag")
            good = r["status"] == 200 and size == a.size and (sha is None or sha == a.sha256)
            ok &= good
            print(f"{'ok  ' if good else 'FAIL'} {a.name:<17} {url}\n     size {size} (pinned {a.size}), sha256 "
                  f"{sha or 'not reported'}{' (matches pin)' if sha == a.sha256 else ''}, ranges {r.get('accept-ranges')}"
                  f"\n     {' -> '.join(r['chain'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
