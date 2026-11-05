"""MLX buffer-cache limit policy: 4 GB for small inputs (today's value), larger for big inputs when the RAM allows.

The shipped CLI / screen runner cap MLX's buffer cache (mx.set_cache_limit) at 4 GB. At 1600+ tokens the pair buffers
are 0.7-5 GB each, so they never fit that cache: every freed pair buffer goes back to the OS and the next allocation
pays page-zeroing again (system CPU was 49-57 % of the wall on the M5 Max). A larger cache only changes how many freed
buffers MLX keeps for reuse, never what is computed.

Policy (n_tokens = the token dimension of the featurized input, i.e. the pair tensor is n_tokens x n_tokens):
    n_tokens <= 1000 : 4 GB (BASE_GB), exactly the old behaviour (no extra mx call is made at all)
    n_tokens  > 1000 : min(32, max(4, total_memory - estimated_peak - margin)),  margin = max(16 GB, 15 % of total)
    estimated_peak   : 7.5e-6 GB x n_tokens^2 (fit of the bench-r2 MLX peaks: 1600 -> 19.2 GB measured 19.9,
                       2400 -> 43.2 measured 43.1; 3200 -> 76.8, measured/estimated 67-77)
Because the margin is >= 16 GB, a machine with <= 20 GB RAM always gets 4 GB. "GB" is 1e9 bytes everywhere (the unit of
the old `int(gb * 1e9)`); total memory is hw.memsize / 1e9, so a 128 GiB Mac counts as 137.4.

Explicit override wins: BOLTZ_MLX_CACHE_GB set (any value, as before) fixes the limit for the whole process and the policy
never touches it; so does a tool that sets its own limit (it never calls `install_startup`, so the policy stays disarmed).

Usage (a site that used to call mx.set_cache_limit(4 GB) at start-up):
    cachelimit.install_startup()                       # start-up, N unknown: 4 GB (or the env override); arms the policy
    ...featurize...
    cachelimit.adapt_feats(feats)                      # N known, before model(feats, ...): re-sets the limit if the policy says so
adapt() is a no-op (no mx call, no output) whenever the policy value equals the limit already in force, which is always
the case for n_tokens <= 1000. It logs one line to stderr each time it changes the limit.
"""
from __future__ import annotations

import os
import sys

ENV = "BOLTZ_MLX_CACHE_GB"
BASE_GB = 4.0        # the shipped limit; also the floor
MAX_GB = 32.0        # the ceiling (the M5 Max A/B: 32 GB removed the allocator churn at 3200 tokens)
SMALL_N = 1000       # at or below this many tokens the limit is BASE_GB, unconditionally
MARGIN_MIN_GB = 16.0
MARGIN_FRAC = 0.15
PEAK_COEF_GB = 7.5e-6  # GB per token^2 above SMALL_N (bench-r2 MLX peaks)

_STATE = {"armed": False, "current_gb": None, "total_gb": None}


def estimate_peak_gb(n_tokens: int) -> float:
    """Estimated MLX peak memory of one prediction (GB): 7.5e-6 x N^2, the fit that holds above 1000 tokens."""
    return PEAK_COEF_GB * float(n_tokens) ** 2


def cache_limit_gb(n_tokens: int, total_mem_gb: float, est_peak_gb: float | None = None) -> float:
    """The policy (pure function): cache limit in GB for an input of n_tokens on a machine with total_mem_gb of RAM."""
    if n_tokens <= SMALL_N:
        return BASE_GB
    if est_peak_gb is None:
        est_peak_gb = estimate_peak_gb(n_tokens)
    margin = max(MARGIN_MIN_GB, MARGIN_FRAC * total_mem_gb)
    return round(min(MAX_GB, max(BASE_GB, total_mem_gb - est_peak_gb - margin)), 2)  # 0.01 GB: 128 - 76.8 - 19.2 is 32.0, not 31.999...


def env_override_gb(environ=None) -> float | None:
    """BOLTZ_MLX_CACHE_GB as a float, None when unset or empty. A non-numeric value raises ValueError (as before)."""
    v = (os.environ if environ is None else environ).get(ENV)
    if v is None or str(v).strip() == "":
        return None
    return float(v)


def total_memory_gb() -> float | None:
    """Physical memory in GB (1e9 bytes): the Metal device's memory_size, else sysconf; None when neither is known."""
    b = 0.0
    try:
        import mlx.core as mx

        d = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        b = float(d.get("memory_size", 0))
    except Exception:  # CPU-only MLX build, no Metal device
        b = 0.0
    if b <= 0:
        try:
            b = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            return None
    return b / 1e9


def n_tokens_of(feats) -> int:
    """Token dimension of a featurized batch (feats["token_pad_mask"] is [batch, N]): the N of the N x N pair tensors."""
    return int(feats["token_pad_mask"].shape[-1])


def _set_limit(gb: float) -> None:
    import mlx.core as mx

    mx.set_cache_limit(int(gb * 1e9))


def install_startup(environ=None) -> float:
    """The start-up call of a prediction process (replaces the old mx.set_cache_limit(int(BOLTZ_MLX_CACHE_GB or 4 GB))).
    Sets the env override if present (policy stays disarmed: explicit wins), else BASE_GB and arms the policy for adapt().
    Returns the limit set in GB. environ={} ignores the process environment (tools that arm without a user override)."""
    ov = env_override_gb(environ)
    gb = BASE_GB if ov is None else ov
    _set_limit(gb)
    _STATE.update(armed=ov is None, current_gb=gb, total_gb=None)
    return gb


def set_explicit(gb: float) -> float:
    """A tool's explicit limit (bench_mlx --cache-limit-gb G): set it, disarm the policy."""
    _set_limit(gb)
    _STATE.update(armed=False, current_gb=gb)
    return gb


def current_gb() -> float | None:
    """The limit this module last set (None before install_startup / set_explicit)."""
    return _STATE["current_gb"]


def adapt(n_tokens: int, total_mem_gb: float | None = None, est_peak_gb: float | None = None, log=None) -> float | None:
    """Apply the policy for an input of n_tokens: call once the token count is known and before the trunk runs (per
    record; cheap). Returns the new limit in GB when it changed, None when nothing was done: policy disarmed (explicit
    override or a tool-set limit), unknown RAM, or the policy value equals the limit in force (always true for
    n_tokens <= 1000 after install_startup). Logs the change to stderr (or log(msg))."""
    if not _STATE["armed"]:
        return None
    total = total_mem_gb
    if total is None:
        if _STATE["total_gb"] is None:
            _STATE["total_gb"] = total_memory_gb()
        total = _STATE["total_gb"]
    gb = BASE_GB if total is None else cache_limit_gb(n_tokens, total, est_peak_gb)
    if gb == _STATE["current_gb"]:
        return None
    _set_limit(gb)
    prev, _STATE["current_gb"] = _STATE["current_gb"], gb
    why = "" if total is None else (f" on {total:.0f} GB RAM, estimated peak "
                                    f"{est_peak_gb if est_peak_gb is not None else estimate_peak_gb(n_tokens):.1f} GB"
                                    if n_tokens > SMALL_N else f" (up to {SMALL_N} tokens: the old limit)")
    msg = f"boltz-mlx: MLX buffer cache limit {gb:.1f} GB (was {prev:.1f}) for {n_tokens} tokens{why}; {ENV} overrides"
    (log or (lambda m: print(m, file=sys.stderr)))(msg)
    return gb


def adapt_feats(feats, **kw) -> float | None:
    """adapt() for a featurized batch (reads the token count only when the policy is armed)."""
    if not _STATE["armed"]:
        return None
    return adapt(n_tokens_of(feats), **kw)
