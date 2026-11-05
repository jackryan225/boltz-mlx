"""Opt-in MSA server (--use_msa_server and friends): CPU only, no GPU, no real network except the one live test.

Unit tests mock the HTTP client (requests.post / requests.get inside upstream's boltz.data.msa.mmseqs2) and block sockets,
so a stray request fails loudly. The live test (BOLTZ_MSA_LIVE=1) asks api.colabfold.com for the MSA of PDB 8eil chain A.
Needs the upstream molecule dictionary (<cache>/mols); skipped when it is not installed.
"""

from __future__ import annotations

import io
import os
import socket
import tarfile
from pathlib import Path

import pytest
import yaml

from boltz_mlx import cli
from boltz_mlx.paths import default_cache

REPO = Path(__file__).resolve().parents[1]
SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
SEQ_B = "GSHMKPGFLYTIGLSNKGMPGLYRLELQVTKGSGKLATSGLWNSSSAKEQVKIAFDYFKANA"
SEQ_8EIL = yaml.safe_load((REPO / "cases" / "nim_8eil.yaml").read_text())["sequences"][0]["protein"]["sequence"]


def _cache() -> Path:
    for c in (default_cache(),):
        if (c / "mols").is_dir():
            return c
    pytest.skip("upstream molecule dictionary (<cache>/mols) not installed")


def _args(*extra: str, data: str = "x.yaml"):
    return cli.parse_args(["predict", data, "--device", "cpu", *extra])


# ----------------------------------------------------------------------------------------------- fake MSA server
class _Resp:
    def __init__(self, js=None, content=b""):
        self._js, self.content, self.text, self.status_code = js, content, "", 200

    def json(self):
        return self._js


def _mut(seq: str) -> str:
    return seq[:-1] + ("G" if seq[-1] == "A" else "A")


class FakeServer:
    """Stands in for requests.post / requests.get as used by boltz.data.msa.mmseqs2.run_mmseqs2."""

    def __init__(self):
        self.posts, self.gets, self.queries = [], [], []

    def post(self, url, data=None, timeout=None, headers=None, auth=None):
        q = data["q"].split()
        self.queries.append([(int(a[1:]), s) for a, s in zip(q[::2], q[1::2])])
        self.posts.append(dict(url=url, mode=data["mode"], headers=headers, auth=auth))
        return _Resp({"status": "COMPLETE", "id": f"T{len(self.posts)}"})

    def get(self, url, timeout=None, headers=None, auth=None):
        self.gets.append(dict(url=url, headers=headers, auth=auth))
        qs = self.queries[-1]
        blocks = lambda tag: "".join(  # noqa: E731  one block per query, separated by \x00 (the server's a3m layout)
            ("\x00" if k else "") + f">{i}\n{s}\n>{tag}{i}\n{_mut(s)}\n" for k, (i, s) in enumerate(qs))
        files = {"pair.a3m": blocks("pair"), "uniref.a3m": blocks("uni"), "bfd.mgnify30.metaeuk30.smag30.a3m": blocks("bfd")}
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, text in files.items():
                b = text.encode()
                ti = tarfile.TarInfo(name)
                ti.size = len(b)
                tar.addfile(ti, io.BytesIO(b))
        return _Resp(content=buf.getvalue())


@pytest.fixture
def server(monkeypatch):
    """Mock HTTP client + no real sockets."""
    import boltz.data.msa.mmseqs2 as M

    s = FakeServer()
    monkeypatch.setattr(M.requests, "post", s.post)
    monkeypatch.setattr(M.requests, "get", s.get)

    def no_net(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", no_net)
    monkeypatch.setattr(socket, "getaddrinfo", no_net)
    return s


def _yaml(tmp_path, seqs: dict[str, str | None], name="job") -> Path:
    """seqs: chain id -> sequence; the MSA is left out (server fills it) unless the id ends in '!' (msa: empty)."""
    ents = []
    for cid, seq in seqs.items():
        p = {"id": cid.rstrip("!"), "sequence": seq}
        if cid.endswith("!"):
            p["msa"] = "empty"
        ents.append({"protein": p})
    f = tmp_path / f"{name}.yaml"
    f.write_text(yaml.safe_dump({"version": 1, "sequences": ents}))
    return f


def _run(args, data: Path, out: Path) -> Path:
    """The featurisation step of `predict` (same call, same kwargs), nothing else."""
    from boltz.main import check_inputs, process_inputs

    cache = _cache()
    process_inputs(data=check_inputs(data), out_dir=out, ccd_path=cache / "ccd.pkl", mol_dir=cache / "mols",
                   **cli.msa_server_kwargs(args), boltz2=True, preprocessing_threads=1, max_msa_seqs=args.max_msa_seqs)
    return out


# ------------------------------------------------------------------------------------------------------- flags
def test_flags_off_by_default_with_upstream_names_and_defaults():
    a = _args()
    assert a.use_msa_server is False
    assert (a.msa_server_url, a.msa_pairing_strategy) == ("https://api.colabfold.com", "greedy")
    assert (a.msa_server_username, a.msa_server_password, a.api_key_header, a.api_key_value) == (None, None, None, None)
    txt = cli.build_parser()._subparsers._group_actions[0].choices["predict"].format_help()
    assert "--use_msa_server" in txt and "default off" in txt


def test_kwargs_off_ignore_environment(monkeypatch):
    monkeypatch.setenv("BOLTZ_MSA_USERNAME", "u")
    monkeypatch.setenv("BOLTZ_MSA_PASSWORD", "p")
    kw = cli.msa_server_kwargs(_args("--msa_server_url=http://elsewhere"))
    assert kw["use_msa_server"] is False and "msa_server_username" not in kw and "api_key_value" not in kw


def test_kwargs_on_pass_flags_and_env_fallback(monkeypatch):
    monkeypatch.setenv("BOLTZ_MSA_USERNAME", "envu")
    monkeypatch.setenv("BOLTZ_MSA_PASSWORD", "envp")
    monkeypatch.setenv("MSA_API_KEY_VALUE", "envkey")
    kw = cli.msa_server_kwargs(_args("--use_msa_server", "--msa_server_url=http://h", "--msa_pairing_strategy=complete",
                                     "--msa_server_username=me", "--api_key_header=X-Foo"))
    assert kw == dict(use_msa_server=True, msa_server_url="http://h", msa_pairing_strategy="complete",
                      msa_server_username="me", msa_server_password="envp", api_key_header="X-Foo", api_key_value="envkey")


def test_predict_hands_flags_to_upstream_featuriser(monkeypatch, tmp_path):
    """predict(): process_inputs gets use_msa_server False by default, and the user's flags when set (no model is loaded)."""
    import boltz.main as bm

    seen = []

    class Stop(Exception):
        pass

    def fake(**kw):
        seen.append(kw)
        raise Stop

    monkeypatch.setattr(bm, "process_inputs", fake)
    f = _yaml(tmp_path, {"A": SEQ_A})
    common = [str(f), f"--out_dir={tmp_path / 'o'}", f"--cache={tmp_path}", "--device", "cpu"]
    for extra, want in (([], False), (["--use_msa_server", "--msa_server_url=http://h"], True)):
        with pytest.raises(Stop):
            cli.predict(cli.parse_args(["predict", *common, *extra]))
        assert seen[-1]["use_msa_server"] is want
    assert seen[-1]["msa_server_url"] == "http://h"
    import json

    st = json.loads((tmp_path / "o" / "boltz_results_job" / "boltz_mlx_settings.json").read_text())
    assert st["msa_server"] == "http://h"


# ---------------------------------------------------------------------------------------- featuriser behaviour
def test_flag_off_makes_no_network_call(server, tmp_path, capsys):
    out = _run(_args(), _yaml(tmp_path, {"A": SEQ_A}), tmp_path / "o")
    assert server.posts == [] and server.gets == []
    assert not list((out / "msa").glob("*.csv"))
    assert "Missing MSA's in input and --use_msa_server flag not set" in capsys.readouterr().out


def test_flag_on_but_every_chain_has_an_msa_makes_no_call(server, tmp_path):
    _run(_args("--use_msa_server"), _yaml(tmp_path, {"A!": SEQ_A}), tmp_path / "o")
    assert server.posts == []


def test_monomer_sends_sequence_and_writes_csv_where_featuriser_reads_it(server, tmp_path):
    out = _run(_args("--use_msa_server"), _yaml(tmp_path, {"A": SEQ_A}), tmp_path / "o")
    assert len(server.posts) == 1  # a monomer needs no paired request
    p = server.posts[0]
    assert p["url"] == "https://api.colabfold.com/ticket/msa" and p["headers"]["User-Agent"] == "boltz" and p["auth"] is None
    assert server.queries == [[(101, SEQ_A)]]
    csv = out / "msa" / "job_0.csv"
    rows = csv.read_text().splitlines()
    assert rows[0] == "key,sequence"
    assert rows[1] == f"-1,{SEQ_A}" and f"-1,{_mut(SEQ_A)}" in rows and len(rows) == 5  # header + (query, hit) from each of uniref and bfd, as upstream concatenates them
    assert (out / "processed" / "msa" / "job_0.npz").exists()  # parsed by the featuriser's own parse_csv
    assert (out / "processed" / "records" / "job.json").exists()


def test_heteromer_pairs_and_only_missing_chains_are_sent(server, tmp_path):
    out = _run(_args("--use_msa_server", "--msa_pairing_strategy=complete"),
               _yaml(tmp_path, {"A": SEQ_A, "B": SEQ_B, "C!": SEQ_A[:30]}), tmp_path / "o")
    assert [p["url"].rsplit("/", 1)[1] for p in server.posts] == ["pair", "msa"]  # paired request first, then unpaired
    assert server.posts[0]["mode"] == "paircomplete-env"
    assert all(q == [(101, SEQ_A), (102, SEQ_B)] for q in server.queries)  # chain C (msa: empty) never leaves the machine
    for k, seq in enumerate((SEQ_A, SEQ_B)):
        rows = (out / "msa" / f"job_{k}.csv").read_text().splitlines()
        assert rows[1:3] == [f"0,{seq}", f"1,{_mut(seq)}"]  # paired rows: key = row index, shared by both chains
        assert any(r.startswith("-1,") for r in rows)


def test_credentials_reach_the_client(server, tmp_path):
    _run(_args("--use_msa_server", "--api_key_header=X-Tok", "--api_key_value=sekrit", "--msa_server_url=http://h:1"),
         _yaml(tmp_path, {"A": SEQ_A}), tmp_path / "o")
    assert server.posts[0]["url"] == "http://h:1/ticket/msa"
    assert server.posts[0]["headers"]["X-Tok"] == "sekrit"
    server.posts.clear()
    _run(_args("--use_msa_server", "--msa_server_username=u", "--msa_server_password=p"), _yaml(tmp_path, {"A": SEQ_B}, "j2"),
         tmp_path / "o2")
    assert (server.posts[0]["auth"].username, server.posts[0]["auth"].password) == ("u", "p")


def test_both_auth_methods_refused_before_any_request(server, tmp_path):
    with pytest.raises(ValueError, match="Cannot use both"):
        _run(_args("--use_msa_server", "--msa_server_username=u", "--msa_server_password=p", "--api_key_value=k"),
             _yaml(tmp_path, {"A": SEQ_A}), tmp_path / "o")
    assert server.posts == []


# ------------------------------------------------------------------------------------------------------- GUI form
def _gui_chain(mode):
    return {"type": "protein", "id": "A", "sequence": SEQ_A, "msa": {"mode": mode}}


def test_gui_msa_server_choice_is_opt_in():
    from boltz_mlx.gui import predict as PR

    for mode in (None, "empty"):  # default: no server, `msa: empty`, no flag
        ck = PR.check_chains({"chains": [dict(_gui_chain(mode), **({} if mode else {"msa": {}}))]})
        assert yaml.safe_load(PR.build_yaml(ck))["sequences"][0]["protein"]["msa"] == "empty"
    ck = PR.check_chains({"chains": [_gui_chain("server")]})
    assert any("sent to the public MSA server" in w for w in ck["warnings"])
    assert "msa" not in yaml.safe_load(PR.build_yaml(ck))["sequences"][0]["protein"]  # absent key = upstream asks the server
    assert PR.public_chains(ck)[0]["msa"] == "server"


def test_gui_predict_argv_flag_only_for_server_chains(tmp_path):
    from boltz_mlx.gui import predict as PR

    options = {"preset": "strict", "steps": 200, "recycling": 3, "affinity_steps": 200, "samples": 1, "seed": 0}
    for mode, want in (("empty", False), ("server", True)):
        rec = PR.stage(PR.check_chains({"chains": [_gui_chain(mode)]}), tmp_path / mode / "inputs")
        assert ("--use_msa_server" in PR.predict_argv(tmp_path / mode, rec, options)) is want


# ------------------------------------------------------------------------------------------------------ live check
@pytest.mark.skipif(os.environ.get("BOLTZ_MSA_LIVE") != "1", reason="live network check: set BOLTZ_MSA_LIVE=1")
def test_live_colabfold_8eil_chain_a(tmp_path):
    """Real request to api.colabfold.com: PDB 8eil chain A (186 aa, public, from cases/nim_8eil.yaml)."""
    assert len(SEQ_8EIL) == 186
    out = _run(_args("--use_msa_server"), _yaml(tmp_path, {"A": SEQ_8EIL}, "nim_8eil"), tmp_path / "o")
    rows = (out / "msa" / "nim_8eil_0.csv").read_text().splitlines()
    print(f"\nLIVE: {len(rows) - 1} MSA rows for 8eil chain A; first sequence == query: {rows[1] == '-1,' + SEQ_8EIL}")
    assert rows[0] == "key,sequence" and rows[1] == f"-1,{SEQ_8EIL}"
    assert len(rows) > 101  # a real family, not just the query (rows keep a3m insertions, so lengths vary: as upstream)
    assert (out / "processed" / "msa" / "nim_8eil_0.npz").exists()
