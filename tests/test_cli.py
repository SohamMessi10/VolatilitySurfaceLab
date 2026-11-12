"""Tests for the ``vsl`` command line interface."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from volsurface.cli import main


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    assert main(["generate", "-o", str(d / "m.json"), "--csv", str(d / "m.csv")]) == 0
    assert main(["calibrate", str(d / "m.json"), "-o", str(d / "fit.json")]) == 0
    return d


def test_generate_calibrate_check(pipeline, capsys):
    capsys.readouterr()
    assert main(["check", str(pipeline / "fit.json")]) == 0
    assert "PASS" in capsys.readouterr().out
    blob = json.loads((pipeline / "fit.json").read_text())
    assert blob["version"] == 1
    assert len(blob["slices"]) == 6
    assert blob["error_vs_true"] is not None
    assert (pipeline / "m.csv").exists()


def test_calibrate_from_csv(pipeline, tmp_path, capsys):
    out = tmp_path / "f.json"
    assert main(["calibrate", str(pipeline / "m.csv"), "-o", str(out)]) == 0
    assert "RMSE" in capsys.readouterr().out
    assert json.loads(out.read_text())["error_vs_true"] is not None


def test_plot_cdn_and_inline(pipeline, tmp_path):
    cdn, inline = tmp_path / "cdn.html", tmp_path / "inline.html"
    base = ["plot", str(pipeline / "fit.json"), "--market", str(pipeline / "m.json")]
    assert main([*base, "-o", str(cdn), "--plotlyjs", "cdn", "--title", "T <b>"]) == 0
    assert main([*base, "-o", str(inline)]) == 0
    text = cdn.read_text(encoding="utf-8")
    assert cdn.stat().st_size < 200_000
    assert "cdn.plot.ly" in text
    assert "scatter3d" in text
    assert "Market IV" in text
    assert "T &lt;b&gt;" in text
    assert inline.stat().st_size > 1_000_000
    assert "cdn.plot.ly/plotly" not in inline.read_text(encoding="utf-8")


def test_run_end_to_end(tmp_path, capsys):
    assert main(["run", "--outdir", str(tmp_path / "o"), "--plotlyjs", "cdn"]) == 0
    for name in ("market.json", "market.csv", "fit.json", "report.json", "surface.html"):
        assert (tmp_path / "o" / name).exists()
    assert json.loads((tmp_path / "o" / "report.json").read_text())["passed"] is True
    assert "PASS" in capsys.readouterr().out


def test_check_detects_calendar_arbitrage(pipeline, tmp_path, capsys):
    blob = json.loads((pipeline / "fit.json").read_text())
    p = blob["slices"][3]["params"]  # T=1: set the minimum variance to 0, below T=0.5
    p["a"] = -p["b"] * p["sigma"] * (1 - p["rho"] ** 2) ** 0.5
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(blob))
    rep = tmp_path / "rep.json"
    capsys.readouterr()
    assert main(["check", str(bad), "--json", str(rep)]) == 1
    out = capsys.readouterr().out
    assert "calendar" in out
    assert "T=0.5 -> 1" in out
    assert "k in [" in out
    assert "FAIL" in out
    report = json.loads(rep.read_text())
    assert report["passed"] is False
    assert report["calendar"]
    # quoted-only still fails because the violation is inside the quotes
    assert main(["check", str(bad), "--quoted-only"]) == 1


def test_missing_and_malformed_inputs(tmp_path, capsys):
    assert main(["calibrate", str(tmp_path / "nope.json")]) == 3
    assert "vsl: error:" in capsys.readouterr().err
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    assert main(["calibrate", str(junk)]) == 3
    assert main(["check", str(junk)]) == 3
    assert main(["plot", str(tmp_path / "nope.json")]) == 3


def test_invalid_parameters(tmp_path, capsys):
    assert main(["generate", "-o", str(tmp_path / "m.json"), "--rho", "2"]) == 3
    err = capsys.readouterr().err
    assert err.startswith("vsl: error:")
    assert "Traceback" not in err
    assert main(["generate", "-o", str(tmp_path / "m.json"), "--expiries", "1,x"]) == 3
    assert main(["generate", "-o", str(tmp_path / "m.json"), "--noise", "-1"]) == 3


def test_usage_errors_and_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["frobnicate"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "vsl" in capsys.readouterr().out


def test_module_help_lists_exit_codes():
    r = subprocess.run(
        [sys.executable, "-m", "volsurface.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert r.returncode == 0
    for word in ("exit codes", "arbitrage detected", "usage error", "input error", "numerical"):
        assert word in r.stdout


def test_outputs_create_missing_parent_directories(tmp_path):
    out = tmp_path / "a" / "b"
    assert main(["generate", "-o", str(out / "m.json"), "--csv", str(out / "c" / "m.csv")]) == 0
    assert main(["calibrate", str(out / "m.json"), "-o", str(out / "f" / "fit.json")]) == 0
    html_path = out / "p" / "s.html"
    fit = str(out / "f" / "fit.json")
    assert main(["plot", fit, "--plotlyjs", "cdn", "-o", str(html_path)]) == 0
    assert html_path.exists()


def test_fetch_without_yfinance_is_an_input_error(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert main(["fetch", "SPY", "-o", "unused.json"]) == 3
    assert "volsurface[fetch]" in capsys.readouterr().err


def test_fetch_writes_market_from_provider(monkeypatch, tmp_path, capsys):
    # No network in tests: replace the provider with a tiny synthetic market.
    import volsurface.fetch as fetch_mod
    from volsurface.market import generate_market

    def fake(symbol, **kwargs):
        m = generate_market()
        return m, {
            "2027-01-15": {
                "T": 0.1,
                "F": 100.0,
                "df": 0.99,
                "method": "parity",
                "n_quotes": len(m.rows),
            }
        }

    monkeypatch.setattr(fetch_mod, "fetch_market_with_info", fake)
    out = tmp_path / "data" / "spy.json"
    assert main(["fetch", "SPY", "-o", str(out)]) == 0
    assert out.exists()
    assert "personal use only" in capsys.readouterr().out


def test_fetch_network_failure_is_a_clean_input_error(monkeypatch, capsys):
    import volsurface.fetch as fetch_mod

    def broken(symbol, **kwargs):
        raise TypeError("argument of type 'NoneType' is not iterable")  # what yfinance does offline

    monkeypatch.setattr(fetch_mod, "fetch_market_with_info", broken)
    assert main(["fetch", "SPY", "-o", "unused.json"]) == 3
    err = capsys.readouterr().err
    assert "could not fetch SPY" in err
    assert "Traceback" not in err
