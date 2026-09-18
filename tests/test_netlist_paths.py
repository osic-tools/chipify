# Copyright (c) 2026 Santiago Hofwimmer
"""Tests for chipify.engines.netlist_paths (relative include resolution).

A deck is written into the scratch dir before it runs, so a relative
`.include` in it used to resolve against that scratch dir and fail. These
tests pin the rewrite that fixes it.

Covers:
- absolutize_netlist_includes: relative -> absolute against the testbench dir,
  the bare-filename (work/ staging) carve-out, `.control`/comment/Jinja/env-var
  exclusions, quoting, continuation folding, idempotence.
- testbench_dir: the single anchor, including a nested tb_path.
- NetlistPathError: every unresolved path reported at once, and surfaced as
  test.template_error without stopping the rest of the sweep.
- The ngspice / vacask / plugin-engine call sites.
"""
from __future__ import annotations

import pytest

np_mod = pytest.importorskip("chipify.engines.netlist_paths")

absolutize = np_mod.absolutize_netlist_includes
NetlistPathError = np_mod.NetlistPathError


def _touch(path) -> object:
    """Create *path* (and its parents) as an empty file, returning it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("* stub\n", encoding="utf-8")
    return path


# ── The core rewrite ─────────────────────────────────────────────────────────

def test_relative_include_becomes_absolute(tmp_path) -> None:
    target = _touch(tmp_path / "models" / "x.lib")
    out = absolutize(".include models/x.lib\n", tmp_path)
    assert out == f".include {target.as_posix()}\n"


def test_parent_relative_include_escapes_the_testbench_dir(tmp_path) -> None:
    tb_sub = tmp_path / "tb" / "sub"
    tb_sub.mkdir(parents=True)
    target = _touch(tmp_path / "tb" / "macros" / "amp.spice")
    out = absolutize(".include ../macros/amp.spice\n", tb_sub)
    assert out == f".include {target.as_posix()}\n"


def test_bare_filename_is_left_untouched(tmp_path) -> None:
    """The documented work/ staging contract: bare names come from FAST_TMP."""
    _touch(tmp_path / "models.lib")
    deck = ".include models.lib\n.lib corner.lib tt\n"
    assert absolutize(deck, tmp_path) == deck


def test_absolute_include_is_left_untouched(tmp_path) -> None:
    """Holds on Windows too — pins _is_absolute_token against os.path.isabs."""
    deck = '.include /foss/pdks/ihp-sg13g2/models/cornerMOSlv.lib\n'
    assert absolutize(deck, tmp_path) == deck
    assert np_mod._is_absolute_token("/foss/pdks/x.lib")
    assert np_mod._is_absolute_token("D:/designs/x.lib")
    assert not np_mod._is_absolute_token("models/x.lib")


def test_lib_section_argument_is_preserved(tmp_path) -> None:
    target = _touch(tmp_path / "models" / "corner.lib")
    out = absolutize(".lib models/corner.lib mos_tt\n", tmp_path)
    assert out == f'.lib {target.as_posix()} mos_tt\n'


def test_lib_section_name_only_is_untouched(tmp_path) -> None:
    deck = ".lib tt\n.endl\n"
    assert absolutize(deck, tmp_path) == deck


def test_control_block_is_never_touched(tmp_path) -> None:
    """Chipify owns .control (wrdata/setplot/echo); leave the whole region."""
    _touch(tmp_path / "models" / "x.lib")
    _touch(tmp_path / "out" / "run.tab")
    deck = (
        ".control\n"
        "wrdata out/run.tab v(a)\n"
        ".inc models/x.lib\n"
        ".endc\n"
    )
    assert absolutize(deck, tmp_path) == deck


def test_rewrite_happens_outside_but_not_inside_control(tmp_path) -> None:
    target = _touch(tmp_path / "models" / "x.lib")
    deck = (
        ".include models/x.lib\n"
        ".control\n"
        ".inc models/x.lib\n"
        ".endc\n"
    )
    out = absolutize(deck, tmp_path).splitlines()
    assert out[0] == f".include {target.as_posix()}"
    assert out[2] == ".inc models/x.lib"


def test_comment_lines_are_untouched(tmp_path) -> None:
    _touch(tmp_path / "models" / "x.lib")
    deck = (
        "* .include models/x.lib\n"
        "// include \"models/x.lib\"\n"
        "; .inc models/x.lib\n"
        "/* include \"models/x.lib\"\n"
        "   still a comment */\n"
    )
    assert absolutize(deck, tmp_path) == deck


def test_jinja_placeholders_are_untouched(tmp_path) -> None:
    deck = ".include {{ mdir }}/x.lib\nwrdata {{ tran_out_path }} v(a)\n"
    assert absolutize(deck, tmp_path) == deck


def test_env_vars_and_tilde_pass_through_without_error(tmp_path) -> None:
    deck = ".include $PDK_ROOT/models/x.lib\n.inc ~/models/x.lib\n"
    assert absolutize(deck, tmp_path) == deck


def test_case_insensitive_directives(tmp_path) -> None:
    target = _touch(tmp_path / "models" / "x.lib")
    out = absolutize(".INCLUDE models/x.lib\n.Lib models/x.lib tt\n", tmp_path)
    assert out.splitlines() == [
        f".INCLUDE {target.as_posix()}",
        f".Lib {target.as_posix()} tt",
    ]


def test_directive_lookalikes_are_not_matched(tmp_path) -> None:
    deck = ".included_by models/x.lib\nincludes models/x.lib\n"
    assert absolutize(deck, tmp_path) == deck


def test_continuation_lines_are_folded(tmp_path) -> None:
    target = _touch(tmp_path / "models" / "x.lib")
    out = absolutize(".include\n+ models/x.lib\n", tmp_path)
    # A rewritten directive collapses onto one physical line.
    assert out == f".include {target.as_posix()}\n"


def test_quotes_are_preserved(tmp_path) -> None:
    target = _touch(tmp_path / "models" / "x.lib")
    out = absolutize('.include "models/x.lib"\n', tmp_path)
    assert out == f'.include "{target.as_posix()}"\n'


def test_quotes_are_added_when_the_resolved_path_has_a_space(tmp_path) -> None:
    base = tmp_path / "my designs"
    target = _touch(base / "models" / "x.lib")
    out = absolutize(".include models/x.lib\n", base)
    assert out == f'.include "{target.as_posix()}"\n'


def test_spectre_include_ahdl_include_and_load(tmp_path) -> None:
    base = tmp_path / "tb"
    base.mkdir()
    inc = _touch(tmp_path / "models" / "x.scs")
    ahdl = _touch(tmp_path / "va" / "diode.va")
    osdi = _touch(tmp_path / "osdi" / "resistor.osdi")
    deck = (
        'include "../models/x.scs" section=tt\n'
        'ahdl_include "../va/diode.va"\n'
        'load "../osdi/resistor.osdi"\n'
    )
    assert absolutize(deck, base).splitlines() == [
        f'include "{inc.as_posix()}" section=tt',
        f'ahdl_include "{ahdl.as_posix()}"',
        f'load "{osdi.as_posix()}"',
    ]


def test_is_idempotent(tmp_path) -> None:
    _touch(tmp_path / "models" / "x.lib")
    once = absolutize(".include models/x.lib\n", tmp_path)
    assert absolutize(once, tmp_path) == once


def test_trailing_newline_and_layout_are_preserved(tmp_path) -> None:
    deck = "* deck\n\nV1 vdd 0 1.8\n"
    assert absolutize(deck, tmp_path) == deck
    assert absolutize("", tmp_path) == ""


# ── Failure reporting ────────────────────────────────────────────────────────

def test_unresolvable_include_raises_with_directive_path_and_anchor(tmp_path) -> None:
    with pytest.raises(NetlistPathError) as exc:
        absolutize(".include ../macros/amp.spice\n", tmp_path, label="tb_amp")
    msg = str(exc.value)
    assert "tb_amp" in msg
    assert ".include" in msg
    assert "../macros/amp.spice" in msg
    assert str(tmp_path) in msg or tmp_path.as_posix() in msg


def test_every_unresolved_path_is_reported_at_once(tmp_path) -> None:
    with pytest.raises(NetlistPathError) as exc:
        absolutize(
            ".include a/one.lib\n.lib b/two.lib tt\ninclude \"c/three.scs\"\n",
            tmp_path,
        )
    msg = str(exc.value)
    assert "3 include path(s)" in msg
    for name in ("a/one.lib", "b/two.lib", "c/three.scs"):
        assert name in msg


def test_a_resolved_path_containing_a_brace_is_rejected(tmp_path) -> None:
    """A brace would be re-rendered by Jinja2 and crash the sweep instead."""
    base = tmp_path / "v{1}"
    _touch(base / "models" / "x.lib")
    with pytest.raises(NetlistPathError, match="brace"):
        absolutize(".include models/x.lib\n", base)


# ── testbench_dir: the single anchor ─────────────────────────────────────────

class _Stub:
    def __init__(self, tb_path: str) -> None:
        self.tb_path = tb_path


def test_testbench_dir_is_tb_dir_for_a_flat_tb_path(tmp_path, monkeypatch) -> None:
    from chipify import settings
    monkeypatch.setattr(settings, "TB_DIR", str(tmp_path))
    assert np_mod.testbench_dir(_Stub("tb_amp")) == tmp_path


def test_testbench_dir_follows_a_nested_tb_path(tmp_path, monkeypatch) -> None:
    from chipify import settings
    monkeypatch.setattr(settings, "TB_DIR", str(tmp_path))
    assert np_mod.testbench_dir(_Stub("sub/tb_amp")) == tmp_path / "sub"


# ── Engine call sites ────────────────────────────────────────────────────────

def _use_tb_dir(monkeypatch, tmp_path) -> None:
    from chipify import settings
    monkeypatch.setattr(settings, "TB_DIR", str(tmp_path))


def _no_xschem(monkeypatch, module) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("run_xschem must not run for an imported netlist")
    monkeypatch.setattr(module, "run_xschem", _boom)


def _make_test(tb: str, value_names: list[str]):
    from chipify.util import Test as TbTest, Value
    t = TbTest(tb, [Value(n, None, None, None) for n in value_names])
    t.netlist_source = "netlist"
    return t


def test_ngspice_import_netlist_absolutizes_include(monkeypatch, tmp_path) -> None:
    from chipify.engines import ngspice as ng_mod
    _use_tb_dir(monkeypatch, tmp_path)
    _no_xschem(monkeypatch, ng_mod)

    target = _touch(tmp_path / "macros" / "amp.spice")
    (tmp_path / "amp.spice").write_text(
        ".include macros/amp.spice\n.control\ntran 1n 100n\n.endc\n",
        encoding="utf-8",
    )
    out = ng_mod.NgspiceSimulator().generate_test_template(
        _make_test("amp", ["gain"]))

    assert f".include {target.as_posix()}" in out
    # The managed capture is still injected on top of the rewritten deck.
    assert "echo MY_DATA:$&gain" in out
    assert "set num_threads=1" in out


def test_vacask_import_netlist_absolutizes_include(monkeypatch, tmp_path) -> None:
    vc_mod = pytest.importorskip("chipify.engines.vacask")
    _use_tb_dir(monkeypatch, tmp_path)
    _no_xschem(monkeypatch, vc_mod)

    target = _touch(tmp_path / "models" / "x.scs")
    (tmp_path / "amp.sim").write_text(
        '* vacask deck\ninclude "models/x.scs"\ntran 1n 100n\n', encoding="utf-8",
    )
    out = vc_mod.VacaskSimulator().generate_test_template(
        _make_test("amp", ["gain"]))
    assert f'include "{target.as_posix()}"' in out


def test_unresolved_include_fails_only_that_testbench(monkeypatch, tmp_path) -> None:
    from chipify.util import Stimuli
    from chipify.engines import ngspice as ng_mod
    simulator = pytest.importorskip("chipify.simulator")
    _use_tb_dir(monkeypatch, tmp_path)
    _no_xschem(monkeypatch, ng_mod)

    _touch(tmp_path / "macros" / "amp.spice")
    (tmp_path / "tb_ok.spice").write_text(
        ".include macros/amp.spice\n.control\ntran 1n 1u\n.endc\n", encoding="utf-8",
    )
    (tmp_path / "tb_bad.spice").write_text(
        ".include macros/missing.spice\n.control\ntran 1n 1u\n.endc\n",
        encoding="utf-8",
    )
    ok = _make_test("tb_ok", ["a"]); ok.engine = "ngspice"
    bad = _make_test("tb_bad", ["b"]); bad.engine = "ngspice"
    stim = Stimuli()
    stim.tests = [ok, bad]

    simulator.generate_templates(stim)

    assert ok.template_error is None and "echo MY_DATA:$&a" in ok.template_str
    assert bad.template_str == ""
    assert "macros/missing.spice" in (bad.template_error or "")


def test_plugin_engine_gets_the_helper_from_the_base_class(tmp_path, monkeypatch) -> None:
    from chipify.engines.base import BaseSimulator
    from chipify import settings
    monkeypatch.setattr(settings, "TB_DIR", str(tmp_path))
    target = _touch(tmp_path / "models" / "x.lib")

    class _Plugin(BaseSimulator):
        name = "plugin-under-test"

        def generate_test_template(self, test) -> str:
            return self.resolve_netlist_paths(".include models/x.lib\n", test)

        def run(self, netlist, timeout_sec=10, test=None, analysis_tab_paths=None):
            return "", None

    out = _Plugin().generate_test_template(_Stub("tb_amp"))
    assert out == f".include {target.as_posix()}\n"
