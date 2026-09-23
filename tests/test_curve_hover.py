# Copyright (c) 2026 Santiago Hofwimmer
"""Hover tooltips on the per-run waveform curves.

Covers ``uikit/services/curve_hover.py`` — the nearest-curve hit test, the
tooltip text, and the redraw lifecycle — plus the two things it leans on: the
shared bubble in ``uikit/services/hover_bubble.py`` and the ``line_map`` the
three overlay plotters return.

No Qt and no display: ``PlotManager.draw_*`` and everything under ``uikit/``
take a bare Figure plus a canvas, so an Agg canvas and a hand-built
``MouseEvent`` positioned through ``ax.transData`` drive the real handlers
deterministically. ``canvas.callbacks.process`` is the documented dispatch and
re-raises handler exceptions, so a crash in the hover path fails the test
instead of being swallowed.
"""
from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.backend_bases import MouseEvent          # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure                     # noqa: E402

from chipify.plot_manager import PlotManager             # noqa: E402
from chipify.uikit.services import curve_hover as ch     # noqa: E402


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _fig(figsize=(6, 4)):
    fig = Figure(figsize=figsize)
    canvas = FigureCanvasAgg(fig)          # MouseEvent must carry THIS canvas
    return fig, canvas


def _manager(canvas, fig, state):
    """A connected manager whose state never changes, plus a drawn background."""
    mgr = ch.CurveHoverManager(canvas, fig, get_state=lambda: state)
    mgr.connect()
    canvas.draw()
    return mgr


def _move(canvas, ax, x, y, **kw):
    """Dispatch a motion event at data coordinates (x, y) of *ax*."""
    px, py = ax.transData.transform((x, y))
    canvas.callbacks.process(
        "motion_notify_event",
        MouseEvent("motion_notify_event", canvas, px, py, **kw),
    )


def _move_px(canvas, px, py, **kw):
    canvas.callbacks.process(
        "motion_notify_event",
        MouseEvent("motion_notify_event", canvas, px, py, **kw),
    )


def _text(mgr):
    """The tooltip's text while it is visible, else None."""
    annot = mgr._bubble.annot
    if annot is None or not annot.get_visible():
        return None
    return annot.get_text()


def _plot(ax, curves):
    """``{Line2D: (run_id, signal)}`` for ``{(run_id, signal): (x, y)}``."""
    line_map = {}
    for (run_id, signal), (x, y) in curves.items():
        line, = ax.plot(x, y)
        line_map[line] = (run_id, signal)
    return line_map


def _results(run_ids, **cols):
    return pd.DataFrame({"run_id": list(run_ids), **cols})


def _stim(**params):
    return SimpleNamespace(params=params, tests=[])


def _write_runs(tmp_path, frames):
    """Write ``{run_id: DataFrame}`` as the ``run_<id>__<tb>.csv`` files on disk."""
    for run_id, frame in frames.items():
        frame.to_csv(tmp_path / f"run_{run_id}__tb.csv", index=False)
    return str(tmp_path)


# ── Hit test: which curve is under the cursor ─────────────────────────────────

def test_the_nearest_curve_wins_not_the_first_drawn():
    """With overlapping Monte-Carlo curves, first-hit would always say run 0."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line_map = _plot(ax, {
        ("000000", "v(out)"): (x, np.zeros_like(x)),
        ("000001", "v(out)"): (x, np.full_like(x, 0.5)),
        ("000002", "v(out)"): (x, np.ones_like(x)),
    })
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0, 1, 2]), None))

    _move(canvas, ax, 0.5, 0.98)          # closest to run 2
    assert _text(mgr).startswith("Run #000002")
    _move(canvas, ax, 0.5, 0.52)          # closest to run 1
    assert _text(mgr).startswith("Run #000001")


def test_a_cursor_away_from_every_curve_shows_nothing():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line_map = _plot(ax, {("000000", "v(out)"): (x, np.zeros_like(x))})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    _move(canvas, ax, 0.5, 0.0)
    assert _text(mgr) is not None
    _move(canvas, ax, 0.5, 0.9)           # far above the curve
    assert _text(mgr) is None


def test_a_sparse_curve_is_hoverable_between_its_samples():
    """An 11-point DC sweep puts its samples tens of pixels apart.

    Nearest-*vertex* hit testing would only answer on the samples themselves;
    the distance is to the drawn segment.
    """
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0.0, 1.0, 11)
    line_map = _plot(ax, {("000000", "v(o)"): (x, 2.0 * x)})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    _move(canvas, ax, 0.55, 1.10)         # mid-segment, no sample nearby
    assert _text(mgr) is not None


def test_hits_are_measured_in_pixels_not_data_units():
    """On a log x axis a fixed data-unit radius is a decade wide at 1 Hz."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    freq = np.logspace(1, 7, 300)
    line_map = _plot(ax, {("000000", "out"): (freq, np.zeros_like(freq))})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    # Two decades away in data units, but right on the curve in pixels.
    _move(canvas, ax, 1e5, 0.0)
    assert _text(mgr) is not None
    # A handful of pixels above the same point: a miss, at any decade.
    px, py = ax.transData.transform((1e5, 0.0))
    _move_px(canvas, px, py + 40)
    assert _text(mgr) is None


def test_non_positive_samples_on_a_log_axis_are_not_hoverable_ghosts():
    """matplotlib maps x<=0 on a log axis to the sentinel -1000, not -inf.

    Left unmasked it plants a phantom sample a thousand decades to the left
    that wins every hit test near the axes' left edge.
    """
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    x = np.array([0.0, -5.0, 10.0, 100.0, 1000.0])
    line_map = _plot(ax, {("000000", "out"): (x, np.zeros_like(x))})
    ax.set_xlim(10, 1000)
    index = ch._CurveIndex(ax, list(line_map.items()))
    assert np.all(index.raw_x > 0)


def test_a_curve_that_is_all_nan_is_drawn_but_not_hoverable():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 50)
    line_map = _plot(ax, {
        ("000000", "broken"): (x, np.full_like(x, np.nan)),
        ("000001", "v(out)"): (x, np.zeros_like(x)),
    })
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0, 1]), None))

    _move(canvas, ax, 0.5, 0.0)
    # argmin over a NaN-poisoned array would have answered with the NaN curve.
    assert _text(mgr).startswith("Run #000001")


def test_holes_in_a_curve_do_not_capture_the_cursor():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 101)
    y = np.zeros_like(x)
    y[40:60] = np.nan                     # a dropout in the middle
    line_map = _plot(ax, {("000000", "v(out)"): (x, y)})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    _move(canvas, ax, 0.05, 0.0)
    assert _text(mgr) is not None         # intact part still answers
    _move(canvas, ax, 0.5, 0.0)
    assert _text(mgr) is None             # the hole is genuinely empty


def test_a_single_sample_curve_is_still_hoverable():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    line_map = _plot(ax, {("000007", "v(out)"): ([0.5], [0.5])})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([7]), None))

    _move(canvas, ax, 0.5, 0.5)
    assert _text(mgr).startswith("Run #000007")


def test_the_reported_sample_is_one_that_exists_in_the_data():
    """A tooltip in an EDA tool must not invent an interpolated value."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0.0, 1.0, 11)
    y = np.round(np.sin(x * 3.0), 6)
    line_map = _plot(ax, {("000000", "v(o)"): (x, y)})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    _move(canvas, ax, 0.52, float(np.interp(0.52, x, y)))
    reported = float(_text(mgr).splitlines()[3].split(": ")[1])
    assert any(abs(reported - float(f"{v:.4g}")) < 1e-9 for v in y)


def test_the_index_survives_a_zoom_without_a_rebuild():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line_map = _plot(ax, {("000000", "v(out)"): (x, x)})
    state = ch.CurveHoverState(line_map, _results([0]), None)
    mgr = _manager(canvas, fig, state)

    _move(canvas, ax, 0.5, 0.5)
    built = mgr._index[ax]
    ax.set_xlim(0.4, 0.6)
    ax.set_ylim(0.4, 0.6)
    canvas.draw()
    _move(canvas, ax, 0.5, 0.5)
    assert _text(mgr) is not None
    assert mgr._index[ax] is built        # only the affine was re-read


# ── Tooltip content ───────────────────────────────────────────────────────────

def test_the_tooltip_mirrors_the_scatter_tooltip():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time (ms)")
    x = np.linspace(0, 1, 101)            # 0.5 is a real sample
    line_map = _plot(ax, {("000003", "v(out)"): (x, np.full_like(x, 1.25))})
    df = _results([3], temp=[27], corner=["tt"], global_pass=[True])
    # Two runs' worth of temperatures so temp counts as swept, one corner so
    # it does not: the scatter tooltip lists only what actually varies.
    df = pd.concat([df, _results([4], temp=[-40], corner=["tt"], global_pass=[False])])
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, df, _stim(temp=[27, -40],
                                                                      corner=["tt"])))
    _move(canvas, ax, 0.5, 1.25)
    assert _text(mgr).splitlines() == [
        "Run #000003",
        "-" * 15,
        "Time (ms): 0.5",
        "v(out): 1.25",
        "-" * 15,
        "PASS",
        "temp: 27",
    ]


def test_a_failing_run_says_so():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 50)
    line_map = _plot(ax, {("000009", "v(out)"): (x, np.zeros_like(x))})
    df = _results([9], global_pass=[False])
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, df, None))

    _move(canvas, ax, 0.5, 0.0)
    assert "FAIL" in _text(mgr)


def test_a_frame_without_a_verdict_column_omits_the_verdict():
    """Defaulting to FAIL, as the scatter tooltip does, would libel every run."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 50)
    line_map = _plot(ax, {("000001", "v(out)"): (x, np.zeros_like(x))})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([1]), None))

    _move(canvas, ax, 0.5, 0.0)
    text = _text(mgr)
    assert "FAIL" not in text and "PASS" not in text


def test_run_detail_map_pads_ids_to_match_the_line_map():
    """``line_map`` keys come from the filenames; ``run_id`` reads back as int."""
    df = _results([4, 12], temp=[-40, 100], global_pass=[True, False])
    details = ch.run_detail_map(df, _stim(temp=[-40, 100]))
    assert details == {
        "000004": ["PASS", "temp: -40"],
        "000012": ["FAIL", "temp: 100"],
    }


def test_run_detail_map_tolerates_a_frame_without_run_ids():
    assert ch.run_detail_map(None, None) == {}
    assert ch.run_detail_map(pd.DataFrame({"x": [1]}), None) == {}


def test_curve_tooltip_lines_formats_values_like_the_scatter_hover():
    from chipify.uikit.services.scatter_hover import fmt_value
    lines = ch.curve_tooltip_lines("7", "v(out)", "Time (µs)", 1.23456789, 2.0,
                                   ["PASS"])
    assert lines[0] == "Run #000007"
    assert lines[2] == f"Time (µs): {fmt_value(1.23456789)}" == "Time (µs): 1.235"
    assert lines[3] == "v(out): 2"
    assert lines[-1] == "PASS"


def test_x_axis_label_falls_back_to_the_shared_sibling():
    """The Bode magnitude pane carries no x label — its sharex partner does."""
    fig, _canvas = _fig()
    ax_mag, ax_phase = fig.subplots(2, 1, sharex=True)
    ax_phase.set_xlabel("Frequency (Hz)")
    assert ch.x_axis_label(ax_mag) == "Frequency (Hz)"
    assert ch.x_axis_label(ax_phase) == "Frequency (Hz)"
    lone = Figure().add_subplot(111)
    assert ch.x_axis_label(lone) == "x"


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def test_no_tooltip_when_the_owner_reports_another_mode():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 50)
    _plot(ax, {("000000", "v(out)"): (x, np.zeros_like(x))})
    mgr = ch.CurveHoverManager(canvas, fig, get_state=lambda: None)
    mgr.connect()
    canvas.draw()

    _move(canvas, ax, 0.5, 0.0)
    assert _text(mgr) is None


def test_invalidate_drops_the_bubble_and_the_index():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 50)
    line_map = _plot(ax, {("000000", "v(out)"): (x, np.zeros_like(x))})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    _move(canvas, ax, 0.5, 0.0)
    assert mgr._bubble.annot is not None and mgr._index
    mgr.invalidate()
    assert mgr._bubble.annot is None and not mgr._index


def test_a_redraw_does_not_leave_the_tooltip_hit_testing_erased_curves():
    """``fig.clf()`` detaches every Line2D the old map held."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 50)
    line_map = _plot(ax, {("000000", "v(out)"): (x, np.zeros_like(x))})
    state = ch.CurveHoverState(line_map, _results([0]), None)
    mgr = _manager(canvas, fig, state)
    _move(canvas, ax, 0.5, 0.0)
    assert _text(mgr) is not None

    fig.clf()
    mgr.invalidate()
    ax2 = fig.add_subplot(111)
    ax2.plot(x, np.zeros_like(x))
    canvas.draw()
    _move(canvas, ax2, 0.5, 0.0)          # curves exist, but none are in the map
    assert _text(mgr) is None


def test_hover_is_suppressed_while_a_drag_or_a_toolbar_tool_owns_the_mouse():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 50)
    line_map = _plot(ax, {("000000", "v(out)"): (x, np.zeros_like(x))})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    _move(canvas, ax, 0.5, 0.0, button=1)             # dragging
    assert _text(mgr) is None
    canvas.widgetlock(object())                       # pan / zoom rectangle active
    _move(canvas, ax, 0.5, 0.0)
    assert _text(mgr) is None
    canvas.widgetlock.release(canvas.widgetlock._owner)
    _move(canvas, ax, 0.5, 0.0)
    assert _text(mgr) is not None


def test_leaving_the_canvas_hides_the_tooltip():
    """Also what keeps an animated bubble out of an Export… taken right after."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 50)
    line_map = _plot(ax, {("000000", "v(out)"): (x, np.zeros_like(x))})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    _move(canvas, ax, 0.5, 0.0)
    assert _text(mgr) is not None
    canvas.callbacks.process(
        "figure_leave_event",
        MouseEvent("figure_leave_event", canvas, 0, 0),
    )
    assert _text(mgr) is None


def test_the_hover_never_redraws_the_whole_figure(monkeypatch):
    """A draw_idle() per mouse move re-renders every curve — the point of blitting."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line_map = _plot(ax, {("000000", "v(out)"): (x, np.zeros_like(x))})
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    calls = []
    monkeypatch.setattr(canvas, "draw_idle", lambda *a, **k: calls.append(1))
    for offset in (0.0, 0.001, 0.002):
        _move(canvas, ax, 0.5 + offset, 0.0)
    _move(canvas, ax, 0.5, 0.9)           # and the hide path
    assert mgr._can_blit and calls == []


# ── Highlighting the hovered curve ────────────────────────────────────────────

def _widths(line_map):
    return {key: line.get_linewidth() for line, key in line_map.items()}


def test_the_hovered_curve_is_drawn_thicker_and_opaque():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line_map = {}
    for run_id, level in (("000000", 0.0), ("000001", 1.0)):
        line, = ax.plot(x, np.full_like(x, level), linewidth=0.8, alpha=0.3)
        line_map[line] = (run_id, "v(out)")
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0, 1]), None))
    before = _widths(line_map)

    _move(canvas, ax, 0.5, 1.0)
    after = _widths(line_map)
    assert after[("000001", "v(out)")] > before[("000001", "v(out)")]
    assert after[("000000", "v(out)")] == before[("000000", "v(out)")]
    hovered = next(l for l, k in line_map.items() if k == ("000001", "v(out)"))
    assert hovered.get_alpha() == 1.0          # out of the auto-faded crowd
    assert hovered.get_zorder() > 2


def test_the_highlight_moves_with_the_cursor_and_leaves_nothing_behind():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line_map = {}
    for run_id, level in (("000000", 0.0), ("000001", 1.0)):
        line, = ax.plot(x, np.full_like(x, level), linewidth=0.8, alpha=0.3)
        line_map[line] = (run_id, "v(out)")
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0, 1]), None))
    before = _widths(line_map)

    _move(canvas, ax, 0.5, 1.0)
    _move(canvas, ax, 0.5, 0.0)               # onto the other curve
    after = _widths(line_map)
    assert after[("000000", "v(out)")] > before[("000000", "v(out)")]
    assert after[("000001", "v(out)")] == before[("000001", "v(out)")]

    _move(canvas, ax, 0.5, 0.5)               # off both curves
    assert _widths(line_map) == before
    assert all(line.get_alpha() == 0.3 for line in line_map)


def test_leaving_the_canvas_restores_the_curve():
    """Otherwise a thickened curve is baked into the very next export."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line, = ax.plot(x, np.zeros_like(x), linewidth=0.8, alpha=0.3)
    line_map = {line: ("000000", "v(out)")}
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([0]), None))

    _move(canvas, ax, 0.5, 0.0)
    assert line.get_linewidth() > 0.8
    canvas.callbacks.process(
        "figure_leave_event", MouseEvent("figure_leave_event", canvas, 0, 0))
    assert line.get_linewidth() == 0.8 and line.get_alpha() == 0.3


def test_invalidate_restores_the_curve_before_the_redraw():
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line, = ax.plot(x, np.zeros_like(x), linewidth=0.8)
    mgr = _manager(canvas, fig,
                   ch.CurveHoverState({line: ("000000", "v(out)")}, _results([0]), None))

    _move(canvas, ax, 0.5, 0.0)
    mgr.invalidate()
    assert line.get_linewidth() == 0.8
    assert mgr._highlight is None


def test_a_highlight_is_never_saved_into_the_blit_background():
    """A saved frame showing the thick curve would smear it until the next draw."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 200)
    line, = ax.plot(x, np.zeros_like(x), linewidth=0.8)
    mgr = _manager(canvas, fig,
                   ch.CurveHoverState({line: ("000000", "v(out)")}, _results([0]), None))

    _move(canvas, ax, 0.5, 0.0)
    assert mgr._highlight is not None
    canvas.draw()                             # e.g. a resize while hovering
    assert mgr._highlight is None             # restored before the frame was kept
    assert line.get_linewidth() == 0.8

    # The real invariant, checked in pixels: whatever background is now held
    # matches a settled frame with no highlight in it. A frame captured while
    # the curve was thick would smear it back on every blit.
    canvas.draw()
    clean = bytes(canvas.buffer_rgba())
    canvas.restore_region(mgr._bg)
    canvas.blit(fig.bbox)
    assert bytes(canvas.buffer_rgba()) == clean


def test_the_highlight_survives_a_rasterized_overlay():
    """Above 200 curves the plotters rasterize; draw_artist must still work."""
    fig, canvas = _fig()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 100)
    line_map = {}
    for i in range(6):
        line, = ax.plot(x, np.full_like(x, i * 0.1), linewidth=0.8, alpha=0.2,
                        rasterized=True)
        line_map[line] = (f"{i:06d}", "v(out)")
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results(range(6)), None))

    _move(canvas, ax, 0.5, 0.3)
    assert mgr._highlight is not None
    assert _text(mgr).startswith("Run #000003")

# ── The plotters' line_map ────────────────────────────────────────────────────

def test_the_overlay_plotters_map_every_curve_they_draw(tmp_path):
    t = np.linspace(0, 1e-3, 50)
    adir = _write_runs(tmp_path, {
        f"{i:06d}": pd.DataFrame({"time": t, "v(out)": t * i, "v(in)": t})
        for i in range(3)
    })
    fig, canvas = _fig()
    line_map = PlotManager.draw_transient_plot(
        fig, canvas, adir, ["000000", "000001", "000002"], ["v(out)", "v(in)"])
    assert len(line_map) == 6
    assert set(line_map.values()) == {
        (rid, sig) for rid in ("000000", "000001", "000002")
        for sig in ("v(out)", "v(in)")
    }


def test_the_bode_map_covers_both_panes(tmp_path):
    """Hovering the phase pane — half the plot — must identify a run too."""
    freq = np.logspace(1, 6, 60)
    adir = _write_runs(tmp_path, {
        "000000": pd.DataFrame({"frequency": freq,
                                "out_mag": np.ones_like(freq),
                                "out_phase": np.zeros_like(freq)}),
    })
    fig, canvas = _fig()
    line_map = PlotManager.draw_bode_plot(fig, canvas, adir, ["000000"], ["out"])
    assert sorted(line_map.values()) == [("000000", "out (mag)"),
                                         ("000000", "out (phase)")]
    ax_mag, ax_phase = fig.axes[0], fig.axes[1]
    assert {line.axes for line in line_map} == {ax_mag, ax_phase}


def test_the_bubble_moves_between_the_bode_panes_without_stranding_one(tmp_path):
    freq = np.logspace(1, 6, 60)
    adir = _write_runs(tmp_path, {
        "000002": pd.DataFrame({"frequency": freq,
                                "out_mag": np.ones_like(freq),
                                "out_phase": np.full_like(freq, -45.0)}),
    })
    fig, canvas = _fig(figsize=(6, 5))
    line_map = PlotManager.draw_bode_plot(fig, canvas, adir, ["000002"], ["out"])
    ax_mag, ax_phase = fig.axes[0], fig.axes[1]
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, _results([2]), None))

    _move(canvas, ax_mag, 1e3, 0.0)                   # 20*log10(1) == 0 dB
    assert mgr._bubble.annot.axes is ax_mag
    assert _text(mgr).splitlines()[3].startswith("out (mag): ")

    _move(canvas, ax_phase, 1e3, -45.0)
    assert mgr._bubble.annot.axes is ax_phase
    assert _text(mgr).splitlines()[2].startswith("Frequency (Hz): ")
    assert len(ax_mag.texts) == 0                     # the old bubble is gone


def test_an_empty_overlay_returns_an_empty_map_that_hovers_at_nothing(tmp_path):
    fig, canvas = _fig()
    line_map = PlotManager.draw_transient_plot(fig, canvas, "", [], [])
    assert line_map == {}
    mgr = _manager(canvas, fig, ch.CurveHoverState(line_map, None, None))
    _move_px(canvas, 200, 150)
    assert _text(mgr) is None


# ── Shared code, not copied code ──────────────────────────────────────────────

def test_both_hover_managers_use_the_same_bubble_and_formatter():
    from chipify.uikit.services import hover_bubble, scatter_hover
    assert scatter_hover.fmt_value is hover_bubble.fmt_value is ch.fmt_value
    assert scatter_hover.ScatterHoverManager(
        None, None, lambda: None)._bubble.__class__ is hover_bubble.HoverBubble


@pytest.mark.parametrize("module", ["curve_hover", "hover_bubble", "scatter_hover"])
def test_the_hover_services_import_no_gui_toolkit(module):
    """CONTRIBUTING invariant #1: uikit/ stays unit-testable without a display."""
    path = (pathlib.Path(__file__).resolve().parent.parent
            / "chipify" / "uikit" / "services" / f"{module}.py")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"PySide6", "PyQt5", "PyQt6", "tkinter", "customtkinter"}
