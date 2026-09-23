# Copyright (c) 2026 Santiago Hofwimmer
"""
scatter_hover.py – Shared hover tooltip + click handling for scatter plots.

One implementation serves both the Advanced Analytics tab (main window) and
the Multi-Plot Dashboard scatter cells. The tooltip bubble itself lives in
:mod:`chipify.uikit.services.hover_bubble`, shared with the waveform-curve
hover, and is dropped via ``invalidate()`` whenever the owner redraws with
``fig.clf()``.

Robustness rules learned from the two previous copies:
- Never assume x/y values are numeric (corner sweeps yield strings) — format
  via ``fmt_value`` and anchor at the rendered scatter offsets instead of the
  raw data values.
"""
from __future__ import annotations

import logging

from chipify.uikit.services.hover_bubble import SEPARATOR, HoverBubble, fmt_value

log = logging.getLogger("chipify.uikit.scatter_hover")

__all__ = ["HoverState", "ScatterHoverManager", "fmt_value"]


class HoverState:
    """Snapshot of everything the hover/click handlers need for one plot."""

    def __init__(self, artist, df, x_col, y_col, stim):
        self.artist = artist   # PathCollection returned by ax.scatter
        self.df = df           # DataFrame row-aligned with the artist offsets
        self.x_col = x_col
        self.y_col = y_col
        self.stim = stim       # util.Stimuli or None


class ScatterHoverManager:
    """Hover tooltip + optional point-click dispatch for one mpl canvas.

    Parameters
    ----------
    canvas, fig:
        The FigureCanvasTkAgg and its Figure.
    get_state:
        Callable returning a :class:`HoverState` when scatter hover is
        currently applicable, else ``None`` (e.g. other plot mode active).
    on_point_click:
        Optional ``(row, state, mpl_event) -> None`` called when a scatter
        point is clicked.
    """

    def __init__(self, canvas, fig, get_state, on_point_click=None):
        self._canvas = canvas
        self._fig = fig
        self._get_state = get_state
        self._on_point_click = on_point_click
        self._bubble = HoverBubble(canvas, fig)

    def connect(self) -> None:
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        if self._on_point_click is not None:
            self._canvas.mpl_connect("button_press_event", self._on_click)

    def invalidate(self) -> None:
        """Forget the annotation — call after ``fig.clf()`` redraws."""
        self._bubble.invalidate()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _hide(self) -> None:
        if self._bubble.hide():
            self._canvas.draw_idle()

    def _hit(self, event):
        """Return ``(state, idx)`` for the point under the cursor, else None."""
        state = self._get_state()
        if state is None or state.artist is None or state.df is None:
            return None
        if not self._fig.axes or event.inaxes != self._fig.axes[0]:
            return None
        try:
            cont, ind = state.artist.contains(event)
        except Exception:
            return None
        hits = ind.get("ind", []) if isinstance(ind, dict) else []
        if not cont or len(hits) == 0:
            return None
        return state, int(hits[0])

    def _row_for(self, state, idx):
        try:
            return state.df.iloc[idx]
        except Exception:
            return None

    def _on_motion(self, event) -> None:
        hit = self._hit(event)
        if hit is None:
            self._hide()
            return
        state, idx = hit
        row = self._row_for(state, idx)
        if row is None:
            self._hide()
            return

        run_id = str(row.get("run_id", row.name))
        status = "PASS" if bool(row.get("global_pass", False)) else "FAIL"
        text_lines = [
            f"Run #{run_id.zfill(6)}",
            SEPARATOR,
            f"{state.x_col}: {fmt_value(row.get(state.x_col, '-'))}",
            f"{state.y_col}: {fmt_value(row.get(state.y_col, '-'))}",
            SEPARATOR,
            status,
        ]
        if state.stim is not None:
            for p in getattr(state.stim, "params", {}).keys():
                try:
                    if p in row and state.df[p].nunique() > 1:
                        text_lines.append(f"{p}: {row[p]}")
                except Exception:
                    continue

        # Anchor at the rendered offsets, not (row[x], row[y]) — also works
        # for categorical/string axes.
        try:
            xy = tuple(state.artist.get_offsets()[idx])
        except Exception:
            self._hide()
            return
        self._bubble.show(self._fig.axes[0], xy, "\n".join(text_lines), event)
        self._canvas.draw_idle()

    def _on_click(self, event) -> None:
        if getattr(event, "button", None) not in (1, 3):
            return
        hit = self._hit(event)
        if hit is None:
            return
        state, idx = hit
        row = self._row_for(state, idx)
        if row is None:
            return
        try:
            self._on_point_click(row, state, event)
        except Exception:
            log.exception("Scatter point click handler failed.")
