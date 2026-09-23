# Copyright (c) 2026 Santiago Hofwimmer
"""
hover_bubble.py – The tooltip bubble shared by every matplotlib hover manager.

One annotation implementation serves the scatter hover
(:mod:`chipify.uikit.services.scatter_hover`) and the waveform-curve hover
(:mod:`chipify.uikit.services.curve_hover`), so the two cannot drift into
looking like different products.

Robustness rules learned from the scatter implementation:
- The annotation must sit above the axes title: titles are drawn after the
  regular axes children at the same default zorder, so an un-raised tooltip
  is painted over by the plot headline.
- Never assume hovered values are numeric (corner sweeps yield strings) —
  format via :func:`fmt_value`.
- The bubble is dropped via ``invalidate()`` whenever the owner redraws with
  ``fig.clf()``; the annotation it held belongs to an axes that no longer
  exists.

No GUI-toolkit imports (matplotlib only) — see CONTRIBUTING, "Architecture
conventions".
"""
from __future__ import annotations

import logging

log = logging.getLogger("chipify.uikit.hover_bubble")

ACCENT = "#3484F0"
OFFSET = 14  # px gap between the hovered point and the bubble corner

#: Rule drawn between the tooltip's sections.
SEPARATOR = "-" * 15


def fmt_value(v) -> str:
    """Format a hover value: 4-significant-digit float, anything else verbatim."""
    try:
        return f"{float(v):.4g}"
    except Exception:
        return str(v)


class HoverBubble:
    """The annotation a hover manager shows, and where it is placed.

    Owns nothing but the bubble: *when* to show it and *what* to repaint
    afterwards are the manager's business, because a scatter plot can afford a
    full ``draw_idle()`` and a 500-curve waveform overlay cannot.

    Parameters
    ----------
    canvas, fig:
        The matplotlib canvas and its Figure.
    animated:
        When true the annotation is marked animated, so ordinary draws skip it
        and the manager paints it itself by blitting. Managers that repaint
        with ``draw_idle()`` leave this false.
    """

    def __init__(self, canvas, fig, *, animated: bool = False) -> None:
        self._canvas = canvas
        self._fig = fig
        self.animated = animated
        self._annot = None

    @property
    def annot(self):
        """The live annotation artist, or ``None`` before the first show."""
        return self._annot

    def invalidate(self) -> None:
        """Forget the annotation — call after ``fig.clf()`` redraws."""
        self._annot = None

    def visible(self) -> bool:
        return self._annot is not None and bool(self._annot.get_visible())

    def hide(self) -> bool:
        """Hide the bubble. Returns whether that actually changed anything."""
        if self.visible():
            self._annot.set_visible(False)
            return True
        return False

    def show(self, ax, xy, text: str, event) -> None:
        """Anchor the bubble at *xy* (data coords of *ax*) and show *text*."""
        annot = self._ensure(ax)
        annot.xy = xy
        annot.set_text(text)
        annot.set_animated(self.animated)
        # Visible *before* placing: ``Text.get_window_extent`` reports a
        # 1-pixel box for a hidden artist, so measuring first meant the
        # edge-flip below never fired and a tall bubble near the top of the
        # canvas was drawn off it.
        annot.set_visible(True)
        self._place(annot, event)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure(self, ax):
        """The annotation for *ax*, created (and moved between axes) on demand."""
        if self._annot is not None and self._annot.axes is ax:
            return self._annot
        if self._annot is not None:
            # Stacked subplots (the Bode magnitude/phase pair): leaving the old
            # bubble behind would strand a visible tooltip on the other axes.
            try:
                self._annot.remove()
            except Exception:  # noqa: BLE001 — already detached by fig.clf()
                pass
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(OFFSET, OFFSET), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.45", fc="#1c1c1c", ec=ACCENT,
                      lw=1, alpha=0.95),
            color="white",
            arrowprops=dict(arrowstyle="-|>", color=ACCENT),
        )
        annot.set_annotation_clip(False)
        # Above all axes children *including* the title (see module docstring).
        annot.set_zorder(1000)
        annot.set_visible(False)
        self._annot = annot
        return annot

    def _place(self, annot, event) -> None:
        """Flip the bubble's offsets so it stays inside the canvas."""
        try:
            renderer = self._canvas.get_renderer()
            w, h = self._canvas.get_width_height()
            annot.set_position((OFFSET, OFFSET))
            annot.set_ha("left")
            annot.set_va("bottom")
            bbox = annot.get_window_extent(renderer)
            x_off = -OFFSET if event.x + OFFSET + bbox.width > w else OFFSET
            y_off = -OFFSET if event.y + OFFSET + bbox.height > h else OFFSET
        except Exception:
            # Renderer not ready — mirror near the top/right axes edges instead.
            try:
                ax_bbox = annot.axes.get_window_extent()
                x_off = -OFFSET if event.x > (ax_bbox.x0 + ax_bbox.width * 0.70) else OFFSET
                y_off = -OFFSET if event.y > (ax_bbox.y0 + ax_bbox.height * 0.70) else OFFSET
            except Exception:
                x_off = y_off = OFFSET
        annot.set_position((x_off, y_off))
        annot.set_ha("right" if x_off < 0 else "left")
        annot.set_va("top" if y_off < 0 else "bottom")
