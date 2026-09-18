# Copyright (c) 2026 Santiago Hofwimmer
"""
netlist_paths.py – Resolve relative include paths in a testbench deck.

Chipify never simulates a netlist where it was authored: the deck is read into
a string, rendered through Jinja2 and written out *fresh* into the scratch dir
(``FAST_TMP`` for ngspice, a per-worker subdir of it for VACASK). A relative
``.include`` / ``.lib`` path would therefore be resolved by the simulator
against that scratch dir, which has no relationship to ``tb/`` — the reason
absolute paths worked and relative ones silently failed.

This module rewrites those paths to absolute at netlist-generation time,
anchored on **the testbench's own directory** (``TB_DIR/<dirname(tb_path)>``):
a relative path is relative to the ``.sch``/``.spice``/``.sim`` file that
contains it, and to nothing else.

A **bare filename** is deliberately left alone — that is the documented
``work/`` staging contract (README, "Model files"), where ``*.lib``/``*.mod``/
``*.inc`` are copied flat into FAST_TMP by :mod:`chipify.engines.staging` and
referenced by name. Tokens holding an environment variable (``$VAR``,
``%VAR%``), a Jinja2 placeholder or a ``~`` are passed through untouched: they
are not ours to resolve.

Anything that *is* a candidate must exist next to the testbench; if it does
not, :class:`NetlistPathError` names every unresolved path at once. The
orchestrator turns that into ``test.template_error`` (see
``simulator.generate_templates``), so only that testbench fails and the rest of
the sweep continues.

Engines reach this through
:meth:`chipify.engines.base.BaseSimulator.resolve_netlist_paths`.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger("chipify.engines.netlist_paths")


class NetlistPathError(ValueError):
    """A relative include path in a deck did not resolve next to the testbench."""


# ── Directive matchers ────────────────────────────────────────────────────────
# Both are anchored at "^" and only ever have their "path" group replaced, so a
# .lib section argument and a spectre "section=tt" are byte-preserved. The
# alternation is longest-first (".include" never matches as ".inc") and the
# lookahead stops ".included_by" / "includes".
_SPICE_RE = re.compile(
    r'^(?P<lead>[ \t]*)(?P<dir>\.(?:include|inc|lib))(?![A-Za-z0-9_])'
    r'(?P<gap>[ \t]+)(?P<path>"[^"\n]*"|\'[^\'\n]*\'|[^\s"\']+)',
    re.IGNORECASE,
)
_SPECTRE_RE = re.compile(
    r'^(?P<lead>[ \t]*)(?P<dir>ahdl_include|include|load)(?![A-Za-z0-9_])'
    r'(?P<gap>[ \t]+)(?P<path>"[^"\n]*"|\'[^\'\n]*\'|[^\s"\']+)',
    re.IGNORECASE,
)

#: Windows drive-letter prefix ("C:/", "D:\") — see _is_absolute_token.
_WIN_DRIVE = re.compile(r'^[A-Za-z]:[\\/]')

#: Env-var and Jinja2 markers. A token carrying one of these is not ours to
#: resolve: ngspice/xschem expand their own, and Jinja2 fills the rest in later.
_OPAQUE = ('$', '%', '{{', '{%', '}}')


def _is_absolute_token(tok: str) -> bool:
    """True if *tok* is an absolute path, independent of the host platform.

    ``os.path.isabs`` must not be used here: netlist text is Linux-flavoured but
    the test suite also runs on Windows, where ``ntpath.isabs("/foss/pdks/x")``
    is True only by accident (drive-relative) and ``posixpath.isabs("D:/x")`` is
    False. That would make the answer depend on the dev machine, not the deck.
    """
    return tok.startswith(('/', '\\')) or bool(_WIN_DRIVE.match(tok))


def _is_rewritable(tok: str) -> bool:
    """True if *tok* is a relative path that names a directory component.

    A bare filename is excluded on purpose (the ``work/`` staging contract), as
    are absolute, ``~``-rooted, env-var and Jinja2 tokens.
    """
    if not tok or tok.startswith('~') or _is_absolute_token(tok):
        return False
    if any(marker in tok for marker in _OPAQUE):
        return False
    return '/' in tok or '\\' in tok


def _abs_posix(base_dir: str | os.PathLike[str], tok: str) -> str:
    """Join *tok* onto *base_dir* and normalize to one separator flavour.

    ``os.path.normpath`` collapses ``..`` lexically. ``Path.resolve()`` is
    deliberately avoided — it dereferences the staged OSDI/model symlinks and
    can produce UNC paths on Windows; ``os.path.relpath`` likewise (it raises
    across drives, cf. ``xschem._safe_rel``).
    """
    joined = os.path.join(str(base_dir), tok.replace('\\', '/'))
    return Path(os.path.normpath(joined)).as_posix()


def _requote(new: str, original_token: str) -> str:
    """Re-apply the original quoting to *new* (adding quotes if it needs them)."""
    quote = original_token[0] if original_token[:1] in ('"', "'") else ''
    if quote:
        return quote + new + quote
    return f'"{new}"' if any(ch.isspace() for ch in new) else new


def _fold_continuations(lines: list[str]) -> list[tuple[int, int, str]]:
    """Group *lines* into ``(start, stop, logical_text)`` spans.

    Folds SPICE ``+`` continuations and spectre trailing backslashes so a
    directive split over several physical lines is matched as one.
    """
    out: list[tuple[int, int, str]] = []
    i, n = 0, len(lines)
    while i < n:
        text, j = lines[i], i + 1
        while j < n:
            if text.rstrip().endswith('\\'):
                text = text.rstrip()[:-1] + ' ' + lines[j].strip()
            elif lines[j].lstrip().startswith('+'):
                text = text + ' ' + lines[j].lstrip()[1:].strip()
            else:
                break
            j += 1
        out.append((i, j, text))
        i = j
    return out


def _rewrite_logical(logical: str, base_dir: str | os.PathLike[str],
                     unresolved: list[str]) -> str | None:
    """Return *logical* with its include path absolutized, or None if unchanged.

    Candidates that do not exist under *base_dir* are appended to *unresolved*
    (as ready-to-print reason lines) and the line is left alone.
    """
    match = _SPICE_RE.match(logical) or _SPECTRE_RE.match(logical)
    if match is None:
        return None

    raw = match.group("path")
    tok = raw[1:-1] if raw[:1] in ('"', "'") else raw
    if not _is_rewritable(tok):
        return None

    directive = match.group("dir")
    resolved = _abs_posix(base_dir, tok)

    if not os.path.isfile(resolved):
        unresolved.append(f"{directive} {tok}   (no file at {resolved})")
        return None
    if '{' in resolved or '}' in resolved:
        # The whole deck is re-rendered by Jinja2 with StrictUndefined, so a
        # brace in the path would turn a path bug into a template crash.
        unresolved.append(
            f"{directive} {tok}   (resolves to {resolved}, which contains a "
            f"brace Jinja2 would try to render)"
        )
        return None

    return (logical[:match.start("path")]
            + _requote(resolved, raw)
            + logical[match.end("path"):])


def testbench_dir(test: object) -> Path:
    """Directory the testbench's own file lives in — the sole resolution anchor.

    ``tb_path`` is authored with ``/`` in the datasheet YAML, so a nested
    ``"sub/tb_x"`` anchors on ``TB_DIR/sub``. ``settings`` is imported here
    rather than at module scope: importing it creates the project folders as a
    side effect, and ``chipify.engines`` is imported by the datasheet schema
    just to validate engine names.
    """
    from chipify import settings
    tb_path = str(getattr(test, "tb_path", "") or "").replace('\\', '/')
    sub = os.path.dirname(tb_path)
    base = Path(settings.TB_DIR)  # tests monkeypatch settings.* as plain strings
    return base / sub if sub else base


def absolutize_netlist_includes(text: str, base_dir: str | os.PathLike[str], *,
                                label: str = "") -> str:
    """Rewrite relative include-family paths in *text* to absolute paths.

    Handles ``.include`` / ``.inc`` / ``.lib`` (SPICE) and ``include`` /
    ``ahdl_include`` / ``load`` (spectre, VACASK). Only a token that is relative
    *and* names a directory component is a candidate; each must exist under
    *base_dir*. Comment lines, spectre block comments and everything between
    ``.control`` and ``.endc`` (chipify-owned ``wrdata``/``setplot``/``echo``)
    are never touched. Idempotent.

    *label* (the tb_path) only decorates the error message.

    Raises :class:`NetlistPathError` listing every candidate that did not
    resolve; the deck is otherwise returned with those paths made absolute.
    """
    lines = text.split('\n')
    result: list[str] = []
    unresolved: list[str] = []
    in_control = False
    in_block_comment = False

    for start, stop, logical in _fold_continuations(lines):
        stripped = logical.lstrip()

        if in_block_comment:
            if '*/' in logical:
                in_block_comment = False
            result.extend(lines[start:stop])
            continue
        if stripped.startswith('/*'):
            if '*/' not in stripped[2:]:
                in_block_comment = True
            result.extend(lines[start:stop])
            continue
        # '*' is a SPICE comment, '//' a spectre one, ';' an ngspice one.
        if stripped.startswith(('*', '//', ';')):
            result.extend(lines[start:stop])
            continue

        low = stripped.lower()
        if low.startswith('.control'):
            in_control = True
        elif low.startswith('.endc'):
            in_control = False
        if in_control:
            result.extend(lines[start:stop])
            continue

        rewritten = _rewrite_logical(logical, base_dir, unresolved)
        if rewritten is None:
            result.extend(lines[start:stop])
        else:
            # A rewritten line folds its continuations into one physical line;
            # harmless, an include directive never needs to be split.
            log.debug("%s: rewrote %r -> %r", label or "<deck>", logical, rewritten)
            result.append(rewritten)

    if unresolved:
        prefix = f"{label}: " if label else ""
        raise NetlistPathError(
            f"{prefix}{len(unresolved)} include path(s) could not be resolved "
            f"against the testbench directory {base_dir}:\n  "
            + "\n  ".join(unresolved)
            + "\nA relative include path is resolved against the directory of "
              "the testbench itself. Use a path relative to that directory, an "
              "absolute path, or put the file in work/ and reference it by bare "
              "filename."
        )

    return '\n'.join(result)
