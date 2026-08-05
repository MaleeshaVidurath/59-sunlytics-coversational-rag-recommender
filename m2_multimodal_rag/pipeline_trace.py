"""
M2 Pipeline Trace — staged, human-readable request logging.

Turns the flat M2 log into the six stages of the M2 architecture diagram, with
per-stage timing and a closing summary, so a reader can follow one request from
query expansion through to the verified output.

    STAGE 1  RETRIEVAL              LLM query expansion → CLIP → FAISS + RRF
    STAGE 2  HARD FILTER + SCORING  filters, boosts/penalties, CF, Kansei KB
    STAGE 3  RE-RANKING             MiniLM cross-encoder → LLM semantic rerank
    STAGE 4  SELECTION              Thompson-sampling bandit + MMR
    STAGE 5  HALLUCINATION GUARD    pre-filter → L1 → L2 → L3/CLIPScore
    STAGE 6  OUTPUT                 verified items + image URLs

Enable with M2_TRACE=1 in .env (or the environment). When disabled every call
here is a no-op and the existing logs print exactly as before.

Existing print() calls inside each phase are NOT replaced — they appear nested
under their stage banner, so both traced and untraced modes stay useful.

Guard output is grouped per item.  Phase 6 verifies items in parallel threads,
which interleaves their logs; capture_item() redirects a thread's stdout into a
per-item buffer and flush_items() replays them grouped, in MMR order. This keeps
the parallel speed-up while producing output that reads sequentially.

Note: the stage/timing state is module-level and assumes one traced request at a
time, which is the intended demo/inspection setting.
"""

import os
import sys
import threading
import time
from contextlib import contextmanager

ENABLED = os.getenv("M2_TRACE", "").strip().lower() in ("1", "true", "yes", "on")

_WIDTH = 78
_STAGE_TOTAL = 6


def _console_supports_unicode() -> bool:
    """
    Box-drawing characters raise UnicodeEncodeError on a cp1252 console
    (plain cmd.exe, or any piped stdout on Windows). Fall back to ASCII there
    rather than crashing the request.
    """
    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "═│┌└▸•→".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


_UNI = _console_supports_unicode()

# (unicode, ascii) pairs chosen so both render at the same width
_H     = "═" if _UNI else "="
_HL    = "─" if _UNI else "-"
_V     = "║" if _UNI else "|"
_TL, _TR = ("╔", "╗") if _UNI else ("+", "+")
_BL, _BR = ("╚", "╝") if _UNI else ("+", "+")
_FL, _FR = ("┌", "┐") if _UNI else ("+", "+")
_GL, _GR = ("└", "┘") if _UNI else ("+", "+")
_STEP  = "▸" if _UNI else ">"
_DOT   = "•" if _UNI else "-"
_ARROW = "→" if _UNI else "->"


# ---------------------------------------------------------------------------
# Per-thread stdout routing (used to group parallel guard output per item)
# ---------------------------------------------------------------------------
class _ThreadRoutedStdout:
    """
    Drop-in stdout replacement. Writes from a thread that has registered a
    buffer go to that buffer; every other write passes through untouched.
    """

    def __init__(self, real):
        self._real = real
        self._buffers: dict[int, list] = {}
        self._lock = threading.Lock()

    def register(self) -> None:
        with self._lock:
            self._buffers[threading.get_ident()] = []

    def collect(self) -> str:
        with self._lock:
            return "".join(self._buffers.pop(threading.get_ident(), []))

    def write(self, s):
        with self._lock:
            buf = self._buffers.get(threading.get_ident())
        if buf is not None:
            buf.append(s)
            return len(s)
        return self._real.write(s)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        # __dict__ lookup avoids infinite recursion if _real is not yet set.
        return getattr(self.__dict__["_real"], name)


_router: _ThreadRoutedStdout | None = None


def _ensure_router() -> _ThreadRoutedStdout:
    global _router
    if _router is None:
        _router = _ThreadRoutedStdout(sys.stdout)
        sys.stdout = _router
    return _router


# ---------------------------------------------------------------------------
# Trace state
# ---------------------------------------------------------------------------
_timings: list[tuple[str, float]] = []
_open_stage: list = []          # [title, t0] while a stage is open
_item_logs: dict[str, str] = {}
_item_titles: dict[str, str] = {}
_request_t0: float = 0.0


def _out(line: str = "") -> None:
    real = _router._real if _router is not None else sys.stdout
    real.write(line + "\n")


def _rule(char: str = _HL) -> str:
    return char * _WIDTH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def begin(action: str, session_id: str | None, user_message: str) -> None:
    """Opens a traced request and prints the header banner."""
    global _request_t0
    if not ENABLED:
        return
    _ensure_router()
    _timings.clear()
    _item_logs.clear()
    _item_titles.clear()
    _open_stage.clear()
    _request_t0 = time.time()

    msg = user_message if len(user_message) <= _WIDTH - 6 else user_message[: _WIDTH - 9] + "..."
    _out()
    _out(_TL + _H * _WIDTH + _TR)
    _out(_V + "  M2 MULTIMODAL RAG - REQUEST TRACE".ljust(_WIDTH) + _V)
    _out(_V + f"  action={action}   session={session_id or 'n/a'}".ljust(_WIDTH) + _V)
    _out(_V + f'  "{msg}"'.ljust(_WIDTH) + _V)
    _out(_BL + _H * _WIDTH + _BR)


def stage_start(n: int, title: str, subtitle: str = "") -> None:
    """Opens stage n. Any stage still open is closed first."""
    if not ENABLED:
        return
    if _open_stage:
        stage_end()
    head = f"{_HL} STAGE {n}/{_STAGE_TOTAL} {_HL}{_HL} {title} "
    if subtitle:
        head += f"{_HL}{_HL} {subtitle} "
    _out()
    _out(_FL + head.ljust(_WIDTH, _HL) + _FR)
    _open_stage.extend([title, time.time()])


def stage_end() -> None:
    """Closes the open stage and records its elapsed time."""
    if not ENABLED or not _open_stage:
        return
    title, t0 = _open_stage[0], _open_stage[1]
    dt = time.time() - t0
    _timings.append((title, dt))
    _open_stage.clear()
    _out(_GL + f"{_HL} {title.lower()} done in {dt:.2f}s ".ljust(_WIDTH, _HL) + _GR)


def step(label: str, detail: str = "") -> None:
    """A named sub-step inside the current stage."""
    if not ENABLED:
        return
    _out(f"   {_STEP} {label}" + (f"   {_ARROW} {detail}" if detail else ""))


def bullet(text: str) -> None:
    """An indented detail line (e.g. one query variant)."""
    if not ENABLED:
        return
    _out(f"       {_DOT} {text}")


# ---------------------------------------------------------------------------
# Per-item capture for the parallel hallucination guard
# ---------------------------------------------------------------------------
@contextmanager
def capture_item(article_id: str, title: str = ""):
    """
    Redirects this thread's stdout into a buffer for `article_id`, so parallel
    guard output can be replayed grouped by item instead of interleaved.
    """
    if not ENABLED:
        yield
        return
    router = _ensure_router()
    router.register()
    _item_titles[article_id] = title
    try:
        yield
    finally:
        _item_logs[article_id] = router.collect()


def flush_items(order: list[str]) -> None:
    """Replays captured per-item guard logs, grouped, in the given order."""
    if not ENABLED:
        return
    for aid in order:
        body = _item_logs.pop(aid, "")
        title = _item_titles.get(aid, "")
        _out()
        _out(f"  {_HL}{_HL} item {aid}" + (f'  "{title}"' if title else "") + " " + _rule()[:20])
        for line in body.splitlines():
            if line.strip():
                _out("  " + line)
    # Anything not in `order` (shouldn't happen) still gets shown.
    for aid, body in list(_item_logs.items()):
        _out(f"  ── item {aid} (unordered)")
        for line in body.splitlines():
            if line.strip():
                _out("  " + line)
        _item_logs.pop(aid, None)


# ---------------------------------------------------------------------------
def end(n_items: int = 0) -> None:
    """Closes any open stage and prints the timing summary."""
    if not ENABLED:
        return
    if _open_stage:
        stage_end()
    total = time.time() - _request_t0
    _out()
    _out(_TL + _H * _WIDTH + _TR)
    _out(_V + f"  TOTAL {total:.2f}s   |   {n_items} verified item(s) returned".ljust(_WIDTH) + _V)
    parts = "   ".join(f"{t.strip().lower()} {d:.2f}s" for t, d in _timings)
    for chunk in _wrap(parts, _WIDTH - 4):
        _out(_V + "  " + chunk.ljust(_WIDTH - 2) + _V)
    _out(_BL + _H * _WIDTH + _BR)
    _out()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split("   "), [], ""
    for w in words:
        if len(cur) + len(w) + 3 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur}   {w}" if cur else w
    if cur:
        lines.append(cur)
    return lines or [""]
