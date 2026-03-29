"""Chat message widgets with gutter attribution.

ChatHistory -- scrollable container
MessageWidget -- horizontal row: GutterLabel + MessageBody
ThinkingIndicator -- braille spinner during provider processing
"""

import re
import time

from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static

from ..theme import PALETTE, get_accent, get_abbreviation


# ---------------------------------------------------------------------------
# Inline markdown helpers (ported from cascade/ui/markdown.py)
# ---------------------------------------------------------------------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADER = re.compile(r"^(#{1,6})\s+(.*)")


def _render_md_line(line: str) -> Text:
    """Convert a single markdown line to styled Rich Text."""
    # Headers
    m = _HEADER.match(line)
    if m:
        level = len(m.group(1))
        style = f"bold {PALETTE.text_bright}" if level <= 2 else f"bold {PALETTE.text_primary}"
        return Text(m.group(2), style=style)

    # Bullet lists
    stripped = line.lstrip()
    if stripped.startswith(("- ", "* ", "+ ")):
        indent = len(line) - len(stripped)
        t = Text()
        t.append(" " * indent)
        t.append(stripped[0], style=PALETTE.text_dim)
        t.append_text(_inline_format(f" {stripped[2:]}"))
        return t

    # Numbered lists
    num_match = re.match(r"^(\s*\d+\.)\s+(.*)", line)
    if num_match:
        t = Text()
        t.append(num_match.group(1), style=PALETTE.text_dim)
        t.append_text(_inline_format(f" {num_match.group(2)}"))
        return t

    return _inline_format(line)


def _inline_format(text: str) -> Text:
    """Apply inline code, bold, italic, link formatting."""
    result = Text()
    spans: list[tuple[int, int, str, str]] = []
    for m in _INLINE_CODE.finditer(text):
        spans.append((m.start(), m.end(), "code", m.group(1)))
    for m in _BOLD.finditer(text):
        spans.append((m.start(), m.end(), "bold", m.group(1)))
    for m in _ITALIC.finditer(text):
        spans.append((m.start(), m.end(), "italic", m.group(1)))
    for m in _LINK.finditer(text):
        spans.append((m.start(), m.end(), "link", m.group(1)))

    spans.sort(key=lambda s: s[0])
    filtered = []
    last_end = 0
    for start, end, kind, content in spans:
        if start >= last_end:
            filtered.append((start, end, kind, content))
            last_end = end

    pos = 0
    for start, end, kind, content in filtered:
        if start > pos:
            result.append(text[pos:start], style=PALETTE.text_primary)
        if kind == "code":
            result.append(f" {content} ", style=f"on {PALETTE.surface} {PALETTE.inline_code}")
        elif kind == "bold":
            result.append(content, style=f"bold {PALETTE.text_bright}")
        elif kind == "italic":
            result.append(content, style=f"italic {PALETTE.text_primary}")
        elif kind == "link":
            result.append(content, style=f"underline {PALETTE.inline_code}")
        pos = end

    if pos < len(text):
        result.append(text[pos:], style=PALETTE.text_primary)

    return result


def render_content(content: str) -> Text:
    """Render multi-line markdown content (prose only, no fenced blocks)."""
    result = Text()
    for i, line in enumerate(content.split("\n")):
        if i > 0:
            result.append("\n")
        result.append_text(_render_md_line(line))
    return result


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class ChatHistory(VerticalScroll):
    """Scrollable container for all conversation messages."""

    DEFAULT_CSS = """
    ChatHistory {
        height: 1fr;
        width: 100%;
        padding: 1 2;
        background: #0d1117;
    }
    """

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self.scroll_relative(y=-6, animate=False, force=True)
        event.stop()
        event.prevent_default()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self.scroll_relative(y=6, animate=False, force=True)
        event.stop()
        event.prevent_default()


class MessageWidget(Widget):
    """A single message row: gutter label + body content."""

    DEFAULT_CSS = """
    MessageWidget {
        height: auto;
        width: 100%;
        padding: 0 0 1 0;
        layout: horizontal;
    }
    """

    def __init__(self, role: str, content: str, tokens: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._role = role
        self._content = content
        self._tokens = tokens

    def compose(self) -> ComposeResult:
        yield GutterLabel(self._role)
        yield MessageBody(self._content)


class GutterLabel(Static):
    """Fixed-width right-aligned label: provider name in accent / 'you' in dim."""

    DEFAULT_CSS = """
    GutterLabel {
        width: 10;
        min-width: 10;
        max-width: 10;
        height: auto;
        text-align: right;
        padding-right: 1;
    }
    """

    def __init__(self, role: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._role = role

    def render(self) -> Text:
        if self._role == "you":
            return Text(f"{'you':>8}", style=f"dim {PALETTE.text_dim}")
        if self._role == "system":
            return Text(f"{'sys':>8}", style=f"dim {PALETTE.text_dim}")
        accent = get_accent(self._role)
        abbr = get_abbreviation(self._role)
        return Text(f"{abbr:>8}", style=f"bold {accent}")


class MessageBody(Static):
    """The message content with inline markdown rendering."""

    DEFAULT_CSS = """
    MessageBody {
        width: 1fr;
        height: auto;
        padding-left: 1;
    }
    """

    def __init__(self, content: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._content = content

    def render(self) -> Text:
        return render_content(self._content)


class ThinkingIndicator(Static):
    """Braille spinner shown while provider is processing."""

    SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: 1;
        width: 100%;
        padding: 0 0 0 11;
    }
    """

    def __init__(self, provider: str = "gemini", label: str = "thinking...", **kwargs) -> None:
        super().__init__(**kwargs)
        self._provider = provider
        self._label = label
        self._idx = 0
        self._timer = None
        self._started_at = time.monotonic()

    def on_mount(self) -> None:
        self._started_at = time.monotonic()
        self._timer = self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        self._idx = (self._idx + 1) % len(self.SPINNER_FRAMES)
        self.refresh()

    def on_unmount(self) -> None:
        if self._timer:
            self._timer.stop()

    def set_label(self, label: str) -> None:
        """Update activity label while the spinner is active."""
        self._label = label
        self.refresh()

    def render(self) -> Text:
        accent = get_accent(self._provider)
        ch = self.SPINNER_FRAMES[self._idx]
        elapsed = max(0, int(time.monotonic() - self._started_at))
        t = Text()
        t.append(ch, style=f"bold {accent}")
        label = self._label.strip()
        if elapsed > 0:
            t.append(f" {label} | {elapsed}s", style=f"dim {PALETTE.text_dim}")
        else:
            t.append(f" {label}", style=f"dim {PALETTE.text_dim}")
        return t
