"""Compact widget showing a completed tool call in the chat history."""

from rich.text import Text
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static

from ..theme import PALETTE


class ToolCallWidget(Widget):
    """A single tool-call row: gutter label + tool body."""

    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        width: 100%;
        padding: 0 0 0 0;
        layout: horizontal;
    }
    """

    def __init__(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._tool_input = tool_input
        self._tool_output = tool_output

    def compose(self) -> ComposeResult:
        yield _ToolGutter()
        yield _ToolBody(self._tool_name, self._tool_input, self._tool_output)


class _ToolGutter(Static):
    """Fixed-width gutter showing 'tool' in dim text."""

    DEFAULT_CSS = """
    _ToolGutter {
        width: 10;
        min-width: 10;
        max-width: 10;
        height: auto;
        text-align: right;
        padding-right: 1;
    }
    """

    def render(self) -> Text:
        return Text(f"{'tool':>8}", style=f"dim {PALETTE.text_dim}")


class _ToolBody(Static):
    """Tool name + truncated args + truncated result."""

    DEFAULT_CSS = """
    _ToolBody {
        width: 1fr;
        height: auto;
        padding-left: 1;
    }
    """

    def __init__(
        self, tool_name: str, tool_input: dict, tool_output: str, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._tool_input = tool_input
        self._tool_output = tool_output

    def render(self) -> Text:
        t = Text()
        t.append(self._tool_name, style=f"bold {PALETTE.file_ops}")

        # Truncated args
        args_str = ""
        if self._tool_input:
            import json
            try:
                args_str = json.dumps(self._tool_input, ensure_ascii=False)
            except Exception:
                args_str = str(self._tool_input)
        if args_str:
            if len(args_str) > 80:
                args_str = args_str[:77] + "..."
            t.append(f" {args_str}", style=f"dim {PALETTE.text_dim}")

        # Truncated result
        result = self._tool_output.strip()
        if result:
            if len(result) > 120:
                result = result[:117] + "..."
            result_oneline = result.replace("\n", " ")
            t.append(" -> ", style=f"dim {PALETTE.text_dim}")
            t.append(result_oneline, style=PALETTE.text_dim)

        return t
