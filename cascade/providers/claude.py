"""Anthropic Claude provider implementation."""

from typing import Optional, Iterator, TYPE_CHECKING
import json
import os
import shutil
import subprocess
import httpx
from .base import BaseProvider, ProviderConfig, ToolEvent, ToolEventCallback
from .registry import register_provider

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


@register_provider("claude")
class ClaudeProvider(BaseProvider):
    """Anthropic Claude API provider.

    Supports both standard API keys and OAuth tokens from Claude Code CLI.
    OAuth tokens (``sk-ant-oat01`` prefix) are proxied through `claude -p`.
    """

    _ACTIVITY_PREFIX = "[[cascade_activity]] "

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        self.client = httpx.Client(timeout=60.0)
        self._use_oauth_cli = config.api_key.startswith("sk-ant-oat01")
        self._claude_bin = shutil.which("claude")
        self._use_cli_proxy = self._use_oauth_cli and bool(self._claude_bin)
        default_activity = "1" if self._use_cli_proxy else "0"
        self._emit_activity = (
            os.getenv("CASCADE_CLAUDE_ACTIVITY", default_activity).lower()
            not in ("0", "false", "no", "off")
        )

    def _headers(self) -> dict:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _activity(self, message: str) -> Optional[str]:
        """Encode a status line into a chunk the TUI can detect."""
        if not self._emit_activity:
            return None
        return f"{self._ACTIVITY_PREFIX}{message}"

    def _stream_via_cli(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream assistant text by proxying through `claude -p`."""
        if not self._claude_bin:
            yield "Error: claude CLI not found in PATH for OAuth mode."
            return

        cmd = [
            self._claude_bin,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--add-dir",
            os.getcwd(),
        ]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        if system:
            cmd.extend(["--system-prompt", system])

        env = os.environ.copy()
        env.setdefault("NO_BROWSER", "true")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as e:
            yield f"Error: {e}"
            return

        saw_text = False
        saw_delta = False
        status_lines: list[str] = []
        start_msg = self._activity(f"starting claude cli in {os.getcwd()}")
        if start_msg:
            yield start_msg

        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                status_lines.append(line)
                status = self._activity(line)
                if status:
                    yield status
                continue

            ev_type = event.get("type")

            if ev_type == "system":
                if event.get("subtype") == "init":
                    model = event.get("model")
                    if isinstance(model, str) and model:
                        status = self._activity(f"model: {model}")
                        if status:
                            yield status
                continue

            if ev_type == "stream_event":
                inner = event.get("event", {})
                inner_type = inner.get("type")
                if inner_type == "content_block_start":
                    block = inner.get("content_block", {})
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "tool")
                        status = self._activity(f"tool: {tool_name}")
                        if status:
                            yield status
                elif inner_type == "content_block_delta":
                    delta = inner.get("delta", {})
                    text = delta.get("text")
                    if isinstance(text, str) and text:
                        saw_delta = True
                        saw_text = True
                        yield text
                elif inner_type == "message_start":
                    usage = inner.get("message", {}).get("usage", {})
                    in_t = usage.get("input_tokens", 0)
                    if isinstance(in_t, int):
                        self._last_usage = (in_t, 0)
                elif inner_type == "message_delta":
                    usage = inner.get("usage", {})
                    out_t = usage.get("output_tokens")
                    if isinstance(out_t, int):
                        prev = self._last_usage or (0, 0)
                        self._last_usage = (prev[0], out_t)
                continue

            if ev_type == "assistant":
                if saw_delta:
                    continue
                message = event.get("message", {})
                for block in message.get("content", []):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            saw_text = True
                            yield text
                usage = message.get("usage", {})
                in_t = usage.get("input_tokens")
                out_t = usage.get("output_tokens")
                if isinstance(in_t, int) and isinstance(out_t, int):
                    self._last_usage = (in_t, out_t)
                continue

            if ev_type == "result":
                usage = event.get("usage", {})
                in_t = usage.get("input_tokens")
                out_t = usage.get("output_tokens")
                if isinstance(in_t, int) and isinstance(out_t, int):
                    self._last_usage = (in_t, out_t)
                duration = event.get("duration_ms")
                if isinstance(duration, int):
                    status = self._activity(f"done in {duration}ms")
                    if status:
                        yield status
                if event.get("is_error"):
                    msg = event.get("result")
                    if isinstance(msg, str) and msg:
                        status_lines.append(msg)
                continue

            if ev_type == "rate_limit_event":
                info = event.get("rate_limit_info", {})
                resets = info.get("resetsAt")
                if isinstance(resets, int):
                    status = self._activity(f"five-hour window resets at {resets}")
                    if status:
                        yield status
                continue

        proc.wait()
        if proc.returncode != 0 and not saw_text:
            message = status_lines[-1] if status_lines else f"claude exited with code {proc.returncode}"
            yield f"Error: {message}"
        elif not saw_text and status_lines:
            yield f"Error: {status_lines[-1]}"

    def ask(self, prompt: str, system: Optional[str] = None) -> str:
        """Get a complete response from Claude."""
        chunks = []
        for chunk in self.stream(prompt, system):
            if isinstance(chunk, str) and chunk.startswith(self._ACTIVITY_PREFIX):
                continue
            chunks.append(chunk)
        return "".join(chunks)

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from Claude."""
        self._last_usage = None
        if self._use_cli_proxy:
            yield from self._stream_via_cli(prompt, system)
            return
        if self._use_oauth_cli and not self._claude_bin:
            yield "Error: Claude OAuth token detected, but claude CLI is not in PATH."
            return

        try:
            url = f"{self.base_url}/messages"
            payload = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens or 2048,
                "temperature": self.config.temperature,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system:
                payload["system"] = system

            with self.client.stream("POST", url, json=payload, headers=self._headers()) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                if "delta" in data and "text" in data["delta"]:
                                    yield data["delta"]["text"]
                            elif data.get("type") == "message_delta":
                                usage = data.get("usage", {})
                                out_tokens = usage.get("output_tokens", 0)
                                if out_tokens:
                                    prev = self._last_usage or (0, 0)
                                    self._last_usage = (prev[0], out_tokens)
                            elif data.get("type") == "message_start":
                                usage = data.get("message", {}).get("usage", {})
                                in_tokens = usage.get("input_tokens", 0)
                                self._last_usage = (in_tokens, 0)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"Error: {str(e)}"

    def ask_with_tools(
        self,
        prompt: str,
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
    ) -> tuple[str, list[dict]]:
        """Claude-native tool calling using tools array + tool_use/tool_result."""
        if self._use_cli_proxy:
            # Claude CLI OAuth mode is proxied as plain text chat.
            return self.ask(prompt, system), []

        from ..tools.executor import ToolExecutor

        executor = ToolExecutor(tools)
        tool_defs = [
            {
                "name": td.name,
                "description": td.description,
                "input_schema": td.parameters,
            }
            for td in tools.values()
        ]

        messages = [{"role": "user", "content": prompt}]
        tool_log = []

        for round_num in range(max_rounds):
            payload = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens or 2048,
                "temperature": self.config.temperature,
                "messages": messages,
                "tools": tool_defs,
            }
            if system:
                payload["system"] = system

            url = f"{self.base_url}/messages"
            try:
                response = self.client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                return f"Error: {e}", tool_log

            # Capture token usage
            usage = data.get("usage", {})
            in_t = usage.get("input_tokens", 0)
            out_t = usage.get("output_tokens", 0)
            if in_t or out_t:
                prev = self._last_usage or (0, 0)
                self._last_usage = (prev[0] + in_t, prev[1] + out_t)

            # Check stop reason
            stop_reason = data.get("stop_reason", "end_turn")

            # Extract text and tool_use blocks
            text_parts = []
            tool_uses = []
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_uses.append(block)

            if not tool_uses or stop_reason != "tool_use":
                return "".join(text_parts), tool_log

            # Append the assistant message with all content blocks
            messages.append({"role": "assistant", "content": data["content"]})

            # Execute each tool call and build tool_result messages
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use["name"]
                tool_input = tool_use.get("input", {})
                tool_id = tool_use["id"]

                if on_tool_event:
                    on_tool_event(ToolEvent(
                        kind="tool_start",
                        tool_name=tool_name,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        tool_input=tool_input,
                    ))

                result = executor.execute(tool_name, tool_input)
                tool_log.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": result,
                })

                if on_tool_event:
                    on_tool_event(ToolEvent(
                        kind="tool_done",
                        tool_name=tool_name,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        tool_input=tool_input,
                        tool_output=result,
                    ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        # Exhausted rounds, return whatever text we have
        return "".join(text_parts) if text_parts else "", tool_log

    def compare(self, prompt: str, system: Optional[str] = None) -> dict:
        """Generate comparison data."""
        response = self.ask(prompt, system)
        return {
            "provider": self.name,
            "model": self.config.model,
            "response": response,
            "length": len(response),
        }

    def __del__(self):
        """Cleanup HTTP client."""
        try:
            self.client.close()
        except Exception:
            pass
