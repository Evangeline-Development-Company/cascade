"""OpenAI provider for GPT-4o, o1, o3, and Codex models."""

import json
import os
import shutil
import subprocess
from typing import Optional, Iterator, TYPE_CHECKING
import httpx
from .base import BaseProvider, ProviderConfig, ToolEventCallback
from .registry import register_provider
from ._openai_tools import openai_ask_with_tools

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


@register_provider("openai")
class OpenAIProvider(BaseProvider):
    """OpenAI API provider - supports custom base_url for Azure/proxies."""

    _ACTIVITY_PREFIX = "[[cascade_activity]] "

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.client = httpx.Client(timeout=60.0)
        self._codex_bin = shutil.which("codex")
        self._use_oauth_cli = self._looks_like_jwt(config.api_key)
        self._use_cli_proxy = (
            self._use_oauth_cli
            and bool(self._codex_bin)
            and self.base_url.rstrip("/") == "https://api.openai.com/v1"
        )
        default_activity = "1" if self._use_cli_proxy else "0"
        self._emit_activity = (
            os.getenv(
                "CASCADE_OPENAI_ACTIVITY",
                os.getenv("CASCADE_CODEX_ACTIVITY", default_activity),
            ).lower()
            not in ("0", "false", "no", "off")
        )

    @staticmethod
    def _looks_like_jwt(token: str) -> bool:
        token = (token or "").strip()
        return token.startswith("eyJ") and token.count(".") >= 2

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _activity(self, message: str) -> Optional[str]:
        """Encode a status line into a chunk the TUI can detect."""
        if not self._emit_activity:
            return None
        return f"{self._ACTIVITY_PREFIX}{message}"

    def _build_cli_prompt(self, prompt: str, system: Optional[str]) -> str:
        """Build a single prompt string for Codex CLI mode."""
        if not system:
            return prompt
        return (
            "System instructions:\n"
            f"{system}\n\n"
            "User request:\n"
            f"{prompt}"
        )

    def _extract_codex_text(self, item: dict) -> str:
        """Extract assistant text from a Codex JSON event item."""
        text = item.get("text")
        if isinstance(text, str) and text:
            return text

        content = item.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "output_text":
                    chunk = block.get("text")
                    if isinstance(chunk, str):
                        parts.append(chunk)
            if parts:
                return "".join(parts)
        return ""

    def _stream_via_cli(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream assistant text by proxying through `codex exec --json`."""
        if not self._codex_bin:
            yield "Error: codex CLI not found in PATH for OAuth mode."
            return

        full_prompt = self._build_cli_prompt(prompt, system)
        cmd = [
            self._codex_bin,
            "exec",
            "--json",
            "--cd",
            os.getcwd(),
        ]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        cmd.append(full_prompt)

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
        status_lines: list[str] = []
        start_msg = self._activity(f"starting codex cli in {os.getcwd()}")
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
            if ev_type == "thread.started":
                thread_id = event.get("thread_id")
                if isinstance(thread_id, str) and thread_id:
                    status = self._activity(f"thread: {thread_id[:12]}...")
                    if status:
                        yield status
                continue

            if ev_type == "turn.started":
                status = self._activity("turn started")
                if status:
                    yield status
                continue

            if ev_type == "error":
                msg = event.get("message")
                if isinstance(msg, str) and msg:
                    status_lines.append(msg)
                    status = self._activity(msg)
                    if status:
                        yield status
                continue

            if ev_type == "item.completed":
                item = event.get("item", {})
                item_type = item.get("type")
                if item_type == "agent_message":
                    text = self._extract_codex_text(item)
                    if text:
                        saw_text = True
                        yield text
                elif item_type == "reasoning":
                    reason = item.get("text")
                    if isinstance(reason, str) and reason:
                        status = self._activity(reason)
                        if status:
                            yield status
                elif item_type == "error":
                    msg = item.get("message")
                    if isinstance(msg, str) and msg:
                        status_lines.append(msg)
                        status = self._activity(msg)
                        if status:
                            yield status
                continue

            if ev_type == "turn.completed":
                usage = event.get("usage", {})
                in_t = usage.get("input_tokens")
                out_t = usage.get("output_tokens")
                if isinstance(in_t, int) and isinstance(out_t, int):
                    self._last_usage = (in_t, out_t)
                status = self._activity("turn completed")
                if status:
                    yield status
                continue

        proc.wait()
        if proc.returncode != 0 and not saw_text:
            message = status_lines[-1] if status_lines else f"codex exited with code {proc.returncode}"
            yield f"Error: {message}"
        elif not saw_text and status_lines:
            yield f"Error: {status_lines[-1]}"

    def ask(self, prompt: str, system: Optional[str] = None) -> str:
        """Get a complete response from OpenAI."""
        chunks = []
        for chunk in self.stream(prompt, system):
            if isinstance(chunk, str) and chunk.startswith(self._ACTIVITY_PREFIX):
                continue
            chunks.append(chunk)
        return "".join(chunks)

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from OpenAI."""
        self._last_usage = None
        if self._use_cli_proxy:
            yield from self._stream_via_cli(prompt, system)
            return
        if self._use_oauth_cli and not self._use_cli_proxy:
            if not self._codex_bin:
                yield "Error: Codex OAuth token detected, but codex CLI is not in PATH."
            else:
                yield (
                    "Error: Codex OAuth token requires the default OpenAI base URL "
                    "(https://api.openai.com/v1)."
                )
            return

        try:
            url = f"{self.base_url}/chat/completions"

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens:
                payload["max_tokens"] = self.config.max_tokens

            with self.client.stream("POST", url, json=payload, headers=self._headers()) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        usage = data.get("usage")
                        if usage:
                            self._last_usage = (
                                usage.get("prompt_tokens", 0),
                                usage.get("completion_tokens", 0),
                            )
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
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
        """OpenAI-native tool calling."""
        if self._use_cli_proxy:
            # Codex CLI OAuth mode is proxied as plain text chat.
            return self.ask(prompt, system), []

        return openai_ask_with_tools(
            client=self.client,
            url=f"{self.base_url}/chat/completions",
            headers=self._headers(),
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            prompt=prompt,
            tools=tools,
            system=system,
            max_rounds=max_rounds,
            on_tool_event=on_tool_event,
        )

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
