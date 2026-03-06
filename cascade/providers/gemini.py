"""Google Gemini provider implementation.

Supports two auth paths:
- Gemini API key (direct HTTP requests)
- Gemini CLI OAuth token (``ya29.*``), proxied through ``gemini -p``
"""

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


@register_provider("gemini")
class GeminiProvider(BaseProvider):
    """Google Gemini API provider."""

    _ACTIVITY_PREFIX = "[[cascade_activity]] "

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta/models"
        self.client = httpx.Client(timeout=60.0)
        # OAuth tokens (from Gemini CLI) start with "ya29." and use Bearer auth
        # API keys use ?key= query param
        self._use_bearer = config.api_key.startswith("ya29.")
        self._gemini_bin = shutil.which("gemini")
        self._use_cli_proxy = self._use_bearer and bool(self._gemini_bin)
        default_activity = "1" if self._use_cli_proxy else "0"
        self._emit_activity = (
            os.getenv("CASCADE_GEMINI_ACTIVITY", default_activity).lower()
            not in ("0", "false", "no", "off")
        )

    def _auth_params(self) -> tuple[dict, dict]:
        """Return (headers, params) for authentication."""
        headers = {"Content-Type": "application/json"}
        if self._use_bearer:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
            return headers, {}
        return headers, {"key": self.config.api_key}

    def _build_cli_prompt(self, prompt: str, system: Optional[str]) -> str:
        """Build a single prompt string for Gemini CLI mode."""
        if not system:
            return prompt
        return (
            "System instructions:\n"
            f"{system}\n\n"
            "User request:\n"
            f"{prompt}"
        )

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
        """Stream assistant text by proxying through `gemini -p`."""
        if not self._gemini_bin:
            yield "Error: gemini CLI not found in PATH for OAuth mode."
            return

        full_prompt = self._build_cli_prompt(prompt, system)
        cmd = [
            self._gemini_bin,
            "-p",
            full_prompt,
            "--output-format",
            "stream-json",
            "--include-directories",
            os.getcwd(),
        ]
        if self.config.model:
            cmd.extend(["--model", self.config.model])

        env = os.environ.copy()
        # In SSH/headless environments this prevents browser-launch auth branches.
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
        non_json_lines: list[str] = []
        start_msg = self._activity(f"starting gemini cli in {os.getcwd()}")
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
                # Gemini CLI can emit plain status lines (e.g., retry notices).
                if line != "Loaded cached credentials.":
                    non_json_lines.append(line)
                    status = self._activity(line)
                    if status:
                        yield status
                continue

            ev_type = event.get("type")
            if ev_type == "init":
                model = event.get("model", "")
                status = self._activity(f"model: {model}")
                if status:
                    yield status
            elif ev_type == "tool_use":
                tool_name = event.get("tool_name", "tool")
                params = event.get("parameters", {})
                params_text = ""
                try:
                    params_text = json.dumps(params, ensure_ascii=True)
                except Exception:
                    params_text = str(params)
                if len(params_text) > 140:
                    params_text = params_text[:137] + "..."
                status = self._activity(f"tool: {tool_name} {params_text}")
                if status:
                    yield status
            elif ev_type == "tool_result":
                tool_id = event.get("tool_id", "")
                status_text = event.get("status", "unknown")
                status = self._activity(f"tool result: {tool_id} ({status_text})")
                if status:
                    yield status
            elif ev_type == "message" and event.get("role") == "assistant":
                text = event.get("content")
                if isinstance(text, str) and text:
                    saw_text = True
                    yield text
            elif ev_type == "result":
                stats = event.get("stats", {})
                if isinstance(stats, dict):
                    in_t = stats.get("input_tokens", 0)
                    out_t = stats.get("output_tokens", 0)
                    if isinstance(in_t, int) and isinstance(out_t, int):
                        self._last_usage = (in_t, out_t)
                    duration = stats.get("duration_ms")
                    if isinstance(duration, int):
                        status = self._activity(f"done in {duration}ms")
                        if status:
                            yield status

        proc.wait()
        if proc.returncode != 0 and not saw_text:
            message = non_json_lines[-1] if non_json_lines else f"gemini exited with code {proc.returncode}"
            yield f"Error: {message}"

    def ask(self, prompt: str, system: Optional[str] = None) -> str:
        """Get a complete response from Gemini."""
        chunks = []
        for chunk in self.stream(prompt, system):
            if isinstance(chunk, str) and chunk.startswith(self._ACTIVITY_PREFIX):
                continue
            chunks.append(chunk)
        return "".join(chunks)

    def stream(self, prompt: str, system: Optional[str] = None) -> Iterator[str]:
        """Stream tokens from Gemini."""
        self._last_usage = None
        if self._use_cli_proxy:
            yield from self._stream_via_cli(prompt, system)
            return

        try:
            url = f"{self.base_url}/{self.config.model}:streamGenerateContent"
            headers, params = self._auth_params()

            contents = []
            if system:
                contents.append({"role": "user", "parts": [{"text": system}]})
                contents.append({"role": "model", "parts": [{"text": "Understood."}]})

            contents.append({"role": "user", "parts": [{"text": prompt}]})

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens or 2048,
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_UNSPECIFIED",
                        "threshold": "BLOCK_NONE",
                    }
                ],
            }

            with self.client.stream("POST", url, json=payload, params=params, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "candidates" in data:
                                for candidate in data["candidates"]:
                                    if "content" in candidate:
                                        for part in candidate["content"].get("parts", []):
                                            if "text" in part:
                                                yield part["text"]
                            usage = data.get("usageMetadata", {})
                            in_t = usage.get("promptTokenCount", 0)
                            out_t = usage.get("candidatesTokenCount", 0)
                            if in_t or out_t:
                                self._last_usage = (in_t, out_t)
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
        """Gemini-native tool calling using function_declarations."""
        if self._use_cli_proxy:
            # Gemini CLI OAuth mode is proxied as plain text chat.
            return self.ask(prompt, system), []

        from ..tools.executor import ToolExecutor

        executor = ToolExecutor(tools)

        # Build Gemini function declarations
        function_declarations = []
        for td in tools.values():
            decl = {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            }
            function_declarations.append(decl)

        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        tool_log = []
        headers, params = self._auth_params()

        for round_num in range(max_rounds):
            url = f"{self.base_url}/{self.config.model}:generateContent"
            payload = {
                "contents": contents,
                "tools": [{"function_declarations": function_declarations}],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens or 2048,
                },
            }

            try:
                response = self.client.post(url, json=payload, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                return f"Error: {e}", tool_log

            # Parse response parts
            candidates = data.get("candidates", [])
            if not candidates:
                return "", tool_log

            parts = candidates[0].get("content", {}).get("parts", [])

            text_parts = []
            function_calls = []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    function_calls.append(part["functionCall"])

            if not function_calls:
                return "".join(text_parts), tool_log

            # Append the model response
            contents.append({"role": "model", "parts": parts})

            # Execute each function call
            response_parts = []
            for fc in function_calls:
                tool_name = fc["name"]
                tool_args = fc.get("args", {})

                if on_tool_event:
                    on_tool_event(ToolEvent(
                        kind="tool_start",
                        tool_name=tool_name,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        tool_input=tool_args,
                    ))

                result = executor.execute(tool_name, tool_args)
                tool_log.append({
                    "tool": tool_name,
                    "input": tool_args,
                    "output": result,
                })

                if on_tool_event:
                    on_tool_event(ToolEvent(
                        kind="tool_done",
                        tool_name=tool_name,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        tool_input=tool_args,
                        tool_output=result,
                    ))

                response_parts.append({
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"result": result},
                    }
                })

            contents.append({"role": "user", "parts": response_parts})

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
