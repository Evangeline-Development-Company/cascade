"""Execute tool calls and return results.

Provides a safe execution wrapper that catches exceptions and returns
structured results for the tool calling loop. Supports hook lifecycle
events for tool_call (pre-execution) and tool_result (post-execution).
"""

import json
from typing import Any, Optional

from .schema import ToolDef
from ..hooks import HookEvent, HookContext, HookRunner


class ToolExecutor:
    """Execute registered tools by name with argument dicts.

    Supports Pi-style tool lifecycle hooks:
    - tool_call: fired before execution, can block or modify arguments
    - tool_result: fired after execution, can modify the result
    """

    def __init__(
        self,
        tools: dict[str, ToolDef],
        hook_runner: Optional[HookRunner] = None,
    ):
        self._tools = dict(tools)
        self._hook_runner = hook_runner

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the result as a JSON string.

        Fires tool_call hook before execution (can block/transform args).
        Fires tool_result hook after execution (can transform result).

        Args:
            tool_name: Name of the tool to call.
            arguments: Keyword arguments for the tool handler.

        Returns:
            JSON-encoded result string. On error, returns a JSON error object.
        """
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        # Fire tool_call hook (can block or transform arguments)
        if self._hook_runner:
            ctx = HookContext(
                event=HookEvent.TOOL_CALL.value,
                tool_name=tool_name,
                tool_input=tuple(arguments.items()),
            )
            hook_result = self._hook_runner.emit(HookEvent.TOOL_CALL, ctx)
            if hook_result is not None:
                if hook_result.block:
                    return json.dumps({
                        "error": f"Tool '{tool_name}' blocked by hook: {hook_result.reason}",
                    })
                if hook_result.transformed_value is not None:
                    if not isinstance(hook_result.transformed_value, dict):
                        return json.dumps({
                            "error": (
                                f"Hook returned invalid arguments type for {tool_name}: "
                                f"{type(hook_result.transformed_value).__name__} (expected dict)"
                            ),
                        })
                    arguments = hook_result.transformed_value

        tool = self._tools[tool_name]
        try:
            result = tool.handler(**arguments)
            result_str = json.dumps({"result": result})

            # Fire tool_result hook (can transform result)
            if self._hook_runner:
                ctx = HookContext(
                    event=HookEvent.TOOL_RESULT.value,
                    tool_name=tool_name,
                    tool_input=tuple(arguments.items()),
                    tool_output=result_str,
                )
                hook_result = self._hook_runner.emit(HookEvent.TOOL_RESULT, ctx)
                if hook_result is not None and hook_result.transformed_value is not None:
                    result_str = json.dumps({"result": hook_result.transformed_value})

            return result_str

        except TypeError as e:
            return json.dumps({"error": f"Invalid arguments for {tool_name}: {e}"})
        except Exception as e:
            return json.dumps({"error": f"Tool {tool_name} failed: {e}"})
