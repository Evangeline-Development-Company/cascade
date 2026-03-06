"""Base provider interface for all AI models."""

from abc import ABC, abstractmethod
from typing import Optional, Iterator, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..tools.schema import ToolDef


@dataclass(frozen=True)
class ToolEvent:
    """Progress event emitted during tool-calling rounds."""

    kind: str  # "tool_start" | "tool_done"
    tool_name: str
    round_num: int
    max_rounds: int
    tool_input: dict = field(default_factory=dict)
    tool_output: str = ""


ToolEventCallback = Optional[Callable[[ToolEvent], None]]


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class BaseProvider(ABC):
    """Abstract base class for all AI providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = self.__class__.__name__
        self._last_usage: Optional[tuple[int, int]] = None

    @property
    def last_usage(self) -> Optional[tuple[int, int]]:
        """Token usage from last ask/stream call: (input_tokens, output_tokens)."""
        return self._last_usage

    @abstractmethod
    def ask(self, prompt: str, system: Optional[str] = None) -> str:
        """Send a single prompt and get a complete response."""
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> Iterator[str]:
        """Stream tokens from the provider."""
        pass

    @abstractmethod
    def compare(self, prompt: str, system: Optional[str] = None) -> dict:
        """Generate and return structured comparison data."""
        pass

    def ask_with_tools(
        self,
        prompt: str,
        tools: dict[str, "ToolDef"],
        system: Optional[str] = None,
        max_rounds: int = 5,
        on_tool_event: ToolEventCallback = None,
    ) -> tuple[str, list[dict]]:
        """Ask with tool calling support.

        Subclasses implement provider-native tool calling. The default
        falls back to a plain ask() with no tool support.

        Args:
            prompt: User message.
            tools: Mapping of tool_name -> ToolDef.
            system: Optional system prompt.
            max_rounds: Maximum tool-calling round trips.
            on_tool_event: Optional callback for tool progress events.

        Returns:
            Tuple of (final_text_response, tool_calls_log).
        """
        return self.ask(prompt, system), []

    def validate(self) -> bool:
        """Validate provider configuration and connectivity."""
        return bool(self.config.api_key and self.config.model)

    def ping(self) -> bool:
        """Test connectivity with a minimal API call. Returns True on success."""
        try:
            result = self.ask("Reply with the single word OK.")
            return bool(result and len(result.strip()) > 0)
        except Exception:
            return False
