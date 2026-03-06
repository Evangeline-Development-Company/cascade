# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Cascade

Cascade is a multi-model AI assistant CLI and TUI. It supports Gemini, Claude, OpenAI, and OpenRouter providers in a single tool with a fullscreen Textual terminal interface and a Click-based CLI.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run the TUI (main entry point)
cascade

# Run CLI commands directly
cascade-cli ask "question"
cascade-cli compare "question" --providers gemini claude
cascade-cli analyze file.py
cascade-cli config
cascade-cli setup
cascade-cli init

# Tests
python -m pytest tests/ -v              # all tests
python -m pytest tests/test_config.py   # single module
python -m pytest tests/test_config.py::test_default_config -v  # single test

# Lint and format
ruff check cascade tests
ruff format cascade tests
```

## Architecture

### Two UI Systems

There are two distinct UI systems that share the same backend:

1. **Textual TUI** (`cascade/app.py` + `cascade/screens/` + `cascade/widgets/`): The fullscreen terminal app. Entry: `cascade` -> `cascade.repl:main()` -> creates `CascadeApp` (CLIApp) -> wraps in `CascadeTUI` -> `.run()`.
2. **Rich/Click CLI** (`cascade/cli.py`): Traditional CLI commands via Click. Entry: `cascade-cli` -> `cascade.cli:cli`.

`CascadeApp` in `cascade/cli.py` is the shared backend (providers, config, hooks, tools, prompt pipeline, agents). `CascadeTUI` in `cascade/app.py` is the Textual wrapper that holds `CascadeState` and the history DB.

### Provider System

All providers extend `BaseProvider` (in `cascade/providers/base.py`) with three required methods: `ask()`, `stream()`, `compare()`. The `ask_with_tools()` method adds tool-calling support with optional `on_tool_event` callback for progress reporting.

Providers self-register via `@register_provider("name")` decorator in `cascade/providers/registry.py`. At startup, `discover_providers()` imports all modules in the providers package, triggering the decorators.

Current providers: `gemini`, `claude`, `openai_provider` (registered as "openai"), `openrouter`.

`BaseProvider.stream()` returns `Iterator[str]` (synchronous). The TUI bridges this to async via `run_worker(thread=True)` + `call_from_thread()` in `MainScreen._provider_worker()`.

### OAuth / CLI Proxy Pattern

Providers detect OAuth tokens and proxy through their respective CLI tools when direct API access isn't possible:

| Provider | OAuth Token Pattern | CLI Binary | Env Override |
|----------|---|---|---|
| Claude | `sk-ant-oat01*` | `claude` | `CASCADE_CLAUDE_ACTIVITY` |
| Gemini | `ya29.*` | `gemini` | `CASCADE_GEMINI_ACTIVITY` |
| OpenAI | JWT (`eyJ...`) | `codex` | `CASCADE_OPENAI_ACTIVITY` |

Each provider sets `_use_oauth_cli` (token looks like OAuth) and `_use_cli_proxy` (OAuth + CLI binary found). When `_use_cli_proxy` is True, `stream()` delegates to `_stream_via_cli()` which spawns the CLI subprocess and parses its JSON stream output. Activity messages use the `[[cascade_activity]] ` prefix so the TUI can display status (model name, tool calls, duration) without polluting the response text.

**Key constraint:** When using CLI proxy mode, `ask_with_tools()` falls back to plain `ask()` -- the CLI handles its own tool calling internally. The TUI's `_should_use_tools()` checks both flags to decide whether to use the streaming path or the tool-calling path.

### Tool System

Tools flow: `BasePlugin.get_tools()` -> `build_tool_registry()` -> `CascadeApp.tool_registry` -> `provider.ask_with_tools()`.

- `cascade/tools/schema.py`: `callable_to_tool_def()` introspects Python function signatures + type hints to build JSON Schema `ToolDef` objects. Extracts param descriptions from Google-style docstrings.
- `cascade/tools/executor.py`: `ToolExecutor` wraps tool execution with error handling, returns JSON-encoded results.
- `cascade/plugins/`: Plugins register via `@register_plugin("name")` decorator. `FileOpsPlugin` provides `read_file`, `write_file`, `list_files`, `append_file`.

In the TUI, `MainScreen._provider_worker()` branches: if `tool_registry` is non-empty and the provider is not using CLI proxy, it calls `_tool_worker()` (non-streaming `ask_with_tools()` with `ToolEvent` callbacks that update the `ThinkingIndicator` and mount `ToolCallWidget` instances). Otherwise it uses the streaming path.

`ToolEvent` (frozen dataclass in `base.py`) carries `kind` ("tool_start"/"tool_done"), `tool_name`, `round_num`, `max_rounds`, `tool_input`, and `tool_output`. The `ToolEventCallback` type alias is `Optional[Callable[[ToolEvent], None]]`.

### Auth and Credential Detection

`cascade/auth.py`: `detect_all()` discovers credentials from CLI config files (`~/.claude/.credentials.json`, `~/.gemini/oauth_creds.json`, `~/.codex/auth.json`) and from `TokenStore` (`~/.config/cascade/tokens/`). Returns `DetectedCredential` dataclasses. `CascadeApp.__init__()` calls `_apply_detected_credentials()` to auto-enable providers.

`cascade/auth_flow.py`: Interactive `login()` dispatcher routes to provider-specific flows. Gemini supports full device-code OAuth2; Claude and OpenAI guide users to their CLI's `login` command; OpenRouter is API-key only.

### Mode System

Four modes map 1:1 to providers with distinct accent colors:
- `design` -> gemini (#b44dff)
- `plan` -> claude (#f0956c)
- `build` -> openai (#34d399)
- `test` -> openrouter (#d94060)

Shift+Tab cycles modes in the TUI. All color constants live in `cascade/theme.py` (TUI) and `cascade/ui/theme.py` (legacy Rich REPL).

### Prompt Pipeline

`PromptPipeline` in `cascade/prompts/layers.py` is an **immutable** pipeline. Each layer has name + content + priority (lower = earlier). Calling `add_layer()` returns a new instance. Layers merge at `build()` time, sorted by priority:

- 10: default system prompt
- 20: design language
- 30: project system prompt (from `.cascade/system_prompt.md`)
- 40: project context files (from `.cascade/context/`)
- 50: user override
- 60: REPL context (uploads, conversation history)

### State and Reactivity (TUI)

`CascadeState` in `cascade/state.py` holds all mutable state. Mutator methods (e.g. `set_provider()`, `add_message()`, `update_tokens()`) post Textual `Message` subclasses to the app's message bus so widgets can react.

### Cross-Model Memory

When switching providers mid-session, conversation context is carried via one of three policies (configured in `config.yaml` under `memory.cross_model_memory`):
- `off`: no context carried
- `summary`: compact handoff summary + provider-local recent turns
- `full`: recent full transcript across all providers

Summary compaction is triggered on provider switches and periodically (every N turns).

### History

`cascade/history/database.py`: SQLite with WAL mode. Two tables: `sessions` and `messages`. DB at `~/.config/cascade/history.db`. Sessions auto-created on first user message; auto-titled from first message content.

### Configuration

YAML at `~/.config/cascade/config.yaml`. Supports `${ENV_VAR}` substitution for API keys. Sections: `providers`, `defaults`, `prompts`, `memory`, `hooks`, `tools`, `workflows`, `integrations`.

### Project Context

`.cascade/` directory (searched up from cwd) provides per-project config:
- `system_prompt.md` - injected into prompt pipeline
- `agents.yaml` - agent definitions (provider/model/temperature overrides)
- `context/` - files auto-loaded into conversation context

### Hooks

Shell commands triggered at lifecycle events (`before_ask`, `after_response`, `on_exit`, `on_error`). Configured in `config.yaml`. Context passed via `CASCADE_*` environment variables.

### Slash Commands (TUI)

Handled by `CommandHandler` in `cascade/commands.py`. Key commands: `/model`, `/mode`, `/fast`, `/agent`, `/verify`, `/review`, `/checkpoint`, `/upload`, `/login`, `/config reload`.

### Known Issues

- `cascade/cli.py`: `chat` Click command has broken indentation (nested inside `compare` function)
- `test_web.py` tests fail due to missing `python-multipart` unless `[web]` extras installed

## Code Conventions

- Python 3.9+ (no walrus operator in hot paths, use `from __future__` sparingly)
- Ruff for linting and formatting (line-length 100)
- Frozen dataclasses for immutable value types (`ProviderConfig`, `ToolEvent`, `AgentDef`, `HookDefinition`, `PromptLayer`, `ProviderTheme`, `Palette`)
- Textual widgets never hardcode colors; all hex values come from `cascade/theme.py`
- OpenAI and OpenRouter share tool-calling logic via `cascade/providers/_openai_tools.py`
