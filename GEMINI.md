# GEMINI.md - Cascade CLI Project Context

## Project Overview
**Cascade** (or `cascade-cli`) is a sophisticated, multi-model AI assistant CLI designed with a focus on aesthetics ("Deep Stream" theme) and developer productivity. It provides a unified interface for interacting with various Large Language Models (LLMs) including Google Gemini, Anthropic Claude, OpenAI, and others via OpenRouter.

### Key Features
- **Multi-Model Support:** Seamless switching between Gemini, Claude, OpenAI, and OpenRouter.
- **Interactive REPL:** A rich interactive chat experience with history management and session summaries.
- **Tool Calling & Plugins:** Extensible plugin system (e.g., `FileOpsPlugin`) allowing models to interact with the local filesystem and other tools.
- **Prompt Layering:** A sophisticated prompt pipeline that merges default system prompts, project-specific context, and user overrides based on priority.
- **Deep Stream Aesthetics:** Beautiful terminal UI powered by `rich`, featuring a signature Cyan (#00d4e5) and Violet (#b44dff) color palette.
- **Analysis Mode:** Direct file analysis capabilities (`cascade analyze <file>`).
- **Comparison Mode:** Compare responses from multiple providers side-by-side.

## Technical Stack
- **Language:** Python 3.9+
- **CLI Framework:** `click`
- **UI & Rendering:** `rich`, `prompt_toolkit`
- **HTTP Client:** `httpx`
- **Configuration:** YAML (`pyyaml`)
- **Testing:** `pytest`
- **Linting:** `ruff`

## Directory Structure
- `cascade/`: Main package directory.
    - `cli.py`: Click CLI entry point and command definitions.
    - `repl.py`: Interactive REPL implementation.
    - `providers/`: LLM provider implementations (Gemini, Claude, etc.) inheriting from `BaseProvider`.
    - `prompts/`: Logic for the layered prompt pipeline.
    - `ui/`: Theme definitions and Rich-based rendering components.
    - `plugins/`: Tool and plugin implementations (e.g., `file_ops.py`).
    - `hooks/`: Event-based hooks (e.g., `BEFORE_ASK`, `AFTER_RESPONSE`).
    - `context/`: Project-level context detection and management.
- `tests/`: Comprehensive test suite.
- `examples/`: Example configurations and code.

## Building and Running

### Installation
```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Running the Application
- **Interactive REPL:** `cascade`
- **Single Question:** `cascade-cli ask "your question"`
- **File Analysis:** `cascade-cli analyze path/to/file.py`
- **Compare Models:** `cascade-cli compare "explain recursion" --providers gemini claude`
- **Setup Wizard:** `cascade-cli setup`

### Development Commands
- **Run Tests:** `pytest`
- **Linting:** `ruff check .`
- **Formatting:** `ruff format .`

## Development Conventions

### Provider Implementation
New providers must inherit from `cascade.providers.base.BaseProvider` and be registered using the `@register_provider("<name>")` decorator. They should implement `ask`, `stream`, and optionally `ask_with_tools`.

### Prompt Layering
The `PromptPipeline` manages system prompts using priorities:
- `PRIORITY_DEFAULT` (100)
- `PRIORITY_PROJECT_SYSTEM` (200)
- `PRIORITY_PROJECT_CONTEXT` (300)
- `PRIORITY_USER_OVERRIDE` (400)
- `PRIORITY_REPL_CONTEXT` (500)

### Configuration
Config is stored at `~/.config/cascade/config.yaml`. Environment variables in the config file are supported using the `${VAR_NAME}` syntax and are resolved at runtime.

### UI & Styling
Always use the established `CYAN` (#00d4e5) and `VIOLET` (#b44dff) accents for UI elements. Prefer `DEFAULT_THEME.palette` for semantic color lookups.
