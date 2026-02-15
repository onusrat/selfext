# SelfExt

A self-extending AI agent that creates its own tools at runtime. Single file, zero dependencies.

SelfExt is an AI agent that can dynamically generate, persist, and manage custom tools backed by scripts. When the agent encounters a task that would benefit from a specialized tool, it creates one — and that tool persists across sessions.

## Features

- **Self-extending** — the AI creates new tools at runtime via `create_tool`
- **Multi-language tools** — custom tools can be bash, sh, python3, or node scripts
- **Tool persistence** — custom tools saved to `~/.selfext/tools/` and survive restarts
- **Single file, zero dependencies** — stdlib only, compiles everywhere
- **CRUD tool management** — create, list, update, and remove custom tools
- **Safety filters** — blocks destructive commands before execution

## Install

```bash
go install github.com/onusrat/selfext@latest
```

Or build from source:

```bash
git clone https://github.com/onusrat/selfext.git
cd selfext
make build
```

## Quick Start

```bash
export SELFEXT_API_KEY=sk-...
./selfext                                  # interactive REPL
./selfext -m "create a tool that checks disk usage"  # one-shot
```

## Usage

### REPL Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/tools` | List all tools (built-in, meta, and custom) |
| `/clear` | Clear conversation history |
| `/quit` | Exit |

### Keys

- `Ctrl+C` — cancel current operation
- `Ctrl+D` — exit REPL

## Tools

### Built-in Tools

| Tool | Description |
|------|-------------|
| `exec` | Execute shell commands |
| `read_file` | Read file contents |
| `write_file` | Write content to files |
| `list_dir` | List directory contents |
| `edit_file` | Replace exact string matches in files |

### Meta-Tools (Self-Extension)

| Tool | Description |
|------|-------------|
| `create_tool` | Create a new custom tool backed by a script |
| `list_custom_tools` | List all registered custom tools |
| `update_tool` | Update an existing custom tool |
| `remove_tool` | Remove a custom tool by name |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SELFEXT_API_KEY` | API key (required) | — |
| `SELFEXT_API_BASE` | API endpoint | `https://api.openai.com/v1` |
| `SELFEXT_MODEL` | LLM model | `gpt-4o` |

### Config File

`~/.selfext/config.json` — supports all options. Priority: env vars > config file > defaults.

```json
{
  "api_key": "sk-...",
  "model": "gpt-4o",
  "max_iterations": 20,
  "session_window": 50
}
```

## Building

```bash
make build          # current platform
make build-all      # linux/amd64, linux/arm64, linux/riscv64, darwin/arm64
make install        # copy to ~/.local/bin
make vet            # run go vet
make fmt            # run go fmt
make clean          # remove artifacts
```

## Creating Custom Tools

The agent creates tools autonomously, but here's what happens under the hood:

1. The AI calls `create_tool` with a name, description, parameter schema, script, and interpreter
2. The script is saved to `~/.selfext/tools/<name>.sh` (or `.py`, `.js`)
3. A JSON definition file is saved alongside it
4. The tool is immediately registered and available

**Parameter passing:** Custom tools receive parameters as environment variables (`TOOL_PARAM_<NAME>`) and JSON on stdin.

**Example flow:**
```
You: "I need a tool that checks if a website is up"
Agent: I'll create a custom tool for that.
       [calls create_tool with a curl-based bash script]
       Created tool 'check_website'. Let me test it.
       [calls check_website with url="https://example.com"]
       example.com is up (HTTP 200).
```

Tools persist in `~/.selfext/tools/` and are reloaded on startup.

## License

MIT
