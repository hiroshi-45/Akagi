# CLAUDE.md — AI Assistant Guide for Akagi

## Project Overview

**Akagi** is a real-time Mahjong AI assistant that analyzes online Mahjong games (primarily Majsoul) using AI models (Mortal/LibRiichi). It uses Playwright to intercept game WebSocket traffic, converts it to the MJAI protocol, feeds events to a local or remote AI bot, and displays AI recommendations in a Textual terminal UI.

**Language:** Python 3.12.5
**UI Framework:** Textual 3.0.0 (terminal UI)
**Browser Automation:** Playwright 1.53.0
**AI/ML:** PyTorch 2.5.1, MJAI protocol

---

## Repository Structure

```
Akagi/
├── run_akagi.py                  # Entry point — sets up logging, calls akagi.main()
├── requirements.txt              # Python dependencies
├── .python-version               # Pins Python 3.12.5
│
├── akagi/                        # Textual UI application
│   ├── akagi.py                  # Main App class, screens, UI logic (~45KB)
│   ├── logging_utils.py          # setup_logger() — creates module-scoped loguru loggers
│   ├── libriichi_helper.py       # LibRiichi helper utilities
│   ├── hooks.py                  # UI hook definitions
│   ├── misc.py                   # Miscellaneous helpers
│   ├── logger.py                 # Module-level logger instance
│   ├── client.tcss               # Textual CSS styles for the UI
│   ├── HELP.md                   # In-app help (English)
│   └── HELP_ZH.md                # In-app help (Chinese)
│
├── playwright_client/            # Browser automation & game protocol layer
│   ├── client.py                 # PlaywrightClient — threading, message queues
│   ├── majsoul.py                # Majsoul-specific Playwright controller (~75KB)
│   ├── x_post.py                 # X (Twitter) OAuth2/PKCE posting
│   ├── slack_listener.py         # Slack Socket Mode integration
│   ├── autoplay/
│   │   ├── autoplay.py           # Autoplay base logic
│   │   ├── autoplay_majsoul.py   # Majsoul-specific autoplay (PyAutoGUI clicks)
│   │   └── util.py               # Autoplay utilities
│   └── bridge/
│       ├── bridge_base.py        # Abstract BridgeBase (parse/build interface)
│       └── majsoul/
│           ├── bridge.py         # Majsoul bridge — converts liqi↔MJAI
│           ├── liqi.py           # liqi protocol handling
│           └── liqi_proto/
│               └── liqi_pb2.py   # Generated protobuf bindings
│
├── mjai_bot/                     # MJAI bot implementations (plugin architecture)
│   ├── controller.py             # Bot Controller — discovery, routing, auto-switching
│   ├── base/
│   │   └── bot.py                # Abstract Bot base class (react() interface)
│   ├── mortal/                   # 4-player Mortal AI bot
│   │   └── bot.py
│   ├── mortal3p/                 # 3-player Mortal AI bot
│   │   └── bot.py
│   ├── mortal_common/            # Shared code for mortal bots
│   ├── akochan_local/            # Alternative bot implementation
│   │   └── bot.py
│   └── strategy/                 # Strategy overlays
│       ├── safety.py             # Safety tile strategy
│       └── last_avoid.py         # Last-avoid strategy
│
├── settings/                     # Configuration management
│   ├── settings.py               # Settings dataclasses, load/save/validate logic
│   ├── settings.json             # Runtime config (gitignored if secrets present)
│   └── settings.schema.json      # JSON Schema for settings validation
│
├── docs/                         # Documentation assets (images, GIFs)
├── logs/                         # Runtime logs (auto-created, gitignored)
├── README.md                     # Main documentation (English)
└── README_ZH.md                  # Main documentation (Chinese)
```

---

## Architecture & Data Flow

```
Majsoul Game (Browser)
        |
        | WebSocket (liqi/protobuf protocol)
        v
playwright_client/majsoul.py     ← Playwright intercepts WebSocket frames
        |
        | raw bytes
        v
playwright_client/bridge/majsoul/bridge.py   ← Converts liqi → MJAI JSON events
        |
        | list[dict] (MJAI events)
        v
mjai_bot/controller.py           ← Routes to correct bot, handles 3P/4P auto-switching
        |
        | JSON string of events
        v
mjai_bot/<bot_name>/bot.py       ← AI model inference (PyTorch / online OT server)
        |
        | JSON action string
        v
akagi/akagi.py (Textual UI)      ← Displays recommendations to user
        |
        | (if autoplay enabled)
        v
playwright_client/autoplay/autoplay_majsoul.py  ← PyAutoGUI clicks
```

---

## Key Abstractions

### Bot Plugin System (`mjai_bot/`)

Bots are auto-discovered at startup. Any subdirectory of `mjai_bot/` containing a `bot.py` with a `Bot` class (subclassing `mjai_bot.base.bot.Bot`) is loaded automatically.

**To add a new bot:**
1. Create `mjai_bot/<bot_name>/bot.py`
2. Define a `Bot` class inheriting from `mjai_bot.base.bot.Bot`
3. Implement the `react(self, events: str) -> str` method

**Bot interface (`mjai_bot/base/bot.py`):**
```python
class Bot:
    def react(self, events: str) -> str:
        # events: JSON string of MJAI event list
        # returns: JSON string of a single MJAI action
        ...
```

The first event in any game must be `start_game` to initialize the bot's player ID.

### Bridge Plugin System (`playwright_client/bridge/`)

Bridges convert platform-specific protocols to/from MJAI. `BridgeBase` defines the interface:
```python
class BridgeBase:
    def parse(self, content: bytes) -> None | list[dict]: ...  # platform → MJAI
    def build(self, command: dict) -> None | bytes: ...         # MJAI → platform
```

Currently only `majsoul/` bridge is implemented.

### Settings (`settings/settings.py`)

Configuration is loaded at import time as a module-level singleton:
```python
# settings/settings.py (bottom of file)
settings: Settings = load_settings()
```

Import and use throughout the codebase:
```python
from settings.settings import settings
```

Settings are validated against `settings/settings.schema.json` (JSON Schema) on every load. If `settings.json` is corrupted, it is backed up as `settings.json.bak` and regenerated with defaults.

**Settings structure:**
```json
{
  "playwright": { "majsoul_url": "...", "viewport": { "width": 1280, "height": 720 } },
  "model": "mortal",
  "theme": "textual-dark",
  "ot_server": { "server": "http://...", "online": false, "api_key": "..." },
  "autoplay": false,
  "auto_switch_model": true
}
```

### Logging (`akagi/logging_utils.py`)

Each module creates its own logger with a timestamped log file:
```python
from akagi.logging_utils import setup_logger
logger = setup_logger("my_module")
# → writes to ./logs/my_module_YYYYMMDD_HHMMSS.log
```

All modules follow this pattern — never use `print()` for diagnostic output.

---

## Development Setup

```bash
# Clone repository
git clone <repo_url>
cd Akagi

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
python -m playwright install

# Place AI model files in mjai_bot/mortal/ and mjai_bot/mortal3p/
# (model files are gitignored — obtain separately)

# Run the application
python run_akagi.py
```

**Python version:** Must use Python 3.12.5 (see `.python-version`).

---

## Running the Application

```bash
python run_akagi.py
```

This:
1. Creates the `logs/` directory if missing
2. Sets up the main logger
3. Launches the Textual UI (`akagi.akagi.main()`)

The Textual UI then starts the Playwright browser, navigates to Majsoul, and begins intercepting game traffic.

---

## Configuration

Edit `settings/settings.json` or use the in-app Settings screen.

| Key | Description |
|-----|-------------|
| `playwright.majsoul_url` | Majsoul game URL |
| `playwright.viewport` | Browser window size |
| `model` | Active bot name (must match a subdirectory in `mjai_bot/`) |
| `theme` | Textual theme (`textual-dark`, `textual-light`, etc.) |
| `ot_server.online` | Use online OT inference server instead of local model |
| `ot_server.server` | OT server URL |
| `ot_server.api_key` | OT server API key |
| `autoplay` | Enable automatic tile clicking via PyAutoGUI |
| `auto_switch_model` | Auto-switch between mortal (4P) and mortal3p (3P) |

---

## Bot Auto-Switching Logic

When `auto_switch_model` is enabled, the Controller detects 3P vs 4P games by inspecting the `start_kyoku` event's `scores` array:
- If `scores[3] == 0` (4th player has 0 points) → 3-player game → switch to `mortal3p`
- Otherwise → 4-player game → switch to `mortal`

This logic lives in `mjai_bot/controller.py:react()`.

---

## MJAI Protocol Reference

MJAI is the standard JSON protocol for Mahjong AI communication. Key event types:

| Event | Direction | Description |
|-------|-----------|-------------|
| `start_game` | → Bot | Initialize game, set player ID |
| `start_kyoku` | → Bot | Start of a hand (round) |
| `tsumo` | → Bot | Player draws a tile |
| `dahai` | → Bot | Player discards a tile |
| `chi` / `pon` / `kan` | → Bot | Meld actions |
| `end_game` | → Bot | Game over |
| `dahai` | Bot → | Bot wants to discard |
| `reach` | Bot → | Bot declares riichi |
| `none` | Bot → | No action |

See [mjai.app](https://github.com/smly/mjai.app) for full protocol specification.

---

## Code Conventions

- **Type hints:** Use throughout — `bot: Bot | None`, `events: list[dict]`, etc.
- **Dataclasses:** Use for configuration objects (see `settings/settings.py`)
- **Logging:** Use `loguru` via `setup_logger()` — never use `print()` or stdlib `logging`
- **Error returns:** Bots return `{"type": "none"}` JSON when no action is appropriate
- **JSON serialization:** Use `separators=(",", ":")` for compact bot communication
- **Module singletons:** Settings and loggers are module-level singletons, imported directly
- **No linting config:** No `.flake8`/`pyproject.toml` — follow the existing style in each file

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `run_akagi.py` | Entry point |
| `akagi/akagi.py` | Main Textual App, all UI screens |
| `akagi/logging_utils.py` | Logger factory |
| `playwright_client/majsoul.py` | Majsoul Playwright controller (largest file) |
| `playwright_client/client.py` | Threading wrapper for Playwright |
| `playwright_client/bridge/majsoul/bridge.py` | liqi ↔ MJAI protocol conversion |
| `mjai_bot/controller.py` | Bot discovery, routing, auto-switching |
| `mjai_bot/base/bot.py` | Abstract bot interface |
| `settings/settings.py` | Settings load/save/validate, singleton |
| `settings/settings.json` | Runtime configuration |
| `settings/settings.schema.json` | JSON Schema for config validation |

---

## Notifications & Integrations

The application supports optional integrations configured via the settings UI:

- **X (Twitter):** OAuth2/PKCE flow in `playwright_client/x_post.py` — posts game results
- **Slack:** Socket Mode in `playwright_client/slack_listener.py` — listens for commands
- **Email:** SMTP-based notifications for game events

These run in background daemon threads and do not block the main UI.

---

## Adding a New Bot

1. Create directory: `mjai_bot/<your_bot_name>/`
2. Create `mjai_bot/<your_bot_name>/bot.py`:

```python
from mjai_bot.base.bot import Bot as BotBase
import json

class Bot(BotBase):
    def __init__(self):
        super().__init__()
        self.model = None

    def react(self, events: str) -> str:
        events_list = json.loads(events)
        for event in events_list:
            if event["type"] == "start_game":
                self.player_id = event["id"]
                self.model = load_model()  # your initialization
                return json.dumps({"type": "none"}, separators=(",", ":"))
            # ... handle other events
        return json.dumps({"type": "none"}, separators=(",", ":"))
```

3. The Controller auto-discovers it on next startup — no registration needed.

---

## Adding a New Game Platform

1. Create `playwright_client/bridge/<platform>/`
2. Implement `bridge.py` with a class inheriting `BridgeBase`:
   ```python
   from playwright_client.bridge.bridge_base import BridgeBase

   class Bridge(BridgeBase):
       def parse(self, content: bytes) -> None | list[dict]:
           # Convert platform protocol → MJAI event list
           ...
       def build(self, command: dict) -> None | bytes:
           # Convert MJAI action → platform protocol bytes
           ...
   ```
3. Wire it up in `playwright_client/majsoul.py` or create a new platform-specific client.

---

## Gitignored Paths

The following are excluded from version control:
- `venv/` — virtual environment
- `logs/` — runtime log files
- `mjai_bot/*/model*` — AI model weights (large binary files)
- `playwright_client/bridge/majsoul/liqi_proto/*.so` — compiled C extensions
- `.playwright/` — Playwright browser data
- `settings/settings.json.bak` — settings backups
