# Loophole

**Adversarial moral-legal code system** — an AI tool that stress-tests your ethical principles by trying to break them.

## The Idea

Real legal systems evolve slowly. A law gets written, someone finds a loophole, a court patches it, someone finds another loophole. This process takes decades. Loophole compresses it into minutes.

You state your moral principles in plain language. An AI legislator drafts a formal legal code from them. Then two adversarial agents attack it:

- **The Loophole Finder** searches for scenarios that are *technically legal* under your code but *morally wrong* according to your principles. Think creative rule-lawyering, exploiting vague definitions, finding gaps the drafters didn't anticipate.

- **The Overreach Finder** searches for the opposite: scenarios your code *prohibits* that you'd actually consider *morally acceptable*. Good Samaritan situations, overbroad rules that catch innocent behavior, emergencies where rigid compliance causes worse outcomes.

When an attack lands, a **Judge agent** tries to patch the code automatically — but only if the fix doesn't break any previous ruling. Every resolved case becomes a permanent constraint, a growing test suite the code must satisfy.

If the Judge can't find a consistent fix — meaning any patch would contradict a prior decision — the case gets **escalated to you**. These escalated cases are guaranteed to be interesting: they represent genuine tensions in your own moral framework, places where your principles actually conflict with each other.

The legal code gets progressively more robust. But the real output isn't the code — it's what you discover about your own beliefs.

## How It Works

```
                    +-----------------+
                    |  Your Moral     |
                    |  Principles     |
                    +--------+--------+
                             |
                             v
                    +--------+--------+
                    |   Legislator    |
                    | (drafts legal   |
                    |  code from      |
                    |  principles)    |
                    +--------+--------+
                             |
                             v
              +--------------+--------------+
              |                             |
    +---------v----------+      +-----------v--------+
    |  Loophole Finder   |      |  Overreach Finder  |
    |  (legal but        |      |  (illegal but      |
    |   immoral)         |      |   moral)           |
    +--------+-----------+      +-----------+--------+
              |                             |
              +-------------+---------------+
                            |
                            v
                   +--------+--------+
                   |     Judge       |
                   | (auto-resolve   |
                   |  or escalate)   |
                   +--------+--------+
                            |
                +-----------+-----------+
                |                       |
        +-------v-------+      +-------v--------+
        | Auto-resolved |      |  Escalated     |
        | (code updated,|      |  to YOU        |
        |  case becomes |      |  (genuine      |
        |  precedent)   |      |   moral        |
        +---------------+      |   dilemma)     |
                               +----------------+
```

Each resolved case — whether by the Judge or by you — becomes binding precedent. The adversarial agents attack again, and the cycle repeats. Round after round, the legal code tightens, and the cases that reach you get harder and more revealing.

## Setup

Requires Python 3.12+ and an Anthropic API key.

```bash
# Clone and install
git clone <repo-url>
cd law
uv sync

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Start a new session

Interactive mode:
```bash
uv run python -m loophole.main
```

Or directly with a domain and principles file:
```bash
uv run python -m loophole.main new --domain privacy -p examples/privacy_principles.txt
```

You'll see the initial legal code, then the adversarial loop begins. Each round:
1. Both adversarial agents attack the current code
2. The Judge processes each case (auto-resolve or escalate)
3. You see a summary and choose to continue, view the code, or stop

When a case is escalated, you'll be prompted to make a decision. Your decision becomes a new constraint that the legal code must respect going forward.

### Resume a session

Sessions auto-save after every case. Pick up where you left off:
```bash
uv run python -m loophole.main resume
```

### Generate a visualization

After a session (or for any past session), generate an HTML report:
```bash
uv run python -m loophole.main visualize
```

This creates a `report.html` in the session directory with:
- Your moral principles and the initial legal code
- A timeline of every adversarial case
- Git-style diffs showing how the code changed after each case
- The final legal code

### List sessions

```bash
uv run python -m loophole.main list
```

### Web UI (optional)

A web interface is available for watching rounds unfold interactively. It wraps the same adversarial loop as the CLI, streams updates over WebSocket, and offers three visual themes (court, terminal, lab).

```bash
# Install web dependencies
uv sync --extra web

# Start the FastAPI backend
uv run uvicorn web_api:app --port 8001

# In another terminal, start the React frontend
cd frontend
npm install
npm start
```

Then open <http://localhost:3000>. The "Try the demo" button loads a pre-baked session with sample cases — no API key required.

## Configuration

Edit `config.yaml` to tune the system:

```yaml
model:
  default: "claude-sonnet-4-20250514"   # Which Claude model to use
  max_tokens: 4096

temperatures:
  legislator: 0.4          # Lower = more precise drafting
  loophole_finder: 0.9     # Higher = more creative attacks
  overreach_finder: 0.9
  judge: 0.3               # Lower = more conservative judgments

loop:
  max_rounds: 10
  cases_per_agent: 3       # How many cases each attacker finds per round

session_dir: "sessions"
```

## Writing Good Principles

The system works best when your principles are:

- **Specific enough to draft from.** "I believe in fairness" is too vague. "Companies should not sell user data without explicit, informed consent" gives the legislator something to work with.
- **Broad enough to have tensions.** If your principles only cover one narrow situation, the adversarial agents won't find interesting cases. Cover the domain from multiple angles.
- **Honest.** The system surfaces conflicts in *your* beliefs. If you state principles you don't actually hold, the escalated cases won't be meaningful.

See `examples/privacy_principles.txt` for a starting point.

## Project Structure

```
loophole/
  main.py              CLI and main adversarial loop
  models.py            Data models (SessionState, Case, LegalCode)
  llm.py               Anthropic SDK wrapper
  prompts.py           All agent prompt templates
  session.py           Session persistence (JSON + markdown)
  visualize.py         HTML report generator
  agents/
    base.py            Base agent class
    legislator.py      Drafts and revises the legal code
    loophole_finder.py Finds legal-but-immoral scenarios
    overreach_finder.py Finds illegal-but-moral scenarios
    judge.py           Auto-resolves cases or escalates

sessions/              One directory per session (auto-created)
examples/              Example moral principles files
config.yaml            Model and loop configuration
```

## Why This Matters

Most attempts to formalize ethics start with the rules and hope they cover everything. Loophole starts with your intuitions and systematically finds where they break down. It's less "solve ethics" and more "discover what you actually believe by watching it fail."

The same architecture applies anywhere humans write rules for AI systems: content moderation policies, LLM system prompts, codes of conduct, safety specifications. Anywhere there's a gap between what the rules say and what the rules mean, Loophole will find it.
