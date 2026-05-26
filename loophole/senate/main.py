from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from loophole.llm import _infer_provider, create_provider
from loophole.senate.agents.biographer import Biographer
from loophole.senate.agents.drafter import Drafter
from loophole.senate.agents.reviser import Reviser
from loophole.senate.agents.tenets import TenetsExtractor
from loophole.senate.agents.voter import Voter
from loophole.senate.iterate import run_iteration
from loophole.senate.library import generate_landing_html, scan_iterations
from loophole.senate.pipeline import analyze_bill, build_constitutions
from loophole.senate.roster import load_roster
from loophole.senate.session import ConstitutionStore, SenateSessionManager
from loophole.senate.visualize import generate_html
from loophole.senate.visualize_iteration import generate_iteration_html

app = typer.Typer(name="loophole-senate", add_completion=False)
console = Console()


def _load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        return yaml.safe_load(config_path.read_text())
    return {
        "model": {"default": "claude-sonnet-4-20250514", "max_tokens": 4096},
        "temperatures": {"biographer": 0.4, "voter": 0.5},
        "session_dir": "sessions",
        "constitutions_dir": "sessions/_senate_constitutions",
        "parallelism": 6,
    }


def _resolve_provider(config: dict, role: str):
    max_tokens = config["model"]["max_tokens"]
    providers = config["model"].get("providers", {})
    if role in providers:
        role_cfg = providers[role]
        return create_provider(
            provider=role_cfg["provider"],
            model=role_cfg["model"],
            max_tokens=max_tokens,
            base_url=role_cfg.get("base_url"),
        )
    # Constituent voters use a cheaper model (haiku) by default — 500 calls/bill
    # at sonnet pricing would be wasteful for what's essentially gut-reaction.
    if role == "constituent":
        haiku_model = config["model"].get("bot") or "claude-haiku-4-5-20251001"
        return create_provider(_infer_provider(haiku_model), haiku_model, max_tokens)
    model = config["model"]["default"]
    return create_provider(_infer_provider(model), model, max_tokens)


def _build_biographer(config: dict) -> Biographer:
    temps = config.get("temperatures", {})
    return Biographer(
        _resolve_provider(config, "biographer"),
        temperature=temps.get("biographer", 0.4),
    )


def _build_voter(config: dict) -> Voter:
    temps = config.get("temperatures", {})
    return Voter(
        _resolve_provider(config, "voter"),
        temperature=temps.get("voter", 0.5),
    )


@app.command("build-constitutions")
def build_constitutions_cmd(
    overwrite: bool = typer.Option(False, "--overwrite", help="Regenerate cached constitutions"),
    only: str = typer.Option(None, "--only", help="Comma-separated short names to build (e.g. Schumer,Cruz)"),
    parallelism: int = typer.Option(None, "--parallelism", "-j", help="Max concurrent LLM calls"),
):
    """Generate (and cache) moral constitutions for every senator."""
    config = _load_config()
    senators = load_roster()
    if only:
        wanted = {s.strip().lower() for s in only.split(",")}
        senators = [s for s in senators if s.short_name.lower() in wanted]
        if not senators:
            console.print(f"[red]No senators matched: {only}[/red]")
            raise typer.Exit(1)
    store = ConstitutionStore(config.get("constitutions_dir", "sessions/_senate_constitutions"))
    biographer = _build_biographer(config)
    workers = parallelism or config.get("parallelism", 6)

    console.print(
        Panel(
            f"[bold]Senate — Biographer[/bold]\n"
            f"Roster: {len(senators)} senators · workers: {workers}",
            border_style="bright_blue",
        )
    )
    build_constitutions(senators, biographer, store, console, overwrite=overwrite, max_workers=workers)
    console.print(f"\n[green]Done.[/green] Constitutions in: [bold]{store.base_dir}[/bold]")


@app.command("analyze-bill")
def analyze_bill_cmd(
    bill_file: str = typer.Argument(..., help="Path to a text file containing the bill"),
    name: str = typer.Option(None, "--name", help="Display name for the bill"),
    parallelism: int = typer.Option(None, "--parallelism", "-j"),
    open_report: bool = typer.Option(True, "--open/--no-open", help="Open HTML report when done"),
):
    """Poll every senator on a bill and produce a visualization."""
    config = _load_config()
    bill_path = Path(bill_file)
    if not bill_path.exists():
        console.print(f"[red]Bill file not found: {bill_file}[/red]")
        raise typer.Exit(1)
    bill_text = bill_path.read_text()
    bill_name = name or bill_path.stem.replace("_", " ").title()

    store = ConstitutionStore(config.get("constitutions_dir", "sessions/_senate_constitutions"))
    constitutions = store.load_all()
    if not constitutions:
        console.print(
            "[red]No constitutions found.[/red] Run [bold]build-constitutions[/bold] first."
        )
        raise typer.Exit(1)

    senators = load_roster()
    session_mgr = SenateSessionManager(config.get("session_dir", "sessions"))
    voter = _build_voter(config)
    workers = parallelism or config.get("parallelism", 6)

    session_id = f"senate_{bill_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    state = session_mgr.create_session(session_id, bill_name, bill_text)

    console.print(
        Panel(
            f"[bold]Senate — Bill Analysis[/bold]\n"
            f"Bill: {bill_name}\n"
            f"Constitutions available: {len(constitutions)} / {len(senators)}",
            border_style="bright_blue",
        )
    )

    state = analyze_bill(
        state, senators, constitutions, voter, session_mgr, console, max_workers=workers
    )

    _display_tally(state)

    out_path = generate_html(state, senators, constitutions)
    console.print(f"\n[bold blue]Report:[/bold blue] {out_path}")

    if open_report:
        import webbrowser

        webbrowser.open(f"file://{Path(out_path).resolve()}")


@app.command("visualize")
def visualize_cmd(
    session_id: str = typer.Argument(None),
    output: str = typer.Option(None, "--output", "-o"),
):
    """Regenerate the HTML report for a previously-run session."""
    config = _load_config()
    session_mgr = SenateSessionManager(config.get("session_dir", "sessions"))

    if not session_id:
        sessions = session_mgr.list_sessions()
        if not sessions:
            console.print("[red]No senate sessions found.[/red]")
            raise typer.Exit(1)
        t = Table(title="Senate Sessions")
        t.add_column("#", style="dim")
        t.add_column("Session ID")
        t.add_column("Bill")
        t.add_column("Reactions")
        for i, s in enumerate(sessions, 1):
            t.add_row(str(i), s["id"], s["bill"], str(s["reactions"]))
        console.print(t)
        choice = Prompt.ask("Select session number")
        session_id = sessions[int(choice) - 1]["id"]

    state = session_mgr.load(session_id)
    store = ConstitutionStore(config.get("constitutions_dir", "sessions/_senate_constitutions"))
    constitutions = store.load_all()
    senators = load_roster()

    out_path = generate_html(state, senators, constitutions, output_path=output)
    console.print(f"[bold green]Report:[/bold green] {out_path}")


@app.command("iterate")
def iterate_cmd(
    prompt: str = typer.Argument(..., help='Plain-English bill idea, e.g. "end daylight savings time"'),
    max_rounds: int = typer.Option(4, "--max-rounds", help="Cap on iteration rounds"),
    target: int = typer.Option(67, "--target", help="Yes-vote target (67 = supermajority, 51 = simple majority)"),
    parallelism: int = typer.Option(None, "--parallelism", "-j"),
    open_report: bool = typer.Option(True, "--open/--no-open"),
    with_constituents: bool = typer.Option(False, "--with-constituents", help="Poll 10 Nemotron-USA personas per state and feed mood into each senator's voter"),
    personas_per_state: int = typer.Option(10, "--personas-per-state", help="How many constituents to sample per state (default 10)"),
):
    """Draft a bill from a plain-English idea, then iterate until passage or stuck."""
    config = _load_config()
    store = ConstitutionStore(config.get("constitutions_dir", "sessions/_senate_constitutions"))
    constitutions = store.load_all()
    if not constitutions:
        console.print(
            "[red]No constitutions found.[/red] Run [bold]build-constitutions[/bold] first."
        )
        raise typer.Exit(1)

    senators = load_roster()
    workers = parallelism or config.get("parallelism", 6)
    temps = config.get("temperatures", {})

    drafter = Drafter(_resolve_provider(config, "drafter"), temperature=temps.get("drafter", 0.4))
    tenets_extractor = TenetsExtractor(_resolve_provider(config, "tenets"), temperature=temps.get("tenets", 0.2))
    reviser = Reviser(_resolve_provider(config, "reviser"), temperature=temps.get("reviser", 0.4))
    voter = _build_voter(config)

    from pathlib import Path
    session_id = f"iterate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = Path(config.get("session_dir", "sessions")) / session_id

    console.print(
        Panel(
            f"[bold]Senate — Iteration[/bold]\n"
            f'User idea: "{prompt}"\n'
            f"Max rounds: {max_rounds} · Target yes: {target}"
            + (f"\n[yellow]Constituents enabled: {personas_per_state}/state[/yellow]" if with_constituents else ""),
            border_style="bright_blue",
        )
    )

    # If --with-constituents, we need the bill text first to poll personas.
    # The Drafter is part of run_iteration() — to avoid restructuring, do a
    # pre-draft pass here, then pass the moods into run_iteration which will
    # re-draft (idempotent at temperature=0.4, but close enough; cheaper than
    # restructuring).
    state_moods = None
    if with_constituents:
        from loophole.senate.agents.constituent import ConstituentVoter
        from loophole.senate.personas import load_persona_pool, sample_personas
        from loophole.senate.pipeline import poll_constituents

        console.print("[bold]Pre-draft for constituent polling...[/bold]")
        pre_bill = drafter.draft(prompt)
        console.print(f"  [dim]Drafted: {pre_bill.name}[/dim]")

        const_voter = ConstituentVoter(
            _resolve_provider(config, "constituent"),
            temperature=temps.get("constituent", 0.6),
        )
        pool = load_persona_pool()
        sampled = sample_personas(pool, per_state=personas_per_state)

        votes, mood_map = poll_constituents(
            sampled=sampled,
            voter=const_voter,
            bill_name=pre_bill.name,
            bill_text=pre_bill.text,
            console=console,
            max_workers=workers * 2,  # haiku is fast, push parallelism
        )
        state_moods = mood_map

        # Persist raw votes alongside the session
        session_dir.mkdir(parents=True, exist_ok=True)
        from loophole.senate.models import ConstituentVoteRecord
        intro_by_uuid = {p.uuid: p.one_line_intro() for ps in sampled.values() for p in ps}
        records = [
            ConstituentVoteRecord(
                persona_uuid=v.persona_uuid,
                state=v.state,
                vote=v.vote,
                reasoning=v.reasoning,
                key_concern=v.key_concern,
                persona_intro=intro_by_uuid.get(v.persona_uuid, ""),
            )
            for v in votes
        ]
        (session_dir / "constituents.json").write_text(
            "[\n" + ",\n".join(r.model_dump_json() for r in records) + "\n]"
        )

    iter_session = run_iteration(
        user_prompt=prompt,
        senators=senators,
        constitutions=constitutions,
        drafter=drafter,
        tenets_extractor=tenets_extractor,
        reviser=reviser,
        voter=voter,
        console=console,
        session_dir=session_dir,
        max_rounds=max_rounds,
        target_yes=target,
        max_workers=workers,
        state_moods=state_moods,
        pre_drafted_bill=pre_bill if with_constituents else None,
    )

    # Attach constituent data to the session
    if with_constituents and state_moods:
        from loophole.senate.models import StateMoodRecord, ConstituentVoteRecord
        intro_by_uuid = {p.uuid: p.one_line_intro() for ps in sampled.values() for p in ps}
        for state, mood in state_moods.items():
            iter_session.state_moods.append(StateMoodRecord(
                state=state,
                yes_count=mood.yes_count,
                no_count=mood.no_count,
                unsure_count=mood.unsure_count,
                label=mood.label,
                sample_voices=[
                    ConstituentVoteRecord(
                        persona_uuid=v.persona_uuid,
                        state=v.state,
                        vote=v.vote,
                        reasoning=v.reasoning,
                        key_concern=v.key_concern,
                        persona_intro=intro_by_uuid.get(v.persona_uuid, ""),
                    )
                    for v in mood.sample_voices[:3]
                ],
            ))
        iter_session.constituents_polled = True
        # Re-save so the JSON has constituent data
        (session_dir / "state.json").write_text(iter_session.model_dump_json(indent=2))

    out_path = generate_iteration_html(iter_session, senators, constitutions, output_path=str(session_dir / "iteration.html"))
    console.print(f"\n[bold blue]Report:[/bold blue] {out_path}")

    # Refresh the library landing page
    base_sessions = Path(config.get("session_dir", "sessions"))
    landing_path = generate_landing_html(
        scan_iterations(base_sessions), senators=senators, output_path=str(base_sessions / "index.html")
    )
    console.print(f"[bold blue]Library:[/bold blue] {landing_path}")

    if open_report:
        import webbrowser
        webbrowser.open(f"file://{Path(landing_path).resolve()}")


@app.command("landing")
def landing_cmd(
    output: str = typer.Option(None, "--output", "-o"),
    open_report: bool = typer.Option(True, "--open/--no-open"),
):
    """Regenerate the library landing page from all iterate_* sessions."""
    config = _load_config()
    base = Path(config.get("session_dir", "sessions"))
    sessions = scan_iterations(base)
    out_path = output or str(base / "index.html")
    out = generate_landing_html(sessions, senators=load_roster(), output_path=out_path)
    console.print(f"[bold green]Landing:[/bold green] {out} ({len(sessions)} sessions)")
    if open_report:
        import webbrowser
        webbrowser.open(f"file://{Path(out).resolve()}")


@app.command("list-constitutions")
def list_constitutions_cmd():
    """Show which senators have cached constitutions."""
    config = _load_config()
    store = ConstitutionStore(config.get("constitutions_dir", "sessions/_senate_constitutions"))
    constitutions = store.load_all()
    senators = load_roster()
    have = set(constitutions.keys())

    t = Table(title=f"Constitutions: {len(have)} / {len(senators)}")
    t.add_column("Senator")
    t.add_column("Party")
    t.add_column("State")
    t.add_column("Status")
    t.add_column("Confidence")
    for s in senators:
        if s.full_name in have:
            c = constitutions[s.full_name]
            t.add_row(s.short_name, s.party.value, s.state, "[green]✓[/green]", c.confidence)
        else:
            t.add_row(s.short_name, s.party.value, s.state, "[red]missing[/red]", "-")
    console.print(t)


def _display_tally(state) -> None:
    tally = state.tally
    t = Table(title=f"Tally — {state.bill_name}", show_header=False)
    t.add_column("Outcome")
    t.add_column("Count")
    for outcome in ["yes", "lean_yes", "undecided", "lean_no", "no"]:
        color = {"yes": "green", "lean_yes": "green", "undecided": "yellow",
                 "lean_no": "red", "no": "red"}[outcome]
        t.add_row(outcome.replace("_", " "), f"[{color}]{tally[outcome]}[/{color}]")
    console.print(t)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Loophole — Senate simulator."""
    if ctx.invoked_subcommand is None:
        console.print(
            Panel(
                "[bold]Loophole — Senate[/bold]\n"
                "Simulate how 100 senators would vote on a bill.",
                border_style="bright_blue",
            )
        )
        console.print("Commands:")
        console.print("  [bold]build-constitutions[/bold]  — synthesize moral constitutions (once)")
        console.print("  [bold]analyze-bill[/bold]         — poll every senator on a bill")
        console.print("  [bold]visualize[/bold]            — regenerate HTML for a session")
        console.print("  [bold]list-constitutions[/bold]   — show cached constitutions")


if __name__ == "__main__":
    app()
