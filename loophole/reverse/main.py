from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from loophole.llm import _infer_provider, create_provider
from loophole.reverse.agents.analyst import Analyst
from loophole.reverse.agents.contradiction_finder import ContradictionFinder
from loophole.reverse.agents.gap_finder import GapFinder
from loophole.reverse.agents.simplifier import Simplifier
from loophole.reverse.models import CaseResolution, CaseType, PrinciplesList, ReverseSession
from loophole.reverse.session import ReverseSessionManager

app = typer.Typer(name="loophole-reverse", add_completion=False)
console = Console()


def _load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {
        "model": {"default": "claude-sonnet-4-20250514", "max_tokens": 4096},
        "temperatures": {
            "legislator": 0.4,
            "loophole_finder": 0.9,
            "overreach_finder": 0.9,
            "judge": 0.3,
        },
        "loop": {"max_rounds": 10, "cases_per_agent": 3},
        "session_dir": "sessions",
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

    model = config["model"]["default"]
    return create_provider(_infer_provider(model), model, max_tokens)


def _build_agents(config: dict) -> dict:
    temps = config["temperatures"]
    cases_per = config["loop"]["cases_per_agent"]

    agents = {
        "analyst": Analyst(
            _resolve_provider(config, "legislator"),
            temperature=temps["legislator"],
        ),
        "contradiction": ContradictionFinder(
            _resolve_provider(config, "loophole_finder"),
            temperature=temps["loophole_finder"],
            cases_per_agent=cases_per,
        ),
        "gap": GapFinder(
            _resolve_provider(config, "overreach_finder"),
            temperature=temps["overreach_finder"],
            cases_per_agent=cases_per,
        ),
    }

    if config.get("simplify", {}).get("enabled", False):
        agents["simplifier"] = Simplifier(
            _resolve_provider(config, "legislator"),
            temperature=temps["legislator"],
        )

    return agents


def _display_principles(principles: PrinciplesList) -> None:
    console.print()
    console.print(
        Panel(
            principles.text,
            title=f"[bold]Extracted Principles v{principles.version}[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    if principles.changelog:
        console.print(f"[dim]Changelog: {principles.changelog}[/dim]")
    console.print()


def _display_finding(finding) -> None:
    if finding.case_type == CaseType.CONTRADICTION:
        color = "red"
        label = "CONTRADICTION"
        sublabel = "Two principles conflict"
    else:
        color = "yellow"
        label = "GAP"
        sublabel = "Missing moral value"

    console.print()
    console.print(
        Panel(
            f"[bold]Scenario:[/bold]\n{finding.scenario}\n\n"
            f"[bold]Problem:[/bold]\n{finding.explanation}\n\n"
            f"[bold]Principles involved:[/bold] {', '.join(finding.principles_involved)}",
            title=f"[{color}]Finding #{finding.id} — {label}[/{color}]",
            subtitle=f"[{color}]{sublabel}[/{color}]",
            border_style=color,
            padding=(1, 2),
        )
    )


def _display_tensions(state: ReverseSession) -> None:
    if not state.tensions:
        console.print("[dim]No tensions identified yet.[/dim]")
        return
    console.print(Rule("[bold magenta] Genuine Tensions [/bold magenta]", style="magenta"))
    for i, t in enumerate(state.tensions, 1):
        label = "CONTRADICTION" if t.case_type == CaseType.CONTRADICTION else "GAP"
        console.print(
            Panel(
                f"[bold]Scenario:[/bold] {t.scenario}\n\n"
                f"[bold]Why it's a tension:[/bold] {t.tension_note}",
                title=f"[magenta]Tension #{i} ({label})[/magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )


def _display_round_summary(state, total, refined, tension):
    console.print()
    table = Table(title=f"Round {state.current_round} Summary", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Findings", str(total))
    table.add_row("Principles refined", f"[green]{refined}[/green]")
    table.add_row("Marked as tension", f"[magenta]{tension}[/magenta]")
    table.add_row("Principles version", f"v{state.current_principles.version}")
    table.add_row("Total tensions", str(len(state.tensions)))
    console.print(table)


def _get_multiline_input(prompt_text: str) -> str:
    console.print(f"\n[bold]{prompt_text}[/bold]")
    console.print("[dim](Enter a blank line when finished)[/dim]")
    lines = []
    while True:
        line = Prompt.ask("", default="")
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _run_simplification(state, agents, session_mgr, config):
    import difflib

    simplifier = agents.get("simplifier")
    if not simplifier or not state.refined_findings:
        return

    console.print(Rule("[bold magenta] Simplification Pass [/bold magenta]", style="magenta"))

    old_length = len(state.current_principles.text)
    console.print(f"[dim]Current principles: {old_length} characters[/dim]")

    proposed = simplifier.simplify(state)
    if not proposed:
        console.print("[yellow]Simplifier could not produce a shorter version.[/yellow]")
        return

    new_length = len(proposed.text)
    reduction = (1 - new_length / old_length) * 100
    console.print(f"[dim]Proposed: {new_length} characters ({reduction:.0f}% reduction)[/dim]")

    diff = difflib.unified_diff(
        state.current_principles.text.splitlines(),
        proposed.text.splitlines(),
        fromfile=f"v{state.current_principles.version}",
        tofile=f"v{proposed.version}",
        lineterm="",
        n=3,
    )
    diff_text = "\n".join(diff)
    if diff_text:
        console.print(Panel(diff_text, title="Simplification Diff", border_style="magenta"))
    if not Confirm.ask("Accept simplified version?", default=True):
        console.print("[yellow]Simplification rejected.[/yellow]")
        return

    state.current_principles = proposed
    state.principles_history.append(proposed)
    session_mgr.save(state)
    console.print(f"[green bold]Simplified! Principles now at v{proposed.version}[/green bold]")


def _run_adversarial_loop(state, agents, session_mgr, config):
    max_rounds = config["loop"]["max_rounds"]
    analyst: Analyst = agents["analyst"]
    contradiction_finder: ContradictionFinder = agents["contradiction"]
    gap_finder: GapFinder = agents["gap"]

    while state.current_round < max_rounds:
        state.current_round += 1
        console.print(Rule(f"[bold] Round {state.current_round} [/bold]", style="cyan"))

        # Phase 1: Find contradictions
        console.print("\n[bold]Searching for contradictions...[/bold]", end="")
        contradictions = contradiction_finder.find(state)
        console.print(f" found [red]{len(contradictions)}[/red]")

        # Phase 2: Find gaps
        console.print("[bold]Searching for gaps...[/bold]", end="")
        gaps = gap_finder.find(state)
        console.print(f" found [yellow]{len(gaps)}[/yellow]")

        all_findings = contradictions + gaps

        if not all_findings:
            console.print(
                "\n[green bold]No new issues found! "
                "The principles list appears comprehensive.[/green bold]"
            )
            if not Confirm.ask("Run another round?", default=False):
                break
            continue

        # Phase 3: Present each finding to the user
        round_refined = 0
        round_tension = 0

        for finding in all_findings:
            state.findings.append(finding)
            _display_finding(finding)

            action = Prompt.ask(
                "[bold]How do you want to handle this?[/bold]",
                choices=["refine", "tension", "skip"],
                default="refine",
            )

            if action == "refine":
                instruction = _get_multiline_input(
                    "How should the principles be updated?"
                )
                finding.resolution = CaseResolution.REFINED
                finding.user_instruction = instruction
                state.user_clarifications.append(
                    f"[Finding #{finding.id}] {instruction}"
                )

                console.print("  [dim]Updating principles...[/dim]")
                revised = analyst.revise(state, finding)
                state.current_principles = revised
                state.principles_history.append(revised)
                console.print(f"  [green]Principles updated -> v{revised.version}[/green]")
                round_refined += 1

            elif action == "tension":
                note = _get_multiline_input(
                    "Why is this genuinely unresolvable? (This becomes part of your output):"
                )
                finding.resolution = CaseResolution.TENSION
                finding.tension_note = note
                state.tensions.append(finding)
                console.print(f"  [magenta]Marked as genuine tension #{len(state.tensions)}[/magenta]")
                round_tension += 1

            else:
                console.print("  [dim]Skipped[/dim]")

            session_mgr.save(state)

        _display_round_summary(state, len(all_findings), round_refined, round_tension)

        # Periodic simplification
        simplify_every = config.get("simplify", {}).get("every_n_rounds", 0)
        if simplify_every > 0 and state.current_round % simplify_every == 0:
            _run_simplification(state, agents, session_mgr, config)

        console.print()
        action = Prompt.ask(
            "[bold]Next?[/bold]",
            choices=["continue", "view principles", "view tensions", "stop"],
            default="continue",
        )
        if action == "view principles":
            _display_principles(state.current_principles)
            if not Confirm.ask("Continue to next round?", default=True):
                break
        elif action == "view tensions":
            _display_tensions(state)
            if not Confirm.ask("Continue to next round?", default=True):
                break
        elif action == "stop":
            break

    # Final simplification
    if config.get("simplify", {}).get("enabled", False):
        _run_simplification(state, agents, session_mgr, config)

    console.print(Rule("[bold green] Session Complete [/bold green]", style="green"))
    _display_principles(state.current_principles)
    console.print()
    _display_tensions(state)
    console.print(
        f"\n[bold]Final stats:[/bold] {len(state.findings)} findings over "
        f"{state.current_round} rounds, principles at v{state.current_principles.version}, "
        f"{len(state.tensions)} genuine tensions identified"
    )
    console.print(f"[dim]Session saved to: sessions/{state.session_id}/[/dim]")

    from loophole.reverse.visualize import generate_html
    report_path = generate_html(state)
    console.print(f"[bold blue]HTML report:[/bold blue] {report_path}")


@app.command()
def new(
    document_name: str = typer.Option(None, "--name", "-n", help="Name for the legal document"),
    legal_text_file: str = typer.Option(None, "--text", "-t", help="Path to a text file with the legal document"),
    simplify: bool = typer.Option(False, "--simplify", help="Enable simplification passes on the principles"),
    simplify_every: int = typer.Option(0, "--simplify-every", help="Simplify every N rounds (0 = only at end)"),
):
    """Start a new reverse morals session — extract principles from a legal text."""
    console.print(
        Panel(
            "[bold]Loophole — Reverse Morals[/bold]\n"
            "Extract the moral DNA of a legal text",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    config = _load_config()
    if simplify:
        config.setdefault("simplify", {})["enabled"] = True
        config["simplify"]["every_n_rounds"] = simplify_every
    agents = _build_agents(config)

    if not document_name:
        document_name = Prompt.ask(
            "\n[bold]Document name[/bold] (e.g., US Constitution, Magna Carta)"
        )

    if legal_text_file:
        legal_text = Path(legal_text_file).read_text(encoding="utf-8").strip()
        console.print(f"[dim]Loaded legal text from {legal_text_file} ({len(legal_text)} chars)[/dim]")
    else:
        legal_text = _get_multiline_input(
            "Paste or type the legal text:"
        )

    session_id = (
        f"reverse_{document_name.lower().replace(' ', '_')}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_mgr = ReverseSessionManager(config["session_dir"])

    # Extract initial principles
    console.print("\n[bold]Extracting moral principles...[/bold]")
    analyst: Analyst = agents["analyst"]

    placeholder = ReverseSession(
        session_id=session_id,
        document_name=document_name,
        legal_text=legal_text,
        current_principles=PrinciplesList(version=0, text=""),
    )
    initial_principles = analyst.extract_initial(placeholder)

    state = session_mgr.create_session(
        session_id, document_name, legal_text, initial_principles
    )
    _display_principles(state.current_principles)

    if Confirm.ask("Begin adversarial analysis?", default=True):
        _run_adversarial_loop(state, agents, session_mgr, config)


@app.command()
def resume(
    session_id: str = typer.Argument(None, help="Session ID to resume"),
    simplify: bool = typer.Option(False, "--simplify", help="Enable simplification passes"),
    simplify_every: int = typer.Option(0, "--simplify-every", help="Simplify every N rounds (0 = only at end)"),
):
    """Resume an existing reverse morals session."""
    config = _load_config()
    if simplify:
        config.setdefault("simplify", {})["enabled"] = True
        config["simplify"]["every_n_rounds"] = simplify_every
    session_mgr = ReverseSessionManager(config["session_dir"])

    if not session_id:
        sessions = session_mgr.list_sessions()
        if not sessions:
            console.print("[red]No reverse sessions found.[/red]")
            raise typer.Exit()

        table = Table(title="Available Reverse Sessions")
        table.add_column("#", style="dim")
        table.add_column("Session ID")
        table.add_column("Document")
        table.add_column("Round")
        table.add_column("Findings")
        table.add_column("Tensions")
        table.add_column("Principles")
        for i, s in enumerate(sessions, 1):
            table.add_row(
                str(i), s["id"], s["document"],
                str(s["round"]), str(s["findings"]),
                str(s["tensions"]), f"v{s['principles_version']}",
            )
        console.print(table)

        choice = Prompt.ask("Select session number")
        session_id = sessions[int(choice) - 1]["id"]

    state = session_mgr.load(session_id)
    agents = _build_agents(config)

    console.print(f"\n[bold]Resuming session:[/bold] {session_id}")
    console.print(
        f"Document: {state.document_name} | Round: {state.current_round} | "
        f"Principles: v{state.current_principles.version} | "
        f"Tensions: {len(state.tensions)}"
    )
    _display_principles(state.current_principles)

    _run_adversarial_loop(state, agents, session_mgr, config)


@app.command(name="list")
def list_sessions():
    """List all reverse morals sessions."""
    config = _load_config()
    session_mgr = ReverseSessionManager(config["session_dir"])
    sessions = session_mgr.list_sessions()

    if not sessions:
        console.print("[dim]No reverse sessions found.[/dim]")
        return

    table = Table(title="Reverse Morals Sessions")
    table.add_column("Session ID")
    table.add_column("Document")
    table.add_column("Round")
    table.add_column("Findings")
    table.add_column("Tensions")
    table.add_column("Principles")
    for s in sessions:
        table.add_row(
            s["id"], s["document"],
            str(s["round"]), str(s["findings"]),
            str(s["tensions"]), f"v{s['principles_version']}",
        )
    console.print(table)


@app.command()
def visualize(
    session_id: str = typer.Argument(None, help="Session ID to visualize"),
    output: str = typer.Option(None, "--output", "-o", help="Output HTML file path"),
):
    """Generate an HTML visualization of a reverse morals session."""
    config = _load_config()
    session_mgr = ReverseSessionManager(config["session_dir"])

    if not session_id:
        sessions = session_mgr.list_sessions()
        if not sessions:
            console.print("[red]No reverse sessions found.[/red]")
            raise typer.Exit()

        table = Table(title="Available Reverse Sessions")
        table.add_column("#", style="dim")
        table.add_column("Session ID")
        table.add_column("Document")
        table.add_column("Tensions")
        for i, s in enumerate(sessions, 1):
            table.add_row(str(i), s["id"], s["document"], str(s["tensions"]))
        console.print(table)

        choice = Prompt.ask("Select session number")
        session_id = sessions[int(choice) - 1]["id"]

    state = session_mgr.load(session_id)

    from loophole.reverse.visualize import generate_html
    report_path = generate_html(state, output_path=output)
    console.print(f"[bold green]Report generated:[/bold green] {report_path}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Loophole Reverse — Extract the moral DNA of a legal text."""
    if ctx.invoked_subcommand is None:
        console.print(
            Panel(
                "[bold]Loophole — Reverse Morals[/bold]\n"
                "Extract the moral DNA of a legal text",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        console.print("  1. [bold]New session[/bold]")
        console.print("  2. [bold]Resume session[/bold]")
        console.print("  3. [bold]List sessions[/bold]")
        console.print("  4. [bold]Exit[/bold]")
        console.print()

        choice = Prompt.ask("Select", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            ctx.invoke(new, document_name=None, legal_text_file=None)
        elif choice == "2":
            ctx.invoke(resume, session_id=None)
        elif choice == "3":
            ctx.invoke(list_sessions)
        else:
            raise typer.Exit()


if __name__ == "__main__":
    app()
