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

from loophole.agents.judge import Judge
from loophole.agents.legislator import Legislator
from loophole.agents.loophole_finder import LoopholeFinder
from loophole.agents.overreach_finder import OverreachFinder
from loophole.agents.simplifier import Simplifier
from loophole.llm import LLMClient, _infer_provider, create_provider
from loophole.models import CaseStatus, CaseType, LegalCode, SessionState
from loophole.session import SessionManager

app = typer.Typer(name="loophole", add_completion=False)
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
    """Create an LLM provider for a given agent role, using per-role overrides if set."""
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

    # Fallback to default model
    model = config["model"]["default"]
    return create_provider(_infer_provider(model), model, max_tokens)


def _build_agents(config: dict) -> dict:
    temps = config["temperatures"]
    cases_per = config["loop"]["cases_per_agent"]

    agents = {
        "legislator": Legislator(_resolve_provider(config, "legislator"), temperature=temps["legislator"]),
        "loophole": LoopholeFinder(_resolve_provider(config, "loophole_finder"), temperature=temps["loophole_finder"], cases_per_agent=cases_per),
        "overreach": OverreachFinder(_resolve_provider(config, "overreach_finder"), temperature=temps["overreach_finder"], cases_per_agent=cases_per),
        "judge": Judge(_resolve_provider(config, "judge"), temperature=temps["judge"]),
    }

    if config.get("simplify", {}).get("enabled", False):
        agents["simplifier"] = Simplifier(
            _resolve_provider(config, "legislator"), temperature=temps["legislator"]
        )

    return agents


def _display_legal_code(code: LegalCode) -> None:
    console.print()
    console.print(
        Panel(
            code.text,
            title=f"[bold]Legal Code v{code.version}[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    if code.changelog:
        console.print(f"[dim]Changelog: {code.changelog}[/dim]")
    console.print()


def _display_case(case_obj) -> None:
    color = "red" if case_obj.case_type == CaseType.LOOPHOLE else "yellow"
    label = "LOOPHOLE" if case_obj.case_type == CaseType.LOOPHOLE else "OVERREACH"
    console.print()
    console.print(
        Panel(
            f"[bold]Scenario:[/bold]\n{case_obj.scenario}\n\n"
            f"[bold]Problem:[/bold]\n{case_obj.explanation}",
            title=f"[{color}]Case #{case_obj.id} — {label}[/{color}]",
            border_style=color,
            padding=(1, 2),
        )
    )


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


def _oversight_review(state, case_obj, result, revised):
    """Let user review an auto-resolution. Returns (accept, modification_text)."""
    import difflib

    console.print()
    console.print(
        Panel(
            f"[bold]Judge reasoning:[/bold]\n{result.reasoning}\n\n"
            f"[bold]Resolution:[/bold]\n{result.resolution_summary or '(see proposed revision)'}",
            title="[yellow]Oversight Review[/yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    diff = difflib.unified_diff(
        state.current_code.text.splitlines(),
        revised.text.splitlines(),
        fromfile=f"v{state.current_code.version}",
        tofile=f"v{revised.version}",
        lineterm="",
        n=3,
    )
    diff_text = "\n".join(diff)
    if diff_text:
        console.print(Panel(diff_text, title="Proposed Changes", border_style="cyan"))

    action = Prompt.ask(
        "[bold]Accept this resolution?[/bold]",
        choices=["accept", "reject", "modify"],
        default="accept",
    )

    if action == "accept":
        return True, None
    if action == "reject":
        return False, None
    modification = _get_multiline_input("Enter your modification instructions:")
    return True, modification


def _run_simplification(state, agents, session_mgr, config):
    """Run a simplification pass on the current legal code."""
    import difflib

    simplifier = agents.get("simplifier")
    judge: Judge = agents["judge"]
    oversight = config.get("oversight", False)

    if not simplifier or not state.resolved_cases:
        return

    console.print(Rule("[bold magenta] Simplification Pass [/bold magenta]", style="magenta"))

    old_length = len(state.current_code.text)
    console.print(f"[dim]Current code: {old_length} characters[/dim]")

    proposed = simplifier.simplify(state)
    if not proposed:
        console.print("[yellow]Simplifier could not produce a shorter version.[/yellow]")
        return

    new_length = len(proposed.text)
    reduction = (1 - new_length / old_length) * 100
    console.print(f"[dim]Proposed: {new_length} characters ({reduction:.0f}% reduction)[/dim]")

    console.print("[dim]Validating simplified version...[/dim]", end="")
    validation = judge.validate(state, proposed.text)

    if not validation.passes:
        console.print(" [red]Validation failed — keeping current version[/red]")
        console.print(f"[dim]{validation.details}[/dim]")
        return

    console.print(" [green]Validation passed[/green]")

    if oversight:
        diff = difflib.unified_diff(
            state.current_code.text.splitlines(),
            proposed.text.splitlines(),
            fromfile=f"v{state.current_code.version}",
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

    state.current_code = proposed
    state.code_history.append(proposed)
    session_mgr.save(state)
    console.print(f"[green bold]Simplified! Code now at v{proposed.version}[/green bold]")


def _run_adversarial_loop(state, agents, session_mgr, config):
    max_rounds = config["loop"]["max_rounds"]
    oversight = config.get("oversight", False)
    legislator: Legislator = agents["legislator"]
    loophole_finder: LoopholeFinder = agents["loophole"]
    overreach_finder: OverreachFinder = agents["overreach"]
    judge: Judge = agents["judge"]

    while state.current_round < max_rounds:
        state.current_round += 1
        console.print(Rule(f"[bold] Round {state.current_round} [/bold]", style="cyan"))

        # Phase 1: Adversarial search
        console.print("\n[bold]Searching for loopholes...[/bold]", end="")
        loopholes = loophole_finder.find(state)
        console.print(f" found [red]{len(loopholes)}[/red]")

        console.print("[bold]Searching for overreach...[/bold]", end="")
        overreaches = overreach_finder.find(state)
        console.print(f" found [yellow]{len(overreaches)}[/yellow]")

        all_cases = loopholes + overreaches

        if not all_cases:
            console.print(
                "\n[green bold]No failures found! "
                "The legal code appears robust against this round of testing.[/green bold]"
            )
            if not Confirm.ask("Run another round to be sure?", default=False):
                break
            continue

        # Phase 2: Judge each case
        round_auto = 0
        round_escalated = 0

        for case_obj in all_cases:
            state.cases.append(case_obj)
            _display_case(case_obj)

            # Judge attempts auto-resolution
            console.print("  [dim]Judge evaluating...[/dim]", end="")
            result = judge.evaluate(state, case_obj)

            if result.resolvable:
                # Validate against test suite
                if result.proposed_revision and state.resolved_cases:
                    console.print(" [dim]validating...[/dim]", end="")

                    # Have the legislator produce the actual revised code
                    case_obj.resolution = result.resolution_summary or result.reasoning
                    case_obj.status = CaseStatus.AUTO_RESOLVED
                    case_obj.resolved_by = "judge"

                    revised = legislator.revise(state, case_obj)

                    validation = judge.validate(state, revised.text)
                    if validation.passes:
                        # Oversight gate
                        if oversight:
                            accept, modification = _oversight_review(
                                state, case_obj, result, revised
                            )
                            if not accept:
                                case_obj.status = CaseStatus.ESCALATED
                                case_obj.resolution = None
                                case_obj.resolved_by = None
                                _escalate(state, case_obj, "User rejected auto-resolution.", legislator)
                                round_escalated += 1
                                session_mgr.save(state)
                                continue
                            if modification:
                                case_obj.resolution = modification
                                revised = legislator.revise(state, case_obj)

                        state.current_code = revised
                        state.code_history.append(revised)
                        console.print(
                            f" [green]Resolved → Code v{revised.version}[/green]"
                        )
                        round_auto += 1
                    else:
                        # Validation failed — escalate
                        case_obj.status = CaseStatus.ESCALATED
                        case_obj.resolution = None
                        case_obj.resolved_by = None
                        console.print(" [red]Validation failed — escalating[/red]")
                        _escalate(state, case_obj, validation.details, legislator)
                        round_escalated += 1
                else:
                    # No prior cases to validate against, or no proposed revision
                    case_obj.resolution = result.resolution_summary or result.reasoning
                    case_obj.status = CaseStatus.AUTO_RESOLVED
                    case_obj.resolved_by = "judge"

                    revised = legislator.revise(state, case_obj)

                    # Oversight gate
                    if oversight:
                        accept, modification = _oversight_review(
                            state, case_obj, result, revised
                        )
                        if not accept:
                            case_obj.status = CaseStatus.ESCALATED
                            case_obj.resolution = None
                            case_obj.resolved_by = None
                            _escalate(state, case_obj, "User rejected auto-resolution.", legislator)
                            round_escalated += 1
                            session_mgr.save(state)
                            continue
                        if modification:
                            case_obj.resolution = modification
                            revised = legislator.revise(state, case_obj)

                    state.current_code = revised
                    state.code_history.append(revised)
                    console.print(
                        f" [green]Resolved → Code v{revised.version}[/green]"
                    )
                    round_auto += 1
            else:
                # Unresolvable — escalate to user
                console.print(" [red bold]Cannot resolve — escalating to you[/red bold]")
                _escalate(state, case_obj, result.conflict_explanation or result.reasoning, legislator)
                round_escalated += 1

            session_mgr.save(state)

        # Round summary
        _display_round_summary(state, len(all_cases), round_auto, round_escalated)

        # Periodic simplification
        simplify_every = config.get("simplify", {}).get("every_n_rounds", 0)
        if simplify_every > 0 and state.current_round % simplify_every == 0:
            _run_simplification(state, agents, session_mgr, config)

        # Continue?
        console.print()
        action = Prompt.ask(
            "[bold]Next?[/bold]",
            choices=["continue", "view code", "stop"],
            default="continue",
        )
        if action == "view code":
            _display_legal_code(state.current_code)
            if not Confirm.ask("Continue to next round?", default=True):
                break
        elif action == "stop":
            break

    # Final simplification pass
    if config.get("simplify", {}).get("enabled", False):
        _run_simplification(state, agents, session_mgr, config)

    console.print(Rule("[bold green] Session Complete [/bold green]", style="green"))
    _display_legal_code(state.current_code)
    console.print(
        f"[bold]Final stats:[/bold] {len(state.cases)} cases over "
        f"{state.current_round} rounds, code at v{state.current_code.version}"
    )
    console.print(
        f"[dim]Session saved to: sessions/{state.session_id}/[/dim]"
    )

    # Generate HTML report
    from loophole.visualize import generate_html
    report_path = generate_html(state)
    console.print(f"[bold blue]HTML report:[/bold blue] {report_path}")
    console.print("[dim]Open it in a browser for a Twitter-ready visualization[/dim]")


def _escalate(state, case_obj, conflict_text, legislator):
    console.print(
        Panel(
            f"[bold]The judge could not resolve this case without breaking prior rulings.[/bold]\n\n"
            f"{conflict_text or 'No additional conflict details.'}",
            title="[red bold]Escalation[/red bold]",
            border_style="red",
            padding=(1, 2),
        )
    )

    decision = _get_multiline_input(
        "How should this case be handled? Your decision becomes a new constraint:"
    )

    case_obj.status = CaseStatus.USER_RESOLVED
    case_obj.resolution = decision
    case_obj.resolved_by = "user"
    state.user_clarifications.append(
        f"[Case #{case_obj.id}] {decision}"
    )

    # Legislator incorporates the user's decision
    console.print("  [dim]Updating legal code...[/dim]")
    revised = legislator.revise(state, case_obj)
    state.current_code = revised
    state.code_history.append(revised)
    console.print(f"  [green]Code updated → v{revised.version}[/green]")


def _display_round_summary(state, total, auto, escalated):
    console.print()
    table = Table(title=f"Round {state.current_round} Summary", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Cases found", str(total))
    table.add_row("Auto-resolved", f"[green]{auto}[/green]")
    table.add_row("Escalated to user", f"[red]{escalated}[/red]")
    table.add_row("Legal code version", f"v{state.current_code.version}")
    table.add_row("Total resolved cases", str(len(state.resolved_cases)))
    console.print(table)


@app.command()
def new(
    domain: str = typer.Option(None, help="Domain for the legal code (e.g., privacy, property, speech)"),
    principles_file: str = typer.Option(None, "--principles", "-p", help="Path to a text file with moral principles"),
    oversight: bool = typer.Option(False, "--oversight", help="Review every auto-judge decision before accepting"),
    simplify: bool = typer.Option(False, "--simplify", help="Enable simplification passes to compress the legal code"),
    simplify_every: int = typer.Option(0, "--simplify-every", help="Simplify every N rounds (0 = only at end)"),
):
    """Start a new Loophole session."""
    console.print(
        Panel(
            "[bold]Loophole[/bold]\n"
            "Adversarial moral-legal code system",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    config = _load_config()
    if oversight:
        config["oversight"] = True
    if simplify:
        config.setdefault("simplify", {})["enabled"] = True
        config["simplify"]["every_n_rounds"] = simplify_every
    agents = _build_agents(config)

    if not domain:
        domain = Prompt.ask("\n[bold]Domain[/bold] (e.g., privacy, property, speech)")

    if principles_file:
        principles = Path(principles_file).read_text(encoding="utf-8").strip()
        console.print(f"[dim]Loaded principles from {principles_file}[/dim]")
    else:
        principles = _get_multiline_input(
            "State your moral principles for this domain:"
        )

    session_id = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_mgr = SessionManager(config["session_dir"])

    # Generate initial legal code
    console.print("\n[bold]Generating initial legal code...[/bold]")
    legislator: Legislator = agents["legislator"]

    # Bootstrap: create a placeholder state for the initial draft
    placeholder = SessionState(
        session_id=session_id,
        domain=domain,
        moral_principles=principles,
        current_code=LegalCode(version=0, text=""),
    )
    initial_code = legislator.draft_initial(placeholder)

    state = session_mgr.create_session(session_id, domain, principles, initial_code)
    _display_legal_code(state.current_code)

    if Confirm.ask("Begin adversarial testing?", default=True):
        _run_adversarial_loop(state, agents, session_mgr, config)


@app.command()
def resume(
    session_id: str = typer.Argument(None, help="Session ID to resume"),
    oversight: bool = typer.Option(False, "--oversight", help="Review every auto-judge decision before accepting"),
    simplify: bool = typer.Option(False, "--simplify", help="Enable simplification passes to compress the legal code"),
    simplify_every: int = typer.Option(0, "--simplify-every", help="Simplify every N rounds (0 = only at end)"),
):
    """Resume an existing session."""
    config = _load_config()
    if oversight:
        config["oversight"] = True
    if simplify:
        config.setdefault("simplify", {})["enabled"] = True
        config["simplify"]["every_n_rounds"] = simplify_every
    session_mgr = SessionManager(config["session_dir"])

    if not session_id:
        sessions = session_mgr.list_sessions()
        if not sessions:
            console.print("[red]No sessions found.[/red]")
            raise typer.Exit()

        table = Table(title="Available Sessions")
        table.add_column("#", style="dim")
        table.add_column("Session ID")
        table.add_column("Domain")
        table.add_column("Round")
        table.add_column("Cases")
        table.add_column("Code Version")
        for i, s in enumerate(sessions, 1):
            table.add_row(
                str(i), s["id"], s["domain"],
                str(s["round"]), str(s["cases"]), f"v{s['code_version']}"
            )
        console.print(table)

        choice = Prompt.ask("Select session number")
        session_id = sessions[int(choice) - 1]["id"]

    state = session_mgr.load(session_id)
    agents = _build_agents(config)

    console.print(f"\n[bold]Resuming session:[/bold] {session_id}")
    console.print(f"Domain: {state.domain} | Round: {state.current_round} | Code: v{state.current_code.version}")
    _display_legal_code(state.current_code)

    _run_adversarial_loop(state, agents, session_mgr, config)


@app.command(name="list")
def list_sessions():
    """List all sessions."""
    config = _load_config()
    session_mgr = SessionManager(config["session_dir"])
    sessions = session_mgr.list_sessions()

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Sessions")
    table.add_column("Session ID")
    table.add_column("Domain")
    table.add_column("Round")
    table.add_column("Cases")
    table.add_column("Code Version")
    for s in sessions:
        table.add_row(
            s["id"], s["domain"],
            str(s["round"]), str(s["cases"]), f"v{s['code_version']}"
        )
    console.print(table)


@app.command()
def visualize(
    session_id: str = typer.Argument(None, help="Session ID to visualize"),
    output: str = typer.Option(None, "--output", "-o", help="Output HTML file path"),
):
    """Generate an HTML visualization of a session."""
    config = _load_config()
    session_mgr = SessionManager(config["session_dir"])

    if not session_id:
        sessions = session_mgr.list_sessions()
        if not sessions:
            console.print("[red]No sessions found.[/red]")
            raise typer.Exit()

        table = Table(title="Available Sessions")
        table.add_column("#", style="dim")
        table.add_column("Session ID")
        table.add_column("Domain")
        table.add_column("Cases")
        for i, s in enumerate(sessions, 1):
            table.add_row(str(i), s["id"], s["domain"], str(s["cases"]))
        console.print(table)

        choice = Prompt.ask("Select session number")
        session_id = sessions[int(choice) - 1]["id"]

    state = session_mgr.load(session_id)

    from loophole.visualize import generate_html
    report_path = generate_html(state, output_path=output)
    console.print(f"[bold green]Report generated:[/bold green] {report_path}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Loophole — Adversarial moral-legal code system."""
    if ctx.invoked_subcommand is None:
        # Interactive menu
        console.print(
            Panel(
                "[bold]Loophole[/bold]\n"
                "Adversarial moral-legal code system",
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
            ctx.invoke(new, domain=None, principles_file=None)
        elif choice == "2":
            ctx.invoke(resume, session_id=None)
        elif choice == "3":
            ctx.invoke(list_sessions)
        else:
            raise typer.Exit()


if __name__ == "__main__":
    app()
