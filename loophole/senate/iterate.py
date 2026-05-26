"""The iterate loop: draft → poll → revise → re-poll → ... until passage or stuck."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from loophole.senate.agents.drafter import Drafter
from loophole.senate.agents.reviser import (
    Reviser,
    cluster_flip_amendments,
    cluster_load_bearing_provisions,
)
from loophole.senate.agents.tenets import TenetsExtractor
from loophole.senate.agents.voter import Voter
from loophole.senate.models import (
    Bill,
    CoreTenets,
    IterationRound,
    IterationSession,
    MoralConstitution,
    Senator,
)
from loophole.senate.personas import StateMood
from loophole.senate.pipeline import poll_chamber


# Pass thresholds
SUPERMAJORITY = 67  # 2/3 — cloture+passage in one
MAJORITY = 51

# Defaults
DEFAULT_MAX_ROUNDS = 4
DEFAULT_TARGET = SUPERMAJORITY


def run_iteration(
    user_prompt: str,
    senators: list[Senator],
    constitutions: dict[str, MoralConstitution],
    drafter: Drafter,
    tenets_extractor: TenetsExtractor,
    reviser: Reviser,
    voter: Voter,
    console: Console,
    session_dir: Path,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    target_yes: int = DEFAULT_TARGET,
    max_workers: int = 6,
    state_moods: dict[str, StateMood] | None = None,
    pre_drafted_bill: Bill | None = None,
) -> IterationSession:
    """Run the full iterate loop. Saves IterationSession to session_dir/state.json."""

    session_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: Draft (or reuse a pre-drafted bill) ----
    console.print(Rule("[bold cyan] Stage 1 — Drafting [/bold cyan]"))
    if pre_drafted_bill is not None:
        bill = pre_drafted_bill
        console.print(f"  [dim]Using pre-drafted bill: {bill.name}[/dim]")
    else:
        bill = drafter.draft(user_prompt)
    console.print(Panel(
        f"[bold]{bill.name}[/bold]\n[dim]{bill.summary}[/dim]\n\n"
        f"{bill.text[:600]}{'...' if len(bill.text) > 600 else ''}",
        title="Draft Bill (v1)",
        border_style="cyan",
    ))

    # ---- Stage 2: Extract tenets ----
    console.print(Rule("[bold cyan] Stage 2 — Tenets [/bold cyan]"))
    tenets = tenets_extractor.extract(bill, user_prompt)
    for i, t in enumerate(tenets.tenets, 1):
        console.print(f"  [bold]{i}.[/bold] {t}")

    # ---- Stage 3: Iterate (hill-climb) ----
    # Each round we poll a candidate bill. If its yes_total beats the current
    # best, it becomes the new trunk. If not, we revert: the next revision
    # is generated against the best-known bill, using the demands surfaced
    # in this regressed round. Stop when target hit, all amendments violate
    # tenets, or N consecutive rounds fail to improve.
    session_id = session_dir.name
    iter_session = IterationSession(
        session_id=session_id,
        user_prompt=user_prompt,
        tenets=tenets,
    )

    def _save():
        (session_dir / "state.json").write_text(iter_session.model_dump_json(indent=2))

    _save()

    best_bill: Bill = bill
    best_yes: int = -1
    best_idx: int | None = None
    best_reactions: list = []  # reactions to the best bill — source of load-bearing provisions
    no_improvement_streak = 0
    MAX_REGRESSIONS = 2  # two consecutive non-improving rounds → declare converged

    for round_idx in range(max_rounds):
        console.print(Rule(f"[bold] Round {round_idx + 1} — Polling chamber on v{bill.version} [/bold]", style="cyan"))

        reactions = poll_chamber(
            senators=senators,
            constitutions=constitutions,
            voter=voter,
            bill_name=bill.name,
            bill_text=bill.text,
            console=console,
            max_workers=max_workers,
            progress_label=f"Round {round_idx + 1}",
            state_moods=state_moods,
        )

        round_record = IterationRound(
            round_number=round_idx,
            bill=bill.model_copy(),
            reactions=reactions,
            target_to_beat=best_yes if best_yes >= 0 else None,
        )

        yes_total = round_record.yes_total
        no_total = round_record.tally["no"] + round_record.tally["lean_no"]
        undecided = round_record.tally["undecided"]

        improved = yes_total > best_yes
        if improved:
            best_bill = bill
            best_yes = yes_total
            best_idx = round_idx
            best_reactions = reactions
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1

        console.print(
            f"\n[bold]Round {round_idx + 1} tally:[/bold] "
            f"[green]{yes_total} yes[/green] / "
            f"[red]{no_total} no[/red] / "
            f"[yellow]{undecided} undecided[/yellow] "
            f"(target {target_yes} · best so far {best_yes})"
        )
        if not improved and round_idx > 0:
            console.print(
                f"  [yellow]↓ Regression: {yes_total} ≤ best {best_yes} — reverting to v{best_bill.version} for next attempt[/yellow]"
            )

        # Stop condition 1: target reached
        if yes_total >= target_yes:
            round_record.outcome = "passed"
            iter_session.rounds.append(round_record)
            iter_session.best_round_idx = best_idx
            iter_session.final_status = "passed"
            _save()
            console.print(
                f"\n[bold green]✓ Bill reached {yes_total}-yes target after {round_idx + 1} rounds.[/bold green]"
            )
            return iter_session

        # Cluster vote-flipping amendments from *this round's* reactions —
        # those demands are still legitimate input even if the bill regressed.
        proposals = cluster_flip_amendments(reactions)

        if not proposals:
            round_record.outcome = "accepted" if improved else "regressed"
            iter_session.rounds.append(round_record)
            iter_session.best_round_idx = best_idx
            iter_session.final_status = "stuck"
            _save()
            console.print(
                f"\n[bold yellow]⚠ No senator offered a vote-flipping amendment — "
                f"converged at best {best_yes}-yes.[/bold yellow]"
            )
            return iter_session

        console.print(
            f"  [dim]{len(proposals)} amendment clusters proposed; "
            f"top would flip {proposals[0].flip_count} senators[/dim]"
        )

        # If this is the last round, don't bother revising — record outcome and exit.
        if round_idx == max_rounds - 1:
            round_record.outcome = "accepted" if improved else "regressed"
            iter_session.rounds.append(round_record)
            iter_session.best_round_idx = best_idx
            iter_session.final_status = "max_rounds"
            _save()
            console.print(
                f"\n[bold yellow]Hit max rounds ({max_rounds}). Best: {best_yes}-yes.[/bold yellow]"
            )
            return iter_session

        # Stop condition 2: too many regressions in a row — give up on hill-climbing.
        if no_improvement_streak >= MAX_REGRESSIONS:
            round_record.outcome = "regressed"
            iter_session.rounds.append(round_record)
            iter_session.best_round_idx = best_idx
            iter_session.final_status = "converged"
            _save()
            console.print(
                f"\n[bold yellow]⚠ {no_improvement_streak} rounds without improvement — "
                f"converged at best {best_yes}-yes.[/bold yellow]"
            )
            return iter_session

        # ---- Revise (always against the BEST bill, not the just-polled one) ----
        # Load-bearing provisions come from the BEST round's yes-voters — those
        # are the supporters whose votes the reviser must protect, since the
        # revision will be polled against that coalition.
        load_bearing = cluster_load_bearing_provisions(best_reactions)
        if load_bearing:
            console.print(
                f"  [dim]Reviser given {len(load_bearing)} load-bearing provisions "
                f"from {best_yes} yes-voters[/dim]"
            )
        console.print("\n[bold]Revising bill...[/bold]")
        revised, applied, rejected, revision_summary = reviser.revise(
            best_bill, tenets, proposals, load_bearing=load_bearing
        )
        round_record.applied = applied
        round_record.rejected = rejected
        round_record.revision_summary = revision_summary
        round_record.outcome = "accepted" if improved else "regressed"
        iter_session.rounds.append(round_record)
        iter_session.best_round_idx = best_idx

        console.print(
            f"  Applied [green]{len(applied)}[/green] amendments, "
            f"rejected [red]{len(rejected)}[/red] for tenet violations"
        )
        if revision_summary:
            console.print(f"  [dim]{revision_summary}[/dim]")

        if not applied:
            iter_session.final_status = "stuck"
            _save()
            console.print(
                f"\n[bold yellow]⚠ All proposed amendments would violate tenets — "
                f"converged at best {best_yes}-yes.[/bold yellow]"
            )
            return iter_session

        bill = revised
        _save()

    iter_session.final_status = "max_rounds"
    iter_session.best_round_idx = best_idx
    _save()
    return iter_session
