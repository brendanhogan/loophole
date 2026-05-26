"""Orchestrates the two senate workloads: building constitutions and analyzing a bill."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console

from loophole.senate.agents.biographer import Biographer
from loophole.senate.agents.constituent import ConstituentVoter
from loophole.senate.agents.voter import Voter
from loophole.senate.models import BillReaction, MoralConstitution, SenateSession, Senator
from loophole.senate.personas import (
    ConstituentVote,
    Persona,
    StateMood,
    aggregate_state_mood,
)
from loophole.senate.session import ConstitutionStore, SenateSessionManager


def build_constitutions(
    senators: list[Senator],
    biographer: Biographer,
    store: ConstitutionStore,
    console: Console,
    overwrite: bool = False,
    max_workers: int = 6,
) -> dict[str, MoralConstitution]:
    """Generate (or load cached) moral constitutions for every senator."""

    constitutions: dict[str, MoralConstitution] = {}
    to_build: list[Senator] = []

    for s in senators:
        if not overwrite and store.has(s.full_name):
            constitutions[s.full_name] = store.load(s.full_name)
        else:
            to_build.append(s)

    if not to_build:
        console.print(f"[green]All {len(senators)} constitutions cached.[/green]")
        return constitutions

    console.print(
        f"[bold]Building {len(to_build)} constitutions[/bold] "
        f"(parallelism {max_workers}, {len(constitutions)} cached)..."
    )

    def _one(senator: Senator) -> tuple[Senator, MoralConstitution | None, str | None]:
        try:
            c = biographer.build(senator)
            store.save(c)
            return senator, c, None
        except Exception as exc:  # pragma: no cover — surfaced to console
            return senator, None, str(exc)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, s): s for s in to_build}
        done = 0
        for fut in as_completed(futures):
            senator, c, err = fut.result()
            done += 1
            if c is None:
                console.print(
                    f"  [red][{done}/{len(to_build)}] {senator.short_name}: {err}[/red]"
                )
                continue
            constitutions[senator.full_name] = c
            tag = {"high": "green", "medium": "yellow", "low": "red"}.get(
                c.confidence, "yellow"
            )
            console.print(
                f"  [dim][{done}/{len(to_build)}][/dim] "
                f"{senator.short_name:<18} [{tag}]conf:{c.confidence}[/{tag}] "
                f"({len(c.top_issues)} issues, {len(c.citations)} citations)"
            )

    return constitutions


def poll_chamber(
    senators: list[Senator],
    constitutions: dict[str, MoralConstitution],
    voter: Voter,
    bill_name: str,
    bill_text: str,
    console: Console,
    max_workers: int = 6,
    progress_label: str = "Polling",
    on_each: callable = None,
    state_moods: dict[str, StateMood] | None = None,
) -> list[BillReaction]:
    """Run the voter agent across every senator for a bill. Returns reactions.

    `on_each(reactions_so_far)` is called after each completion if provided.
    `state_moods` (optional) passes constituent context into each senator's
    voter call — the senator sees their state's mood when deciding.
    """

    eligible = [s for s in senators if s.full_name in constitutions]
    missing = [s.short_name for s in senators if s.full_name not in constitutions]
    if missing:
        console.print(
            f"[yellow]Skipping {len(missing)} without constitutions: "
            f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}[/yellow]"
        )

    console.print(
        f"[bold]{progress_label}[/bold] {len(eligible)} senators on '{bill_name}' "
        f"(parallelism {max_workers}){' · with constituent context' if state_moods else ''}..."
    )

    def _one(senator: Senator) -> tuple[Senator, BillReaction | None, str | None]:
        try:
            r = voter.react(
                senator=senator,
                constitution=constitutions[senator.full_name],
                bill_name=bill_name,
                bill_text=bill_text,
                state_mood=state_moods.get(senator.state) if state_moods else None,
            )
            return senator, r, None
        except Exception as exc:  # pragma: no cover
            return senator, None, str(exc)

    reactions: list[BillReaction] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, s): s for s in eligible}
        done = 0
        for fut in as_completed(futures):
            senator, r, err = fut.result()
            done += 1
            if r is None:
                console.print(
                    f"  [red][{done}/{len(eligible)}] {senator.short_name}: {err}[/red]"
                )
                continue
            reactions.append(r)
            vote_color = {
                "yes": "green",
                "lean_yes": "green",
                "undecided": "yellow",
                "lean_no": "red",
                "no": "red",
            }.get(r.vote.value, "white")
            console.print(
                f"  [dim][{done}/{len(eligible)}][/dim] "
                f"{senator.short_name:<18} [{vote_color}]{r.vote.value:<9}[/{vote_color}] "
                f"conf:{r.confidence:.2f}"
            )
            if on_each is not None:
                on_each(reactions)

    return reactions


def poll_constituents(
    sampled: dict[str, list[Persona]],
    voter: ConstituentVoter,
    bill_name: str,
    bill_text: str,
    console: Console,
    max_workers: int = 12,
) -> tuple[list[ConstituentVote], dict[str, StateMood]]:
    """Run the ConstituentVoter across every sampled persona.

    Returns (all votes, per-state mood). The state mood map is what the
    senator voter consumes; the full vote list is kept for the viz.
    """
    total = sum(len(ps) for ps in sampled.values())
    console.print(
        f"[bold]Polling {total} constituents[/bold] across {len(sampled)} states "
        f"(parallelism {max_workers})..."
    )

    work: list[Persona] = []
    for personas in sampled.values():
        work.extend(personas)

    def _one(p: Persona) -> tuple[Persona, ConstituentVote | None, str | None]:
        try:
            return p, voter.vote(p, bill_name, bill_text), None
        except Exception as exc:
            return p, None, str(exc)

    votes: list[ConstituentVote] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, p): p for p in work}
        done = 0
        for fut in as_completed(futures):
            persona, vote, err = fut.result()
            done += 1
            if vote is None:
                console.print(
                    f"  [red][{done}/{total}] {persona.state}/{persona.uuid[:6]}: {err}[/red]"
                )
                continue
            votes.append(vote)
            if done % 25 == 0 or done == total:
                console.print(f"  [dim][{done}/{total}][/dim] {persona.state} {vote.vote}")

    by_state = aggregate_state_mood(votes)
    # Print quick state-level summary
    sample_states = sorted(by_state.keys())[:6]
    for s in sample_states:
        m = by_state[s]
        console.print(
            f"  [dim]{s}:[/dim] yes={m.yes_count} no={m.no_count} unsure={m.unsure_count} → [bold]{m.label}[/bold]"
        )
    return votes, by_state


def analyze_bill(
    state: SenateSession,
    senators: list[Senator],
    constitutions: dict[str, MoralConstitution],
    voter: Voter,
    session_mgr: SenateSessionManager,
    console: Console,
    max_workers: int = 6,
) -> SenateSession:
    """Run the voter agent across every senator for the bill in `state`."""

    def _save(reactions):
        state.reactions = reactions
        session_mgr.save(state)

    reactions = poll_chamber(
        senators=senators,
        constitutions=constitutions,
        voter=voter,
        bill_name=state.bill_name,
        bill_text=state.bill_text,
        console=console,
        max_workers=max_workers,
        on_each=_save,
    )
    state.reactions = reactions
    session_mgr.save(state)
    return state
