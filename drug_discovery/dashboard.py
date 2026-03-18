"""Professional terminal dashboard for the ZANE AI Drug Discovery platform."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class DashboardSnapshot:
    """Single dashboard state snapshot."""

    run_id: str
    model_type: str
    mode: str
    molecules_screened: int
    molecules_generated: int
    active_jobs: int
    hit_rate: float
    avg_qed: float
    avg_sa: float
    best_binding: float
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    latency_ms: float


def _build_header(snapshot: DashboardSnapshot) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = Text("ZANE", style="bold cyan")
    subtitle = Text("AI Drug Discovery Operations Dashboard", style="bold white")
    meta = Text(
        f"Run: {snapshot.run_id}  |  Model: {snapshot.model_type.upper()}  |  Mode: {snapshot.mode}  |  {now}",
        style="dim",
    )
    return Panel(Align.center(Group(title, subtitle, meta)), border_style="cyan", box=box.ROUNDED)


def _build_kpi_table(snapshot: DashboardSnapshot) -> Panel:
    table = Table(show_header=False, box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")

    table.add_row("Molecules Screened", f"[bold green]{snapshot.molecules_screened:,}[/bold green]")
    table.add_row("Generated Candidates", f"{snapshot.molecules_generated:,}")
    table.add_row("Active Jobs", f"{snapshot.active_jobs}")
    table.add_row("Hit Rate", f"{snapshot.hit_rate:.2%}")
    table.add_row("Avg QED", f"{snapshot.avg_qed:.3f}")
    table.add_row("Avg SA", f"{snapshot.avg_sa:.2f}")
    table.add_row("Best Binding (kcal/mol)", f"[bold magenta]{snapshot.best_binding:.2f}[/bold magenta]")
    table.add_row("Inference Latency", f"{snapshot.latency_ms:.1f} ms")

    return Panel(table, title="Operational KPIs", border_style="green", box=box.ROUNDED)


def _build_training_panel(snapshot: DashboardSnapshot) -> Panel:
    progress_ratio = max(0.0, min(1.0, snapshot.epoch / max(snapshot.total_epochs, 1)))
    filled = int(progress_ratio * 30)
    progress_bar = "[" + "#" * filled + "-" * (30 - filled) + "]"

    text = Text()
    text.append(f"Epoch Progress : {snapshot.epoch}/{snapshot.total_epochs}\n", style="white")
    text.append(f"{progress_bar} {progress_ratio:.1%}\n\n", style="cyan")
    text.append(f"Train Loss     : {snapshot.train_loss:.4f}\n", style="yellow")
    text.append(f"Validation Loss: {snapshot.val_loss:.4f}\n", style="yellow")

    status = "Stable" if snapshot.val_loss <= snapshot.train_loss * 1.2 else "Drift Risk"
    status_style = "green" if status == "Stable" else "red"
    text.append(f"Model Health   : {status}", style=status_style)

    return Panel(text, title="Training Monitor", border_style="yellow", box=box.ROUNDED)


def _build_candidates_table() -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Candidate", style="cyan")
    table.add_column("QED", justify="right")
    table.add_column("SA", justify="right")
    table.add_column("Binding", justify="right", style="magenta")
    table.add_column("Status", justify="center")

    sample_rows = [
        (1, "ZN-A14", 0.84, 2.9, -10.6, "Promote"),
        (2, "ZN-B03", 0.79, 3.4, -9.8, "Promote"),
        (3, "ZN-C22", 0.75, 4.2, -9.2, "Review"),
        (4, "ZN-D07", 0.69, 4.9, -8.7, "Hold"),
        (5, "ZN-E11", 0.65, 5.3, -8.1, "Hold"),
    ]

    for rank, cid, qed, sa, bind, status in sample_rows:
        style = "green" if status == "Promote" else "yellow" if status == "Review" else "red"
        table.add_row(str(rank), cid, f"{qed:.2f}", f"{sa:.1f}", f"{bind:.1f}", f"[{style}]{status}[/{style}]")

    return Panel(table, title="Top Candidate Queue", border_style="magenta", box=box.ROUNDED)


def _build_alerts_panel(snapshot: DashboardSnapshot) -> Panel:
    alerts = [
        "Docking workers healthy across all nodes.",
        "No assay pipeline failures in the last hour.",
        "Quality gate: ADMET threshold compliance above target.",
    ]

    if snapshot.val_loss > snapshot.train_loss * 1.2:
        alerts.insert(0, "Validation loss rising above expected band.")

    text = Text()
    for line in alerts:
        bullet = "• "
        style = "red" if "rising" in line else "white"
        text.append(f"{bullet}{line}\n", style=style)

    return Panel(text, title="System Alerts", border_style="blue", box=box.ROUNDED)


def render_dashboard(snapshot: DashboardSnapshot) -> Layout:
    """Build a complete terminal dashboard layout for the given snapshot."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="main"),
        Layout(name="footer", size=7),
    )

    layout["header"].update(_build_header(snapshot))

    layout["main"].split_row(Layout(name="left"), Layout(name="right"))
    layout["left"].split_column(Layout(name="kpis"), Layout(name="train"))

    layout["left"]["kpis"].update(_build_kpi_table(snapshot))
    layout["left"]["train"].update(_build_training_panel(snapshot))
    layout["right"].update(_build_candidates_table())

    layout["footer"].update(_build_alerts_panel(snapshot))
    return layout


def _next_snapshot(previous: DashboardSnapshot) -> DashboardSnapshot:
    molecules_screened = previous.molecules_screened + random.randint(40, 110)
    molecules_generated = previous.molecules_generated + random.randint(5, 20)
    active_jobs = max(2, min(16, previous.active_jobs + random.choice([-1, 0, 1])))
    hit_rate = min(0.55, max(0.08, previous.hit_rate + random.uniform(-0.01, 0.012)))
    avg_qed = min(0.95, max(0.30, previous.avg_qed + random.uniform(-0.01, 0.008)))
    avg_sa = min(8.5, max(1.8, previous.avg_sa + random.uniform(-0.12, 0.1)))
    best_binding = min(-6.0, max(-13.0, previous.best_binding + random.uniform(-0.2, 0.15)))
    epoch = min(previous.total_epochs, previous.epoch + 1)
    train_loss = max(0.0001, previous.train_loss + random.uniform(-0.02, 0.01))
    val_loss = max(0.0001, previous.val_loss + random.uniform(-0.02, 0.015))
    latency_ms = min(240.0, max(25.0, previous.latency_ms + random.uniform(-8.0, 8.0)))

    return DashboardSnapshot(
        run_id=previous.run_id,
        model_type=previous.model_type,
        mode=previous.mode,
        molecules_screened=molecules_screened,
        molecules_generated=molecules_generated,
        active_jobs=active_jobs,
        hit_rate=hit_rate,
        avg_qed=avg_qed,
        avg_sa=avg_sa,
        best_binding=best_binding,
        epoch=epoch,
        total_epochs=previous.total_epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        latency_ms=latency_ms,
    )


def run_dashboard(live: bool = True, refresh_seconds: float = 1.0, iterations: int = 30) -> None:
    """Run the ZANE terminal dashboard in static or live mode."""
    console = Console()

    snapshot = DashboardSnapshot(
        run_id="ZANE-2026-0318-A",
        model_type="gnn",
        mode="Autonomous Discovery",
        molecules_screened=12480,
        molecules_generated=930,
        active_jobs=8,
        hit_rate=0.183,
        avg_qed=0.742,
        avg_sa=3.88,
        best_binding=-10.2,
        epoch=18,
        total_epochs=40,
        train_loss=0.192,
        val_loss=0.217,
        latency_ms=61.5,
    )

    if not live:
        console.print(render_dashboard(snapshot))
        return

    with Live(render_dashboard(snapshot), refresh_per_second=8, console=console, screen=True) as live_view:
        for _ in range(max(iterations, 1)):
            time.sleep(max(refresh_seconds, 0.2))
            snapshot = _next_snapshot(snapshot)
            live_view.update(render_dashboard(snapshot))


if __name__ == "__main__":
    run_dashboard(live=True)
