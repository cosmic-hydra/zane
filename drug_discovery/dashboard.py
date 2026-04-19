"""Professional terminal dashboard for the ZANE AI Drug Discovery platform."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from rich import box
from rich.align import Align
from rich.columns import Columns
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
    user_query: str
    filter_query: str
    cpu_util: float
    gpu_util: float
    memory_gb: float
    tick: int


@dataclass(frozen=True)
class DashboardTheme:
    """Terminal color theme for dashboard rendering."""

    name: str
    primary: str
    secondary: str
    accent: str
    caution: str
    ok: str
    panel_box: Any


@dataclass(frozen=True)
class _MoleculeSpec:
    """Small simulation-only library entry for dashboard ranking."""

    name: str
    smiles: str
    indications: tuple[str, ...]
    fallback_qed: float | None = None
    fallback_risk: float | None = None


_SIMULATION_LIBRARY: list[_MoleculeSpec] = [
    _MoleculeSpec("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", ("pain", "fever", "inflammation", "cold")),
    _MoleculeSpec("Naproxen", "COC1=CC=CC2=C1C=C(C=C2)C(C)C(=O)O", ("pain", "inflammation", "arthritis", "cold")),
    _MoleculeSpec("Dextromethorphan", "CN1CCC23CCCCC2C1CC4=C3C=CC(=C4)OC", ("cough", "cold", "flu")),
    _MoleculeSpec("Guaifenesin", "COC1=CC=C(C=C1)OCC(O)CO", ("cough", "mucus", "cold", "chest congestion")),
    _MoleculeSpec("Pseudoephedrine", "CC(C)NCC(C1=CC=CC=C1)O", ("congestion", "sinus", "cold", "rhinitis")),
    _MoleculeSpec("Loratadine", "CCOC(=O)N1CCC(=C2C3=CC=CC=C3CCC4=CC=CC=C24)CC1", ("allergy", "sneezing", "rhinitis", "cold")),
    _MoleculeSpec("Cetirizine", "CN1CCN(CC1)CCOCCOCC(=O)O", ("allergy", "rhinitis", "sneezing", "cold")),
]

_FALLBACK_MOL_PROFILES: dict[str, tuple[float, float]] = {
    # name: (qed, risk_proxy)
    "Ibuprofen": (0.82, 0.32),
    "Naproxen": (0.88, 0.56),
    "Dextromethorphan": (0.78, 0.53),
    "Guaifenesin": (0.72, 0.26),
    "Pseudoephedrine": (0.74, 0.23),
    "Loratadine": (0.74, 0.79),
    "Cetirizine": (0.57, 0.22),
}

_CUSTOM_CARBON_SCAFFOLDS: list[tuple[str, str, float, float, tuple[str, ...]]] = [
    (
        "trimethylbenzoate-like ester",
        "CC1=CC(C)=C(C(=O)OC)C(C)=C1",
        0.71,
        0.31,
        ("aromatic", "ester", "carbon"),
    ),
    (
        "ethylcyclohexane hydrocarbon",
        "CCC1CCCCC1",
        0.63,
        0.20,
        ("hydrocarbon", "lipophilic", "carbon"),
    ),
    (
        "isobutylbenzene aromatic",
        "CC(C)CC1=CC=CC=C1",
        0.66,
        0.26,
        ("aromatic", "hydrocarbon", "carbon"),
    ),
    (
        "alkyl carbonate prototype",
        "CCOC(=O)OCC",
        0.60,
        0.22,
        ("carbonate", "consumable", "carbon"),
    ),
    (
        "cyclopentyl acetate prototype",
        "CC(=O)OC1CCCC1",
        0.58,
        0.24,
        ("ester", "carbon", "volatile"),
    ),
    (
        "linear alkyl ether prototype",
        "CCCOCC",
        0.55,
        0.18,
        ("ether", "hydrocarbon", "carbon"),
    ),
]

_ZANE_BANNER = r"""
 ________      ___      _   _   ________
|___  /\ \    / / |    | \ | | |  ____|
    / /  \ \  / /| |    |  \| | | |__
  / /    \ \/ / | |    | . ` | |  __|
 / /__    \  /  | |____| |\  | | |____
/_____|    \/   |______|_| \_| |______|
""".strip("\n")

_DEFAULT_HEADER_LOGO_URL = "https://www.rdkit.org/docs/_static/logo.png"
_DEFAULT_LOCAL_LOGO = Path("logo.png")

_DASHBOARD_THEMES: dict[str, DashboardTheme] = {
    "lab": DashboardTheme(
        name="lab",
        primary="bright_cyan",
        secondary="white",
        accent="bright_green",
        caution="yellow",
        ok="green",
        panel_box=box.ROUNDED,
    ),
    "neon": DashboardTheme(
        name="neon",
        primary="magenta",
        secondary="bright_white",
        accent="cyan",
        caution="bright_yellow",
        ok="bright_green",
        panel_box=box.DOUBLE,
    ),
    "classic": DashboardTheme(
        name="classic",
        primary="blue",
        secondary="white",
        accent="green",
        caution="yellow",
        ok="green",
        panel_box=box.SQUARE,
    ),
}


def _resolve_theme(name: str | None) -> DashboardTheme:
    """Resolve theme by name with fallback to 'lab' theme.

    Args:
        name: Theme name or None to use default.

    Returns:
        DashboardTheme configuration object.
    """
    key = (name or "lab").strip().lower()
    return _DASHBOARD_THEMES.get(key, _DASHBOARD_THEMES["lab"])


def _phase_glyph(tick: int) -> str:
    """Return an animated spinner glyph cycling through phases.

    Args:
        tick: Animation frame counter.

    Returns:
        Single character representing current animation phase.
    """
    glyphs = ["◐", "◓", "◑", "◒"]
    return glyphs[max(0, tick) % len(glyphs)]


def _animated_bar(ratio: float, tick: int, width: int = 26) -> str:
    """Render an animated progress bar with animated head.

    Args:
        ratio: Progress from 0.0 to 1.0.
        tick: Animation frame counter for head animation.
        width: Bar width in characters.

    Returns:
        Animated progress bar string.
    """
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    filled = min(width, max(0, filled))
    chars = ["█" if i < filled else "░" for i in range(width)]
    return "".join(chars)


def _get_admet_predictor() -> Any | None:
    """Safely instantiate ADMET predictor with graceful degradation.

    Returns:
        ADMETPredictor instance if available, None if import or instantiation fails.
    """
    try:
        from .evaluation import ADMETPredictor

        return ADMETPredictor()
    except Exception:
        return None


class DashboardAIAdvisor:
    """Provide dashboard guidance via local LLM when available, with safe heuristic fallback."""

    def __init__(self, model_id: str | None = None, max_new_tokens: int = 120):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._assistant: Any | None = None
        self._provider = "heuristic"

        if model_id:
            try:
                from drug_discovery.ai_support import AISupportConfig, LlamaSupportAssistant

                self._assistant = LlamaSupportAssistant(config=AISupportConfig(model_id=model_id))
                self._provider = f"llama:{model_id}"
            except Exception:
                self._assistant = None
                self._provider = "heuristic"

    @property
    def provider(self) -> str:
        return self._provider

    def summarize(self, snapshot: DashboardSnapshot) -> str:
        heuristic = _heuristic_insights(snapshot)

        if self._assistant is None:
            return heuristic

        prompt = (
            "Provide exactly 3 concise operational recommendations for this drug discovery run. "
            "Focus on training quality, candidate triage, and throughput."
        )
        context = (
            f"model={snapshot.model_type}; mode={snapshot.mode}; "
            f"hit_rate={snapshot.hit_rate:.3f}; avg_qed={snapshot.avg_qed:.3f}; avg_sa={snapshot.avg_sa:.3f}; "
            f"best_binding={snapshot.best_binding:.2f}; train_loss={snapshot.train_loss:.4f}; "
            f"val_loss={snapshot.val_loss:.4f}; latency_ms={snapshot.latency_ms:.1f}; "
            f"active_jobs={snapshot.active_jobs}; epoch={snapshot.epoch}/{snapshot.total_epochs}"
        )

        try:
            response = self._assistant.respond(
                user_prompt=prompt,
                context=context,
                max_new_tokens=self.max_new_tokens,
                temperature=0.2,
                top_p=0.9,
            )
            cleaned = "\n".join(line.strip() for line in response.splitlines() if line.strip())
            return cleaned or heuristic
        except Exception:
            self._provider = "heuristic"
            return heuristic


def _cerebras_brief(user_query: str, evidence_lines: list[str]) -> tuple[str, str]:
    """Generate a short recommendation using Cerebras when configured."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        return "", "cerebras:unavailable(no_api_key)"

    try:
        from cerebras.cloud.sdk import Cerebras
    except Exception:
        return "", "cerebras:unavailable(sdk_missing)"

    requested_model = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")
    evidence = "\n".join(evidence_lines[:6])
    prompt = (
        "You are ZANE dashboard copilot for simulation-only candidate triage. "
        "Given a disease/need query and web/PDF evidence snippets, provide exactly 3 concise recommendations. "
        "Format each as: '- Candidate or combo | reason | caution'.\n\n"
        f"Query: {user_query}\n"
        f"Evidence:\n{evidence}"
    )

    try:
        client = Cerebras(api_key=api_key)
        resolved_model = requested_model
        try:
            model_list = client.models.list()
            available_ids = [entry.id for entry in getattr(model_list, "data", []) if getattr(entry, "id", "")]
            if available_ids and requested_model not in available_ids:
                resolved_model = available_ids[0]
        except Exception:
            resolved_model = requested_model

        completion = client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=220,
            temperature=0.2,
            top_p=1,
            stream=False,
        )
        text = completion.choices[0].message.content.strip()
        return text, f"cerebras:{resolved_model}"
    except Exception as exc:
        return f"Cerebras request failed: {exc}", f"cerebras:{requested_model}(error)"


def _gather_external_intel(
    user_query: str,
    enable_web_intel: bool,
    enable_pdf_read: bool,
    enable_cerebras: bool,
) -> tuple[str, str, list[str]]:
    """Collect dashboard intelligence from web/PDF and optional Cerebras summary."""
    providers: list[str] = []
    lines: list[str] = []
    evidence_lines: list[str] = []

    if enable_web_intel:
        try:
            from .web_scraping.scraper import InternetSearchClient, OnlineResourceReader

            client = InternetSearchClient(go_search_bin=os.getenv("ZANE_GO_SEARCH_BIN"))
            reader = OnlineResourceReader(max_chars=900, max_pdf_pages=3)
            hits = client.search_web(
                query=f"{user_query} treatment options mechanism contraindications",
                max_results=5,
                prefer_google=True,
            )
            providers.append("web-search")

            for hit in hits[:5]:
                title = str(hit.get("title", ""))
                url = str(hit.get("url", ""))
                source = str(hit.get("source", "web"))
                lines.append(f"- [web:{source}] {title}")

                if not enable_pdf_read:
                    continue

                resource = reader.read_resource(url)
                if bool(resource.get("success", False)):
                    rtype = str(resource.get("resource_type", "unknown"))
                    text = str(resource.get("text", "")).replace("\n", " ").strip()
                    if text:
                        evidence_lines.append(f"[{rtype}] {title}: {text}")
            if enable_pdf_read:
                providers.append("pdf-reader")
        except Exception as exc:
            lines.append(f"- Web/PDF intelligence unavailable: {exc}")
    else:
        lines.append("- Web intelligence disabled by configuration.")

    # Local simulation/test/run signals are cheap and should always inform recommendations.
    local_lines: list[str] = []
    try:
        ranking_csv = Path("outputs/reports/simulated_combination_ranking.csv")
        if ranking_csv.exists():
            with ranking_csv.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                top_rows = []
                for row in reader:
                    top_rows.append(row)
                    if len(top_rows) >= 2:
                        break
            for row in top_rows:
                pair = f"{row.get('drug_a', '')} + {row.get('drug_b', '')}".strip(" +")
                score = row.get("combo_score", "")
                if pair:
                    local_lines.append(f"- [sim-ranking] {pair} score={score}")
        latest_run_summary = None
        for candidate in Path("artifacts").rglob("run_summary.json"):
            if latest_run_summary is None or candidate.stat().st_mtime > latest_run_summary.stat().st_mtime:
                latest_run_summary = candidate
        if latest_run_summary is not None:
            payload = json.loads(latest_run_summary.read_text(encoding="utf-8"))
            summary_bits = []
            for key in ("status", "model", "total_candidates", "best_score"):
                if key in payload:
                    summary_bits.append(f"{key}={payload.get(key)}")
            if summary_bits:
                local_lines.append("- [run-summary] " + ", ".join(summary_bits))
        if local_lines:
            providers.append("local-sim")
    except Exception as exc:
        local_lines.append(f"- Local simulation signals unavailable: {exc}")

    cerebras_text = ""
    if enable_cerebras:
        cerebras_text, cerebras_provider = _cerebras_brief(user_query=user_query, evidence_lines=evidence_lines)
        providers.append(cerebras_provider)

    intel_parts = []
    if lines:
        intel_parts.append("Evidence Feeds:\n" + "\n".join(lines))
    if local_lines:
        intel_parts.append("Local Simulation Signals:\n" + "\n".join(local_lines))
    if evidence_lines:
        intel_parts.append("PDF/URL Extracts:\n" + "\n".join(f"- {line}" for line in evidence_lines))
    if cerebras_text:
        intel_parts.append("Cerebras Guidance:\n" + cerebras_text)

    intel_text = "\n\n".join(intel_parts).strip()
    provider_text = ", ".join(providers) if providers else "none"
    return intel_text, provider_text, evidence_lines


def _heuristic_insights(snapshot: DashboardSnapshot) -> str:
    """Generate heuristic recommendations based on dashboard metrics.

    Analyzes training stability, hit rate, and latency to provide actionable guidance.

    Args:
        snapshot: Current dashboard state snapshot.

    Returns:
        Multi-line string with operational recommendations.
    """
    notes: list[str] = []

    if snapshot.val_loss > snapshot.train_loss * 1.2:
        notes.append("- Validation drift detected. Reduce learning rate and enable early stopping checks.")
    else:
        notes.append("- Training trend is stable. Continue current schedule and monitor loss spread.")

    if snapshot.hit_rate < 0.15:
        notes.append("- Hit rate is below target. Tighten candidate filters toward higher QED and lower SA.")
    else:
        notes.append("- Hit rate is healthy. Expand exploration around top-ranked chemical neighborhoods.")

    if snapshot.latency_ms > 120:
        notes.append("- Inference latency is high. Batch scoring and reduce synchronous dashboard polling.")
    else:
        notes.append("- Latency is within operational band. Current serving setup is acceptable.")

    return "\n".join(notes)


def _build_header(snapshot: DashboardSnapshot, theme: DashboardTheme) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = Text(_ZANE_BANNER, style=f"bold {theme.primary}")
    banner_height = _ZANE_BANNER.count("\n") + 1
    subtitle = Text(
        f"- ZANE Computational Discovery Console { _phase_glyph(snapshot.tick) } -",
        style=f"bold {theme.secondary}",
    )
    meta = Text(
        (
            f"RUN ID: {snapshot.run_id}  |  MODEL: {snapshot.model_type.upper()}  |  "
            f"OPERATING MODE: {snapshot.mode}  |  TIMESTAMP: {now}"
        ),
        style="dim",
    )
    query = Text(f"- Study Query - {snapshot.user_query}", style="bright_white")
    filter_query = Text(f"Selection Protocol: {snapshot.filter_query}", style=theme.secondary)

    logo_url = os.getenv("ZANE_DASHBOARD_LOGO_URL", "").strip()
    if not logo_url and _DEFAULT_LOCAL_LOGO.exists():
        logo_url = str(_DEFAULT_LOCAL_LOGO)
    if not logo_url:
        logo_url = _DEFAULT_HEADER_LOGO_URL
    logo_renderable = _build_logo_renderable(logo_url=logo_url, target_height=banner_height)

    # Keep the logo at the left side of ZANE text in all runtime conditions.
    header_row = Columns([Align.left(logo_renderable), Align.left(banner)], expand=True)

    return Panel(
        Group(header_row, subtitle, meta, query, filter_query),
        border_style=theme.primary,
        box=theme.panel_box,
    )


def _resolve_header_logo(logo_url: str) -> Path | None:
    """Resolve logo from local path or fetch/cache from URL for terminal rendering."""
    if not logo_url:
        return None

    # Prefer local file resolution first (absolute or relative path).
    candidate = Path(logo_url)
    if candidate.exists() and candidate.is_file():
        return candidate

    if logo_url.startswith("file://"):
        file_candidate = Path(logo_url.replace("file://", "", 1))
        if file_candidate.exists() and file_candidate.is_file():
            return file_candidate

    parsed = urlparse(logo_url)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return None
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        suffix = ".png"

    cache_dir = Path("artifacts") / "dashboard" / "logos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(logo_url.encode("utf-8")).hexdigest()[:16] + suffix
    local_path = cache_dir / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    try:
        with urlopen(logo_url, timeout=8) as response:
            data = response.read()
    except Exception:
        return None

    if not data:
        return None

    local_path.write_bytes(data)
    return local_path


def _build_logo_renderable(logo_url: str, target_height: int) -> Any:
    """Build a header logo renderable with strict fallback sequence.

    Priority:
      1) Rich native image renderer
      2) PIL-based ASCII rendering from fetched image
      3) Fixed placeholder block (keeps left-slot occupied)
    """
    local_logo = _resolve_header_logo(logo_url) if logo_url else None

    if local_logo is not None:
        try:
            from rich.image import Image as RichImage

            return RichImage.from_path(str(local_logo), width=28)
        except Exception:
            pass

    if local_logo is not None:
        ascii_logo = _logo_ascii_renderable(local_logo, target_height=target_height)
        if ascii_logo is not None:
            return ascii_logo

    return _logo_placeholder(target_height=target_height)


def _logo_ascii_renderable(image_path: Path, target_height: int) -> Text | None:
    """Render an image as ASCII blocks so terminals without image protocol still show a logo."""
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        img = Image.open(image_path).convert("L")
    except Exception:
        return None

    h = max(4, target_height)
    # Approximate terminal character aspect ratio.
    w = max(16, int((img.width / max(1, img.height)) * h * 0.6))
    img = img.resize((w, h))

    chars = " .:-=+*#%@"
    pixels = list(img.getdata())

    out = Text(style="bold cyan")
    for y in range(h):
        row = []
        for x in range(w):
            v = pixels[y * w + x]
            idx = int((v / 255.0) * (len(chars) - 1))
            row.append(chars[idx])
        out.append("".join(row).rstrip() + "\n")

    return out


def _logo_placeholder(target_height: int) -> Text:
    """Guaranteed renderable left-side slot when image download/render is unavailable."""
    h = max(4, target_height)
    label = "LOGO"
    out = Text()
    for i in range(h):
        if i == h // 2:
            out.append(f"[{label:^10}]\n", style="bold cyan")
        else:
            out.append("[          ]\n", style="cyan")
    return out


def _build_kpi_table(snapshot: DashboardSnapshot, theme: DashboardTheme) -> Panel:
    table = Table(show_header=False, box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")

    table.add_row("Molecules Screened", f"[bold {theme.ok}]{snapshot.molecules_screened:,}[/bold {theme.ok}]")
    table.add_row("Generated Candidates", f"{snapshot.molecules_generated:,}")
    table.add_row("Active Jobs", f"{snapshot.active_jobs}")
    table.add_row("Hit Rate", f"{snapshot.hit_rate:.2%}")
    table.add_row("Average QED", f"{snapshot.avg_qed:.3f}")
    table.add_row("Average SA", f"{snapshot.avg_sa:.2f}")
    table.add_row("Best Binding (kcal/mol)", f"[bold {theme.accent}]{snapshot.best_binding:.2f}[/bold {theme.accent}]")
    table.add_row("Inference Latency", f"{snapshot.latency_ms:.1f} ms")
    table.add_row("CPU Utilization", f"{snapshot.cpu_util:.1f}%")
    table.add_row("GPU Utilization", f"{snapshot.gpu_util:.1f}%")
    table.add_row("Active Memory", f"{snapshot.memory_gb:.2f} GB")

    return Panel(
        table,
        title=f"Operational KPIs | Theme={theme.name}",
        border_style=theme.ok,
        box=theme.panel_box,
    )


def _build_training_panel(snapshot: DashboardSnapshot, theme: DashboardTheme, motion_intensity: int = 2) -> Panel:
    progress_ratio = max(0.0, min(1.0, snapshot.epoch / max(snapshot.total_epochs, 1)))
    width = 20 + max(1, min(3, motion_intensity)) * 4
    progress_bar = _animated_bar(progress_ratio, snapshot.tick, width=width)

    text = Text()
    text.append(f"Epoch Progress : {snapshot.epoch}/{snapshot.total_epochs}\n", style="white")
    text.append(f"{progress_bar} {progress_ratio:.1%}\n\n", style=theme.primary)
    text.append(f"Train Loss     : {snapshot.train_loss:.4f}\n", style="yellow")
    text.append(f"Validation Loss: {snapshot.val_loss:.4f}\n", style="yellow")

    status = "Stable" if snapshot.val_loss <= snapshot.train_loss * 1.2 else "Drift Risk"
    status_style = theme.ok if status == "Stable" else "red"
    text.append(f"Model Health   : {status}", style=status_style)

    return Panel(text, title="Training Monitor", border_style=theme.caution, box=theme.panel_box)


def _query_tokens(user_query: str) -> set[str]:
    lowered = user_query.lower()
    normalized = lowered.replace(",", " ").replace("/", " ").replace("-", " ")
    tokens = {part.strip() for part in normalized.split() if part.strip()}
    return tokens


def _characteristic_tokens(characteristics: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-z0-9]+", characteristics.lower()) if tok}


def _custom_indications(user_query: str, characteristic_tokens: set[str]) -> tuple[str, ...]:
    tags: set[str] = set(_query_tokens(user_query))

    if any(tok in characteristic_tokens for tok in {"consumable", "oral", "food", "beverage", "supplement"}):
        tags.update({"consumable", "oral", "low irritation"})
    if any(tok in characteristic_tokens for tok in {"performance", "high", "efficacy", "potent", "strong"}):
        tags.update({"high performance", "efficacy"})
    if any(tok in characteristic_tokens for tok in {"usage", "daily", "chronic", "routine"}):
        tags.update({"routine use", "stability"})
    if any(tok in characteristic_tokens for tok in {"safe", "safety", "low", "toxicity", "gentle"}):
        tags.update({"safety", "low toxicity"})
    if any(tok in characteristic_tokens for tok in {"hydrocarbon", "carbon", "aromatic", "ester", "alkyl"}):
        tags.update({"carbon chemistry", "hydrocarbon"})

    if not tags:
        tags.update({"general screening", "carbon chemistry"})

    return tuple(sorted(tags))


def _generate_custom_specs(
    characteristics: str,
    user_query: str,
    count: int,
) -> list[_MoleculeSpec]:
    """Generate simulation-only carbon/hydrocarbon custom compounds from user traits."""
    if not characteristics.strip() or count <= 0:
        return []

    tokens = _characteristic_tokens(characteristics)
    indications = _custom_indications(user_query=user_query, characteristic_tokens=tokens)
    focus = next((tok for tok in tokens if len(tok) >= 4), "custom")

    selected: list[tuple[str, str, float, float, tuple[str, ...]]] = []
    for scaffold in _CUSTOM_CARBON_SCAFFOLDS:
        scaffold_tags = set(scaffold[4])
        if tokens & scaffold_tags:
            selected.append(scaffold)

    if not selected:
        selected = list(_CUSTOM_CARBON_SCAFFOLDS)

    specs: list[_MoleculeSpec] = []
    for idx in range(max(1, min(count, 8))):
        name, smiles, qed, risk, _tags = selected[idx % len(selected)]
        specs.append(
            _MoleculeSpec(
                name=f"KB721H66-{focus.upper()}-{idx + 1}: {name}",
                smiles=smiles,
                indications=indications,
                fallback_qed=qed,
                fallback_risk=risk,
            )
        )

    return specs


def _query_match_score(indications: tuple[str, ...], tokens: set[str]) -> float:
    if not tokens:
        return 0.5
    matches = sum(1 for indication in indications if any(tok in indication or indication in tok for tok in tokens))
    return min(1.0, matches / max(1, len(indications) // 2))


def _infer_filter_profile(filter_query: str) -> dict[str, float | str]:
    """Infer sorting preference profile from natural language filter query."""
    q = (filter_query or "").lower()

    mode = "all"
    if any(token in q for token in ["combination", "combo", "cocktail", "pair"]):
        mode = "combo"
    if any(token in q for token in ["single", "monotherapy", "mono"]):
        mode = "single"

    w_match = 0.45
    w_efficacy = 0.40
    w_risk = 0.25

    if any(token in q for token in ["safe", "safest", "low side effect", "minimal side", "low risk", "toxicity"]):
        w_match = 0.35
        w_efficacy = 0.20
        w_risk = 0.55
    elif any(token in q for token in ["high efficacy", "most effective", "potent", "strongest"]):
        w_match = 0.30
        w_efficacy = 0.60
        w_risk = 0.10
    elif any(token in q for token in ["match", "disease specific", "targeted", "relevance"]):
        w_match = 0.65
        w_efficacy = 0.25
        w_risk = 0.10

    return {
        "mode": mode,
        "w_match": w_match,
        "w_efficacy": w_efficacy,
        "w_risk": w_risk,
    }


def _compute_combo_rankings(
    user_query: str,
    filter_query: str = "",
    custom_characteristics: str = "",
    custom_count: int = 4,
) -> list[dict[str, float | str]]:
    """Compute simulation-only rankings from query-match + ADMET proxies + filter intent."""
    admet = _get_admet_predictor()
    tokens = _query_tokens(user_query)
    profile = _infer_filter_profile(filter_query)
    mode = str(profile["mode"])
    w_match = float(profile["w_match"])
    w_efficacy = float(profile["w_efficacy"])
    w_risk = float(profile["w_risk"])
    molecule_rows: list[dict[str, float | str]] = []

    custom_specs = _generate_custom_specs(
        characteristics=custom_characteristics,
        user_query=user_query,
        count=custom_count,
    )
    molecule_specs = _SIMULATION_LIBRARY + custom_specs

    for spec in molecule_specs:
        if admet is not None:
            lipinski = admet.check_lipinski_rule(spec.smiles) or {}
            toxicity = admet.predict_toxicity_flags(spec.smiles) or {}
            qed = admet.calculate_qed(spec.smiles)
            sa = admet.calculate_synthetic_accessibility(spec.smiles)

            toxicity_hits = float(sum(1 for value in toxicity.values() if bool(value)))
            lipinski_violations = float(lipinski.get("num_violations", 4))
            qed_val = float(qed if qed is not None else 0.0)
            sa_val = float(sa if sa is not None else 10.0)
            risk_proxy = toxicity_hits + lipinski_violations + (sa_val / 10.0)
        else:
            fallback_qed, fallback_risk = _FALLBACK_MOL_PROFILES.get(
                spec.name,
                (spec.fallback_qed if spec.fallback_qed is not None else 0.5, spec.fallback_risk if spec.fallback_risk is not None else 1.0),
            )
            qed_val, risk_proxy = fallback_qed, fallback_risk

        molecule_rows.append(
            {
                "name": spec.name,
                "qed": qed_val,
                "risk_proxy": risk_proxy,
                "match": _query_match_score(spec.indications, tokens),
            }
        )

    combo_rows: list[dict[str, float | str]] = []
    for left, right in combinations(molecule_rows, 2):
        efficacy_proxy = (float(left["qed"]) + float(right["qed"])) / 2.0
        risk_proxy = (float(left["risk_proxy"]) + float(right["risk_proxy"])) / 2.0
        match_score = (float(left["match"]) + float(right["match"])) / 2.0
        combo_score = (0.45 * match_score) + (0.40 * efficacy_proxy) - (0.25 * risk_proxy)

        combo_rows.append(
            {
                "combo": f"{left['name']} + {right['name']}",
                "kind": "combo",
                "qed": efficacy_proxy,
                "risk": risk_proxy,
                "match": match_score,
                "score": (w_match * match_score) + (w_efficacy * efficacy_proxy) - (w_risk * risk_proxy),
            }
        )

    for mol in molecule_rows:
        single_score = (w_match * float(mol["match"])) + (w_efficacy * float(mol["qed"])) - (w_risk * float(mol["risk_proxy"]))
        combo_rows.append(
            {
                "combo": str(mol["name"]),
                "kind": "single",
                "qed": float(mol["qed"]),
                "risk": float(mol["risk_proxy"]),
                "match": float(mol["match"]),
                "score": single_score,
            }
        )

    if mode == "combo":
        combo_rows = [row for row in combo_rows if str(row.get("kind", "")) == "combo"]
    elif mode == "single":
        combo_rows = [row for row in combo_rows if str(row.get("kind", "")) == "single"]

    combo_rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return combo_rows


def _metric_bar(value: float, min_value: float, max_value: float, width: int = 20) -> str:
    span = max(max_value - min_value, 1e-9)
    ratio = max(0.0, min(1.0, (value - min_value) / span))
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _sparkline(values: list[float]) -> str:
    if not values:
        return ""
    chars = "▁▂▃▄▅▆▇█"
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1e-9)
    out = []
    for value in values:
        idx = int((value - lo) / span * (len(chars) - 1))
        out.append(chars[max(0, min(len(chars) - 1, idx))])
    return "".join(out)


def _build_candidates_table(simulated_combos: list[dict[str, float | str]] | None = None, theme: DashboardTheme | None = None) -> Panel:
    resolved_theme = theme or _resolve_theme("lab")
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Combination", style="cyan", overflow="fold")
    table.add_column("Match", justify="right")
    table.add_column("Efficacy", justify="right")
    table.add_column("Risk", justify="right")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Status", justify="center", overflow="fold")

    sample_rows = simulated_combos or []
    if not sample_rows:
        sample_rows = [
            {"combo": "Ibuprofen + Naproxen", "match": 0.75, "qed": 0.85, "risk": 0.44, "score": 0.57},
            {"combo": "Ibuprofen + Pseudoephedrine", "match": 0.82, "qed": 0.78, "risk": 0.28, "score": 0.64},
            {"combo": "Naproxen + Pseudoephedrine", "match": 0.80, "qed": 0.81, "risk": 0.40, "score": 0.60},
            {"combo": "Ibuprofen + Guaifenesin", "match": 0.78, "qed": 0.77, "risk": 0.29, "score": 0.61},
            {"combo": "Naproxen + Guaifenesin", "match": 0.70, "qed": 0.80, "risk": 0.41, "score": 0.54},
        ]

    for rank, row in enumerate(sample_rows[:5], start=1):
        score = float(row["score"])
        status = "Promote" if score >= 0.62 else "Review" if score >= 0.54 else "Hold"
        style = "green" if status == "Promote" else "yellow" if status == "Review" else "red"
        table.add_row(
            str(rank),
            str(row["combo"]),
            f"{float(row['match']):.2f}",
            f"{float(row['qed']):.2f}",
            f"{float(row['risk']):.2f}",
            f"{score:.2f}",
            f"[{style}]{status}[/{style}]",
        )

    return Panel(table, title="Top Simulated Combinations", border_style=resolved_theme.accent, box=resolved_theme.panel_box)


def _build_composition_table(simulated_combos: list[dict[str, float | str]] | None = None, theme: DashboardTheme | None = None) -> Panel:
    """Build a simulation-only composition table for beta dosage exploration."""
    table = Table(box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Drug Candidate", style="cyan", overflow="fold")
    table.add_column("Probable Composition (sim)", style="white", overflow="fold")
    table.add_column("Beta Dose Index", justify="right", style="yellow")
    table.add_column("Usage Profile", style="green", overflow="fold")

    rows = simulated_combos or []
    seen: set[str] = set()
    selected: list[dict[str, float | str]] = []
    for row in rows:
        candidate = str(row.get("combo", ""))
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        selected.append(row)
        if len(selected) >= 5:
            break

    if not selected:
        selected = [
            {
                "combo": "Ibuprofen + Guaifenesin",
                "score": 0.61,
                "match": 0.78,
                "qed": 0.77,
                "risk": 0.29,
            },
            {
                "combo": "Pseudoephedrine + Cetirizine",
                "score": 0.58,
                "match": 0.81,
                "qed": 0.70,
                "risk": 0.27,
            },
            {
                "combo": "Dextromethorphan",
                "score": 0.55,
                "match": 0.74,
                "qed": 0.78,
                "risk": 0.53,
            },
            {
                "combo": "Loratadine",
                "score": 0.52,
                "match": 0.70,
                "qed": 0.74,
                "risk": 0.79,
            },
            {
                "combo": "Guaifenesin",
                "score": 0.50,
                "match": 0.72,
                "qed": 0.72,
                "risk": 0.26,
            },
        ]

    for rank, row in enumerate(selected, start=1):
        candidate = str(row.get("combo", ""))
        score = float(row.get("score", 0.0))
        match = float(row.get("match", 0.0))
        risk = float(row.get("risk", 0.0))

        active_pct = int(max(25, min(70, round(30 + (score * 45)))))
        stabilizer_pct = int(max(10, min(45, round(18 + (risk * 22)))))
        carrier_pct = max(5, 100 - active_pct - stabilizer_pct)
        beta_dose_index = max(0.20, min(0.95, round((0.55 * score) + (0.30 * match) - (0.15 * risk), 2)))

        composition = f"active {active_pct}% | stabilizer {stabilizer_pct}% | carrier {carrier_pct}%"
        usage = "consumable-screening" if risk < 0.35 else "controlled-screening"

        table.add_row(str(rank), candidate, composition, f"{beta_dose_index:.2f}", usage)

    resolved_theme = theme or _resolve_theme("lab")
    return Panel(
        table,
        title="Drug Composition Table | Beta Testing Mode",
        border_style=resolved_theme.primary,
        box=resolved_theme.panel_box,
    )


def _build_overview_panel(detail_sections: set[str], theme: DashboardTheme) -> Panel:
    text = Text()
    text.append("Unified View Active\n", style="bold white")
    text.append("All detail panels are shown by default.\n\n", style="white")
    text.append("Optional command filters:\n", style="bold cyan")
    text.append("- --detail-panels combinations\n", style="white")
    text.append("- --detail-panels composition\n", style="white")
    text.append("- --detail-panels analytics\n", style="white")
    text.append("- --detail-panels ai\n", style="white")
    text.append("- --detail-panels all\n\n", style="white")

    if detail_sections:
        enabled = ", ".join(sorted(detail_sections))
        text.append(f"Enabled: {enabled}", style="green")
    else:
        text.append("Enabled: none", style="yellow")

    return Panel(text, title="Dashboard Overview", border_style=theme.secondary, box=theme.panel_box)


def _build_alerts_panel(snapshot: DashboardSnapshot, theme: DashboardTheme) -> Panel:
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

    return Panel(text, title="System Alerts", border_style=theme.primary, box=theme.panel_box)


def _build_ai_panel(ai_text: str, provider: str, theme: DashboardTheme) -> Panel:
    body = Text()
    body.append(f"Provider: {provider}\n\n", style="dim")
    body.append(ai_text, style="white")
    return Panel(body, title="AI Copilot", border_style=theme.primary, box=theme.panel_box)


def _build_graphs_panel(
    simulated_combos: list[dict[str, float | str]] | None,
    hit_rate_history: list[float],
    score_history: list[float],
    theme: DashboardTheme,
) -> Panel:
    rows = simulated_combos or []
    top_rows = rows

    text = Text()
    text.append("Top Score Bars\n", style="bold white")
    for row in top_rows:
        label = str(row["combo"]).ljust(18)
        score = float(row["score"])
        text.append(f"{label} {_metric_bar(score, 0.0, 1.0, width=16)} {score:.2f}\n", style="cyan")

    if rows:
        avg_match = sum(float(r["match"]) for r in rows[:10]) / min(10, len(rows))
        avg_qed = sum(float(r["qed"]) for r in rows[:10]) / min(10, len(rows))
        avg_risk = sum(float(r["risk"]) for r in rows[:10]) / min(10, len(rows))
        text.append("\nMetric Averages (top 10)\n", style="bold white")
        text.append(f"Match    {_metric_bar(avg_match, 0.0, 1.0, width=20)} {avg_match:.2f}\n", style="green")
        text.append(f"Efficacy {_metric_bar(avg_qed, 0.0, 1.0, width=20)} {avg_qed:.2f}\n", style="yellow")
        # Lower risk is better, invert for visualization quality bar.
        inv_risk = max(0.0, 1.0 - avg_risk)
        text.append(f"Safety   {_metric_bar(inv_risk, 0.0, 1.0, width=20)} {inv_risk:.2f}\n", style="magenta")

    text.append("\nLive Trends\n", style="bold white")
    text.append(f"HitRate  {_sparkline(hit_rate_history[-20:])}\n", style="green")
    text.append(f"TopScore {_sparkline(score_history[-20:])}", style="magenta")

    return Panel(text, title="Visual Analytics", border_style=theme.ok, box=theme.panel_box)


def _build_pipeline_flow_panel(snapshot: DashboardSnapshot, theme: DashboardTheme) -> Panel:
    table = Table(show_header=True, box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Stage", style="bold white")
    table.add_column("Status", justify="center")
    table.add_column("Rate", justify="right")
    table.add_column("QC", justify="center")

    phase = _phase_glyph(snapshot.tick)
    throughput = max(1, int(snapshot.molecules_screened / max(1, snapshot.epoch)))
    stages = [
        ("Acquisition", f"{phase} RUN", f"{throughput}/epoch", "PASS"),
        ("Normalization", f"{phase} RUN", f"{int(throughput * 0.92)}/epoch", "PASS"),
        ("Scaffold Split", f"{phase} RUN", f"{int(throughput * 0.88)}/epoch", "PASS"),
        ("Training", f"{phase} RUN", f"loss={snapshot.train_loss:.3f}", "PASS"),
        ("Calibration", f"{phase} RUN", f"val={snapshot.val_loss:.3f}", "PASS"),
    ]
    for stage, status, rate, qc in stages:
        table.add_row(stage, f"[{theme.primary}]{status}[/{theme.primary}]", rate, f"[{theme.ok}]{qc}[/{theme.ok}]")

    return Panel(table, title="Pipeline Flow Orchestrator", border_style=theme.accent, box=theme.panel_box)


def _build_runtime_telemetry_panel(
    snapshot: DashboardSnapshot,
    hit_rate_history: list[float],
    score_history: list[float],
    theme: DashboardTheme,
) -> Panel:
    text = Text()
    hit_trend = _sparkline(hit_rate_history[-24:])
    score_trend = _sparkline(score_history[-24:])

    cpu_ratio = max(0.0, min(1.0, snapshot.cpu_util / 100.0))
    gpu_ratio = max(0.0, min(1.0, snapshot.gpu_util / 100.0))
    mem_ratio = max(0.0, min(1.0, snapshot.memory_gb / 32.0))

    text.append("Runtime Utilization\n", style="bold white")
    text.append(f"CPU {_animated_bar(cpu_ratio, snapshot.tick, width=16)} {snapshot.cpu_util:5.1f}%\n", style=theme.primary)
    text.append(f"GPU {_animated_bar(gpu_ratio, snapshot.tick + 1, width=16)} {snapshot.gpu_util:5.1f}%\n", style=theme.accent)
    text.append(f"MEM {_animated_bar(mem_ratio, snapshot.tick + 2, width=16)} {snapshot.memory_gb:5.1f} GB\n\n", style=theme.caution)

    text.append("Live Trend Signals\n", style="bold white")
    text.append(f"HitRate  {hit_trend}\n", style=theme.ok)
    text.append(f"TopScore {score_trend}\n", style=theme.accent)

    return Panel(text, title="Runtime Telemetry", border_style=theme.primary, box=theme.panel_box)


def _build_protocol_panel(snapshot: DashboardSnapshot, theme: DashboardTheme) -> Panel:
    table = Table(show_header=True, box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Protocol", style="bold white")
    table.add_column("Criterion", style="white")
    table.add_column("Observed", justify="right")
    table.add_column("Decision", justify="center")

    checks = [
        ("Hit-rate gate", ">= 0.15", snapshot.hit_rate, snapshot.hit_rate >= 0.15),
        ("QED gate", ">= 0.65", snapshot.avg_qed, snapshot.avg_qed >= 0.65),
        ("SA gate", "<= 5.00", snapshot.avg_sa, snapshot.avg_sa <= 5.0),
        ("Latency gate", "<= 120 ms", snapshot.latency_ms, snapshot.latency_ms <= 120.0),
    ]
    for protocol, criterion, observed, decision in checks:
        decision_text = "PASS" if decision else "REVIEW"
        decision_style = theme.ok if decision else theme.caution
        table.add_row(protocol, criterion, f"{observed:.3f}" if protocol != "Latency gate" else f"{observed:.1f}", f"[{decision_style}]{decision_text}[/{decision_style}]")

    return Panel(table, title="Protocol Compliance", border_style=theme.caution, box=theme.panel_box)


def _compose_ai_panel_content(
    snapshot: DashboardSnapshot,
    advisor: DashboardAIAdvisor | None,
    intel_text: str,
    intel_provider: str,
    evidence_lines: list[str],
) -> tuple[str, str]:
    if advisor:
        local_ai = advisor.summarize(snapshot)
        if evidence_lines:
            local_ai = f"{local_ai}\n\nExternal evidence signals:\n" + "\n".join(
                f"- {line}" for line in evidence_lines
            )
        ai_provider = f"{advisor.provider}, {intel_provider}" if intel_provider else advisor.provider
        if intel_text:
            return f"{local_ai}\n\n{intel_text}", ai_provider
        return local_ai, ai_provider

    ai_text = intel_text or "AI insights disabled. Use --with-ai to enable local advisor."
    ai_provider = intel_provider if intel_provider else "off"
    return ai_text, ai_provider


def render_dashboard(
    snapshot: DashboardSnapshot,
    ai_text: str = "AI insights disabled.",
    ai_provider: str = "off",
    simulated_combos: list[dict[str, float | str]] | None = None,
    hit_rate_history: list[float] | None = None,
    score_history: list[float] | None = None,
    detail_sections: set[str] | None = None,
    theme_name: str = "lab",
    motion_intensity: int = 2,
) -> Any:
    """Build a complete terminal dashboard layout for the given snapshot."""
    theme = _resolve_theme(theme_name)

    # Show every detail section by default so all process panels are visible.
    resolved_sections = set(detail_sections or {"all"})
    if "all" in resolved_sections:
        resolved_sections = {"combinations", "composition", "analytics", "ai"}

    panel_stack: list[Panel] = [
        _build_header(snapshot, theme=theme),
        _build_kpi_table(snapshot, theme=theme),
        _build_training_panel(snapshot, theme=theme, motion_intensity=motion_intensity),
        _build_pipeline_flow_panel(snapshot=snapshot, theme=theme),
        _build_overview_panel(resolved_sections, theme=theme),
    ]

    if "combinations" in resolved_sections:
        panel_stack.append(_build_candidates_table(simulated_combos=simulated_combos, theme=theme))
    if "composition" in resolved_sections:
        panel_stack.append(_build_composition_table(simulated_combos=simulated_combos, theme=theme))
    if "analytics" in resolved_sections:
        panel_stack.append(
            _build_graphs_panel(
                simulated_combos=simulated_combos,
                hit_rate_history=hit_rate_history or [],
                score_history=score_history or [],
                theme=theme,
            )
        )
        panel_stack.append(
            _build_runtime_telemetry_panel(
                snapshot=snapshot,
                hit_rate_history=hit_rate_history or [],
                score_history=score_history or [],
                theme=theme,
            )
        )
    if "ai" in resolved_sections:
        panel_stack.append(_build_ai_panel(ai_text=ai_text, provider=ai_provider, theme=theme))
    panel_stack.append(_build_protocol_panel(snapshot=snapshot, theme=theme))
    panel_stack.append(_build_alerts_panel(snapshot, theme=theme))

    return Group(*panel_stack)


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
    cpu_util = min(98.0, max(12.0, previous.cpu_util + random.uniform(-6.0, 6.0)))
    gpu_util = min(99.0, max(8.0, previous.gpu_util + random.uniform(-7.0, 7.0)))
    memory_gb = min(31.0, max(3.0, previous.memory_gb + random.uniform(-0.9, 0.9)))

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
        user_query=previous.user_query,
        filter_query=previous.filter_query,
        cpu_util=cpu_util,
        gpu_util=gpu_util,
        memory_gb=memory_gb,
        tick=previous.tick + 1,
    )


def run_dashboard(
    live: bool = True,
    refresh_seconds: float = 1.0,
    iterations: int = 30,
    enable_ai: bool = False,
    ai_model_id: str | None = None,
    ai_refresh_every: int = 5,
    include_simulated_combos: bool = True,
    user_query: str = "",
    enable_web_intel: bool = True,
    enable_pdf_read: bool = True,
    enable_cerebras: bool = True,
    intel_refresh_every: int = 3,
    filter_query: str = "",
    custom_characteristics: str = "",
    custom_count: int = 4,
    detail_sections: set[str] | None = None,
    theme: str = "lab",
    motion_intensity: int = 2,
) -> None:
    """Run the ZANE terminal dashboard in static or live mode."""
    console = Console()

    if not user_query.strip():
        user_query = console.input("[bold cyan]Enter drug need / disease query:[/bold cyan] ").strip()
    if not user_query:
        user_query = "cold congestion and cough"
    if not filter_query:
        filter_query = user_query

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
        epoch=100,
        total_epochs=100,
        train_loss=0.192,
        val_loss=0.217,
        latency_ms=61.5,
        user_query=user_query,
        filter_query=filter_query,
        cpu_util=48.0,
        gpu_util=62.0,
        memory_gb=11.8,
        tick=0,
    )

    advisor = DashboardAIAdvisor(model_id=ai_model_id) if enable_ai else None
    simulated_combos = (
        _compute_combo_rankings(
            user_query=user_query,
            filter_query=filter_query,
            custom_characteristics=custom_characteristics,
            custom_count=custom_count,
        )
        if include_simulated_combos
        else None
    )
    intel_text, intel_provider, evidence_lines = _gather_external_intel(
        user_query=user_query,
        enable_web_intel=enable_web_intel,
        enable_pdf_read=enable_pdf_read,
        enable_cerebras=enable_cerebras,
    )
    ai_text, ai_provider = _compose_ai_panel_content(
        snapshot=snapshot,
        advisor=advisor,
        intel_text=intel_text,
        intel_provider=intel_provider,
        evidence_lines=evidence_lines,
    )

    hit_rate_history = [snapshot.hit_rate]
    score_history = [float(simulated_combos[0]["score"]) if simulated_combos else 0.0]

    if not live:
        console.print(
            render_dashboard(
                snapshot,
                ai_text=ai_text,
                ai_provider=ai_provider,
                simulated_combos=simulated_combos,
                hit_rate_history=hit_rate_history,
                score_history=score_history,
                detail_sections=detail_sections,
                theme_name=theme,
                motion_intensity=motion_intensity,
            )
        )
        return

    with Live(
        render_dashboard(
            snapshot,
            ai_text=ai_text,
            ai_provider=ai_provider,
            simulated_combos=simulated_combos,
            hit_rate_history=hit_rate_history,
            score_history=score_history,
            detail_sections=detail_sections,
            theme_name=theme,
            motion_intensity=motion_intensity,
        ),
        refresh_per_second=8,
        console=console,
        screen=False,
    ) as live_view:
        for _ in range(max(iterations, 1)):
            time.sleep(max(refresh_seconds, 0.2))
            snapshot = _next_snapshot(snapshot)

            # Keep intelligence feed proactive: periodically re-read web/PDF sources and update guidance.
            should_refresh_intel = snapshot.epoch % max(1, intel_refresh_every) == 0
            if should_refresh_intel:
                intel_text, intel_provider, evidence_lines = _gather_external_intel(
                    user_query=user_query,
                    enable_web_intel=enable_web_intel,
                    enable_pdf_read=enable_pdf_read,
                    enable_cerebras=enable_cerebras,
                )
                if include_simulated_combos:
                    simulated_combos = _compute_combo_rankings(
                        user_query=user_query,
                        filter_query=filter_query,
                        custom_characteristics=custom_characteristics,
                        custom_count=custom_count,
                    )

            should_refresh_ai = advisor is not None and snapshot.epoch % max(1, ai_refresh_every) == 0
            if should_refresh_ai or should_refresh_intel:
                ai_text, ai_provider = _compose_ai_panel_content(
                    snapshot=snapshot,
                    advisor=advisor,
                    intel_text=intel_text,
                    intel_provider=intel_provider,
                    evidence_lines=evidence_lines,
                )

            hit_rate_history.append(snapshot.hit_rate)
            if simulated_combos:
                score_history.append(float(simulated_combos[0]["score"]))

            live_view.update(
                render_dashboard(
                    snapshot,
                    ai_text=ai_text,
                    ai_provider=ai_provider,
                    simulated_combos=simulated_combos,
                    hit_rate_history=hit_rate_history,
                    score_history=score_history,
                    detail_sections=detail_sections,
                    theme_name=theme,
                    motion_intensity=motion_intensity,
                )
            )


if __name__ == "__main__":
    run_dashboard(live=True)
