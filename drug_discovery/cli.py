"""
Command-line interface for Drug Discovery Platform
"""

import argparse
import json
import sys

from drug_discovery.dashboard import run_dashboard


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="AI Drug Discovery Platform CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict properties for a molecule")
    predict_parser.add_argument("smiles", help="SMILES string of the molecule")
    predict_parser.add_argument("--model", default="gnn", choices=["gnn", "transformer", "ensemble"])
    predict_parser.add_argument("--checkpoint", help="Path to model checkpoint")

    # ADMET command
    admet_parser = subparsers.add_parser("admet", help="Analyze ADMET properties")
    admet_parser.add_argument("smiles", help="SMILES string of the molecule")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--model", default="gnn", choices=["gnn", "transformer", "ensemble"])
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--split-strategy", default="random", choices=["random", "scaffold"])
    train_parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default: auto)")

    # Collect data command
    collect_parser = subparsers.add_parser("collect", help="Collect molecular data")
    collect_parser.add_argument(
        "--sources",
        nargs="+",
        default=["pubchem", "chembl"],
        choices=["pubchem", "chembl", "approved_drugs", "drugbank"],
    )
    collect_parser.add_argument("--limit", type=int, default=1000)
    collect_parser.add_argument("--drugbank-file", default=None, help="Path to DrugBank CSV/TSV export")

    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        aliases=["start", "go"],
        help="Show professional ZANE terminal dashboard",
    )
    dashboard_parser.add_argument("--static", action="store_true", help="Render one static dashboard frame")
    dashboard_parser.add_argument("--refresh", type=float, default=1.0, help="Live refresh interval in seconds")
    dashboard_parser.add_argument("--iterations", type=int, default=30, help="Number of live refresh cycles")
    dashboard_parser.add_argument("--with-ai", action="store_true", help="Enable AI copilot insights panel")
    dashboard_parser.add_argument(
        "--ai-model-id",
        default="artifacts/llama/tinyllama-1.1b-chat",
        help="Local or remote model id/path for dashboard AI suggestions",
    )
    dashboard_parser.add_argument(
        "--ai-refresh-every",
        type=int,
        default=5,
        help="Refresh AI recommendations every N epochs",
    )
    dashboard_parser.add_argument(
        "--intel-refresh-every",
        type=int,
        default=3,
        help="Re-read web/PDF/Cerebras intelligence every N epochs in live mode",
    )
    dashboard_parser.add_argument(
        "--no-sim-combos",
        action="store_true",
        help="Disable simulated drug-combination panel in dashboard",
    )
    dashboard_parser.add_argument(
        "--query",
        default="",
        help="Natural-language disease/need query used to rank simulated dashboard candidates",
    )
    dashboard_parser.add_argument(
        "--filter-query",
        default="",
        help="Natural-language ranking preference (e.g. 'safest combos' or 'highest efficacy single drug')",
    )
    dashboard_parser.add_argument(
        "--interactive-query",
        action="store_true",
        help="Prompt for disease/need query before rendering dashboard",
    )
    dashboard_parser.add_argument(
        "--no-web-intel",
        action="store_true",
        help="Disable in-dashboard web searching/scraping for query evidence",
    )
    dashboard_parser.add_argument(
        "--no-pdf-intel",
        action="store_true",
        help="Disable in-dashboard PDF/URL resource reading",
    )
    dashboard_parser.add_argument(
        "--no-cerebras",
        action="store_true",
        help="Disable in-dashboard Cerebras API guidance",
    )
    dashboard_parser.add_argument(
        "--guided",
        action="store_true",
        help="Launch dashboard with easy step-by-step prompts",
    )
    dashboard_parser.add_argument(
        "--custom-characteristics",
        default="",
        help=(
            "Simulation-only custom compound characteristics (e.g. 'consumable hydrocarbon high performance'). "
            "Used to generate virtual carbon/hydrocarbon candidates."
        ),
    )
    dashboard_parser.add_argument(
        "--custom-count",
        type=int,
        default=4,
        help="Number of custom virtual compounds to generate from characteristics (1-8).",
    )
    dashboard_parser.add_argument(
        "--detail-panels",
        nargs="+",
        choices=["combinations", "composition", "analytics", "ai", "all"],
        default=[],
        help=(
            "Show detailed dashboard sections on demand. "
            "Example: --detail-panels analytics ai (default is simple overview)."
        ),
    )
    dashboard_parser.add_argument(
        "--theme",
        default="lab",
        choices=["lab", "neon", "classic"],
        help="Dashboard color theme preset.",
    )
    dashboard_parser.add_argument(
        "--motion-intensity",
        type=int,
        default=2,
        help="Animation intensity level (1-3).",
    )

    # AI support command (Meta Llama)
    support_parser = subparsers.add_parser("assist", help="Use Meta Llama for AI support")
    support_parser.add_argument("prompt", help="Question or task for the AI assistant")
    support_parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Hugging Face model id (default: Meta Llama 3.2 1B Instruct)",
    )
    support_parser.add_argument("--context", default=None, help="Optional context string")
    support_parser.add_argument("--max-new-tokens", type=int, default=256)
    support_parser.add_argument("--temperature", type=float, default=0.7)
    support_parser.add_argument("--top-p", type=float, default=0.9)

    # Synthesis research command (internet + AI)
    synth_parser = subparsers.add_parser(
        "synthesis-research", help="Plan synthesis with internet research and AI guidance"
    )
    synth_parser.add_argument("smiles", help="Target molecule SMILES")
    synth_parser.add_argument("--target", default=None, help="Optional target protein")
    synth_parser.add_argument("--max-depth", type=int, default=5)
    synth_parser.add_argument("--max-results", type=int, default=5)
    synth_parser.add_argument("--max-resource-reads", type=int, default=3)
    synth_parser.add_argument("--no-internet", action="store_true", help="Disable internet research")
    synth_parser.add_argument("--no-ai", action="store_true", help="Disable AI synthesis guidance")
    synth_parser.add_argument("--no-resource-read", action="store_true", help="Disable URL/PDF reading")

    boltzgen_parser = subparsers.add_parser("boltzgen", help="Run BoltzGen binder design pipeline")
    boltzgen_parser.add_argument("spec", help="Path to a BoltzGen design specification YAML")
    boltzgen_parser.add_argument(
        "--output", default="outputs/boltzgen/run", help="Output directory for BoltzGen artifacts"
    )
    boltzgen_parser.add_argument(
        "--protocol",
        default="protein-anything",
        choices=[
            "protein-anything",
            "peptide-anything",
            "protein-small_molecule",
            "nanobody-anything",
            "antibody-anything",
            "protein-redesign",
        ],
        help="BoltzGen protocol to use",
    )
    boltzgen_parser.add_argument("--num-designs", type=int, default=50, help="Number of intermediate designs to generate")
    boltzgen_parser.add_argument("--budget", type=int, default=10, help="Final number of designs after filtering")
    boltzgen_parser.add_argument("--steps", nargs="+", default=None, help="Optional subset of BoltzGen steps to run")
    boltzgen_parser.add_argument("--devices", type=int, default=None, help="Number of devices to request")
    boltzgen_parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse intermediate files when present to avoid regenerating designs",
    )
    boltzgen_parser.add_argument("--cache-dir", default=None, help="Cache directory for BoltzGen downloads")
    boltzgen_parser.add_argument("--top-k", type=int, default=5, help="Number of designs to show in summary output")
    boltzgen_parser.add_argument(
        "--score-key",
        default=None,
        help="Optional metric key to sort the summary (e.g., refolding_rmsd or filter_rank)",
    )

    args = parser.parse_args()

    if args.command == "predict":
        predict_properties(args)
    elif args.command == "admet":
        analyze_admet(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "collect":
        collect_data(args)
    elif args.command in {"dashboard", "start", "go"}:
        show_dashboard(args)
    elif args.command == "assist":
        run_ai_support(args)
    elif args.command == "synthesis-research":
        run_synthesis_research(args)
    elif args.command == "boltzgen":
        run_boltzgen(args)
    else:
        parser.print_help()


def predict_properties(args):
    """Predict molecular properties"""
    from drug_discovery import DrugDiscoveryPipeline

    print(f"Predicting properties for: {args.smiles}")

    if args.checkpoint:
        pipeline = DrugDiscoveryPipeline(model_type=args.model)
        pipeline.load(args.checkpoint)
        properties = pipeline.predict_properties(args.smiles)

        print("\nPredicted Properties:")
        for key, value in properties.items():
            print(f"  {key}: {value}")
    else:
        print("Error: --checkpoint required for prediction")
        sys.exit(1)


def analyze_admet(args):
    """Analyze ADMET properties"""
    from drug_discovery.evaluation import ADMETPredictor

    print(f"Analyzing ADMET for: {args.smiles}")

    admet = ADMETPredictor()

    # Lipinski
    lipinski = admet.check_lipinski_rule(args.smiles)
    if lipinski:
        print(f"\nLipinski's Rule: {'PASS' if lipinski['passes'] else 'FAIL'}")
        print(f"Violations: {lipinski['num_violations']}")

    # QED
    qed = admet.calculate_qed(args.smiles)
    if qed:
        print(f"\nDrug-likeness (QED): {qed:.3f}")

    # SA
    sa = admet.calculate_synthetic_accessibility(args.smiles)
    if sa:
        print(f"Synthetic Accessibility: {sa:.2f}/10")


def train_model(args):
    """Train a new model"""
    from drug_discovery import DrugDiscoveryPipeline
    from drug_discovery.utils import set_seed

    print(f"Training {args.model} model...")
    set_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    pipeline = DrugDiscoveryPipeline(model_type=args.model)

    # Collect data
    data = pipeline.collect_data()

    # Prepare datasets
    train_loader, test_loader = pipeline.prepare_datasets(
        data,
        batch_size=args.batch_size,
        seed=args.seed,
        split_strategy=args.split_strategy,
        num_workers=args.num_workers,
    )

    # Train
    pipeline.train(train_loader, test_loader, num_epochs=args.epochs)

    # Save
    pipeline.save(f"./checkpoints/{args.model}_model.pt")
    print(f"\nModel saved to ./checkpoints/{args.model}_model.pt")


def collect_data(args):
    """Collect molecular data"""
    print(f"Collecting data from: {', '.join(args.sources)}")

    from drug_discovery.data import DataCollector

    collector = DataCollector()
    data = collector.merge_datasets([])

    for source in args.sources:
        if source == "pubchem":
            df = collector.collect_from_pubchem(limit=args.limit)
        elif source == "chembl":
            df = collector.collect_from_chembl(limit=args.limit)
        elif source == "approved_drugs":
            df = collector.collect_approved_drugs()
        elif source == "drugbank":
            df = collector.collect_from_drugbank(file_path=args.drugbank_file, limit=args.limit)
        else:
            continue

        data = collector.merge_datasets([data, df])

    print(f"\nCollected {len(data)} unique molecules")

    # Save
    data.to_csv("./data/collected_data.csv", index=False)
    print("Saved to ./data/collected_data.csv")


def show_dashboard(args):
    """Display ZANE terminal dashboard."""
    query = "" if args.interactive_query else args.query

    if args.guided:
        print("\nZANE Guided Dashboard Setup")
        print("Press Enter to accept defaults shown in [brackets].\n")

        need = input("What drug need/disease are you exploring? [cold cough congestion]: ").strip()
        query = need or "cold cough congestion"

        filt = input(
            "How should candidates be sorted? [safest combinations with minimal side effects]: "
        ).strip()
        filter_query = filt or "safest combinations with minimal side effects"

        live_answer = input("Run live dashboard updates? [Y/n]: ").strip().lower()
        live_mode = live_answer not in {"n", "no"}

        with_ai_answer = input("Enable local AI copilot (can be heavier)? [y/N]: ").strip().lower()
        with_ai = with_ai_answer in {"y", "yes"}

        web_answer = input("Enable web search + website reading? [Y/n]: ").strip().lower()
        web_intel = web_answer not in {"n", "no"}

        pdf_answer = input("Enable PDF reading? [Y/n]: ").strip().lower()
        pdf_intel = pdf_answer not in {"n", "no"}

        cerebras_answer = input("Enable Cerebras API guidance? [Y/n]: ").strip().lower()
        cerebras = cerebras_answer not in {"n", "no"}

        custom_characteristics = input(
            "Custom compound characteristics (optional, simulation-only) []: "
        ).strip()
        custom_count_raw = input("How many custom compounds to generate? [4]: ").strip()
        custom_count = 4
        if custom_count_raw:
            try:
                custom_count = max(1, min(8, int(custom_count_raw)))
            except ValueError:
                custom_count = 4

        detail_raw = input(
            "Detail panels to show (combinations/composition/analytics/ai/all) [none]: "
        ).strip()
        detail_sections = {part.strip().lower() for part in detail_raw.split() if part.strip()}
        valid_sections = {"combinations", "composition", "analytics", "ai", "all"}
        detail_sections = {item for item in detail_sections if item in valid_sections}

        run_dashboard(
            live=live_mode,
            refresh_seconds=args.refresh,
            iterations=args.iterations,
            enable_ai=with_ai,
            ai_model_id=args.ai_model_id,
            ai_refresh_every=args.ai_refresh_every,
            include_simulated_combos=not args.no_sim_combos,
            user_query=query,
            enable_web_intel=web_intel,
            enable_pdf_read=pdf_intel,
            enable_cerebras=cerebras,
            intel_refresh_every=args.intel_refresh_every,
            filter_query=filter_query,
            custom_characteristics=custom_characteristics,
            custom_count=custom_count,
            detail_sections=detail_sections,
            theme=args.theme,
            motion_intensity=max(1, min(3, args.motion_intensity)),
        )
        return

    run_dashboard(
        live=not args.static,
        refresh_seconds=args.refresh,
        iterations=args.iterations,
        enable_ai=args.with_ai,
        ai_model_id=args.ai_model_id,
        ai_refresh_every=args.ai_refresh_every,
        include_simulated_combos=not args.no_sim_combos,
        user_query=query,
        enable_web_intel=not args.no_web_intel,
        enable_pdf_read=not args.no_pdf_intel,
        enable_cerebras=not args.no_cerebras,
        intel_refresh_every=args.intel_refresh_every,
        filter_query=args.filter_query,
        custom_characteristics=args.custom_characteristics,
        custom_count=max(1, min(8, args.custom_count)),
        detail_sections=set(args.detail_panels or []),
        theme=args.theme,
        motion_intensity=max(1, min(3, args.motion_intensity)),
    )


def run_ai_support(args):
    """Generate AI support response using Meta Llama."""
    from drug_discovery.ai_support import AISupportConfig, LlamaSupportAssistant

    assistant = LlamaSupportAssistant(config=AISupportConfig(model_id=args.model_id))

    try:
        response = assistant.respond(
            user_prompt=args.prompt,
            context=args.context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("\nZANE AI Support Response:\n")
    print(response)


def run_synthesis_research(args):
    """Plan synthesis with internet research and optional AI guidance."""
    from drug_discovery.synthesis import RetrosynthesisPlanner

    planner = RetrosynthesisPlanner()
    result = planner.plan_synthesis_with_research(
        target_smiles=args.smiles,
        target_protein=args.target,
        max_depth=args.max_depth,
        max_research_results=args.max_results,
        use_internet=not args.no_internet,
        use_ai_chat=not args.no_ai,
        read_online_resources=not args.no_resource_read,
        max_resource_reads=args.max_resource_reads,
    )

    print("\nSynthesis Research Result:\n")
    print(json.dumps(result, indent=2))


def run_boltzgen(args):
    """Run the BoltzGen binder design workflow via CLI wrapper."""
    from drug_discovery.boltzgen_adapter import BoltzGenRunner

    runner = BoltzGenRunner(cache_dir=args.cache_dir)
    try:
        result = runner.run(
            design_spec=args.spec,
            output_dir=args.output,
            protocol=args.protocol,
            num_designs=args.num_designs,
            budget=args.budget,
            steps=args.steps,
            devices=args.devices,
            reuse=args.reuse,
            parse_results=True,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = runner.summarize_metrics(result.metrics, top_k=args.top_k, score_key=args.score_key)
    payload = {
        "success": result.success,
        "command": result.command,
        "output_dir": str(result.output_dir),
        "metrics_file": str(result.metrics_file) if result.metrics_file else None,
        "summary": summary,
    }
    if result.stdout.strip():
        payload["stdout"] = result.stdout.strip()
    if result.stderr.strip():
        payload["stderr"] = result.stderr.strip()

    print(json.dumps(payload, indent=2))

    if not result.success:
        sys.exit(result.returncode or 1)


if __name__ == "__main__":
    main()
