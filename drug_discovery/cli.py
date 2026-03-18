"""
Command-line interface for Drug Discovery Platform
"""

import argparse
import json
import sys

from drug_discovery import DrugDiscoveryPipeline
from drug_discovery.ai_support import AISupportConfig, LlamaSupportAssistant
from drug_discovery.dashboard import run_dashboard
from drug_discovery.evaluation import ADMETPredictor
from drug_discovery.synthesis import RetrosynthesisPlanner


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

    # Collect data command
    collect_parser = subparsers.add_parser("collect", help="Collect molecular data")
    collect_parser.add_argument("--sources", nargs="+", default=["pubchem", "chembl"])
    collect_parser.add_argument("--limit", type=int, default=1000)

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Show professional ZANE terminal dashboard")
    dashboard_parser.add_argument("--static", action="store_true", help="Render one static dashboard frame")
    dashboard_parser.add_argument("--refresh", type=float, default=1.0, help="Live refresh interval in seconds")
    dashboard_parser.add_argument("--iterations", type=int, default=30, help="Number of live refresh cycles")

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

    args = parser.parse_args()

    if args.command == "predict":
        predict_properties(args)
    elif args.command == "admet":
        analyze_admet(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "collect":
        collect_data(args)
    elif args.command == "dashboard":
        show_dashboard(args)
    elif args.command == "assist":
        run_ai_support(args)
    elif args.command == "synthesis-research":
        run_synthesis_research(args)
    else:
        parser.print_help()


def predict_properties(args):
    """Predict molecular properties"""
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
    print(f"Training {args.model} model...")

    pipeline = DrugDiscoveryPipeline(model_type=args.model)

    # Collect data
    data = pipeline.collect_data()

    # Prepare datasets
    train_loader, test_loader = pipeline.prepare_datasets(data, batch_size=args.batch_size)

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
        else:
            continue

        data = collector.merge_datasets([data, df])

    print(f"\nCollected {len(data)} unique molecules")

    # Save
    data.to_csv("./data/collected_data.csv", index=False)
    print("Saved to ./data/collected_data.csv")


def show_dashboard(args):
    """Display ZANE terminal dashboard."""
    run_dashboard(live=not args.static, refresh_seconds=args.refresh, iterations=args.iterations)


def run_ai_support(args):
    """Generate AI support response using Meta Llama."""
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


if __name__ == "__main__":
    main()
