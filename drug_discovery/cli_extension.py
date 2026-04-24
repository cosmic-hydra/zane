"""CLI Extension — registers all 2026 Q2 upgrade subcommands."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_commands(subparsers):
    p = subparsers.add_parser("train-advanced", help="Train with AMP/EMA/warmup")
    p.add_argument("--model", choices=["egnn", "schnet", "gnn", "transformer"], default="egnn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--use-amp", action="store_true", default=True)
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument(
        "--scheduler", choices=["cosine_warm_restarts", "plateau", "one_cycle"], default="cosine_warm_restarts"
    )
    p.set_defaults(func=cmd_train_advanced)

    p = subparsers.add_parser("evaluate-scientific", help="Scaffold k-fold CV with stats")
    p.add_argument("--model", default="egnn")
    p.add_argument("--dataset", required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/reports/scientific_eval.json")
    p.set_defaults(func=cmd_evaluate_scientific)

    p = subparsers.add_parser("generate-diffusion", help="SE(3) diffusion generation")
    p.add_argument("--num-molecules", type=int, default=10)
    p.add_argument("--num-atoms", type=int, default=20)
    p.add_argument("--noise-steps", type=int, default=1000)
    p.set_defaults(func=cmd_generate_diffusion)

    p = subparsers.add_parser("generate-gflownet", help="GFlowNet diverse generation")
    p.add_argument("--num-molecules", type=int, default=10)
    p.add_argument("--max-atoms", type=int, default=38)
    p.add_argument("--temperature", type=float, default=1.0)
    p.set_defaults(func=cmd_generate_gflownet)

    p = subparsers.add_parser("dock", help="Molecular docking")
    p.add_argument("--receptor", required=True)
    p.add_argument("--ligands", nargs="+", required=True)
    p.add_argument("--backend", choices=["vina", "diffdock", "gnina"], default="vina")
    p.add_argument("--top-k", type=int, default=10)
    p.set_defaults(func=cmd_dock)

    p = subparsers.add_parser("admet-advanced", help="Multi-modal ADMET")
    p.add_argument("smiles", nargs="+")
    p.add_argument("--endpoints", nargs="+", default=["solubility", "herg_inhibition", "bioavailability"])
    p.set_defaults(func=cmd_admet_advanced)

    p = subparsers.add_parser("optimize-mobo", help="Multi-objective Bayesian optimization")
    p.add_argument("--objectives", nargs="+", default=["binding_affinity", "selectivity", "solubility"])
    p.add_argument("--iterations", type=int, default=50)
    p.set_defaults(func=cmd_optimize_mobo)

    p = subparsers.add_parser("integrations-extended", help="Check extended tool statuses")
    p.set_defaults(func=cmd_integrations_extended)


def cmd_train_advanced(args):
    from drug_discovery.models import get_model
    from drug_discovery.training.advanced_training import AdvancedTrainingConfig

    model = get_model(args.model, hidden_dim=128, num_layers=6)
    config = AdvancedTrainingConfig(
        epochs=args.epochs, learning_rate=args.lr, use_amp=args.use_amp, use_ema=args.use_ema, scheduler=args.scheduler
    )
    print(f"Model: {args.model} | Epochs: {args.epochs} | LR: {args.lr} | AMP: {args.use_amp} | EMA: {args.use_ema}")
    print("Trainer ready. Call trainer.fit(train_loader, val_loader) with your data.")


def cmd_evaluate_scientific(args):
    from drug_discovery.validation.scientific_validation import set_global_seed

    set_global_seed(args.seed)
    print(f"Scientific eval: model={args.model} folds={args.n_folds} seed={args.seed} output={args.output}")


def cmd_generate_diffusion(args):
    print(f"Diffusion: {args.num_molecules} molecules x {args.num_atoms} atoms, {args.noise_steps} steps")


def cmd_generate_gflownet(args):
    print(f"GFlowNet: {args.num_molecules} molecules, max {args.max_atoms} atoms, T={args.temperature}")


def cmd_dock(args):
    print(f"Docking: {args.backend} | {len(args.ligands)} ligands | top-{args.top_k}")


def cmd_admet_advanced(args):
    print(f"Advanced ADMET: {len(args.smiles)} molecules | Endpoints: {args.endpoints}")


def cmd_optimize_mobo(args):
    print(f"MOBO: {args.objectives} | {args.iterations} iters")


def cmd_integrations_extended(args):
    from drug_discovery.integrations_extended import integration_report

    print(integration_report())
