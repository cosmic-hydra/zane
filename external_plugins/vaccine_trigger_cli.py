import argparse
import importlib
import json
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VaccineTrigger")

def run_generate_mrna(args):
    """
    On-demand orchestration for mRNA vaccine design.
    Loads heavy modules only when needed.
    """
    logger.info(f"Triggering mRNA generation for {args.viral_fasta}...")
    
    # Dynamic imports to save memory/VRAM
    try:
        from external_plugins.vaccinology.mhc_epitope_mapper import PatientEpitopeMapper
        from external_plugins.vaccinology.mrna_compiler import ThermodynamicmRNACompiler
        from external_plugins.vaccinology.prefusion_stabilizer import PrefusionLockEngine
    except ImportError as e:
        logger.error(f"Failed to load heavy vaccinology modules: {str(e)}")
        sys.exit(1)

    # 1. Epitope Mapping
    mapper = PatientEpitopeMapper()
    # Mock HLA alleles if not provided
    hla_alleles = ["HLA-A*02:01"]
    if args.patient_hla and os.path.exists(args.patient_hla):
        with open(args.patient_hla, 'r') as f:
            hla_alleles = json.load(f).get("alleles", hla_alleles)
            
    antigen_seq = mapper.select_optimal_antigen_payload(args.viral_fasta, hla_alleles)
    
    # 2. mRNA Compilation
    compiler = ThermodynamicmRNACompiler()
    payload = compiler.compile_full_payload(antigen_seq)
    
    # 3. Structural Stabilization (if PDB is provided)
    stability_report = {}
    if args.viral_pdb:
        stabilizer = PrefusionLockEngine()
        mutations = stabilizer.suggest_stabilizing_mutations(args.viral_pdb)
        ddg = stabilizer.calculate_mutation_ddg(args.viral_pdb, mutations)
        stability_report = {
            "stabilizing_mutations": mutations,
            "predicted_ddg": ddg,
            "conformation": "prefusion_locked"
        }

    # Final Package
    blueprint = {
        "vaccine_type": "mRNA",
        "target_fasta": args.viral_fasta,
        "optimized_payload": payload,
        "structural_validation": stability_report,
        "patient_context": {"hla_alleles": hla_alleles}
    }
    
    output_file = args.output or "vaccine_blueprint.json"
    with open(output_file, 'w') as f:
        json.dump(blueprint, f, indent=4)
        
    logger.info(f"Vaccine design complete. Blueprint saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="ZANE External Vaccine Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Vaccine commands")

    # Generate mRNA Vaccine Command
    mrna_parser = subparsers.add_parser("generate-mrna", help="Design an mRNA vaccine payload")
    mrna_parser.add_argument("--viral-fasta", required=True, help="Path to viral protein FASTA")
    mrna_parser.add_argument("--viral-pdb", help="Path to viral protein PDB structure")
    mrna_parser.add_argument("--patient-hla", help="Path to JSON file with patient HLA alleles")
    mrna_parser.add_argument("--output", help="Output JSON path")

    args = parser.parse_args()

    if args.command == "generate-mrna":
        run_generate_mrna(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
