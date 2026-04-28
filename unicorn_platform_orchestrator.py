import os

import typer

from infrastructure.cloud_lab.os_kernel import OSKernel
from infrastructure.cryptography.zkp_marketplace import ZKPMarketplace
from models.biologics.crispr_foundry import CRISPRFoundry

app = typer.Typer(help="Unicorn Platform Orchestrator for ZANE")

class UnicornOrchestrator:
    def __init__(self):
        self.zkp = ZKPMarketplace()
        self.kernel = OSKernel()
        self.foundry = CRISPRFoundry()
        # Wire closed-loop callback
        self.kernel.set_zkp_callback(
            lambda results: self.zkp.mint_royalties("cloud_lab_contributor", 5.0, "ingest_proof")
        )

    def orchestrate(self, target_seq: str):
        typer.echo("Starting Unicorn Platform workflow...")

        # 1. ZKP Federated Data Bourse
        data_fed = self.zkp.federate_data({"target_sequence": target_seq, "domain": "crispr_design"})
        typer.echo(f"Federated data: {data_fed['status']}")

        # 2. De Novo CRISPR Foundry
        nuclease = self.foundry.generate_nuclease(target_seq)
        nuclease_opt = self.foundry.optimize_offtarget(nuclease)
        rmsd = self.foundry.verify_structure(nuclease_opt, target_seq)
        if rmsd > 2.0:
            raise typer.Exit(code=1, message=f"Structural verification failed: RMSD {rmsd:.2f}A")
        typer.echo(f"CRISPR nuclease verified (RMSD: {rmsd:.2f}A)")

        # 3. Cloud-Lab OS dispatch
        protocol = f"Validate and produce CRISPR effector: {nuclease_opt}"
        job_spec = self.kernel.compile_labop(protocol)
        results = self.kernel.dispatch_bacalhau(job_spec)
        self.kernel.ingest_results(results)
        typer.echo(f"Cloud-Lab execution complete: {results.get('status', 'mock')}")

        # 4. ZK Training Proof & Royalties
        proof = self.zkp.prove_zk_training({"nuclease": nuclease_opt}, "crispr_foundry_1")
        self.zkp.mint_royalties("esm3_team", 10.0, proof)
        typer.echo("ZK proof generated & royalties distributed")

        typer.echo("Unicorn workflow completed successfully!")

@app.command()
def run(
    target_seq: str = "ATCGATCGATCG...",
    fpga_device: str | None = typer.Option(None, "--fpga-device", help="FPGA/ASIC device path (sets env)")
):
    """
    Orchestrate Unicorn Modules 14-16:
    ZKP Data Bourse -> CRISPR Foundry -> Cloud-Lab OS -> Royalties
    """
    if fpga_device:
        os.environ["FPGA_DEVICE"] = fpga_device
        typer.echo(f"FPGA mode: {fpga_device}")

    try:
        orch = UnicornOrchestrator()
        orch.orchestrate(target_seq)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

def main():
    app()

if __name__ == "__main__":
    main()
