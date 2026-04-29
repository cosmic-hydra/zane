"""
Examples for 2024 drug discovery breakthroughs integration.
"""

import asyncio
from drug_discovery.alphafold3.alphafold3_docking import AlphaFold3Docking
from drug_discovery.rfdiffusion.protein_design import RFDiffusionDesigner
# etc.

async def main():
    # AF3 proxy
    af3 = AlphaFold3Docking("protein_seq")
    results = await af3.dock_batch(["CCO", "CN"])
    print(results)

    # RF design
    rf = RFDiffusionDesigner()
    designs = rf.design_batch(["motif1"])
    print(designs)

    # CRISPR base edit
    from models.biologics.crispr_base_editor import CRISPRBaseEditor
    editor = CRISPRBaseEditor()
    edit = editor.base_edit("ATCG", 1, "A", "G")
    print(edit)

    # ADC
    from models.nextgen_adcs.adc_optimizer import ADCOptimizer
    adc = ADCOptimizer()
    res = adc.optimize("DM1", "Her2-shuttle")
    print(res)

if __name__ == "__main__":
    asyncio.run(main())