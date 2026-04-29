&quot;&quot;&quot;
Examples for 2024 drug discovery breakthroughs integration.
&quot;&quot;&quot;

import asyncio
from drug_discovery.alphafold3.alphafold3_docking import AlphaFold3Docking
from drug_discovery.rfdiffusion.protein_design import RFDiffusionDesigner
# etc.

async def main():
    # AF3 proxy
    af3 = AlphaFold3Docking(&quot;protein_seq&quot;)
    results = await af3.dock_batch([&quot;CCO&quot;, &quot;CN&quot;])
    print(results)

    # RF design
    rf = RFDiffusionDesigner()
    designs = rf.design_batch([&quot;motif1&quot;])
    print(designs)

    # CRISPR base edit
    from models.biologics.crispr_base_editor import CRISPRBaseEditor
    editor = CRISPRBaseEditor()
    edit = editor.base_edit(&quot;ATCG&quot;, 1, &quot;A&quot;, &quot;G&quot;)
    print(edit)

    # ADC
    from models.nextgen_adcs.adc_optimizer import ADCOptimizer
    adc = ADCOptimizer()
    res = adc.optimize(&quot;DM1&quot;, &quot;Her2-shuttle&quot;)
    print(res)

if __name__ == &quot;__main__&quot;:
    asyncio.run(main())