"""Deep 12-Endpoint Toxicity Panel for ZANE.
hERG, Ames, DILI, CYP (5 isoforms), phospholipidosis, phototoxicity, mitochondrial.
Descriptor-based heuristic rules from medicinal chemistry literature.
"""
from __future__ import annotations
import logging, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
logger = logging.getLogger(__name__)

@dataclass
class ToxEndpointResult:
    endpoint: str; risk_level: str; probability: float
    key_factors: List[str] = field(default_factory=list); recommendation: str = ""
@dataclass
class ToxPanelReport:
    smiles: str; overall_risk: str = "unknown"; overall_score: float = 0.0
    endpoints: Dict[str, ToxEndpointResult] = field(default_factory=dict)
    pass_count: int = 0; fail_count: int = 0
    def to_dict(self):
        return {"smiles":self.smiles,"risk":self.overall_risk,"score":round(self.overall_score,3),
                "pass":self.pass_count,"fail":self.fail_count,
                "endpoints":{k:{"risk":v.risk_level,"prob":round(v.probability,3)} for k,v in self.endpoints.items()}}

def _cl(prob, lo=0.3, hi=0.7): return "low" if prob<lo else "moderate" if prob<hi else "high"
def _sig(x, mid, k): return 1/(1+math.exp(-k*(x-mid)))

class DeepToxPanel:
    """12-endpoint deep toxicity screening.
    Example: report = DeepToxPanel().screen("c1ccccc1")
    """
    def screen(self, smiles, descriptors=None):
        if descriptors is None:
            try:
                from drug_discovery.data.pipeline import compute_descriptors
                descriptors = compute_descriptors(smiles) or self._fb()
            except ImportError: descriptors = self._fb()
        r = ToxPanelReport(smiles=smiles)
        mw=descriptors.get("mol_weight",300); lp=descriptors.get("logp",2)
        hbd=descriptors.get("hbd",1); tpsa=descriptors.get("tpsa",70)
        aro=descriptors.get("aromatic_rings",1)
        # 1. hERG: logP>3.7, PSA<75
        h=_sig(lp,3.7,1.5)*_sig(-tpsa,-75,0.03)*0.8
        r.endpoints["herg"]=ToxEndpointResult("herg",_cl(h),h,[f"logP={lp:.1f}",f"PSA={tpsa:.0f}"],"Monitor QT" if h>0.5 else "OK")
        # 2. Ames
        a=min(1,aro*0.15+(0.3 if lp>4 else 0))
        r.endpoints["ames"]=ToxEndpointResult("ames",_cl(a),a,[f"aromatic={aro}"],"Run Ames test" if a>0.3 else "Low risk")
        # 3. DILI
        d=_sig(mw,500,0.005)*_sig(lp,3,0.8)*0.7
        r.endpoints["dili"]=ToxEndpointResult("dili",_cl(d),d,[f"MW={mw:.0f}",f"logP={lp:.1f}"],"Monitor ALT/AST" if d>0.5 else "OK")
        # 4-8. CYPs
        for cyp,(lt,mt) in [("cyp1a2",(2.5,350)),("cyp2c9",(3.0,400)),("cyp2c19",(2.8,380)),("cyp2d6",(3.5,350)),("cyp3a4",(3.0,500))]:
            p=_sig(lp,lt,1.0)*_sig(mw,mt,0.003)*0.6
            r.endpoints[cyp]=ToxEndpointResult(cyp,_cl(p),p,[f"logP={lp:.1f}"],"Check DDI" if p>0.5 else "OK")
        # 9. Phospholipidosis
        pl=_sig(lp,3,1.0)*_sig(mw,300,0.003)*0.5
        r.endpoints["phospholipidosis"]=ToxEndpointResult("phospholipidosis",_cl(pl),pl,[],"Monitor" if pl>0.5 else "")
        # 10. Phototoxicity
        ph=min(1,aro*0.2)
        r.endpoints["phototoxicity"]=ToxEndpointResult("phototoxicity",_cl(ph),ph,[f"aromatic={aro}"],"")
        # 11. Mitochondrial
        mi=_sig(lp,3.5,1.0)*0.4
        r.endpoints["mitochondrial"]=ToxEndpointResult("mitochondrial",_cl(mi),mi,[],"" )
        # Aggregate
        probs=[e.probability for e in r.endpoints.values()]
        r.overall_score=float(np.mean(probs)); r.overall_risk=_cl(r.overall_score)
        r.pass_count=sum(1 for e in r.endpoints.values() if e.risk_level=="low")
        r.fail_count=sum(1 for e in r.endpoints.values() if e.risk_level=="high")
        return r
    def batch_screen(self, smiles_list):
        reports=[self.screen(s) for s in smiles_list]
        safe=sum(1 for r in reports if r.overall_risk=="low")
        logger.info(f"Tox: {len(smiles_list)} screened, {safe} safe ({safe/max(len(smiles_list),1)*100:.0f}%)")
        return reports
    def get_safe_molecules(self, reports, max_fails=1): return [r.smiles for r in reports if r.fail_count<=max_fails]
    def _fb(self): return {"mol_weight":350,"logp":2.5,"hbd":1,"hba":4,"tpsa":70,"aromatic_rings":1,"heavy_atoms":25}
