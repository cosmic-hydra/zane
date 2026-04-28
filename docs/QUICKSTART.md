# ZANE Quick Start (5 Minutes)

> Get from zero to ranked drug candidates in 5 minutes.

---

## 1. Install (60 seconds)

```bash
git clone https://github.com/cosmic-hydra/zane.git
cd zane
pip install -e .
```

---

## 2. Collect Molecules (60 seconds)

```bash
zane collect --sources pubchem chembl --limit 200
```

Or use your own SMILES list in Python:

```python
smiles = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)[nH]cc2", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
```

---

## 3. Screen for Safety (60 seconds)

```python
from drug_discovery.evaluation.structural_alerts import StructuralAlertScreener
from drug_discovery.evaluation.deep_tox_panel import DeepToxPanel

# Remove dangerous molecules
screener = StructuralAlertScreener()
clean = screener.get_clean_molecules(screener.batch_screen(smiles))
print(f"{len(clean)}/{len(smiles)} passed structural alerts")

# Deep toxicity check
panel = DeepToxPanel()
safe = panel.get_safe_molecules(panel.batch_screen(clean))
print(f"{len(safe)}/{len(clean)} passed toxicity screening")
```

---

## 4. Score & Rank (60 seconds)

```python
from drug_discovery.evaluation.clinical_success_predictor import ClinicalSuccessPredictor

predictor = ClinicalSuccessPredictor()
profiles = predictor.batch_assess(safe)
ranked = predictor.rank_by_success(profiles)

for p in ranked[:5]:
    print(f"  {p.smiles[:30]:30s} CSS={p.clinical_success_score:.3f} Risk={p.risk_level}")
```

---

## 5. Dashboard (60 seconds)

```bash
zane dashboard --static --detail-panels all
```

Or with a disease query:

```bash
zane go --static --query "respiratory infection" --detail-panels all --with-ai
```

---

## What's Next?

| Goal | Command / Code |
|------|---------------|
| Train a model | `zane train --model transformer --epochs 50` |
| Train with advanced pipeline | `zane train-advanced --model egnn --epochs 100 --use-amp` |
| Generate new molecules | `zane generate-diffusion --num-molecules 50` |
| Diverse generation | `zane generate-gflownet --num-molecules 50` |
| Dock against a protein | `zane dock --receptor protein.pdb --ligands *.sdf` |
| Run full pipeline | `zane dashboard --static --detail-panels all` |
| Scientific evaluation | `zane evaluate-scientific --model egnn --dataset chembl` |
| Check integrations | `zane integrations-extended` |

---

## Full Fail-Fast Pipeline (Python)

```python
from drug_discovery.evaluation.failfast_pipeline import FailFastPipeline

pipeline = FailFastPipeline()
results = pipeline.run(smiles_list)

# See where candidates were eliminated
report = pipeline.attrition_report(results)
print(f"Pass rate: {report['pass_rate']*100:.0f}%")
print(f"Eliminated by stage: {report['eliminated_by_stage']}")

# Get survivors ranked by score
survivors = pipeline.get_survivors(results)
for r in survivors[:10]:
    print(f"  #{r.rank} {r.smiles[:30]} score={r.final_score:.3f}")
```
