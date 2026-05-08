from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import os
import uuid
import shutil
import asyncio
import random
from drug_discovery.commercial.fda_drug_matcher import CommercialDrugMapper
from zane_apex_entrypoint import execute_zane_pipeline

app = FastAPI(title="ZANE Client Gateway")

# Initialize the commercial mapper
mapper = CommercialDrugMapper()
mapper.load_fda_orange_book("data/fda_orange_book.csv")

class Compound(BaseModel):
    smiles: str
    dosage: str
    timing: str
    purpose: str
    toxicity_level: str

class CommercialMatch(BaseModel):
    closest_drug: str
    similarity: float
    commercial_dose: str
    extra_compounds: List[str]
    missing_compounds: List[str]

class TherapeuticBlueprint(BaseModel):
    compounds: List[Compound]
    commercial_match: CommercialMatch

@app.post("/api/v1/generate_blueprint", response_model=TherapeuticBlueprint)
async def generate_blueprint(
    name: str = Form(...),
    dob: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    location: str = Form(...),
    target_purpose: str = Form(...),
    current_treatments: str = Form(""),
    lifestyle: str = Form(...),
    hereditary_problems: str = Form(""),
    health_report: UploadFile = File(...)
):
    """
    Triggers the ZANE Zero-Mortality engine and returns a detailed therapeutic blueprint.
    Only personal data relevant to biology is passed to ZANE.
    """
    # 1. Save uploaded health report
    temp_id = str(uuid.uuid4())
    upload_dir = f"temp_uploads/{temp_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    report_path = os.path.join(upload_dir, health_report.filename)
    with open(report_path, "wb") as buffer:
        shutil.copyfileobj(health_report.file, buffer)

    # 2. Trigger ZANE Pipeline (Simulated)
    # We pass biology-relevant data: name, dob, location, purpose, treatments, lifestyle, hereditary
    # We DO NOT pass phone and email.
    metadata = {
        "name": name,
        "dob": dob,
        "location": location,
        "treatments": current_treatments,
        "lifestyle": lifestyle,
        "hereditary": hereditary_problems
    }
    await execute_zane_pipeline(report_path, target_purpose, metadata=metadata)
    
    # 3. Generate Multi-Compound Blueprint (10-20 compounds)
    num_compounds = random.randint(10, 20)
    mock_compounds = []
    
    # Primary compound
    primary_smiles = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    mock_compounds.append(Compound(
        smiles=primary_smiles,
        dosage="14.5mg",
        timing="08:30 AM",
        purpose=f"Primary inhibitor for {target_purpose}",
        toxicity_level="Ultra-Low (0.02 LD50)"
    ))
    
    # Adjuvant compounds
    for i in range(num_compounds - 1):
        mock_compounds.append(Compound(
            smiles=f"SMILES_ADJ_{i}_{uuid.uuid4().hex[:6]}",
            dosage=f"{random.uniform(1, 10):.1f}mg",
            timing=f"{random.randint(8, 22):02d}:00",
            purpose="Metabolic synergy / Adjuvant",
            toxicity_level="Non-toxic"
        ))
    
    # 4. Find Commercial Match for the primary compound
    comm_match_data = mapper.find_closest_commercial_match(primary_smiles)
    
    # 5. Compare with multi-compound ZANE drug
    comp_analysis = mapper.compare_compounds([c.dict() for c in mock_compounds], comm_match_data)
    
    comm_match = CommercialMatch(
        closest_drug=comm_match_data['closest_drug'],
        similarity=comm_match_data['similarity'],
        commercial_dose=comm_match_data['commercial_dose'],
        extra_compounds=comp_analysis['extra_compounds'],
        missing_compounds=comp_analysis['missing_compounds']
    )
    
    # 6. Final Blueprint
    return TherapeuticBlueprint(
        compounds=mock_compounds,
        commercial_match=comm_match
    )
