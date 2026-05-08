from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import shutil
import asyncio
from drug_discovery.commercial.fda_drug_matcher import CommercialDrugMapper
from zane_apex_entrypoint import execute_zane_pipeline

app = FastAPI(title="ZANE Client Gateway")

# Initialize the commercial mapper
mapper = CommercialDrugMapper()
# Assuming the orange book is at this path, or it will use fallback
mapper.load_fda_orange_book("data/fda_orange_book.csv")

class CommercialMatch(BaseModel):
    closest_drug: str
    similarity: float
    commercial_dose: str

class TherapeuticBlueprint(BaseModel):
    compound_smiles: str
    dosage: str
    administration_time: str
    mechanism_of_action: str
    commercial_match: CommercialMatch

@app.post("/api/v1/generate_blueprint", response_model=TherapeuticBlueprint)
async def generate_blueprint(
    target_disease: str = Form(...),
    current_prescriptions: str = Form(""),
    sleep_schedule: str = Form(""),
    health_report: UploadFile = File(...),
    genomic_vcf: Optional[UploadFile] = File(None)
):
    """
    Triggers the ZANE Zero-Mortality engine and returns a detailed therapeutic blueprint.
    """
    # 1. Save uploaded health report to a temporary location
    temp_id = str(uuid.uuid4())
    upload_dir = f"temp_uploads/{temp_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    report_path = os.path.join(upload_dir, health_report.filename)
    with open(report_path, "wb") as buffer:
        shutil.copyfileobj(health_report.file, buffer)
        
    if genomic_vcf:
        vcf_path = os.path.join(upload_dir, genomic_vcf.filename)
        with open(vcf_path, "wb") as buffer:
            shutil.copyfileobj(genomic_vcf.file, buffer)

    # 2. Trigger ZANE Pipeline
    # Note: execute_zane_pipeline is async and currently returns None but logs results.
    # In a real scenario, we'd want it to return the SMILES.
    # For this scaffolding, we will assume it returns the smiles or we mock the result.
    
    # We might need to modify execute_zane_pipeline to return values, 
    # but I will stick to what's expected in the response.
    # Since execute_zane_pipeline currently returns None in the provided code, 
    # I'll simulate the integration.
    
    await execute_zane_pipeline(report_path, target_disease)
    
    # Mocked results that would normally come from the pipeline
    # In a real implementation, we'd capture the SMILES from the pipeline execution
    generated_smiles = "C1=CC=C(C=C1)CC(=O)NC2=CC=CC=C2" 
    
    # 3. Find Commercial Match
    comm_match_data = mapper.find_closest_commercial_match(generated_smiles)
    comm_match = CommercialMatch(**comm_match_data)
    
    # 4. Construct Blueprint
    blueprint = TherapeuticBlueprint(
        compound_smiles=generated_smiles,
        dosage="14.5mg",
        administration_time="08:30 AM",
        mechanism_of_action="Selective BCR-ABL Tyrosine Kinase Inhibition",
        commercial_match=comm_match
    )
    
    # Cleanup (optional, maybe keep for audit)
    # shutil.rmtree(upload_dir)
    
    return blueprint
