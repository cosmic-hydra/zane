from fastapi import FastAPI, HTTPException, Body
from typing import Dict, Any, List
from pydantic import BaseModel
import torch

try:
    from fhir.resources.bundle import Bundle
    from fhir.resources.patient import Patient
    from fhir.resources.observation import Observation
    _FHIR_SUPPORT = True
except ImportError:
    _FHIR_SUPPORT = False

app = FastAPI(title="ZANE FHIR Interoperability Gateway")

class FHIRIngestionAPI:
    """
    Standardized HL7 FHIR Gateway for secure clinical data ingestion.
    Converts FHIR R4/R5 resources into tensor representations for model inference.
    """
    
    def __init__(self):
        pass

    def parse_bundle_to_tensor(self, bundle_data: Dict[str, Any]) -> torch.Tensor:
        """
        Parses a FHIR Bundle into a normalized PyTorch tensor.
        Extracts phenotypes (Observations) and maps them to a feature vector.
        """
        if not _FHIR_SUPPORT:
            raise RuntimeError("FHIR resources library not installed.")

        bundle = Bundle.parse_obj(bundle_data)
        
        # Example feature mapping: [Age, BMI, SBP, DBP, Glucose]
        features = [0.0] * 5
        
        if bundle.entry:
            for entry in bundle.entry:
                resource = entry.resource
                if isinstance(resource, Observation):
                    code = resource.code.coding[0].code if resource.code.coding else ""
                    value = resource.valueQuantity.value if resource.valueQuantity else 0.0
                    
                    # LOINC mapping (Simplified)
                    if code == "39156-5":  # BMI
                        features[1] = float(value)
                    elif code == "8480-6":  # SBP
                        features[2] = float(value)
                    elif code == "8462-4":  # DBP
                        features[3] = float(value)
                    elif code == "2339-0":  # Glucose
                        features[4] = float(value)
                
                elif isinstance(resource, Patient):
                    if resource.birthDate:
                        age = (2024 - resource.birthDate.year) # Simplified age
                        features[0] = float(age)

        return torch.tensor([features], dtype=torch.float32)

@app.post("/api/v1/fhir/patient_bundle")
async def ingest_patient_bundle(payload: Dict[str, Any] = Body(...)):
    """
    Accepts HL7 FHIR Patient Bundles for trial simulation ingestion.
    """
    if not _FHIR_SUPPORT:
        raise HTTPException(status_code=501, detail="FHIR validation engine not available.")

    try:
        # Validate FHIR Schema
        Bundle.validate(payload)
        
        gateway = FHIRIngestionAPI()
        tensor = gateway.parse_bundle_to_tensor(payload)
        
        return {
            "status": "success",
            "message": "FHIR Bundle validated and tensorized",
            "tensor_shape": list(tensor.shape),
            "data_preview": tensor.tolist()[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid FHIR Bundle: {str(e)}")

# Placeholder for integration with trial_simulation_engine
def get_fhir_gateway():
    return FHIRIngestionAPI()
