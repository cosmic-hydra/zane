import pytest
from drug_discovery.data.rdkit_utils import smiles_to_mols
from drug_discovery.generation.torchdrug_generator import TorchDrugGenerator
from drug_discovery.screening.admet_models import ADMETScreen

def test_rdkit_utils():
    smiles = [&quot;CCO&quot;]
    mols = smiles_to_mols(smiles)
    assert len(mols) == 1

def test_torchdrug_generate():
    gen = TorchDrugGenerator()
    smiles = gen.generate(10)
    assert len(smiles) == 10

def test_admet_screen():
    screen = ADMETScreen()
    preds = screen.predict([&quot;CCO&quot;])
    assert 'herg' in preds