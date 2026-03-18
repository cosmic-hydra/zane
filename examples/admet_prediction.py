"""
Example: ADMET Property Prediction
"""

from drug_discovery.evaluation import ADMETPredictor
import pandas as pd


def main():
    # Initialize ADMET predictor
    admet = ADMETPredictor()

    # Example molecules
    molecules = {
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Penicillin': 'CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O',
    }

    results = []

    print("=== ADMET Property Predictions ===\n")

    for name, smiles in molecules.items():
        print(f"Analyzing: {name}")
        print(f"SMILES: {smiles}")

        # Lipinski's Rule of Five
        lipinski = admet.check_lipinski_rule(smiles)
        if lipinski is None:
            print("\nLipinski's Rule of Five:\n  Could not compute properties")
            print("\n" + "=" * 50 + "\n")
            continue
        print(f"\nLipinski's Rule of Five:")
        print(f"  Pass: {lipinski['passes']}")
        print(f"  Violations: {lipinski['num_violations']}")
        if lipinski['violations']:
            print(f"  Issues: {', '.join(lipinski['violations'])}")

        # Drug-likeness
        qed = admet.calculate_qed(smiles)
        if qed is None:
            print("\nDrug-likeness (QED): N/A")
            qed = 0.0
        else:
            print(f"\nDrug-likeness (QED): {qed:.3f}")
        print(f"  Interpretation: {'Good' if qed > 0.5 else 'Poor'}")

        # Synthetic Accessibility
        sa_score = admet.calculate_synthetic_accessibility(smiles)
        if sa_score is None:
            sa_score = 10.0
        print(f"\nSynthetic Accessibility: {sa_score:.2f}/10")
        print(f"  Interpretation: {'Easy' if sa_score < 5 else 'Difficult'}")

        # Toxicity flags
        toxicity = admet.predict_toxicity_flags(smiles)
        if toxicity is None:
            toxicity = {}
        print(f"\nToxicity Flags:")
        for flag, value in toxicity.items():
            print(f"  {flag}: {'Yes' if value else 'No'}")

        # Calculate properties
        props = lipinski['properties']
        print(f"\nMolecular Properties:")
        print(f"  MW: {props['molecular_weight']:.2f}")
        print(f"  LogP: {props['logp']:.2f}")
        print(f"  H-bond donors: {props['h_bond_donors']}")
        print(f"  H-bond acceptors: {props['h_bond_acceptors']}")

        print("\n" + "="*50 + "\n")

        # Store results
        results.append({
            'name': name,
            'smiles': smiles,
            'lipinski_pass': lipinski['passes'],
            'qed': qed,
            'sa_score': sa_score,
            'mw': props['molecular_weight'],
            'logp': props['logp'],
        })

    # Create summary DataFrame
    df = pd.DataFrame(results)
    print("\n=== Summary Table ===")
    print(df.to_string(index=False))

    # Filter drug-like molecules
    drug_like = df[(df['lipinski_pass'] == True) & (df['qed'] > 0.5)]
    print(f"\n✓ Drug-like molecules: {len(drug_like)}/{len(df)}")
    print(drug_like['name'].tolist())


if __name__ == "__main__":
    main()
