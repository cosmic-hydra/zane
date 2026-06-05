"""
Example: Basic Usage of Drug Discovery Pipeline
"""

from drug_discovery import DrugDiscoveryPipeline


def main():
    # Initialize pipeline
    print("Initializing Drug Discovery Pipeline...")
    pipeline = DrugDiscoveryPipeline(
        model_type='gnn',  # Options: 'gnn', 'transformer', 'ensemble'
        device='cuda'  # or 'cpu'
    )

    # Step 1: Collect data from public sources
    print("\nStep 1: Collecting molecular data...")
    data = pipeline.collect_data(
        sources=['pubchem', 'chembl', 'approved_drugs'],
        limit_per_source=500  # Start small for demo
    )

    print(f"Collected {len(data)} molecules")
    print(data.head())

    # Step 2: Prepare datasets
    print("\nStep 2: Preparing datasets...")
    train_loader, test_loader = pipeline.prepare_datasets(
        data=data,
        smiles_col='smiles',
        target_col=None,  # Unsupervised for now
        test_size=0.2,
        batch_size=32
    )

    # Step 3: Build and train model
    print("\nStep 3: Training model...")
    pipeline.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=10,  # Use more epochs in production
        learning_rate=1e-4
    )

    # Step 4: Predict properties for a molecule
    print("\nStep 4: Predicting properties...")
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    properties = pipeline.predict_properties(aspirin_smiles)

    print("\nPredicted properties for Aspirin:")
    for key, value in properties.items():
        print(f"  {key}: {value}")

    # Step 5: Generate drug candidates
    print("\nStep 5: Generating drug candidates...")
    candidates = pipeline.generate_candidates(
        target_protein="EGFR",
        num_candidates=10
    )

    print("\nTop drug candidates:")
    print(candidates[['smiles', 'qed_score', 'lipinski_pass']].head())

    # Step 6: Evaluate model
    print("\nStep 6: Evaluating model...")
    pipeline.evaluate(test_loader)

    # Save pipeline
    print("\nSaving pipeline...")
    pipeline.save('./checkpoints/pipeline.pt')

    print("\n✓ Pipeline demonstration complete!")


if __name__ == "__main__":
    main()
