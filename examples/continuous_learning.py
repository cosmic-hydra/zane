"""
Example: Advanced Usage - Continuous Learning
"""

from drug_discovery import DrugDiscoveryPipeline
from drug_discovery.training import ContinuousLearner
import time

def main():
    # Initialize pipeline
    pipeline = DrugDiscoveryPipeline(model_type='gnn')

    # Initial training
    print("Initial training phase...")
    data = pipeline.collect_data(limit_per_source=1000)
    train_loader, val_loader = pipeline.prepare_datasets(data)
    pipeline.train(train_loader, val_loader, num_epochs=20)

    # Setup continuous learning
    if pipeline.trainer is None:
        raise RuntimeError("Trainer is not initialized after training.")

    continuous_learner = ContinuousLearner(
        trainer=pipeline.trainer,
        data_collector=pipeline.data_collector,
        retrain_threshold=500  # Retrain after 500 new samples
    )

    # Simulate continuous learning loop
    print("\nStarting continuous learning...")
    for iteration in range(3):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Collect new data
        new_data = pipeline.data_collector.collect_from_chembl(limit=200)

        # Add samples to continuous learner
        should_retrain = continuous_learner.add_samples(len(new_data))

        if should_retrain:
            print("Retraining model with new data...")

            # Combine old and new data
            combined_data = pipeline.data_collector.merge_datasets([data, new_data])
            train_loader, val_loader = pipeline.prepare_datasets(combined_data)

            # Retrain
            continuous_learner.retrain(train_loader, val_loader, num_epochs=10)

            # Update data
            data = combined_data

        # Make predictions on new molecules
        if not new_data.empty and 'smiles' in new_data.columns:
            sample_smiles = new_data['smiles'].iloc[0]
            properties = pipeline.predict_properties(sample_smiles)
            print(f"\nPredictions for new molecule:")
            print(f"  SMILES: {sample_smiles}")
            print(f"  QED: {properties.get('qed_score', 'N/A')}")

        time.sleep(1)

    print("\n✓ Continuous learning demonstration complete!")


if __name__ == "__main__":
    main()
