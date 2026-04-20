"""
Configuration for Drug Discovery Pipeline
"""

# Model Configurations
MODEL_CONFIGS = {
    'gnn': {
        'node_features': 8,
        'edge_features': 3,
        'hidden_dim': 128,
        'num_layers': 4,
        'num_heads': 4,
        'dropout': 0.2,
        'output_dim': 1,
        'pooling': 'attention'
    },
    'transformer': {
        'input_dim': 2048,
        'hidden_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'output_dim': 1,
    },
    'mpnn': {
        'node_features': 8,
        'edge_features': 3,
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.2,
        'output_dim': 1
    }
}

# Training Configurations
TRAINING_CONFIG = {
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'patience': 10,
    'test_size': 0.2,
    'validation_split': 0.1
}

# Data Collection Configurations
DATA_CONFIG = {
    'sources': ['pubchem', 'chembl', 'approved_drugs', 'web_scraping'],
    'limit_per_source': 1000,
    'cache_dir': './data/cache',
    'min_molecular_weight': 100,
    'max_molecular_weight': 900,
}

# ADMET Thresholds
ADMET_THRESHOLDS = {
    'qed_min': 0.5,
    'sa_max': 6.0,
    'lipinski_max_violations': 1,
    'molecular_weight_max': 500,
    'logp_max': 5,
}

# Paths
PATHS = {
    'cache_dir': './data/cache',
    'checkpoint_dir': './checkpoints',
    'logs_dir': './logs',
    'results_dir': './results',
}
