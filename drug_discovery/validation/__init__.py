"""ZANE Validation — Scientific validation and statistical testing."""

__all__ = []
try:
    from drug_discovery.validation.elite_protocols import (
        EliteValidationSuite as EliteValidationSuite,
    )
    from drug_discovery.validation.scientific_validation import (
        CLASSIFICATION_METRICS as CLASSIFICATION_METRICS,
    )
    from drug_discovery.validation.scientific_validation import (
        REGRESSION_METRICS as REGRESSION_METRICS,
    )
    from drug_discovery.validation.scientific_validation import (
        ExperimentReport as ExperimentReport,
    )
    from drug_discovery.validation.scientific_validation import (
        bootstrap_ci as bootstrap_ci,
    )
    from drug_discovery.validation.scientific_validation import (
        compute_metrics as compute_metrics,
    )
    from drug_discovery.validation.scientific_validation import (
        config_hash as config_hash,
    )
    from drug_discovery.validation.scientific_validation import (
        paired_ttest as paired_ttest,
    )
    from drug_discovery.validation.scientific_validation import (
        scaffold_kfold as scaffold_kfold,
    )
    from drug_discovery.validation.scientific_validation import (
        scaffold_split as scaffold_split,
    )
    from drug_discovery.validation.scientific_validation import (
        set_global_seed as set_global_seed,
    )
    from drug_discovery.validation.scientific_validation import (
        wilcoxon_test as wilcoxon_test,
    )

    __all__.extend(
        [
            "CLASSIFICATION_METRICS",
            "REGRESSION_METRICS",
            "EliteValidationSuite",
            "ExperimentReport",
            "bootstrap_ci",
            "compute_metrics",
            "config_hash",
            "paired_ttest",
            "scaffold_kfold",
            "scaffold_split",
            "set_global_seed",
            "wilcoxon_test",
        ]
    )
except ImportError:
    pass
