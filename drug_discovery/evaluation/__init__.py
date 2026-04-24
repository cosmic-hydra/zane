"""ZANE Evaluation — Guarded imports for uncertainty, ADMET, legacy predictors."""

import logging

logger = logging.getLogger(__name__)
__all__ = []

try:
    from drug_discovery.evaluation.uncertainty import (
        ConformalPredictor,
        DeepEnsemble,
        MCDropoutPredictor,
        UncertaintyConfig,
        expected_calibration_error,
        regression_calibration_error,
    )

    __all__.extend(
        [
            "MCDropoutPredictor",
            "DeepEnsemble",
            "ConformalPredictor",
            "UncertaintyConfig",
            "expected_calibration_error",
            "regression_calibration_error",
        ]
    )
except ImportError as e:
    logger.debug(f"Uncertainty not available: {e}")

try:
    from drug_discovery.evaluation.advanced_admet import (
        ADMET_ENDPOINTS,
        ADMETConfig,
        AdvancedADMETPredictor,
        compute_admet_profile,
    )

    __all__.extend(["AdvancedADMETPredictor", "ADMETConfig", "ADMET_ENDPOINTS", "compute_admet_profile"])
except ImportError as e:
    logger.debug(f"Advanced ADMET not available: {e}")

try:
    from drug_discovery.evaluation.predictor import ADMETPredictor, ModelEvaluator, PropertyPredictor

    __all__.extend(["ADMETPredictor", "ModelEvaluator", "PropertyPredictor"])
except ImportError:
    pass

try:
    from drug_discovery.evaluation.torchdrug_scorer import TorchDrugScorer

    __all__.append("TorchDrugScorer")
except ImportError:
    pass
