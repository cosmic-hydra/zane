"""ZANE Evaluation — Guarded imports for uncertainty, ADMET, legacy predictors."""

import logging

logger = logging.getLogger(__name__)
__all__ = []

try:
    from drug_discovery.evaluation.uncertainty import (
        ConformalPredictor as ConformalPredictor,
    )
    from drug_discovery.evaluation.uncertainty import (
        DeepEnsemble as DeepEnsemble,
    )
    from drug_discovery.evaluation.uncertainty import (
        MCDropoutPredictor as MCDropoutPredictor,
    )
    from drug_discovery.evaluation.uncertainty import (
        UncertaintyConfig as UncertaintyConfig,
    )
    from drug_discovery.evaluation.uncertainty import (
        expected_calibration_error as expected_calibration_error,
    )
    from drug_discovery.evaluation.uncertainty import (
        regression_calibration_error as regression_calibration_error,
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
        ADMET_ENDPOINTS as ADMET_ENDPOINTS,
    )
    from drug_discovery.evaluation.advanced_admet import (
        ADMETConfig as ADMETConfig,
    )
    from drug_discovery.evaluation.advanced_admet import (
        AdvancedADMETPredictor as AdvancedADMETPredictor,
    )
    from drug_discovery.evaluation.advanced_admet import (
        compute_admet_profile as compute_admet_profile,
    )

    __all__.extend(["AdvancedADMETPredictor", "ADMETConfig", "ADMET_ENDPOINTS", "compute_admet_profile"])
except ImportError as e:
    logger.debug(f"Advanced ADMET not available: {e}")

try:
    from drug_discovery.evaluation.predictor import (
        ADMETPredictor as ADMETPredictor,
    )
    from drug_discovery.evaluation.predictor import (
        ModelEvaluator as ModelEvaluator,
    )
    from drug_discovery.evaluation.predictor import (
        PropertyPredictor as PropertyPredictor,
    )

    __all__.extend(["ADMETPredictor", "ModelEvaluator", "PropertyPredictor"])
except ImportError:
    pass

try:
    from drug_discovery.evaluation.torchdrug_scorer import TorchDrugScorer as TorchDrugScorer

    __all__.append("TorchDrugScorer")
except ImportError:
    pass
