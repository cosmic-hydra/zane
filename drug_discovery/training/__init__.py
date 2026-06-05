"""ZANE Training — Training loops and advanced utilities."""

from .cryptography import EncryptionProvider as EncryptionProvider
from .cryptography import PrivacyControl as PrivacyControl

__all__ = [
    "EncryptionProvider",
    "PrivacyControl",
]

try:
    from .federated_learning import FederatedServer as FederatedServer
    from .federated_learning import RobustFedAvg as RobustFedAvg
    from .federated_node import FederatedClient as FederatedClient

    __all__.extend(["FederatedClient", "FederatedServer", "RobustFedAvg"])
except ImportError:
    pass

try:
    from drug_discovery.training.advanced_training import (
        EMA as EMA,
    )
    from drug_discovery.training.advanced_training import (
        AdvancedTrainer as AdvancedTrainer,
    )
    from drug_discovery.training.advanced_training import (
        AdvancedTrainingConfig as AdvancedTrainingConfig,
    )
    from drug_discovery.training.advanced_training import (
        EarlyStopping as EarlyStopping,
    )
    from drug_discovery.training.advanced_training import (
        WarmupScheduler as WarmupScheduler,
    )

    __all__.extend(["EMA", "AdvancedTrainer", "AdvancedTrainingConfig", "EarlyStopping", "WarmupScheduler"])
except ImportError:
    pass

try:
    from drug_discovery.training.trainer import SelfLearningTrainer as SelfLearningTrainer

    if "SelfLearningTrainer" not in __all__:
        __all__.append("SelfLearningTrainer")
except Exception:
    pass
