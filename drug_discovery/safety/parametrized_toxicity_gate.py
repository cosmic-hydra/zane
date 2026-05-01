"""Parametrized Toxicity Gate with Configurable Thresholds.

Replaces hardcoded threshold values with configuration objects,
allowing regulatory tier adjustment without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToxicityThresholdConfig:
    """Parametrized toxicity thresholds (no hardcoding)."""
    
    # hERG-related thresholds
    herg_threshold: float = 0.3  # hERG inhibition probability
    herg_warning_threshold: float = 0.2  # Trigger additional scrutiny
    cyp3a4_inhibition_threshold: float = 0.5  # CYP3A4 substrate probability
    cyp2d6_inhibition_threshold: float = 0.4  # CYP2D6 inhibition
    
    # Mutagenicity thresholds
    ames_threshold: float = 0.3  # Ames mutagenicity
    
    # Hepatotoxicity thresholds
    hepatotox_threshold: float = 0.4  # Hepatotoxicity probable
    drug_induced_liver_injury_threshold: float = 0.35  # DILI risk
    
    # Cytotoxicity thresholds
    cytotox_threshold: float = 0.4  # General cytotoxicity
    
    # Physicochemical thresholds
    logp_max: float = 5.0
    logp_min: float = -1.0
    tpsa_max: float = 140.0
    tpsa_min: float = 0.0
    mw_max: float = 500.0
    mw_min: float = 50.0
    rotatable_bonds_max: int = 10
    
    # Bioavailability flags
    require_lipinski_compliance: bool = True
    lipinski_violations_max: int = 1  # Allow up to 1 violation
    
    # Regulatory flags
    require_no_known_toxicophores: bool = True
    allow_experimental_optimization: bool = False
    
    # Quality flags
    min_prediction_confidence: float = 0.5  # 0-1 scale
    require_vendor_approval: bool = False
    
    def __post_init__(self):
        """Validate threshold ranges."""
        if not (0 <= self.herg_threshold <= 1):
            raise ValueError("herg_threshold must be in [0, 1]")
        if not (0 <= self.ames_threshold <= 1):
            raise ValueError("ames_threshold must be in [0, 1]")
        if not (0 <= self.hepatotox_threshold <= 1):
            raise ValueError("hepatotox_threshold must be in [0, 1]")
        if self.logp_min >= self.logp_max:
            raise ValueError("logp_min must be < logp_max")
        if self.mw_min >= self.mw_max:
            raise ValueError("mw_min must be < mw_max")
    
    @classmethod
    def from_regulatory_tier(cls, tier: str) -> ToxicityThresholdConfig:
        """Create thresholds for regulatory submission tier.
        
        Args:
            tier: One of ['discovery', 'lead_optimization', 'ind', 'nda']
        """
        if tier == "discovery":
            return cls(
                herg_threshold=0.5,
                ames_threshold=0.4,
                hepatotox_threshold=0.5,
                min_prediction_confidence=0.3,
            )
        elif tier == "lead_optimization":
            return cls(
                herg_threshold=0.4,
                ames_threshold=0.35,
                hepatotox_threshold=0.45,
                min_prediction_confidence=0.4,
            )
        elif tier == "ind":
            return cls(
                herg_threshold=0.25,
                ames_threshold=0.15,
                hepatotox_threshold=0.2,
                min_prediction_confidence=0.6,
            )
        elif tier == "nda":
            return cls(
                herg_threshold=0.15,
                ames_threshold=0.1,
                hepatotox_threshold=0.1,
                min_prediction_confidence=0.75,
            )
        else:
            raise ValueError(f"Unknown regulatory tier: {tier}")


class ParametrizedToxicityGate:
    """Toxicity gate with configurable thresholds.
    
    Usage::
    
        # Standard thresholds
        gate = ParametrizedToxicityGate()
        
        # IND submission thresholds (stricter)
        config = ToxicityThresholdConfig.from_regulatory_tier("ind")
        gate = ParametrizedToxicityGate(config)
        
        # Custom thresholds
        config = ToxicityThresholdConfig(
            herg_threshold=0.2,
            ames_threshold=0.1,
        )
        gate = ParametrizedToxicityGate(config)
        
        result = gate.evaluate(
            herg_prob=0.1,
            ames_prob=0.05,
            hepatotox_prob=0.15,
        )
        if not result['passed']:
            print(f"Rejected: {result['reasons']}")
    """
    
    def __init__(self, config: Optional[ToxicityThresholdConfig] = None):
        """Initialize with thresholds (no hardcoding)."""
        self.config = config or ToxicityThresholdConfig()
    
    def evaluate(
        self,
        herg_prob: Optional[float] = None,
        ames_prob: Optional[float] = None,
        hepatotox_prob: Optional[float] = None,
        cytotox_prob: Optional[float] = None,
        logp: Optional[float] = None,
        tpsa: Optional[float] = None,
        mw: Optional[float] = None,
        rotatable_bonds: Optional[int] = None,
        confidence: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate molecule against configured thresholds.
        
        Args:
            herg_prob: hERG inhibition probability [0-1]
            ames_prob: Ames mutagenicity probability [0-1]
            hepatotox_prob: Hepatotoxicity probability [0-1]
            cytotox_prob: Cytotoxicity probability [0-1]
            logp: Lipophilicity
            tpsa: Topological polar surface area
            mw: Molecular weight
            rotatable_bonds: Number of rotatable bonds
            confidence: Model prediction confidence [0-1]
            **kwargs: Additional properties
            
        Returns:
            Dictionary with 'passed' (bool) and 'reasons' (list of rejection reasons)
        """
        passed = True
        reasons = []
        warnings = []
        
        # Check confidence
        if confidence is not None:
            if confidence < self.config.min_prediction_confidence:
                passed = False
                reasons.append(
                    f"Prediction confidence too low: {confidence:.2f} "
                    f"(minimum: {self.config.min_prediction_confidence})"
                )
        
        # Check hERG
        if herg_prob is not None:
            if herg_prob > self.config.herg_threshold:
                passed = False
                reasons.append(
                    f"hERG inhibition too high: {herg_prob:.3f} "
                    f"(threshold: {self.config.herg_threshold})"
                )
            elif herg_prob > self.config.herg_warning_threshold:
                warnings.append(
                    f"hERG inhibition warning: {herg_prob:.3f} "
                    f"(caution threshold: {self.config.herg_warning_threshold})"
                )
        
        # Check Ames
        if ames_prob is not None:
            if ames_prob > self.config.ames_threshold:
                passed = False
                reasons.append(
                    f"Ames mutagenicity too high: {ames_prob:.3f} "
                    f"(threshold: {self.config.ames_threshold})"
                )
        
        # Check hepatotoxicity
        if hepatotox_prob is not None:
            if hepatotox_prob > self.config.hepatotox_threshold:
                passed = False
                reasons.append(
                    f"Hepatotoxicity too high: {hepatotox_prob:.3f} "
                    f"(threshold: {self.config.hepatotox_threshold})"
                )
        
        # Check cytotoxicity
        if cytotox_prob is not None:
            if cytotox_prob > self.config.cytotox_threshold:
                passed = False
                reasons.append(
                    f"Cytotoxicity too high: {cytotox_prob:.3f} "
                    f"(threshold: {self.config.cytotox_threshold})"
                )
        
        # Check physicochemical properties
        if logp is not None:
            if not (self.config.logp_min <= logp <= self.config.logp_max):
                passed = False
                reasons.append(
                    f"LogP out of range: {logp:.2f} "
                    f"(allowed: [{self.config.logp_min}, {self.config.logp_max}])"
                )
        
        if tpsa is not None:
            if not (self.config.tpsa_min <= tpsa <= self.config.tpsa_max):
                passed = False
                reasons.append(
                    f"TPSA out of range: {tpsa:.1f} "
                    f"(allowed: [{self.config.tpsa_min}, {self.config.tpsa_max}])"
                )
        
        if mw is not None:
            if not (self.config.mw_min <= mw <= self.config.mw_max):
                passed = False
                reasons.append(
                    f"Molecular weight out of range: {mw:.1f} "
                    f"(allowed: [{self.config.mw_min}, {self.config.mw_max}])"
                )
        
        if rotatable_bonds is not None:
            if rotatable_bonds > self.config.rotatable_bonds_max:
                passed = False
                reasons.append(
                    f"Too many rotatable bonds: {rotatable_bonds} "
                    f"(maximum: {self.config.rotatable_bonds_max})"
                )
        
        return {
            "passed": passed,
            "reasons": reasons,
            "warnings": warnings,
            "active_thresholds": {
                "herg": self.config.herg_threshold,
                "ames": self.config.ames_threshold,
                "hepatotox": self.config.hepatotox_threshold,
            },
        }
    
    def update_thresholds(self, **kwargs: Any) -> None:
        """Update threshold values without recreating object.
        
        Args:
            **kwargs: Threshold parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise AttributeError(f"Unknown threshold: {key}")
