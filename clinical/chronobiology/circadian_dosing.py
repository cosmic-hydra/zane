import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CircadianDosingOptimizer:
    """
    Optimizes drug dosing schedules based on patient-specific circadian rhythms 
    derived from wearable telemetry.
    """
    def __init__(self):
        self.telemetry_data: Optional[pd.DataFrame] = None
        self.circadian_params: Dict[str, float] = {}

    def ingest_wearable_telemetry(self, timeseries_csv: str) -> None:
        """
        Parses 24-hour continuous biometric data to calculate the patient's circadian phase.
        Expected columns: timestamp, heart_rate, body_temperature, activity_level.
        """
        try:
            df = pd.read_csv(timeseries_csv)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
            self.telemetry_data = df
            
            # Fit a cosinor model to body temperature to find the acrophase (peak)
            # T(t) = M + A * cos(2*pi*t/24 - phi)
            # Dim Light Melatonin Onset (DLMO) is typically ~7 hours before the temperature nadir
            self._fit_circadian_model(df)
            
        except Exception as e:
            logger.error(f"Error ingesting telemetry: {str(e)}")
            # Default fallback for healthy adult
            self.circadian_params = {"acrophase": 16.0, "mesor": 37.0, "amplitude": 0.5}

    def _fit_circadian_model(self, df: pd.DataFrame):
        """Fits a cosinor model to the telemetry data."""
        t = df['hour'].values
        y = df['body_temperature'].values if 'body_temperature' in df else df['heart_rate'].values
        
        def model(params, t):
            mesor, amp, phi = params
            return mesor + amp * np.cos(2 * np.pi * t / 24 - phi)
        
        def residuals(params, t, y):
            return model(params, t) - y
        
        # Initial guess
        x0 = [np.mean(y), np.std(y), 0]
        res = least_squares(residuals, x0, args=(t, y))
        
        self.circadian_params = {
            "mesor": res.x[0],
            "amplitude": res.x[1],
            "acrophase": res.x[2] % (2 * np.pi) * 24 / (2 * np.pi)
        }
        logger.info(f"Circadian phase detected: Acrophase at {self.circadian_params['acrophase']:.2f}h")

    def calculate_optimal_tmax(self, target_receptor_peak_hour: float = 8.0) -> float:
        """
        Calculates the optimal hour to administer the drug.
        peak_plasma_concentration (Cmax) should align with the peak circadian expression 
        of the target disease receptor.
        """
        # Adjust target peak based on patient's specific phase shift
        # Standard healthy acrophase is approx 16:00 (4 PM)
        phase_shift = self.circadian_params.get("acrophase", 16.0) - 16.0
        
        # Shift the target receptor peak by the patient's individual clock shift
        individualized_target_hour = (target_receptor_peak_hour + phase_shift) % 24
        
        # If the drug takes 'absorption_delay' hours to reach Tmax
        absorption_delay = 2.0 # Assume 2 hours for standard oral delivery
        
        optimal_dosing_time = (individualized_target_hour - absorption_delay) % 24
        
        logger.info(f"Optimal dosing time calculated: {optimal_dosing_time:.2f}h "
                    f"to reach peak at {individualized_target_hour:.2f}h")
        
        return optimal_dosing_time
