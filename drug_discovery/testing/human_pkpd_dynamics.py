import numpy as np
from scipy.integrate import odeint
from typing import List, Dict, Any

class HumanPKPDEngine:
    """
    Simulates Human Pharmacokinetics (PK) and Pharmacodynamics (PD).
    Uses a 3-compartment ODE model and Sigmoidal Emax PD mapping.
    """

    def __init__(self):
        # Default PK parameters (normalized for human adult)
        self.params = {
            "ka": 0.5,     # Absorption rate (1/hr)
            "ke": 0.1,     # Elimination rate (1/hr)
            "k12": 0.05,   # Central to peripheral 1 (1/hr)
            "k21": 0.03,   # Peripheral 1 to central (1/hr)
            "k13": 0.02,   # Central to peripheral 2 (1/hr)
            "k31": 0.01,   # Peripheral 2 to central (1/hr)
            "Vc": 15.0,    # Apparent volume of central compartment (L)
        }

    def _model_3comp(self, y, t, ka, ke, k12, k21, k13, k31, Vc):
        """
        System of ODEs for 3-compartment PK model.
        y[0]: Amount in absorption depot (mg)
        y[1]: Amount in central compartment (mg)
        y[2]: Amount in peripheral 1 (mg)
        y[3]: Amount in peripheral 2 (mg)
        """
        A_depot, A_central, A_peri1, A_peri2 = y
        
        # dA/dt
        dy0 = -ka * A_depot
        dy1 = ka * A_depot - (ke + k12 + k13) * A_central + k21 * A_peri1 + k31 * A_peri2
        dy2 = k12 * A_central - k21 * A_peri1
        dy3 = k13 * A_central - k31 * A_peri2
        
        return [dy0, dy1, dy2, dy3]

    def simulate_plasma_concentration(
        self, 
        dose_mg: float, 
        duration_hrs: int = 48, 
        points: int = 100
    ) -> np.ndarray:
        """
        Solves the 3-compartment model over the specified timeline.
        Returns central compartment concentration (mg/L).
        """
        t = np.linspace(0, duration_hrs, points)
        y0 = [dose_mg, 0.0, 0.0, 0.0]  # Initial state
        
        args = (
            self.params["ka"], self.params["ke"], 
            self.params["k12"], self.params["k21"], 
            self.params["k13"], self.params["k31"], 
            self.params["Vc"]
        )
        
        sol = odeint(self._model_3comp, y0, t, args=args)
        
        # Concentration C = Amount / Volume
        c_central = sol[:, 1] / self.params["Vc"]
        return t, c_central

    def evaluate_pd_efficacy(
        self, 
        concentrations: np.ndarray, 
        emax: float = 100.0, 
        ec50: float = 1.5, 
        gamma: float = 2.0
    ) -> np.ndarray:
        """
        Sigmoidal Emax model to map concentration to therapeutic effect (0-100%).
        E = (Emax * C^gamma) / (EC50^gamma + C^gamma)
        """
        effect = (emax * np.power(concentrations, gamma)) / (np.power(ec50, gamma) + np.power(concentrations, gamma))
        return np.nan_to_num(effect)

    def run_full_simulation(self, dose_mg: float) -> Dict[str, Any]:
        """Runs end-to-end PK/PD validation."""
        t, c_central = self.simulate_plasma_concentration(dose_mg)
        effects = self.evaluate_pd_efficacy(c_central)
        
        return {
            "time_points": t.tolist(),
            "plasma_concentrations": c_central.tolist(),
            "therapeutic_effects": effects.tolist(),
            "c_max": float(np.max(c_central)),
            "t_max": float(t[np.argmax(c_central)]),
            "auc": float(np.trapz(c_central, t))
        }
