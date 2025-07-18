from typing import Tuple, Optional

from panqec.error_models import PauliErrorModel
from panqec.codes import StabilizerCode
from panqec.bpauli import pauli_to_bsf

import numpy as np
from scipy.special import erfinv


def get_std(p: float):
    """
    Find the standard deviation of a gauss distribution such that the likelihood
    of incorrect decision is p.

    "Incorrect decision" here means measuring a value outside the interval
    [-sqrt(pi)/2, sqrt(pi)/2], as described in https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054.

    Parameters
    ----------
    p : float
        Probability of making an incorrect decision based on sampling from
        the Gaussian distribution.
    """
    std = np.sqrt(np.pi/8.0) / erfinv(1.0-p)
    return std


def sample_from_gauss(std: float, rng: Optional[np.random.Generator]=None) -> Tuple:
    """
    Draw a sample from a normal distribution, and return a tuple containing the
    measured deviation Delta_m and whether the sample produced an error.
    
    The sample produces an error if the measured value is outside the interval
    [-sqrt(pi)/2, sqrt(pi)/2], as described in https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054.

    Parameters
    ----------
    std : float
        Standard deviation of Gaussian distribution.
    rng : np.random.Generator   
        Random number generator to use for sampling. If None: Defaults to
        np.random.default_rng().
    
    Returns
    -------
    float
        The minimized deviation Delta_m.
    int
        The error. 0 if no error occured, 1 if it did.
    """
    rng = np.random.default_rng() if rng is None else rng

    dist = np.sqrt(np.pi) # Distance between q-values
    th = dist / 2.0 # Threshold for producing an error
    x = rng.normal(loc=0.0, scale=std)

    if np.abs(x) <= th:
        Delta_m = np.abs(x)
        return (Delta_m, 0) # 0 = no error
    else:
        Delta_m = np.abs(dist - x)
        return (Delta_m, 1) # 1 = error

class GaussPauliErrorModel(PauliErrorModel):
    """
    Error model for a QEC model with error channels produced by Gaussian likelihoods as
    described by Fukui, Tomita and Okamoto in
        
    "High-Threshold Fault-Tolerant Quantum Computation with Analog Quantum Error Correction"

    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054
    """
    def generate(self, code: StabilizerCode, error_rate: float, rng=None):
        """
        Generate errors. Copied and slightly modified from PanQEC.
        """
        rng = np.random.default_rng() if rng is None else rng

        p = error_rate
        std = get_std(p)
        self.std = std

        error_pauli = ""
        Delta_m_arr = np.zeros(code.n)
        for i in range(code.n):
            Delta_m, binary_error = sample_from_gauss(std, rng)
            Delta_m_arr[i] = Delta_m
            
            pauli = "X" if binary_error==1 else "I"
            error_pauli = error_pauli + pauli

        self.Delta_m_arr = Delta_m_arr
        error = pauli_to_bsf(error_pauli)

        return error


if __name__ == "__main__":
    from panqec.codes import Toric2DCode
    from colorama import Back, Style
    
    code = Toric2DCode(6)
    error_model = GaussPauliErrorModel(1.0, 0.0, 0.0)
    error_rate = 0.1
    error = error_model.generate(code, error_rate)

    spaceing = 11
    print(f"{"Delta_m":^11}|{"error":^11}")
    print("-"*11 + "+" + "-"*11)
    for i in range(code.n):
        err = error[i]
        Delta_m = error_model.Delta_m_arr[i]

        if err == 1:
            color_style = Back.RED
        else:
            color_style = Style.RESET_ALL

        print(color_style + f"{Delta_m:^11.4}|{err:^11}" + Style.RESET_ALL)

    print(np.all(error[code.n:]==0))