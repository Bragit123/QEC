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
    eps = 1e-10
    std = np.sqrt(np.pi/8.0) / (erfinv(1.0-p) + eps)
    return std


def sample_from_gauss(std_X:float, std_Z: float, rng: Optional[np.random.Generator]=None) -> Tuple:
    """
    Draw a sample from a normal distribution, and return a tuple containing the
    measured deviation Delta_m and whether the sample produced an error.
    
    The sample produces an error if the measured value is outside the interval
    [-sqrt(pi)/2, sqrt(pi)/2], as described in
    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054.

    Parameters
    ----------
    std : ndarray
        3-dimensional array of the standard deviation in the X, Y and Z
        directions respectively.
    rng : np.random.Generator   
        Random number generator to use for sampling. If None: Defaults to
        np.random.default_rng().
    
    Returns
    -------
    ndarray
        The minimized deviation Delta_m.
    str
        The error. 0 if no error occured, 1 if it did.
    """
    rng = np.random.default_rng() if rng is None else rng

    dist = np.sqrt(np.pi) # Distance between q-values
    th = dist / 2.0 # Threshold for producing an error

    x = rng.normal(loc=0.0, scale=std_X)
    z = rng.normal(loc=0.0, scale=std_Z)

    x_abs = np.abs(x)
    z_abs = np.abs(z)

    if x_abs > th and z_abs <= th:
        Delta_X_m = np.abs(dist - x_abs)
        Delta_Z_m = np.abs(z_abs)
        pauli_err = "X"
    elif x_abs <= th and z_abs > th:
        Delta_X_m = np.abs(x_abs)
        Delta_Z_m = np.abs(dist - z_abs)
        pauli_err = "Z"
    elif x_abs > th and z_abs > th:
        Delta_X_m = np.abs(dist - x_abs)
        Delta_Z_m = np.abs(dist - z_abs)
        pauli_err = "Y"
    else:
        Delta_X_m = np.abs(x_abs)
        Delta_Z_m = np.abs(z_abs)
        pauli_err = "I"
    
    return (pauli_err, Delta_X_m, Delta_Z_m)


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

        r_x, r_y, r_z = self.direction

        ## Check whether we should generate X-errors, Z-errors or both.
        x_and_z_errors = r_x==0.5 and r_y == 0.0 and r_z == 0.5
        only_x_errors = r_x==1.0 and r_y == 0.0 and r_z == 0.0
        only_z_errors = r_x==0.0 and r_y == 0.0 and r_z == 1.0
        if x_and_z_errors:
            q = 1 - np.sqrt(1-error_rate)
            qx = q
            qz = q
        elif only_x_errors:
            qx = error_rate
            qz = 0
        elif only_z_errors:
            qx = 0
            qz = error_rate
        else:
            raise ValueError(f"Due to the way errors are sampled in GaussPauliErrorModel there are only three valid inputs for (r_x,r_y,r_z):\n  - (1.0, 0.0, 0.0)\n  - (0.0, 0.0, 1.0)\n  - (0.5, 0.0, 0.5)\nYour input: ({r_x:.2f}, {r_y:.2f}, {r_z:.2f})")

        std_X = get_std(qx)
        std_Z = get_std(qz)

        self.std_X = std_X
        self.std_Z = std_Z

        error_pauli = ""
        Delta_X_m_arr = np.zeros(code.n)
        Delta_Z_m_arr = np.zeros(code.n)
        for i in range(code.n):
            pauli, Delta_X_m, Delta_Z_m = sample_from_gauss(std_X, std_Z, rng)
            Delta_X_m_arr[i] = Delta_X_m
            Delta_Z_m_arr[i] = Delta_Z_m

            error_pauli = error_pauli + pauli

        self.Delta_X_m_arr = Delta_X_m_arr
        self.Delta_Z_m_arr = Delta_Z_m_arr
        error = pauli_to_bsf(error_pauli)

        return error


if __name__ == "__main__":
    from panqec.codes import Toric2DCode
    from colorama import Back, Style
    
    code = Toric2DCode(6)
    error_model = GaussPauliErrorModel(0.5, 0.0, 0.5)
    error_rate = 0.4
    error = error_model.generate(code, error_rate)
    n_err = len(error)

    print(f"{"Delta_X_m":^11}|{"Delta_Z_m":^11}|{"X error":^11}|{"Z error":^11}")
    print("-"*11 + "+" + "-"*11 + "+" + "-"*11 + "+" + "-"*11)
    for i in range(code.n):
        err_X = error[i]
        err_Z = error[n_err//2 + i]
        Delta_X_m = error_model.Delta_X_m_arr[i]
        Delta_Z_m = error_model.Delta_Z_m_arr[i]
        if err_X == 1 and err_Z == 1:
            color_style = Back.GREEN
        elif err_X == 1:
            color_style = Back.RED
        elif err_Z == 1:
            color_style = Back.BLUE
        else:
            color_style = Style.RESET_ALL

        print(color_style + f"{Delta_X_m:^11.4}|{Delta_Z_m:^11.4}|{err_X:^11}|{err_Z:^11}" + Style.RESET_ALL)

    print(np.all(error[code.n:]==0))