"""
Based on the Gaussian likelihood error correction as described by Fukui, Tomita and
Okamoto in

"High-Threshold Fault-Tolerant Quantum Computation with Analog Quantum Error Correction"

https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054
"""

from typing import Tuple
import functools

from panqec.error_models import PauliErrorModel
from panqec.codes import StabilizerCode
from panqec.bpauli import pauli_to_bsf

import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton
from scipy.special import erfinv


def p_corr(std):
    """
    Likelihood of correct decision as a function of the standard deviation of
    the Gaussian distribution.
    """
    def gauss(x):
        return 1/(np.sqrt(2*np.pi)*std) * np.exp(-x*x/(2*std*std))
    
    th = np.sqrt(np.pi)/2
    return quad(gauss, -th, th)[0]


# def get_std(p):
#     """
#     Find the standard deviation of a gauss distribution such that the likelihood
#     of incorrect decision is p
#     """
#     def zero_func(std):
#         return p_corr(std) + p - 1
    
#     return newton(zero_func, 1)

def get_std(p):
    """
    Find the standard deviation of a gauss distribution such that the likelihood
    of incorrect decision is p
    """
    # return np.sqrt(np.pi/8.0) * erfinv(p)
    std = np.sqrt(np.pi/8.0) / erfinv(1.0-p)
    return std


def sample_from_gauss(std, rng=None):
    """
    Draw a sample from a normal distribution, and return a tuple containing the
    measured deviation Delta_m and whether 
    """
    rng = np.random.default_rng() if rng is None else rng

    dist = np.sqrt(np.pi) # Distance between q-values
    th = dist / 2.0 # Threshold for producing an error
    x = rng.normal(loc=0.0, scale=std)
    Delta_bar = np.abs(x)

    # if Delta_bar <= th:
    #     Delta_m = Delta_bar
    #     return (Delta_m, 0, Delta_bar) # 0 = no error
    # else:
    #     Delta_m = dist - Delta_bar
    #     return (Delta_m, 1, Delta_bar) # 1 = error
    if np.abs(x) <= th:
        Delta_m = np.abs(x)
        return (Delta_m, 0, Delta_bar) # 0 = no error
    else:
        Delta_m = np.abs(dist - x)
        return (Delta_m, 1, Delta_bar) # 1 = error

class GaussPauliErrorModel(PauliErrorModel):
    def generate(self, code: StabilizerCode, error_rate: float, rng=None):
        rng = np.random.default_rng() if rng is None else rng

        p = error_rate
        std = get_std(p)
        self.std = std

        error_pauli = ""
        Delta_m_arr = np.zeros(code.n)
        Delta_bar_arr = np.zeros(code.n)
        for i in range(code.n):
            Delta_m, binary_error, Delta_bar = sample_from_gauss(std, rng)
            Delta_m_arr[i] = Delta_m
            Delta_bar_arr[i] = Delta_bar
            
            pauli = "X" if binary_error==1 else "I"
            error_pauli = error_pauli + pauli

        self.Delta_m_arr = Delta_m_arr
        self.Delta_bar_arr = Delta_bar_arr
        error = pauli_to_bsf(error_pauli)

        return error

    @functools.lru_cache()
    def probability_distribution(
        self, code: StabilizerCode, error_rate: float
    ) -> Tuple:
        n = code.n
        r_x, r_y, r_z = self.direction

        p: dict = {}
        p['I'] = (1 - error_rate) * np.ones(n)
        p['X'] = (r_x * error_rate) * np.ones(n)
        p['Y'] = (r_y * error_rate) * np.ones(n)
        p['Z'] = (r_z * error_rate) * np.ones(n)

        if self._deformation_name is not None:
            for i in range(code.n):
                deformation = code.get_deformation(
                    code.qubit_coordinates[i], self._deformation_name,
                    **self._deformation_kwargs
                )
                previous_p = {pauli: p[pauli][i] for pauli in ['X', 'Y', 'Z']}
                for pauli in ['X', 'Y', 'Z']:
                    p[pauli][i] = previous_p[deformation[pauli]]

        return p['I'], p['X'], p['Y'], p['Z']


if __name__ == "__main__":
    from panqec.codes import Toric2DCode
    from colorama import Back, Style
    
    code = Toric2DCode(6)
    error_model = GaussPauliErrorModel(1.0, 0.0, 0.0)
    error_rate = 0.1
    error = error_model.generate(code, error_rate)

    spaceing = 11
    print(f"{"Delta_bar":^11}|{"Delta_m":^11}|{"error":^11}")
    print("-"*11 + "+" + "-"*11 + "+" + "-"*11)
    for i in range(code.n):
        err = error[i]
        Delta_m = error_model.Delta_m_arr[i]
        Delta_bar = error_model.Delta_bar_arr[i]

        if err == 1:
            color_style = Back.RED
        else:
            color_style = Style.RESET_ALL

        print(color_style + f"{Delta_bar:^11.4}|{Delta_m:^11.4}|{err:^11}" + Style.RESET_ALL)

    print(np.all(error[code.n:]==0))