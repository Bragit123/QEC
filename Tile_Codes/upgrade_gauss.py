from typing import Tuple, Optional, Callable

from panqec.error_models import PauliErrorModel
from panqec.codes import StabilizerCode
from panqec.bpauli import pauli_to_bsf

import numpy as np
from scipy.special import erfinv

from ldpc.bplsd_decoder import BpLsdDecoder
from decoders import BeliefPropagationLSDDecoder


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


def sample_from_gauss3D(std_X:float, std_Z: float, rng: Optional[np.random.Generator]=None) -> Tuple:
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
    from colorama import Back
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
        print(Back.RED + f"{Delta_X_m:^10.4f}|{Delta_Z_m:^10.4f}|{pauli_err:^10}" + Back.RESET)
        return (pauli_err, Delta_X_m, Delta_Z_m)
    if x_abs <= th and z_abs > th:
        Delta_X_m = np.abs(x_abs)
        Delta_Z_m = np.abs(dist - z_abs)
        pauli_err = "Z"
        print(Back.BLUE + f"{Delta_X_m:^10.4f}|{Delta_Z_m:^10.4f}|{pauli_err:^10}" + Back.RESET)
        return (pauli_err, Delta_X_m, Delta_Z_m)
    elif x_abs > th and z_abs > th:
        Delta_X_m = np.abs(dist - x_abs)
        Delta_Z_m = np.abs(dist - z_abs)
        pauli_err = "Y"
        print(Back.GREEN + f"{Delta_X_m:^10.4f}|{Delta_Z_m:^10.4f}|{pauli_err:^10}" + Back.RESET)
        return (pauli_err, Delta_X_m, Delta_Z_m)
    else:
        Delta_X_m = np.abs(x_abs)
        Delta_Z_m = np.abs(z_abs)
        pauli_err = "I"
        print(f"{Delta_X_m:^10.4f}|{Delta_Z_m:^10.4f}|{pauli_err:^10}")
        return (pauli_err, Delta_X_m, Delta_Z_m)


def gauss_likelihood(std:float) -> Callable[[float], float]:
    """
    Gaussian distribution.

    Parameters
    ----------
    std : float
        Standard deviation of the Gaussian distribution.
    
    Returns
    -------
    Callable
        Gaussian distribution function with the given standard deviation.
        It takes in a real number Delta and returns the Gaussian function
        evaluated at Delta.
    """
    def f(Delta:float):
        return 1/(np.sqrt(2*np.pi)*std) * np.exp(-Delta*Delta / (2*std*std))
    return f


def get_error_channel(std: float, Delta_m_arr: np.ndarray) -> np.ndarray:
    """
    Computes the error_channel to pass through to the decoder from the standard
    deviation of the Gaussian distribution and an array of measured values Delta_m.

    Parameters
    ----------
    std : float
        Standard deviation of the Gaussian distribution.
    Delta_m_arr : ndarray
        Array of measured deviations when generating errors.
    
    Returns
    -------
    ndarray
        The error channel to send into the decoder.
    """
    f_correct = gauss_likelihood(std)
    
    f_correct_Delta = f_correct(Delta_m_arr)
    f_incorrect_Delta = f_correct(np.sqrt(np.pi) - Delta_m_arr)
    f_norm = f_correct_Delta + f_incorrect_Delta
    error_channel = f_incorrect_Delta / f_norm

    return error_channel


class GaussPauliErrorModel3D(PauliErrorModel):
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

        ## Get the standard deviation of XYZ-errors as a 3-array.
        px, py, pz = error_rate * np.array(self.direction)
        qx = px + py
        qz = pz + py
        std_X = get_std(qx)
        std_Z = get_std(qz)

        self.std_X = std_X
        self.std_Z = std_Z

        error_pauli = ""
        Delta_X_m_arr = np.zeros(code.n)
        Delta_Z_m_arr = np.zeros(code.n)
        for i in range(code.n):
            pauli, Delta_X_m, Delta_Z_m = sample_from_gauss3D(std_X, std_Z, rng)
            Delta_X_m_arr[i] = Delta_X_m
            Delta_Z_m_arr[i] = Delta_Z_m

            error_pauli = error_pauli + pauli

        self.Delta_X_m_arr = Delta_X_m_arr
        self.Delta_Z_m_arr = Delta_Z_m_arr
        error = pauli_to_bsf(error_pauli)

        return error


class GaussBeliefPropagationLSDDecoder3D(BeliefPropagationLSDDecoder):
    """
    Decoder for a QEC model with error channels produced by Gaussian likelihoods as
    described by Fukui, Tomita and Okamoto in
        
    "High-Threshold Fault-Tolerant Quantum Computation with Analog Quantum Error Correction"

    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.021054
    """
    def initialize_decoders(self):
        std_X = self.error_model.std_X
        std_Z = self.error_model.std_Z
        Delta_X_m_arr = self.error_model.Delta_X_m_arr
        Delta_Z_m_arr = self.error_model.Delta_Z_m_arr
        error_channel_X = get_error_channel(std_X, Delta_X_m_arr)
        error_channel_Z = get_error_channel(std_Z, Delta_Z_m_arr)

        is_css = self.code.is_css
        if is_css:
            self.z_decoder = BpLsdDecoder(
                self.code.Hx,
                # error_rate=0.0, ##### NOTE: Only considering X-errors so far!
                error_channel=error_channel_Z,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                schedule="serial",
                lsd_method="lsd_cs",  # Choose from: "lsd_e", "lsd_cs", "lsd_0"
                lsd_order=self._osd_order,          
            )

            self.x_decoder = BpLsdDecoder(
                self.code.Hz,
                error_channel=error_channel_X,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                schedule="serial",
                lsd_method="lsd_cs",  # Choose from: "lsd_e", "lsd_cs", "lsd_0"
                lsd_order=self._osd_order
            )

        else:
            raise ValueError("Gaussian decoder should be CSS.")
            # self.decoder = BpLsdDecoder(
            #     self.code.stabilizer_matrix,
            #     error_channel=error_channel,
            #     max_iter=self._max_bp_iter,
            #     bp_method=self._bp_method,
            #     ms_scaling_factor=0.,
            #     lsd_method="lsd_cs",  # Choose from: "lsd_e", "lsd_cs", "lsd_0"
            #     lsd_order=self._osd_order
            # )
        self._initialized = True
    

    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""

        if not self._initialized:
            self.initialize_decoders()

        is_css = self.code.is_css
        n_qubits = self.code.n
        syndrome = np.array(syndrome, dtype=int)

        if is_css:
            syndrome_z = self.code.extract_z_syndrome(syndrome)
            syndrome_x = self.code.extract_x_syndrome(syndrome)

        pi, px, py, pz = self.get_probabilities()

        std_X = self.error_model.std_X
        std_Z = self.error_model.std_Z
        Delta_X_m_arr = self.error_model.Delta_X_m_arr
        Delta_Z_m_arr = self.error_model.Delta_Z_m_arr
        probabilities_x = get_error_channel(std_X, Delta_X_m_arr)
        probabilities_z = get_error_channel(std_Z, Delta_Z_m_arr)
        # probabilities_z = np.zeros(self.code.n)

        probabilities = np.hstack([probabilities_z, probabilities_x])

        if is_css:
            # Update probabilities (in case the distribution is new at each
            # iteration)
            self.x_decoder.update_channel_probs(probabilities_x)
            self.z_decoder.update_channel_probs(probabilities_z)

            # Decode Z errors
            z_correction = self.z_decoder.decode(syndrome_x)

            # Bayes update of the probability
            if self._channel_update:
                print("UPDATE PROB")
                new_x_probs = self.update_probabilities(
                    z_correction, px, py, pz, direction="z->x"
                )
                self.x_decoder.update_channel_probs(new_x_probs)

            # Decode X errors
            x_correction = self.x_decoder.decode(syndrome_z)

            correction = np.concatenate([x_correction, z_correction])
        else:
            # Update probabilities (in case the distribution is new at each
            # iteration)
            self.decoder.update_channel_probs(probabilities)

            # Decode all errors
            self.decoder.decode(syndrome)
            correction = self.decoder.osdw_decoding
            correction = np.concatenate(
                [correction[n_qubits:], correction[:n_qubits]]
            )

        return correction
