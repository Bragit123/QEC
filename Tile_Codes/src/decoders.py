from typing import Callable

from ldpc.bplsd_decoder import BpLsdDecoder
from panqec.decoders import BeliefPropagationOSDDecoder

import numpy as np
from colorama import Back, Style


def gauss_likelihood(std:float) -> Callable[[float], float]:
    """
    Returns a Gaussian distribution function with the given standard deviation.

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


class BeliefPropagationLSDDecoder(BeliefPropagationOSDDecoder):
    """
    Largely copied from PanQEC's BeliefPropagationOSDDecoder, but changed from
    OSD to LSD.
    """
    def initialize_decoders(self):
        is_css = self.code.is_css

        if is_css:
            self.z_decoder = BpLsdDecoder(
                self.code.Hx,
                error_rate=self.error_rate,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                schedule="serial",
                lsd_method="lsd_cs",  # Choose from: "lsd_e", "lsd_cs", "lsd_0"
                lsd_order=self._osd_order
            )

            self.x_decoder = BpLsdDecoder(
                self.code.Hz,
                error_rate=self.error_rate,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                schedule="serial",
                lsd_method="lsd_cs",  # Choose from: "lsd_e", "lsd_cs", "lsd_0"
                lsd_order=self._osd_order
            )

        else:
            self.decoder = BpLsdDecoder(
                self.code.stabilizer_matrix,
                error_rate=self.error_rate,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                lsd_method="lsd_cs",  # Choose from: "lsd_e", "lsd_cs", "lsd_0"
                lsd_order=self._osd_order
            )
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

        probabilities_x = px + py
        probabilities_z = pz + py

        probabilities = np.hstack([probabilities_z, probabilities_x])

        if is_css:
            # Update probabilities (in case the distribution is new at each iteration)
            self.x_decoder.update_channel_probs(probabilities_x)
            self.z_decoder.update_channel_probs(probabilities_z)

            # Decode Z errors
            z_correction = self.z_decoder.decode(syndrome_x)

            # Bayes update of the probability
            if self._channel_update:
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


class GaussBeliefPropagationLSDDecoder(BeliefPropagationLSDDecoder):
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
                new_x_probs = self.update_probabilities(
                    z_correction, px, py, pz, direction="z->x"
                )
                self.x_decoder.update_channel_probs(new_x_probs)

            # Decode X errors
            x_correction = self.x_decoder.decode(syndrome_z)

            correction = np.concatenate([x_correction, z_correction])
        else:
            # Update probabilities (in case the distribution is new at each iteration)
            self.decoder.update_channel_probs(probabilities)

            # Decode all errors
            self.decoder.decode(syndrome)
            correction = self.decoder.osdw_decoding
            correction = np.concatenate(
                [correction[n_qubits:], correction[:n_qubits]]
            )

        return correction


def test_decoder():
    from panqec.codes import XCubeCode
    from panqec.error_models import PauliErrorModel
    import time
    rng = np.random.default_rng()

    L = 20
    code = XCubeCode(L, L, L)

    error_rate = 0.1
    r_x, r_y, r_z = [0.15, 0.15, 0.7]
    error_model = PauliErrorModel(r_x, r_y, r_z)

    print("Create stabilizer matrix")
    code.stabilizer_matrix

    print("Create Hx and Hz")
    code.Hx
    code.Hz

    print("Create logicals")
    code.logicals_x
    code.logicals_z

    print("Instantiate BP-OSD")
    decoder = BeliefPropagationLSDDecoder(
        code, error_model, error_rate, osd_order=0, max_bp_iter=1000
    )

    # Start timer
    start = time.time()

    n_iter = 1
    accuracy = 0
    for i in range(n_iter):
        print(f"\nRun {code.label} {i}...")
        print("Generate errors")
        error = error_model.generate(code, error_rate, rng=rng)
        print("Calculate syndrome")
        syndrome = code.measure_syndrome(error)
        print("Decode")
        correction = decoder.decode(syndrome)
        print("Get total error")
        total_error = (correction + error) % 2

        codespace = code.in_codespace(total_error)
        success = not code.is_logical_error(total_error) and codespace
        print(success)
        accuracy += success

    accuracy /= n_iter
    print("Average time per iteration", (time.time() - start) / n_iter)
    print("Logical error rate", 1 - accuracy)


if __name__ == '__main__':
    test_decoder()
