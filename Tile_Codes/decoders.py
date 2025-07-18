import numpy as np
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bp_decoder import BpDecoder
from panqec.decoders import BeliefPropagationOSDDecoder
from scipy.integrate import quad

def gauss_likelihood(std:float):
    def f(Delta):
        return 1/(np.sqrt(2*np.pi)*std) * np.exp(-Delta*Delta / (2*std*std))
    return f


class BeliefPropagationLSDDecoder(BeliefPropagationOSDDecoder):
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
    def initialize_decoders(self):
        std = self.error_model.std
        Delta_m_arr = self.error_model.Delta_m_arr
        th = np.sqrt(np.pi) / 2.0
        f_correct = gauss_likelihood(std)
        f_norm = quad(f_correct, 0.0, th)[0]
        f_correct_Delta = f_correct(Delta_m_arr) / f_norm
        f_incorrect_Delta = 1.0 - f_correct_Delta
        # f_incorrect_Delta = f_correct(np.sqrt(np.pi) - Delta_m_arr)
        # f_incorrect_Delta = f_correct(Delta_m_arr)
        # print(f"{np.min(Delta_m_arr):6.3} | {np.max(Delta_m_arr):6.3} | {np.mean(Delta_m_arr):6.3}")
        # print(f"{np.min(f_incorrect_Delta):10.3} | {np.max(f_incorrect_Delta):10.3} | {np.mean(f_incorrect_Delta):10.3}")
        # f_incorrect_Delta = np.zeros(self.code.n)


        is_css = self.code.is_css
        if is_css:
            self.z_decoder = BpLsdDecoder(
            # self.z_decoder = BpDecoder(
                self.code.Hx,
                # error_rate=self.error_rate,
                error_channel = f_incorrect_Delta,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                schedule="serial",
                lsd_method="lsd_cs",  # Choose from: "lsd_e", "lsd_cs", "lsd_0"
                lsd_order=self._osd_order,          
            )

            self.x_decoder = BpLsdDecoder(
            # self.x_decoder = BpDecoder(
                self.code.Hz,
                # error_rate=self.error_rate,
                error_channel = f_incorrect_Delta,
                max_iter=self._max_bp_iter,
                bp_method=self._bp_method,
                ms_scaling_factor=0.,
                schedule="serial",
                lsd_method="lsd_cs",  # Choose from: "lsd_e", "lsd_cs", "lsd_0"
                lsd_order=self._osd_order
            )

        else:
            self.decoder = BpLsdDecoder(
            # self.decoder = BpDecoder(
                self.code.stabilizer_matrix,
                # error_rate=self.error_rate,
                error_channel = f_incorrect_Delta,
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
        # self.initialize_decoders()

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
