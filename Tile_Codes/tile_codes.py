import numpy as np
from panqec.codes import StabilizerCode
from panqec.gui import GUI
from bposd.css import css_code


class TileCodes(StabilizerCode):
    dimension = 2

    # def __init__(L_x, L_y=None, L_z=None, B: int = 3, x_tile: List = None):
    #     """ tile = list of coordinates for 'delta' in get_stabilizer """
    #     super().__init__(L_x, L_y, L_z)
        
    #     delta_X = [
    #         (1,0),
    #         (4,1),
    #         (5,2),
    #         (5,4),
    #         (0,5),
    #         (2,5),
    #     ]

    #     delta_Z = []
    #     for i in range(len(delta_X)):
    #         x = delta_X[i][0]
    #         y = delta_X[i][1]
    #         delta_Z.append((2*(B-1)-x, 2*(B-1)-y))
            

    @property
    def label(self):
        return f"Tile Code {self.size[0]}x{self.size[1]}"
    
    def get_qubit_coordinates(self):
        B = 3
        coordinates = []
        Lx, Ly = self.size
        
        # Qubits oriented along x-axis
        for x in range(2*(B-1)+1, 2*(Lx+B-1), 2):
            for y in range(2*(B-1), 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        # Qubits oriented along y-axis
        for x in range(2*(B-1), 2*(Lx+B-1), 2):
            for y in range(2*(B-1)+1, 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        return coordinates
    
    def qubit_axis(self, location):
        x, y = location

        if (y % 2 == 0) and (not x % 2 == 0):
            axis = "x"
        
        elif (x % 2 == 0) and (not y % 2 == 0):
            axis = "y"
        
        else:
            raise ValueError("Location in qubit_axis() must be an edge.")

        return axis
    
    def get_stabilizer_coordinates(self):
        coordinates = []
        Lx, Ly = self.size
        B = 3
        
        # X errors (red/black dots)
        for x in range(2*(B-1), 2*Lx, 2):
            for y in range(0, 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        # Z errors (blue/black dots, but placed on faces)
        for x in range(1, 2*(Lx+B-1), 2):
            for y in range(2*(B-1)+1, 2*Ly, 2):
                coordinates.append((x, y))
        
        return coordinates

    def stabilizer_type(self, location):
        x, y = location
        if (x % 2 == 0) and (y % 2 == 0):
            return "vertex"
        elif (x % 2 == 1) and (y % 2 == 1):
            return "face"
        else:
            raise ValueError("stabilizer_type() must return either 'vertex' or 'face'")

    def get_stabilizer(self, location):
        if self.stabilizer_type(location) == "vertex":
            pauli = "X"
        elif self.stabilizer_type(location) == "face":
            pauli = "Z"
        else:
            raise ValueError("stabilizer_type() must return either 'vertex' or 'face'")
        
        x, y = location

        # delta specifies the positions of the qubits involved in the stabilizer
        # relative to the stabilizer position
        if self.stabilizer_type(location) == "vertex":
            # X type
            delta = [
                (1, 0),
                (4, 1),
                (5, 2),
                (5, 4),
                (0, 5),
                (2, 5)
            ]
        else:
            # Z type
            # Note: The stabilizer is defined on the face. Thus the delta gets an extra
            # minus one in both the x- and y-coordinates compared to the X stabilizer.
            delta = [
                (2, -1),
                (4, -1),
                (-1, 0),
                (-1, 2),
                (0, 3),
                (3, 4)
            ]
        
        B = 3
        operator = dict()
        for d in delta:
            qubit_location = (x + d[0], y + d[1])

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli
        
        return operator

    def get_logicals_x(self):
        ham_mat = self._get_hamming_matrix()
        n_stabilizers_X = self.n_stabilizers // 2
        Hx = ham_mat[:n_stabilizers_X, :]
        Hz = ham_mat[n_stabilizers_X:, :]
        bposd_code = css_code(Hx, Hz)
        Lx = bposd_code.lx

        logicals = []
        for i in range(Lx.shape[0]):
            operator = dict()
            indices = np.argwhere(Lx[i,:] == 1)[:,1]
            for ind in indices:
                coord = self.get_qubit_coordinates()[ind]
                operator[coord] = "X"
            logicals.append(operator)
        
        return logicals

    def get_logicals_z(self):
        ham_mat = self._get_hamming_matrix()
        n_stabilizers_X = self.n_stabilizers // 2
        Hx = ham_mat[:n_stabilizers_X, :]
        Hz = ham_mat[n_stabilizers_X:, :]
        bposd_code = css_code(Hx, Hz)
        Lz = bposd_code.lz

        logicals = []
        for i in range(Lz.shape[0]):
            operator = dict()
            indices = np.argwhere(Lz[i,:] == 1)[:,1]
            for ind in indices:
                coord = self.get_qubit_coordinates()[ind]
                operator[coord] = "Z"
            logicals.append(operator)
        
        return logicals
    
    def qubit_representation(self, location, rotated_picture=False):
        return super().qubit_representation(location, rotated_picture, json_file="tile_code.json")
    
    def stabilizer_representation(self, location, rotated_picture=False):
        res_dict = super().stabilizer_representation(location, rotated_picture, json_file="tile_code.json")
        if self.stabilizer_type(location) == "face":
            x, y = location
            location = [x-1, y-1]
            res_dict["location"] = location
        else:
            x, y = location
            location = [x, y, 1]
            res_dict["location"] = location
        return res_dict
    
    def _get_Z_tile(self, X_tile):
        B = 3
        dB = B-1 # Number of "extra" stabilizers on each side

        Z_tile = np.zeros(X_tile.shape, dtype=int)
        Z_tile[[0,1]] = (dB - X_tile)[[1,0]] # Index swapping since horizontal X errors give vertical Z errors and vice versa
        return Z_tile

    def _get_tile_indices(self, ind):
        """
        Input: Index of stabilizer (row in Hamming matrix, with X tiles first then Z tiles)
        Output: List of indices of qubit errors (columns in Hamming matrix)
        """
        X_tile = np.array([
            [[0,0],[2,1],[2,2]],
            [[2,0],[0,2],[1,2]]
        ], dtype=int)

        L = self.L_x
        B = 3
        dB = B-1 # Number of "extra" stabilizers on each side

        L_small = L - dB # Length of the short side of red/blue dots
        L_big = L + dB # Length of the long side of red/blue dots

        n_stabilizers_X = L_small*L_big # Number of X (or Z) stabilizers
        if ind < n_stabilizers_X:
            # X stabilizer
            x = ind // L_big
            y = ind % L_big - dB # Subtract dB to get coordinates relative to qubits
            tile = X_tile
        else:
            # Z stabilizer
            ind = ind - n_stabilizers_X
            x = ind // L_small - dB # Subtract dB to get coordinates relative to qubits
            y = ind % L_small
            tile = self._get_Z_tile(X_tile)

        indices = []
        for is_vert, delta_i in enumerate(tile):
            for d in delta_i:
                dx, dy = (d[0], d[1])
                qx, qy = (x + dx, y + dy)
                valid_x = qx >= 0 and qx < L
                valid_y = qy >= 0 and qy < L
                is_qubit = valid_x and valid_y
                if is_qubit:
                    ind = qx*L + qy + L*L*is_vert # Add L^2 if the qubits are vertical
                    indices.append(ind)

        indices = np.array(indices)
        return indices
    
    def _get_hamming_matrix(self):
        L = self.L_x
        B = 3
        dB = B-1 # Number of "extra" stabilizers on each side

        L_small = L - dB # Length of the short side of red/blue dots
        L_big = L + dB # Length of the long side of red/blue dots

        n_qubits = 2*L*L
        n_stabilizers = 2*L_small*L_big # Number of X (or Z) stabilizers
        ham_mat = np.zeros((n_stabilizers, n_qubits), dtype=int)
        
        for i in range(n_stabilizers):
            indices = self._get_tile_indices(i)
            ham_mat[i,indices] = 1
        
        return ham_mat


# gui = GUI()
# gui.add_code(TileCodes, "Tile Code")
# gui.run(port=5000)

# code = TileCodes(12)

# import numpy as np
# from panqec.error_models import PauliErrorModel
# error_model = PauliErrorModel(0.5, 0.0, 0.5)
# errors = error_model.generate(code, 0.1)

# n_err = len(errors)
# n = n_err // 2
# x_err = errors[:n]
# z_err = errors[n:]

# log_x = np.zeros(n_err)
# qubit_coords = code.get_qubit_coordinates()
# for ind, coord in enumerate(qubit_coords):
#     if coord[1] == 4:
#         print(f"X: {ind}: {coord}")
#         log_x[ind] = 1

# print(code.in_codespace(log_x))
# print(code.is_logical_error(log_x))

# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# from panqec.decoders import BeliefPropagationOSDDecoder
# from panqec.error_models import PauliErrorModel
# from panqec.simulation import DirectSimulation, BatchSimulation
# from panqec.analysis import Analysis

# error_model = PauliErrorModel(1/3, 1/3, 1/3)

# p_vals = np.linspace(0.1, 0.6, 15).tolist()
# L_vals = [8, 12, 16]

# batch_sim = BatchSimulation("sim_output_tile_codes.json")

# for L in L_vals:
#     code = TileCodes(L)
#     for p in p_vals:
#         decoder = BeliefPropagationOSDDecoder(code, error_model, p)
#         dir_sim = DirectSimulation(code, error_model, decoder, p)
#         batch_sim.append(dir_sim)

# n_trials = 1000
# batch_sim.run(n_trials, progress=tqdm)

# analysis = Analysis("sim_output_tile_codes.json")

# fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

# plt.sca(ax[0])
# analysis.plot_thresholds()
# plt.sca(ax[1])
# analysis.plot_thresholds(sector='X')
# plt.sca(ax[2])
# analysis.plot_thresholds(sector='Z')
# fig.savefig("thresholds_tile_codes.pdf", bbox_inches="tight")

# analysis.plot_thresholds(pdf="thresh_tile_codes.pdf")

from ldpc import BpDecoder
from ldpc import BpOsdDecoder

L = 6; B = 3; dB = B-1
L_small = L-dB; L_big = L+dB

n_qubits = 2*L*L
n_stabilizers_X = L_small*L_big

code = TileCodes(L)

H = code._get_hamming_matrix()
Hx = H[:n_stabilizers_X, :]
Hz = H[n_stabilizers_X:, :]
p = 0.1

bp_osd_X = BpOsdDecoder(
    Hx,
    error_rate=p,
    bp_method="product_sum",
    max_iter=7,
    schedule="serial",
    osd_method="osd_cs",
    osd_order=2
)
bp_osd_Z = BpOsdDecoder(
    Hz,
    error_rate=p,
    bp_method="product_sum",
    max_iter=7,
    schedule="serial",
    osd_method="osd_cs",
    osd_order=2
)

error = np.random.binomial(1, p, 2*n_qubits)
error_X = error[:n_qubits]
error_Z = error[n_qubits:]

syndrome_X = (Hx @ error_Z) % 2
syndrome_Z = (Hz @ error_X) % 2
syndrome = np.concatenate([syndrome_X, syndrome_Z])

pan_syndrome = code.measure_syndrome(error)

correction_Z = bp_osd_X.decode(syndrome_X)
correction_X = bp_osd_Z.decode(syndrome_Z)
correction = np.concatenate([correction_X, correction_Z])

residual_error = (error + correction) % 2

print(code.in_codespace(residual_error))


# print(H.shape)
# print(error.shape)
# print(error_X.shape)
# print(error_Z.shape)
# print(syndrome.shape)
# print(syndrome_X.shape)
# print(syndrome_Z.shape)
# print(correction.shape)
# print(correction_X.shape)
# print(correction_Z.shape)
# print(residual_error_X.shape)
# print(residual_error_Z.shape)

# error_X = error[:n_qubits]
# error_Z = error[n_qubits:]

# syndrome_X = Hx @ error_X % 2
# syndrome_Z = Hz @ error_Z % 2
# syndrome = np.append(syndrome_X, syndrome_Z)

# correction = bpd.decode(syndrome)
# correction_X = bpd_X.decode(syndrome_X)
# correction_Z = bpd_Z.decode(syndrome_Z)

# print(correction)

# residual_error_X = (correction_X + error_X) % 2
# residual_error_Z = (correction_Z + error_Z) % 2

# residual_error = np.append(residual_error_X, residual_error_Z)
# residual_error = np.append(residual_error_Z, residual_error_X)

# print(residual_error)
# print(code.measure_syndrome(residual_error))
# print(code.in_codespace(residual_error))
# print(code.logical_errors(residual_error))
# print(code.is_logical_error(residual_error))