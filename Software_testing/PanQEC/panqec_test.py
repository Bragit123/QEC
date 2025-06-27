from panqec.codes import Toric2DCode
from panqec.error_models import PauliErrorModel
from panqec.decoders import MatchingDecoder
import matplotlib.pyplot as plt

import numpy as np

np.random.seed(100)

code = Toric2DCode(4)

# print(f"[[n,k,d]]=[[{code.n},{code.k},{code.d}]]")

error_model = PauliErrorModel(0.5, 0.0, 0.5)
p = 0.1

decoder = MatchingDecoder(code, error_model, p)

rng = np.random.default_rng(seed=100)
errors = error_model.generate(code, p, rng=rng)
syndrome = code.measure_syndrome(errors)

# print(f"Errors ({len(errors)}): ", errors)
# print(f"Syndrome ({len(syndrome)}): ", syndrome)

# Obtain the correction by decoding the syndrome
correction = decoder.decode(syndrome)
# print("Correction: ", correction)

# Get the remaining error once the correction has been applied
residual_error = (correction + errors) % 2
# print("Residual error: ", residual_error)

# Check whether the residual error is in the codespace
# (i.e. whether there is any remaining excitation)
in_codespace = code.in_codespace(residual_error)
# print("Is in codespace: ", in_codespace)

# Get the logical errors
logical_errors = code.logical_errors(residual_error)
# print("Logical errors: ", logical_errors)

# Check whether the decoding succeeded
success = not code.is_logical_error(residual_error) and in_codespace
# print("Success: ", success)

# print(code.stabilizer_matrix)
# print(code.stabilizer_matrix.toarray())

# Check that the code is CSS
# print("Is CSS: ", code.is_css)

# Extract Hx and Hz
# print("Hx shape", code.Hx.shape)
# print("Hz shape", code.Hz.shape)

error_model = PauliErrorModel(0.2, 0.3, 0.5)
p = 0.1
errors = error_model.generate(code, p)
syndrome = code.measure_syndrome(errors)

# vertex_syndrome = code.extract_x_syndrome(syndrome)
# face_syndrome = code.extract_z_syndrome(syndrome)

# print("Vertex syndrome", vertex_syndrome)
# print("Face syndrome", face_syndrome)

# print("Logicals X", code.logicals_x)
# print("Logicals Z", code.logicals_z)

# plt.imshow(vertex_syndrome.reshape((4, 4)))
# plt.savefig("vertex_syndrome.pdf")
# plt.imshow(face_syndrome.reshape((4, 4)))
# plt.savefig("face_syndrome.pdf")
# plt.imshow(errors[:32].reshape((8, 4)))
# plt.savefig("error_X_h.pdf")
# plt.imshow(errors[32:].reshape((8, 4)))
# plt.savefig("error_Z.pdf")

import numpy as np
qubit_coord = code.get_qubit_coordinates()
stab_coord = code.get_stabilizer_coordinates()
errors_X = errors[:32]
errors_Z = errors[32:]
err_arg_X = np.argwhere(errors_X==1)
err_arg_Z = np.argwhere(errors_Z==1)
# ver_arg = np.argwhere(vertex_syndrome==1)
# face_arg = np.argwhere(face_syndrome==1)
synd_arg = np.argwhere(syndrome==1)

err_coord_X = []
err_coord_Z = []
# ver_coord = []
# face_coord = []
synd_coord = []
for ind in err_arg_X:
    ind = ind[0]
    err_coord_X.append(qubit_coord[ind])
for ind in err_arg_Z:
    ind = ind[0]
    err_coord_Z.append(qubit_coord[ind])
# for ind in ver_arg:
#     ind = ind[0]
#     ver_coord.append(stab_coord[ind])
# for ind in face_arg:
#     ind = ind[0]
#     face_coord.append(stab_coord[ind])
for ind in synd_arg:
    ind = ind[0]
    synd_coord.append(stab_coord[ind])

# print(ver_arg)
# print(synd_arg[16:])
# print(face_arg)
# print(synd_arg[:16])

print(len(syndrome), syndrome)
print(len(stab_coord))

print(len(synd_arg), synd_arg)
print(len(err_arg_X), err_arg_X)
print(len(err_arg_Z), err_arg_Z)

print(err_coord_X)
print(err_coord_Z)
print(synd_coord)

print(qubit_coord)
print(stab_coord)