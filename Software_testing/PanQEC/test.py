import numpy as np
from panqec.codes.surface_2d import Planar2DCode
from panqec.error_models import PauliErrorModel
from panqec.decoders import MatchingDecoder

code = Planar2DCode(6)
# print(f"[[n,k,d]] = [[{code.n}, {code.k}, {code.d}]]")

error_model = PauliErrorModel(1/3, 1/3, 1/3)
p = 0.1

decoder = MatchingDecoder(code, error_model, error_rate=p)

errors = error_model.generate(code, p)
syndrome = code.measure_syndrome(errors)

correction = decoder.decode(syndrome)

residual_error = (correction + errors) % 2
in_codespace = code.in_codespace(residual_error)
print("In codespace after correction: ", in_codespace)

logical_errors = code.logical_errors(residual_error)
print("Logical errors: ", logical_errors)


is_logical_error = code.is_logical_error(residual_error)
success = (not is_logical_error) and in_codespace

print("Success: ", success)

# print(errors)
# print(syndrome)

# print("X_errors")
# for err in np.argwhere(errors[:code.n]==1)[:,0]:
#     print(code.get_qubit_coordinates()[err])
# print("Z_errors")
# for err in np.argwhere(errors[code.n:]==1)[:,0]:
#     print(code.get_qubit_coordinates()[err])
# print()
# print("Syndromes")
# for syn in np.argwhere(syndrome==1)[:,0]:
#     print(code.get_stabilizer_coordinates()[syn])


import os
if os.path.exists("planar_2d_results.json"):
    os.remove("planar_2d_results.json")

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from panqec.simulation import DirectSimulation, BatchSimulation
from panqec.analysis import Analysis

L_vals = [4, 6, 12, 18, 24]
p_vals = np.linspace(0.1, 0.30, 10).tolist()

batch_sim = BatchSimulation("planar_2d_results.json", method="direct")


for L in L_vals:
    code = Planar2DCode(L)
    for p in p_vals:
        error_model = PauliErrorModel(0.5, 0.0, 0.5)
        decoder = MatchingDecoder(code, error_model, p)
        dir_sim = DirectSimulation(code, error_model, decoder, p)
        batch_sim.append(dir_sim)

n_trials = 1000
batch_sim.run(n_trials, progress=tqdm)

analysis = Analysis("planar_2d_results.json")

results = analysis.get_results()
thresh = analysis.thresholds

cols = ["code", "error_model", "p_th_sd", "p_th_fss", "p_th_fss_left", "p_th_fss_right", "p_th_fss_se"]
# print(results)

# print(results.columns)
print(thresh[cols])

fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

plt.sca(ax[0])
analysis.plot_thresholds()
plt.sca(ax[1])
analysis.plot_thresholds(sector="X")
plt.sca(ax[2])
analysis.plot_thresholds(sector="Z")
fig.savefig("planar_2d_thresholds.pdf", bbox_inches="tight")