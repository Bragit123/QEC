from panqec.codes import Toric2DCode
from panqec.error_models import PauliErrorModel
from panqec.simulation import DirectSimulation, BatchSimulation
from panqec.analysis import Analysis

from error_models import GaussPauliErrorModel
from decoders import BeliefPropagationLSDDecoder, GaussBeliefPropagationLSDDecoder

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


Code = Toric2DCode
Decoder = BeliefPropagationLSDDecoder
Gauss_Decoder = GaussBeliefPropagationLSDDecoder

pauli_model = PauliErrorModel(1.0, 0.0, 0.0)
gauss_model = GaussPauliErrorModel(1.0, 0.0, 0.0)

L_vals = [12]
p_vals = np.linspace(0.01, 0.25, 10)

pauli_json = "json_pauli.json"
gauss_json = "json_gauss.json"
real_gauss_json = "json_real_gauss.json"

json_paths = [pauli_json, gauss_json, real_gauss_json]
if os.path.exists(pauli_json):
    delete = input("JSON files exist. Would you like to delete them? (y/n): ")
    delete = True if delete.lower() == "y" else False
    if delete:
        for json_path in json_paths:
            if os.path.exists(json_path):
                os.remove(json_path)

plot_path = "gauss_test.pdf"

batch_sim_pauli = BatchSimulation(pauli_json)
batch_sim_gauss = BatchSimulation(gauss_json)
batch_sim_real_gauss = BatchSimulation(real_gauss_json)

for L in L_vals:
    code = Code(L)
    for p in p_vals:
        p = float(p)
        pauli_decoder = Decoder(code, pauli_model, p)
        dir_sim_pauli = DirectSimulation(code, pauli_model, pauli_decoder, p)
        batch_sim_pauli.append(dir_sim_pauli)
        
        gauss_decoder = Decoder(code, gauss_model, p)
        dir_sim_gauss = DirectSimulation(code, gauss_model, gauss_decoder, p)
        batch_sim_gauss.append(dir_sim_gauss)
        
        real_gauss_decoder = Gauss_Decoder(code, gauss_model, p)
        real_dir_sim_gauss = DirectSimulation(code, gauss_model, real_gauss_decoder, p)
        batch_sim_real_gauss.append(real_dir_sim_gauss)

n_trials = 100
print("Running Pauli simulation:")
batch_sim_pauli.run(n_trials, progress=tqdm)
print("Running Gauss simulation:")
batch_sim_gauss.run(n_trials, progress=tqdm)
print("Running Gauss (with decoder) simulation:")
batch_sim_real_gauss.run(n_trials, progress=tqdm)

analysis_pauli = Analysis(pauli_json)
analysis_gauss = Analysis(gauss_json)
analysis_real_gauss = Analysis(real_gauss_json)

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
# plt.sca(ax[0,0])
# analysis_pauli.plot_thresholds(include_threshold_estimate=False)
# plt.sca(ax[0,1])
# analysis_pauli.plot_thresholds(sector="X", include_threshold_estimate=False)
# plt.sca(ax[1,0])
# analysis_gauss.plot_thresholds(include_threshold_estimate=False)
# plt.sca(ax[1,1])
# analysis_gauss.plot_thresholds(sector="X", include_threshold_estimate=False)
# plt.sca(ax[2,0])
# analysis_real_gauss.plot_thresholds(include_threshold_estimate=False)
# plt.sca(ax[2,1])
# analysis_real_gauss.plot_thresholds(sector="X", include_threshold_estimate=False)
plt.sca(ax[0])
analysis_gauss.plot_thresholds(include_threshold_estimate=False)
analysis_real_gauss.plot_thresholds(include_threshold_estimate=False)
plt.sca(ax[1])
analysis_gauss.plot_thresholds(sector="X", include_threshold_estimate=False)
analysis_real_gauss.plot_thresholds(sector="X", include_threshold_estimate=False)

fig.savefig(plot_path, bbox_inches="tight")


results_gauss = analysis_gauss.get_results()
results_real_gauss = analysis_real_gauss.get_results()

p_vals_gauss = results_gauss["error_rate"]
p_vals_real_gauss = results_real_gauss["error_rate"]
p_est_gauss = results_gauss["p_est"]
p_se_gauss = results_gauss["p_se"]
p_est_real_gauss = results_real_gauss["p_est"]
p_se_real_gauss = results_real_gauss["p_se"]
plt.figure()
plt.xlabel("Physical error rate $p$")
plt.ylabel("Logical error rate $p_L$")
plt.plot(p_vals_gauss, p_est_gauss, "o-", color="blue", label="Without Gaussian decoder")
plt.errorbar(p_vals_gauss, p_est_gauss, yerr=p_se_gauss, capsize=5, color="blue")
plt.plot(p_vals_real_gauss, p_est_real_gauss, "o-", color="green", label="With Gaussian decoder")
plt.errorbar(p_vals_real_gauss, p_est_real_gauss, yerr=p_se_real_gauss, capsize=5, color="green")
plt.legend()
plt.savefig(plot_path, bbox_inches="tight")