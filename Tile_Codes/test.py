from panqec.codes import Toric2DCode
from panqec.error_models import PauliErrorModel
from panqec.simulation import DirectSimulation, BatchSimulation
from panqec.analysis import Analysis

from tile_codes import TileCode_B3_W6
from error_models import GaussPauliErrorModel
from decoders import BeliefPropagationLSDDecoder, GaussBeliefPropagationLSDDecoder

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from panqec.bpauli import bsf_to_pauli

Err_mods = [
    # PauliErrorModel,
    GaussPauliErrorModel_XZsampling,
    # GaussPauliErrorModel_gaussfirst,
    # GaussPauliErrorModel_uniformfirst
]
# r_xs = [0.5, 1/3, 0.2, 0.1, 0.3, 0.7, 0.9]
# r_ys = [0.0, 1/3, 0.7, 0.0, 0.0, 0.0, 0.0]
# r_zs = [0.5, 1/3, 0.1, 0.9, 0.7, 0.3, 0.1]
r_x = 0.99999
r_y = 0.0
r_z = 0.00001

code = Toric2DCode(6)
# p = 0.7
n_p = 20
p_vals = np.linspace(0.0, 1.0, 20)
p_computed = np.zeros(n_p)
r_x_computed = np.zeros(n_p)
r_y_computed = np.zeros(n_p)
r_z_computed = np.zeros(n_p)

n_iter = 1000

from colorama import Style
# print(f"{"Error model":<36}|{"r":^26}|{"n":^23}|{"p":^43}|{"numerical r":^32}|{"Difference in r":^32}")
# print("-"*36 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*5 + "+" + "-"*5 + "+" + "-"*5 + "+" + "-"*5 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10)
print(f"{"Real p":^15}|{"Computed p":^15}|{"Absolute error":^15}||{"r_(x,y,z)":^47}|{"Absolute errors of r_(x,y,z)":^47}")
# print("-"*36 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*8 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10)

for Err_mod in Err_mods:
    print("-"*15 + "+" + "-"*15 + "+" + "-"*15 + "++" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15)
    # for row, r_x, r_y, r_z in zip(range(len(r_xs)), r_xs, r_ys, r_zs):
    for row, p in enumerate(p_vals):
        error_model = Err_mod(r_x, r_y, r_z)
        n_ixyz = np.zeros((n_iter, 4), dtype=int)
        for i in range(n_iter):
            error = error_model.generate(code, p)
            err_pauli = bsf_to_pauli(error)
            
            n_ixyz[i,0] = err_pauli.count("I")
            n_ixyz[i,1] = err_pauli.count("X")
            n_ixyz[i,2] = err_pauli.count("Y")
            n_ixyz[i,3] = err_pauli.count("Z")

            # n_tot = len(err_pauli)
            # p_ixyz = n_ixyz / n_tot
        
        n_tot_ixyz = np.sum(n_ixyz, axis=0)
        n_tot = np.sum(n_tot_ixyz)
        n_tot_xyz = np.sum(n_tot_ixyz[1:])
        p_ixyz = n_tot_ixyz / n_tot
        if n_tot_xyz > 0:
            r_xyz = n_tot_ixyz[1:] / n_tot_xyz
        else:
            r_xyz = np.empty(3)
            r_xyz[:] = np.nan

        r_x_computed[row] = r_xyz[0]
        r_y_computed[row] = r_xyz[1]
        r_z_computed[row] = r_xyz[2]

        p_identity = p_ixyz[0]
        p_error = np.sum(p_ixyz[1:])
        p_computed[row] = p_error
        
        if row % 2 == 0:
            color_style = Style.NORMAL
        else:
            color_style = Style.DIM
        # row_string = color_style
        # row_string += f"{Err_mod.__name__:<36}"
        # row_string += Style.NORMAL + "|" + color_style
        # row_string += f"{r_x:^8.2}"
        # row_string += Style.NORMAL + "|" + color_style
        # row_string += f"{r_y:^8.2}"
        # row_string += Style.NORMAL + "|" + color_style
        # row_string += f"{r_z:^8.2}"
        # for n_i in n_tot_ixyz:
        #     row_string += f"|{n_i:^5}"

        # for p_i in p_ixyz:
        #     row_string += Style.NORMAL + "|" + color_style
        #     row_string += f"{p_i:^10.4f}"
        
        # for r_i in r_xyz:
        #     row_string += Style.NORMAL + "|" + color_style
        #     row_string += f"{r_i:^10.4}"
        
        # r_diff = np.abs(np.array([r_xyz[0]-r_x, r_xyz[1]-r_y, r_xyz[2]-r_z]))
        # for r_d in r_diff:
        #     row_string += Style.NORMAL + "|" + color_style
        #     row_string += f"{r_d:10.4}"
        
        p_identity = p_ixyz[0]
        p_error = np.sum(p_ixyz[1:])
        p_computed[row] = p_error
        row_string = color_style + f"{p_vals[row]:^15.4f}"
        row_string += Style.NORMAL + "|"
        row_string += color_style + f"{p_error:^15.4f}"
        row_string += Style.NORMAL + "|"
        row_string += color_style + f"{np.abs(p_vals[row]-p_error):^15.4f}"
        
        row_string += Style.NORMAL + "||"
        row_string += color_style + f"{r_xyz[0]:^15.4f}"
        row_string += Style.NORMAL + "|"
        row_string += color_style + f"{r_xyz[1]:^15.4f}"
        row_string += Style.NORMAL + "|"
        row_string += color_style + f"{r_xyz[2]:^15.4f}"
        row_string += Style.NORMAL + "|"
        row_string += color_style + f"{np.abs(r_xyz[0]-r_x):^15.4f}"
        row_string += Style.NORMAL + "|"
        row_string += color_style + f"{np.abs(r_xyz[1]-r_y):^15.4f}"
        row_string += Style.NORMAL + "|"
        row_string += color_style + f"{np.abs(r_xyz[2]-r_z):^15.4f}"
        
        row_string += Style.RESET_ALL
        print(row_string)

p_err = np.abs(p_vals - p_computed)
fig, axs = plt.subplots(nrows=2, ncols=2)
fig.suptitle("Computed error rates p from Gaussian PauliErrorModel", y=0.95)
axs[0,0].plot(p_vals, p_computed, "b", label="Computed p")
axs[0,0].axline((0,0),(1,1), linestyle="dashed", color="black", label="Expected p")
axs[0,0].legend()
axs[0,1].plot(p_vals, p_err, "b", label="Absolute error")
axs[0,1].legend()
axs[0,0].set_ylabel("Computed $p_C$")
axs[0,1].set_ylabel("$|p_E-p_C|$")
for ax in axs[0,:]:
    ax.set_xlabel("Expected $p_E$")

r_x_err = np.abs(r_x - r_x_computed)
r_z_err = np.abs(r_z - r_z_computed)
# r_xz_diff = np.abs(r_x_computed - r_z_computed)
r_xz_diff = np.abs(r_x_computed/(r_x_computed + r_z_computed))
# r_xz_diff = r_x_computed - r_z_computed
axs[1,0].set_title("Computed $r$-values")
axs[1,0].plot(p_vals, r_x_computed, "r", label="$r_x$")
axs[1,0].plot(p_vals, r_y_computed, color="orange", label="$r_y$")
axs[1,0].plot(p_vals, r_z_computed, "b", label="$r_z$")
axs[1,0].legend()
# axs[1,1].plot(p_vals, r_x_err, "b", label="$r_x$")
# axs[1,1].plot(p_vals, r_z_err, color="orange", label="$r_z$")
axs[1,1].plot(p_vals, r_xz_diff, "b", label="$|r_x - r_z|$")
axs[1,1].legend()
for ax in axs[1,:]:
    ax.set_xlabel("Input $p$")

fig.savefig("gauss.pdf", bbox_inches="tight")

# names = ["Regular-Pauli", "XZ-Sampling", "Uniform-First", "Gauss-First"]

# Code = Toric2DCode
# # Code = TileCode_B3_W6
# Pauli_Decoder = BeliefPropagationLSDDecoder
# # Gauss_Decoder = GaussBeliefPropagationLSDDecoder
# Gauss_Decoder_XZsampling = GaussBeliefPropagationLSDDecoder_XZsampling
# Gauss_Decoder_uniformfirst = GaussBeliefPropagationLSDDecoder_uniformfirst
# Gauss_Decoder_gaussfirst = GaussBeliefPropagationLSDDecoder_gaussfirst
# Decoders = [Pauli_Decoder, Gauss_Decoder_XZsampling, Gauss_Decoder_uniformfirst, Gauss_Decoder_gaussfirst]

# rng = np.random.default_rng(100)
# # pauli_model = PauliErrorModel(1.0, 0.0, 0.0)
# # gauss_model = GaussPauliErrorModel(1.0, 0.0, 0.0)
# pauli_model = PauliErrorModel(0.3, 0.2, 0.5)
# gauss_model_XZsampling = GaussPauliErrorModel_XZsampling(0.3, 0.2, 0.5)
# gauss_model_uniformfirst = GaussPauliErrorModel_uniformfirst(0.3, 0.2, 0.5)
# gauss_model_gaussfirst = GaussPauliErrorModel_gaussfirst(0.3, 0.2, 0.5)
# error_models = [pauli_model, gauss_model_XZsampling, gauss_model_uniformfirst, gauss_model_gaussfirst]

# L_vals = [8, 12, 16]
# p_vals = np.linspace(0.001, 0.98, 20)

# # pauli_json = "TEMP_json_pauli.json"
# # # gauss_json = "TEMP_json_gauss.json"
# # gauss_XZsampling = "TEMP_json_gauss_XZsampling.json"
# # gauss_uniformfirst = "TEMP_json_gauss_uniformfirst.json"
# # gauss_gaussfirst = "TEMP_json_gauss_gaussfirst.json"
# json_paths = [
#     "TEMP_json_pauli.json",
#     "TEMP_json_gauss_XZsampling.json",
#     "TEMP_json_gauss_uniformfirst.json",
#     "TEMP_json_gauss_gaussfirst.json"
# ]

# if os.path.exists(json_paths[0]):
#     delete = input("JSON files exist. Would you like to delete them and make new simulations? (y/n): ")
#     delete = True if delete.lower() == "y" else False
#     if delete:
#         for json_path in json_paths:
#             if os.path.exists(json_path):
#                 os.remove(json_path)

# plot_path = "gauss_test.pdf"

# # batch_sim_pauli = BatchSimulation(pauli_json)
# # # batch_sim_gauss = BatchSimulation(gauss_json)
# # batch_sim_gauss_XZsampling = BatchSimulation(gauss_XZsampling)
# # batch_sim_gauss_uniformfirst = BatchSimulation(gauss_uniformfirst)
# # batch_sim_gauss_gaussfirst = BatchSimulation(gauss_gaussfirst)
# batch_sims = []
# for json_path in json_paths:
#     batch_sims.append(BatchSimulation(json_path))

# for L in L_vals:
#     code = Code(L)
#     for p in p_vals:
#         p = float(p)

#         for error_model, Decoder, batch_sim in zip(error_models, Decoders, batch_sims):
#             decoder = Decoder(code, error_model, p)
#             dir_sim = DirectSimulation(code, error_model, decoder, p, rng=rng)
#             batch_sim.append(dir_sim)
#         # pauli_decoder = Decoder(code, pauli_model, p)
#         # dir_sim_pauli = DirectSimulation(code, pauli_model, pauli_decoder, p, rng=rng)
#         # batch_sim_pauli.append(dir_sim_pauli)
        
#         # gauss_decoder = Decoder(code, gauss_model_XZsampling, p)
#         # dir_sim_gauss = DirectSimulation(code, gauss_model_XZsampling, gauss_decoder, p, rng=rng)
#         # batch_sim_gauss.append(dir_sim_gauss)
        
#         # real_gauss_decoder = Gauss_Decoder_XZsampling(code, gauss_model_XZsampling, p)
#         # real_dir_sim_gauss = DirectSimulation(code, gauss_model_XZsampling, real_gauss_decoder, p, rng=rng)
#         # batch_sim_gauss_XZsampling.append(real_dir_sim_gauss)

# n_trials = 500

# for name, batch_sim in zip(names, batch_sims):
#     print(f"Running {name} simulation:")
#     batch_sim.run(n_trials, progress=tqdm)
# # print("Running Pauli simulation:")
# # batch_sim_pauli.run(n_trials, progress=tqdm)
# # print("Running Gauss simulation:")
# # batch_sim_gauss.run(n_trials, progress=tqdm)
# # print("Running Gauss (with decoder) simulation:")
# # batch_sim_gauss_XZsampling.run(n_trials, progress=tqdm)

# analyses = []
# for json_path in json_paths:
#     analysis = Analysis(json_path)
#     analyses.append(analysis)
# # analysis_pauli = Analysis(pauli_json)
# # analysis_gauss = Analysis(gauss_json)
# # analysis_real_gauss = Analysis(gauss_XZsampling)

# # fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(15, 5))
# # plt.sca(ax[0,0])
# # analysis_pauli.plot_thresholds(include_threshold_estimate=False)
# # plt.sca(ax[0,1])
# # analysis_pauli.plot_thresholds(sector="X", include_threshold_estimate=False)
# # plt.sca(ax[1,0])
# # analysis_gauss.plot_thresholds(include_threshold_estimate=False)
# # plt.sca(ax[1,1])
# # analysis_gauss.plot_thresholds(sector="X", include_threshold_estimate=False)
# # plt.sca(ax[2,0])
# # analysis_real_gauss.plot_thresholds(include_threshold_estimate=False)
# # plt.sca(ax[2,1])
# # analysis_real_gauss.plot_thresholds(sector="X", include_threshold_estimate=False)
# # plt.sca(ax[0])
# # analysis_gauss.plot_thresholds(include_threshold_estimate=False)
# # analysis_real_gauss.plot_thresholds(include_threshold_estimate=False)
# # plt.sca(ax[1])
# # analysis_gauss.plot_thresholds(sector="X", include_threshold_estimate=False)
# # analysis_real_gauss.plot_thresholds(sector="X", include_threshold_estimate=False)

# # fig.savefig(plot_path, bbox_inches="tight")

# plt.figure()
# analyses[0].calculate_thresholds()
# colors = ["magenta", "blue", "red", "green"]
# plt.figure()
# plt.title("Logical error rate for X (dashed), Z (dotted), and combined (solid) sector. ")
# plt.xlabel("Physical error rate $p$")
# plt.ylabel("Logical error rate $p_L$")

# k = code.k
# pseudo_threshold = 1-(1-p_vals)**k
# plt.plot(p_vals, pseudo_threshold, color="black", linestyle="dashed", label="Pseudo-Threshold")
# # max_ind = np.argmin(np.abs(p_vals - 0.7))
# max_ind = len(p_vals)
# for analysis, name, color in zip(analyses, names, colors):
#     analysis.calculate_sector_thresholds() # Necessary for p_est_X and p_est_Z to be available
#     results = analysis.get_results()
#     p_vals = results["error_rate"]
#     p_est = results["p_est"]
#     p_se = results["p_se"]
#     p_est_X = results["p_est_X"]
#     p_est_Z = results["p_est_Z"]
    
#     L_num = len(L_vals)
#     p_num = len(p_vals)
#     L_slice_ind = int(p_num / L_num)
#     for i, L in enumerate(L_vals):
#         start_ind = i*L_slice_ind
#         end_ind = (i+1)*L_slice_ind
#         plt.plot(p_vals[start_ind:end_ind], p_est[start_ind:end_ind], "o-", color=color, label=name)
#         plt.errorbar(p_vals[start_ind:end_ind], p_est[start_ind:end_ind], yerr=p_se[start_ind:end_ind], capsize=5, color=color)

#     # plt.plot(p_vals[:max_ind], p_est_X[:max_ind], color=color, linestyle="dashed")
#     # plt.plot(p_vals[:max_ind], p_est_Z[:max_ind], color=color, linestyle="dotted")

# plt.legend()
# plt.savefig(plot_path, bbox_inches="tight")

# import pandas as pd
# import seaborn as sns

# colors = ["magenta", "blue", "red", "green"]
# k = code.k
# pseudo_threshold = 1-(1-p_vals)**k

# results_list = []
# for name, analysis in zip(names, analyses):
#     analysis.calculate_sector_thresholds() # Necessary for p_est_X and p_est_Z to be available
#     results_i = analysis.get_results()
#     results_i["name"] = name
#     results_list.append(results_i)

# results = pd.concat(results_list)
# results["L"] = results["code_params"].apply(lambda d: d.get("L_x")) # Extract L-value to its own column

# # plt.figure()

# # Filter results:
# max_p = 0.7
# if max_p is not None:
#     results = results[results["error_rate"] <= max_p]

# sns.set_theme()
# are_subplots = False
# if are_subplots:
#     g = sns.relplot(data=results, kind="line", x="error_rate", y="p_est", style="L", col="name")
# else:
#     g = sns.relplot(data=results, kind="line", x="error_rate", y="p_est", style="L", hue="name")

# # Build the color mapping used by Seaborn
# palette = sns.color_palette()  # default palette
# unique_names = results["name"].unique()
# color_map = dict(zip(unique_names, palette))  # maps name → color

# # Add error bars with matching color
# if are_subplots:
#     for ax, (name_val, subdata) in zip(g.axes.flat, results.groupby("name")):
#         for L_val, group in subdata.groupby("L"):
#             ax.errorbar(
#                 group["error_rate"],
#                 group["p_est"],
#                 yerr=group["p_se"],
#                 fmt='none',
#                 capsize=3,
#                 alpha=0.7,
#                 ecolor="gray"
#             )
# else:
#     for ax in g.axes.flat:
#         for (L_val, name_val), group in results.groupby(["L", "name"]):
#             ax.errorbar(
#                 group["error_rate"],
#                 group["p_est"],
#                 yerr=group["p_se"],
#                 fmt='none',
#                 capsize=3,
#                 alpha=0.7,
#                 ecolor=color_map[name_val]
#             )

# g.set(
#     title="Logical error rate $p_L$ as a function of physical error rate $p$.",
#     xlabel="Physical error rate $p$",
#     ylabel="Logical error rate $p_L$"
# )
# g.savefig(plot_path)
# # plt.legend()
# # plt.savefig(plot_path, bbox_inches="tight")

# # plt.plot(p_vals, pseudo_threshold, color="black", linestyle="dashed", label="Pseudo-Threshold")
# # # max_ind = np.argmin(np.abs(p_vals - 0.7))
# # max_ind = len(p_vals)
# # for analysis, name, color in zip(analyses, names, colors):
# #     analysis.calculate_sector_thresholds() # Necessary for p_est_X and p_est_Z to be available
# #     results = analysis.get_results()
# #     p_vals = results["error_rate"]
# #     p_est = results["p_est"]
# #     p_se = results["p_se"]
# #     p_est_X = results["p_est_X"]
# #     p_est_Z = results["p_est_Z"]
    
# #     L_num = len(L_vals)
# #     p_num = len(p_vals)
# #     L_slice_ind = int(p_num / L_num)
# #     for i, L in enumerate(L_vals):
# #         start_ind = i*L_slice_ind
# #         end_ind = (i+1)*L_slice_ind
# #         plt.plot(p_vals[start_ind:end_ind], p_est[start_ind:end_ind], "o-", color=color, label=name)
# #         plt.errorbar(p_vals[start_ind:end_ind], p_est[start_ind:end_ind], yerr=p_se[start_ind:end_ind], capsize=5, color=color)

# #     # plt.plot(p_vals[:max_ind], p_est_X[:max_ind], color=color, linestyle="dashed")
# #     # plt.plot(p_vals[:max_ind], p_est_Z[:max_ind], color=color, linestyle="dotted")

# # plt.legend()
# # plt.savefig(plot_path, bbox_inches="tight")



# # results_pauli = analysis_pauli.get_results()
# # results_gauss = analysis_gauss.get_results()
# # results_real_gauss = analysis_real_gauss.get_results()

# # p_vals_pauli = results_pauli["error_rate"]
# # p_est_pauli = results_pauli[f"p_est"]
# # p_se_pauli = results_pauli[f"p_se"]

# # p_est_pauli_X = results_pauli[f"p_est_X"]
# # p_est_pauli_Z = results_pauli[f"p_est_Z"]

# # p_vals_gauss = results_gauss["error_rate"]
# # p_est_gauss = results_gauss[f"p_est"]
# # p_se_gauss = results_gauss[f"p_se"]

# # p_est_gauss_X = results_gauss[f"p_est_X"]
# # p_est_gauss_Z = results_gauss[f"p_est_Z"]

# # p_vals_real_gauss = results_real_gauss["error_rate"]
# # p_est_real_gauss = results_real_gauss[f"p_est"]
# # p_se_real_gauss = results_real_gauss[f"p_se"]

# # p_est_real_gauss_X = results_real_gauss[f"p_est_X"]
# # p_est_real_gauss_Z = results_real_gauss[f"p_est_Z"]

# # plt.figure()
# # plt.title("Logical error rate for X (dashed), Z (dotted), and combined (solid) sector. ")
# # plt.xlabel("Physical error rate $p$")
# # plt.ylabel("Logical error rate $p_L$")
# # plt.plot(p_vals_pauli, p_est_pauli, "o-", color="black", label="Normal Pauli")
# # plt.errorbar(p_vals_pauli, p_est_pauli, yerr=p_se_pauli, capsize=5, color="black")
# # plt.plot(p_vals_gauss, p_est_gauss, "o-", color="blue", label="Without Gaussian decoder")
# # plt.errorbar(p_vals_gauss, p_est_gauss, yerr=p_se_gauss, capsize=5, color="blue")
# # plt.plot(p_vals_real_gauss, p_est_real_gauss, "o-", color="green", label="With Gaussian decoder")
# # plt.errorbar(p_vals_real_gauss, p_est_real_gauss, yerr=p_se_real_gauss, capsize=5, color="green")

# # plt.plot(p_vals_pauli, p_est_pauli_X, color="black", linestyle="dashed")
# # plt.plot(p_vals_pauli, p_est_pauli_Z, color="black", linestyle="dotted")
# # plt.plot(p_vals_gauss, p_est_gauss_X, color="blue", linestyle="dashed")
# # plt.plot(p_vals_gauss, p_est_gauss_Z, color="blue", linestyle="dotted")
# # plt.plot(p_vals_real_gauss, p_est_real_gauss_X, color="green", linestyle="dashed")
# # plt.plot(p_vals_real_gauss, p_est_real_gauss_Z, color="green", linestyle="dotted")

# # plt.legend()
# # plt.savefig(plot_path, bbox_inches="tight")