import tile_codes as tc
from decoders import BeliefPropagationLSDDecoder

from upgrade_gauss import \
    GaussPauliErrorModel_XZsampling,\
    GaussPauliErrorModel_gaussfirst,\
    GaussPauliErrorModel_uniformfirst,\
    GaussBeliefPropagationLSDDecoder_XZsampling,\
    GaussBeliefPropagationLSDDecoder_gaussfirst,\
    GaussBeliefPropagationLSDDecoder_uniformfirst

import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Type, Optional, Union
from tqdm import tqdm # Progress bar

from panqec.codes import StabilizerCode, Toric2DCode
from panqec.error_models import PauliErrorModel
from panqec.decoders import BaseDecoder, BeliefPropagationOSDDecoder
from panqec.simulation import DirectSimulation, BatchSimulation
from panqec.analysis import Analysis

OUTPUT_DIR = "Simulation_Outputs/" # Directory to store JSON files containing the output from simulations.
PLOT_DIR = "Threshold_Plots/" # Directory to store threshold plots created from the simulations.

def get_sim_name(
        sim_input: dict
):
    """
    Returns a name that is used for the output-data and plot files, such as "thresholds_error_0.5_0.0_0.5__L_8_12_16_20__p_0.001_0.3_20__trials_1000".

    Parameters
    ----------
    Code : StabilizerCode class
        The Tile Code to run the simulation for. This must be a subclass
        of the TileCode class.
    sim_input : dict
        Dictionary of simulation parameters. Must include the following:
        - "E_X", "E_Y", "E_Z" = Distribution of X,Y and Z-errors. Must sum to 1.
        - "p_min", "p_max" = Min- and max-values for the error probability p of physical qubits.
        - "n_p" = Number of p-values to include in the simulation.
        - "L_vals" = List of code-sizes to consider.
        - "n_trials" = Number of Monte Carlo iterations to run the simulation for.
    Decoder : BaseDecoder class
        Decoder to use for the simulation. Must be a subclass of PanQECs BaseDecoder.
    """
    ## Get code name
    Code = sim_input["Code"]
    code_name = Code.__name__
    tc_len = len("TileCode")
    if code_name == "Toric2DCode":
        code_name = "toric"
    elif code_name[:tc_len] == "TileCode":
        code_name = "tc" + code_name[tc_len+1:tc_len+3] ######## MUST BE FIXED!!!!
    
    ## Get error model name
    Error_Model = sim_input["Error_Model"]
    error_model_name = Error_Model.__name__
    if error_model_name == "PauliErrorModel":
        error_model_name = "pauli"
    elif error_model_name.startswith("Gauss"):
        if error_model_name.endswith("XZsampling"):
            error_end = "XZ"
        elif error_model_name.endswith("gaussfirst"):
            error_end = "gauss"
        elif error_model_name.endswith("uniformfirst"):
            error_end = "uni"
        error_model_name = "Gpauli_" + error_end

    ## Get decoder name
    Decoder = sim_input["Decoder"]
    decoder_name = Decoder.__name__
    if decoder_name == "BeliefPropagationOSDDecoder":
        decoder_name = "bposd"
    elif decoder_name == "BeliefPropagationLSDDecoder":
        decoder_name = "bplsd"    
    elif decoder_name.startswith("Gauss"):
        if decoder_name.endswith("XZsampling"):
            decoder_end = "XZ"
        elif decoder_name.endswith("gaussfirst"):
            decoder_end = "gauss"
        elif decoder_name.endswith("uniformfirst"):
            decoder_end = "uni"
        
        gauss_bp_len = len("GaussBeliefPropagationLSDDecoder")
        if decoder_name[:gauss_bp_len] == "GaussBeliefPropagationLSDDecoder":
            decoder_start = "Gbplsd"
        elif decoder_name[:gauss_bp_len] == "GaussBeliefPropagationOSDDecoder":
            decoder_start = "Gbposd"
        
        decoder_name = decoder_start + "_" + decoder_end
    
    ## Get text for remaining parameters
    E_X, E_Y, E_Z = (sim_input["E_X"], sim_input["E_Y"], sim_input["E_Z"])
    error_text = f"e_{E_X:.2}_{E_Y:.2}_{E_Z:.2}"
    
    L_text = f"L"
    for L in sim_input["L_vals"]:
        L_text = L_text + "_" + str(L)

    p_text = f"p_{sim_input["p_min"]:.2}_{sim_input["p_max"]:.2}_{sim_input["n_p"]}"
    
    trials_text = f"t_{sim_input["n_trials"]}"

    ## Combine parameter strings into one string.
    sim_name = f"{code_name}__{error_model_name}__{decoder_name}__{error_text}__{L_text}__{p_text}__{trials_text}"

    return sim_name


def run_single_simulation(
        sim_input: dict,
        p_logarithmic: Optional[bool] = False
):
    """
    Runs a threshold simulation, and stores it into a JSON file.

    Parameters
    ----------
    Code : StabilizerCode class
        The Tile Code to run the simulation for. This must be a subclass
        of the TileCode class.
    sim_input : dict
        Dictionary of simulation parameters. Must include the following:
        - "E_X", "E_Y", "E_Z" = Distribution of X,Y and Z-errors. Must sum to 1.
        - "p_min", "p_max" = Min- and max-values for the error probability p of physical qubits.
        - "n_p" = Number of p-values to include in the simulation.
        - "L_vals" = List of code-sizes to consider.
        - "n_trials" = Number of Monte Carlo iterations to run the simulation for.
    Decoder : BaseDecoder class
        Decoder to use for the simulation. Must be a subclass of PanQECs BaseDecoder.
    p_logarithmic : bool
        If True, p_values are distributed logarithmically instead of linearly.
    """
    Code = sim_input["Code"]
    Error_Model = sim_input["Error_Model"]
    Decoder = sim_input["Decoder"]
    try:
        assert issubclass(Code, StabilizerCode)
    except:
        raise TypeError("Code must be a subclass of PanQEC's StabilizerCode.")
    try:
        assert issubclass(Decoder, BaseDecoder)
    except:
        raise TypeError("Decoder must be a subclass of PanQEC's BaseDecoder.")
    
    ## Extract input parameters
    E_X, E_Y, E_Z = (sim_input["E_X"], sim_input["E_Y"], sim_input["E_Z"])
    p_min = sim_input["p_min"]
    p_max = sim_input["p_max"]
    n_p = sim_input["n_p"]
    L_vals = sim_input["L_vals"]
    n_trials = sim_input["n_trials"]

    ## Set remaining parameters
    error_model = Error_Model(E_X, E_Y, E_Z)
    if p_logarithmic:
        p_vals = np.logspace(np.log10(p_min), np.log10(p_max), n_p)
    else:
        p_vals = np.linspace(p_min, p_max, n_p)

    sim_name = get_sim_name(sim_input)
    output_path = OUTPUT_DIR + sim_name + ".json"

    if os.path.exists(output_path):
        if args.overwrite and not args.keep:
            delete_existing_json = True
        elif args.keep and not args.overwrite:
            delete_existing_json = False
        else:
            user_input = input(f"FILE ALREADY EXISTS: {output_path} | Would you like to delete it and run a new simulation? (y/n): ")
            delete_existing_json = True if user_input.lower() == "y" else False
        
        if delete_existing_json:
            os.remove(output_path)

    ## Run simulation
    batch_sim = BatchSimulation(output_path)

    for L in L_vals:
        code = Code(L)
        for p in p_vals:
            p = float(p)
            decoder = Decoder(code, error_model, p)
            dir_sim = DirectSimulation(code, error_model, decoder, p)
            batch_sim.append(dir_sim)
    
    batch_sim.run(n_trials, progress=tqdm)

    return output_path


def split_inputs(sim_inputs: dict):
    varying_keys = []
    value_len = []
    fixed_keys = []
    for k, v in sim_inputs.items():
        if isinstance(v, list) and k != "L_vals":
            # Check if one of the inputs except L_vals is a list
            varying_keys.append(k)
            value_len.append(len(v))
        else:
            fixed_keys.append(k)
    
    if len(set(value_len)) == 0:
        # No varying inputs, so no splitting needed.
        return [sim_inputs]
    elif len(set(value_len)) == 1:
        sim_input_list = []
        for i in range(value_len[0]):
            sim_input = dict()
            for k in fixed_keys:
                sim_input[k] = sim_inputs[k]
            for k in varying_keys:
                sim_input[k] = sim_inputs[k][i]
            sim_input_list.append(sim_input)
    else:
        raise ValueError("All lists in sim_inputs (except L_vals) must be of equal length.")
    
    return sim_input_list
    

def run_simulations(
        sim_inputs: dict,
        p_logarithmic: bool = False
):
    sim_input_list = split_inputs(sim_inputs)
    n_sims = len(sim_input_list)
    
    json_paths = []
    for i, sim_input in enumerate(sim_input_list):
        print(f"Running simulation {i+1}/{n_sims}:")
        json_path = run_single_simulation(sim_input, p_logarithmic)
        json_paths.append(json_path)
    
    return json_paths


def extract_data(
        json_paths: Union[str, List[str]],
        names: Optional[Union[str, List[str]]]
):
    try:
        assert isinstance(json_paths, str) or isinstance(json_paths, list) # Check that json_paths is a string or list
        assert (type(json_paths) == type(names)) # Check that json_paths and names are the same type.
    except:
        raise TypeError("json_paths and names must be of the same type: either a string or a list of strings.")
    
    if isinstance(json_paths, str):
        json_paths = [json_paths]
        names = [names]
    
    results_list = []
    for json_path, name in zip(json_paths, names):
        analysis = Analysis(json_path)
        analysis.calculate_sector_thresholds() # Necessary for p_est_X and p_est_Z to be available
        results_i = analysis.get_results()
        results_i["name"] = name
        results_list.append(results_i)
    
    results = pd.concat(results_list, ignore_index=True)
    L_x = results["code_params"].apply(lambda d: d.get("L_x"))
    L_y = results["code_params"].apply(lambda d: d.get("L_y"))
    if np.all(L_x == L_y):
        results["L"] = L_x
    else:
        results["L_x"] = L_x
        results["L_y"] = L_y
    
    # ## Append pseudo threshold to end of dataframe
    # p_vals = results["error_rate"].unique()
    # k = results.loc[0, "k"]
    # ps_th = 1.0 - (1.0 - p_vals)**k # Pseudo threshold
    # ps_th_dict = {
    #     "error_rate": p_vals,
    #     "p_est": ps_th, # Put it in p_est for seaborn-plotting to work
    #     "name": "Pseudo Threshold",
    #     "L": results["L"].min(),
    # }
    # ps_th_df = pd.DataFrame(ps_th_dict)
    # results = pd.concat([results, ps_th_df])

    return results


def plot_results(data: pd.DataFrame, filename: str):
    """
    Analyze output from a simulation, and create the threshold plots. The resulting plot is saved in the plot directory.

    Parameters
    ----------
    Code : Stabilizer class
        The Tile Code to run the simulation for. This must be a subclass
        of the TileCode class.
    sim_input : dict
        Dictionary of simulation parameters. Must include the following:
        - "E_X", "E_Y", "E_Z" = Distribution of X,Y and Z-errors. Must sum to 1.
        - "p_min", "p_max" = Min- and max-values for the error probability p of physical qubits.
        - "n_p" = Number of p-values to include in the simulation.
        - "L_vals" = List of code-sizes to consider.
        - "n_trials" = Number of Monte Carlo iterations to run the simulation for.
    Decoder : BaseDecoder class
        Decoder to use for the simulation. Must be a subclass of PanQECs BaseDecoder.
    xscale : str
        Input to matplotlibs ax.set_xscale().
    yscale : str
        Input to matplotlibs ax.set_yscale().
    include_threshold_estimate : bool
        Passed into Analysis.plot_thresholds(). If True: computes and plots threshold region.
    """
    if not filename.endswith(".pdf"):
        filename = filename + ".pdf"

    plot_path = PLOT_DIR + filename # Path for saving the threshold plot.

    sns.set_theme()

    L = None
    if "L" in data.columns:
        if len(data["L"].unique()) > 1:
            L = "L"
    else:
        L = "L_x"
    
    name = None
    if len(data["name"].unique()) > 1:
        name = "name"

    ## Plot data
    g = sns.relplot(data, kind="line", x="error_rate", y="p_est", hue=name, style=L)
    
    p_vals = data["error_rate"].unique()
    k = data.loc[0, "k"]
    ps_th = 1.0 - (1.0 - p_vals)**k # Pseudo threshold
    g.ax.plot(p_vals, ps_th, color="gray", linestyle="dashed")
    g.ax.text(0.05, 0.7, "Pseudo Threshold\n(gray dashed line)", color="gray", rotation=45, transform=g.ax.transAxes)

    g.set(
        title="Logical error rate $p_L$ as a function of physical error rate $p$. (Pseudo threshold in gray)",
        xlabel="Physical error rate $p$",
        ylabel="Logical error rate $p_L$"
    )
    g.savefig(plot_path, bbox_inches="tight")


if __name__ == "__main__":
    ## Set simulation parameters
    # Code = Toric2DCode
    Code = tc.TileCode_B3_W6
    Error_Models = [
        PauliErrorModel,
        GaussPauliErrorModel_XZsampling,
        GaussPauliErrorModel_gaussfirst,
        GaussPauliErrorModel_uniformfirst
    ]
    Decoders = [
        BeliefPropagationLSDDecoder,
        GaussBeliefPropagationLSDDecoder_XZsampling,
        GaussBeliefPropagationLSDDecoder_gaussfirst,
        GaussBeliefPropagationLSDDecoder_uniformfirst
    ]
    sim_inputs = {
        "Code": Code,
        "Error_Model": Error_Models,
        "Decoder": Decoders,
        "E_X": 0.3,
        "E_Y": 0.2,
        "E_Z": 0.5,
        "p_min": 0.001,
        "p_max": 0.5,
        "n_p": 5,
        "L_vals": [8, 12, 16],
        "n_trials": 10
    }

    # Run and analyze simulation
    names = ["Pauli", "XZ sampling", "Gauss first", "Uniform first"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", help="Run new simulations, overwriting any existing JSON files without asking.", action="store_true")
    parser.add_argument("-k", "--keep", help="Don't run new simulations for already existing JSON files.", action="store_true")
    args = parser.parse_args()
    if args.keep and args.overwrite:
        print("WARNING: Cannot both overwrite and keep JSON files. Reverting back to asking at each JSON file.")

    json_paths = run_simulations(sim_inputs)
    
    filename = "gauss_test_tilecode"
    data = extract_data(json_paths, names)
    plot_results(data, filename)