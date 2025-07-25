import tile_codes as tc
from error_models import GaussPauliErrorModel
from decoders import BeliefPropagationLSDDecoder, GaussBeliefPropagationLSDDecoder

import itertools
import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns

from typing import List, Optional, Union
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
    code = sim_input["Code"].__name__
    tc_len = len("TileCode")
    if code == "Toric2DCode":
        code = "toric"
    elif code[:tc_len] == "TileCode":
        code = code.replace("_", "") # Remove underscores for shorter name
        code = "tc" + code[tc_len:] # Shorten TileCode to tc. Now: TileCode_B3_W6 -> tcB3W6
    
    ## Get error model and decoder names
    error_model = sim_input["error_model"]
    decoder = sim_input["decoder"]
    
    ## Get text for remaining parameters
    E_X, E_Y, E_Z = (sim_input["E_X"], sim_input["E_Y"], sim_input["E_Z"])
    error_text = f"e_{E_X:.2}_{E_Y:.2}_{E_Z:.2}"
    
    L_text = f"L"
    for L in sim_input["L_vals"]:
        L_text = L_text + "_" + str(L)

    p_text = f"p_{sim_input["p_min"]:.2}_{sim_input["p_max"]:.2}_{sim_input["n_p"]}"
    
    trials_text = f"t_{sim_input["n_trials"]}"

    ## Combine parameter strings into one string.
    sim_name = f"{code}__{error_model}__{decoder}__{error_text}__{L_text}__{p_text}__{trials_text}"

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
    ## Extract and check validity of code
    Code = sim_input["Code"]
    try:
        assert issubclass(Code, StabilizerCode)
    except:
        raise TypeError(f"Code must be a subclass of PanQEC's StabilizerCode. Got {Code}")

    ## Extract and check validity of error model and decoder
    error_model_name = sim_input["error_model"]
    decoder_name = sim_input["decoder"]
    
    if error_model_name == "pauli":
        Error_Model = PauliErrorModel
        if decoder_name == "bposd":
            Decoder = BeliefPropagationOSDDecoder
        elif decoder_name == "bplsd":
            Decoder = BeliefPropagationLSDDecoder
        else:
            raise ValueError(f"sim_input['decoder'] must be either 'bposd' or 'bplsd'. Got '{decoder_name}'.")
    elif error_model_name == "gauss":
        Error_Model = GaussPauliErrorModel
        if decoder_name == "bposd":
            raise NotImplementedError("Gaussian BPOSD not implemented.")
        elif decoder_name == "bplsd":
            Decoder = GaussBeliefPropagationLSDDecoder
        else:
            raise ValueError(f"sim_input['decoder'] must be either 'bposd' or 'bplsd'. Got '{decoder_name}'.")
    else:
        raise ValueError(f"sim_input['error_model'] must be either 'pauli' or 'gauss'. Got '{error_model_name}'.")

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
    # return sim_input_list
    varying_keys = []
    varying_values = []
    fixed_keys = []
    fixed_values = []
    for k, v in sim_inputs.items():
        if isinstance(v, list) and k != "L_vals":
            # Check if one of the inputs except L_vals is a list
            varying_keys.append(k)
            varying_values.append(v)
        else:
            fixed_keys.append(k)
            fixed_values.append(v)
    
    sim_input_list = []
    if len(varying_keys) == 0:
        # No varying inputs, so no splitting needed.
        sim_input_list.append(sim_inputs)
    else:
        # Get a list of all combinations of the varying parameters
        varying_product = list(itertools.product(*varying_values))
        for k_values in varying_product:
            sim_input = dict()
            for k, v in zip(varying_keys, k_values):
                sim_input[k] = v
            for k, v in zip(fixed_keys, fixed_values):
                sim_input[k] = v
            sim_input_list.append(sim_input)
    
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
        json_paths: Union[str, List[str]]
):  
    if isinstance(json_paths, str):
        json_paths = [json_paths]
    
    results_list = []
    for json_path in json_paths:
        analysis = Analysis(json_path)
        analysis.calculate_sector_thresholds() # Necessary for p_est_X and p_est_Z to be available
        results_i = analysis.get_results()
        results_list.append(results_i)
    
    results = pd.concat(results_list, ignore_index=True)
    L_x = results["code_params"].apply(lambda d: d.get("L_x"))
    L_y = results["code_params"].apply(lambda d: d.get("L_y"))
    if np.all(L_x == L_y):
        results["L"] = L_x
    else:
        raise NotImplementedError("Only the case when L_x = L_y is implemented.")

    return results


def plot_results(
        data: pd.DataFrame,
        filename: str,
        style: Optional[str] = None,
        hue: Optional[str] = None,
        col: Optional[str] = None
):
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

    ## Plot data
    g = sns.relplot(data, kind="line", x="error_rate", y="p_est", hue=hue, style=style, col=col)

    cols = data[col].unique()

    for i in range(len(cols)):
        ## Plot pseudo threshold
        ax = g.axes.flat[i]
        filtered_data = data.loc[data[col] == cols[i]]
        k = filtered_data["k"].iloc[0]
        p_vals = filtered_data["error_rate"].unique()
        ps_th = 1.0 - (1.0 - p_vals)**k # Pseudo threshold
        ax.plot(p_vals, ps_th, color="gray", linestyle="dashed")

    g.figure.suptitle("Logical error rate $p_L$ as a function of physical error rate $p$. (Pseudo threshold in gray)")
    g.set(
        xlabel="Physical error rate $p$",
        ylabel="Logical error rate $p_L$"
    )
    g.figure.tight_layout()

    g.savefig(plot_path, bbox_inches="tight")

    return plot_path


if __name__ == "__main__":
    ## Set simulation parameters
    Codes = [Toric2DCode, tc.TileCode_B3_W6]
    error_models = [
        "pauli",
        "gauss"
    ]
    decoders = "bplsd"
    sim_inputs = {
        "Code": Codes,
        "error_model": error_models,
        "decoder": decoders,
        "E_X": 0.5,
        "E_Y": 0.0,
        "E_Z": 0.5,
        "p_min": 0.001,
        "p_max": 0.5,
        "n_p": 20,
        "L_vals": [8, 12, 16],
        "n_trials": 500
    }

    # Run and analyze simulation
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", help="Run new simulations, overwriting any existing JSON files without asking.", action="store_true")
    parser.add_argument("-k", "--keep", help="Don't run new simulations for already existing JSON files.", action="store_true")
    args = parser.parse_args()
    if args.keep and args.overwrite:
        print("WARNING: Cannot both overwrite and keep JSON files. Reverting back to asking at each JSON file.")

    json_paths = run_simulations(sim_inputs)
    
    filename = "gauss_test_new"
    data = extract_data(json_paths)
    plot_path = plot_results(data, filename, style="L", hue="error_model", col="code")

    print(f"Results plotted in {plot_path}")