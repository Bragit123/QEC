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

OUTPUT_DIR = "../Simulation_Outputs/" # Directory to store JSON files containing the output from simulations.
PLOT_DIR = "../Plots/" # Directory to store threshold plots created from the simulations.

def get_sim_name(sim_input: dict) -> str:
    """
    Returns a name that is used for the JSON files containing the simulation output.

    Parameters
    ----------
    sim_input : dict
        Dictionary of simulation parameters. Must include the following:
        - "Code" (StabilizerCode) = The QEC code (class, not object) to use in the simulation.
        - "error_model" (str) = The name of the error model to use. "pauli" for PauliErrorModel
        and "gauss" for GaussPauliErrorModel.
        - "decoder" (str) = The name of the decoder to use. "bposd" for OSD- and "bplsd" for LSD-
        belief propagation. Will use regular or Gaussian depending on the specified error model.
        - "r_xyz" (list) = List containing the r_x, r_y and r_z values to give as input to the
        error model. These are the distribution of how often the differen type of errors will
        occur, and must therefore sum to one.
        - "p_vals" (ndarray) = Array of error rates to simulate for.
        - "L_vals" (list) = List of code sizes to consider.
        - "n_trials" (int) = Number of Monte Carlo simulations to run the simulation for.
    
    Returns
    -------
    str
        Name of the simulation, containing information about all the different input
        parameters. This is used to name the JSON files produced from the simulation.
        
        Example name:
        toric__pauli__bplsd__r_0.5_0.0_0.5__L_8_12_16__p_0.001_0.5_20__t_500.json
        
        This name tells us that the simulation is run on the Toric2DCode, using the
        PauliErrorModel and BeliefPropagationLSDDecoder with r_x=0.5, r_y=0.0,
        r_z=0.5; going through the L values 8, 12 and 16; for 20 error rates p from
        0.001 to 0.5; and running 500 Monte Carlo simulations.
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
    r_x, r_y, r_z = sim_input["r_xyz"]
    error_text = f"e_{r_x:.2}_{r_y:.2}_{r_z:.2}"
    
    L_text = f"L"
    for L in sim_input["L_vals"]:
        L_text = L_text + "_" + str(L)

    p_vals = sim_input["p_vals"]
    p_min = np.min(p_vals)
    p_max = np.max(p_vals)
    n_p = len(p_vals)
    p_text = f"r_{p_min:.2}_{p_max:.2}_{n_p}"
    
    trials_text = f"t_{sim_input["n_trials"]}"

    ## Combine parameter strings into one string.
    sim_name = f"{code}__{error_model}__{decoder}__{error_text}__{L_text}__{p_text}__{trials_text}"

    return sim_name


def run_single_simulation(sim_input: dict) -> str:
    """
    Runs a single simulation, and stores it into a JSON file.

    Parameters
    ----------
    sim_input : dict
        Dictionary of simulation parameters. Must include the following:
        - "Code" (StabilizerCode) = The QEC code (class, not object) to use in the simulation.
        - "error_model" (str) = The name of the error model to use. "pauli" for PauliErrorModel
        and "gauss" for GaussPauliErrorModel.
        - "decoder" (str) = The name of the decoder to use. "bposd" for OSD- and "bplsd" for LSD-
        belief propagation. Will use regular or Gaussian depending on the specified error model.
        - "r_xyz" (list) = List containing the r_x, r_y and r_z values to give as input to the
        error model. These are the distribution of how often the differen type of errors will
        occur, and must therefore sum to one.
        - "p_vals" (ndarray) = Array of error rates to simulate for.
        - "L_vals" (list) = List of code sizes to consider.
        - "n_trials" (int) = Number of Monte Carlo simulations to run the simulation for.
    
    Returns
    -------
    str
        Path to the JSON file where the output from the simulation is stored.
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
    r_x, r_y, r_z = sim_input["r_xyz"]
    p_vals = sim_input["p_vals"]
    L_vals = sim_input["L_vals"]
    n_trials = sim_input["n_trials"]

    ## Set remaining parameters
    error_model = Error_Model(r_x, r_y, r_z)

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


def split_inputs(sim_inputs: dict) -> List[dict]:
    """
    Takes a dictionary containing simulation inputs for multiple simulations, and returns
    a list of dictionaries containing the simulation inputs for a single simulation.

    If one or more of the values in sim_inputs are lists (not including the code sizes
    "L_vals" or the error distribution "r_xyz", which are lists also for single
    simulations), this function considers all possible combinations of them and creates
    input dictionaries for each simulation.

    For instance: If the value of "Code" is [Code1, Code2, Code3] and the value of
    "error_model" is ["pauli", "gauss"], but all other entries are as usual for singular
    simulations, the function will return a list of six dictionaries, where "Code" and
    "error_model" takes the following combinations:
    
    - Code1, "pauli"
    - Code1, "gauss"
    - Code2, "pauli"
    - Code2, "gauss"
    - Code3, "pauli"
    - Code3, "gauss"

    and the remaining parameters are equal for all six dictionaries.

    Parameters
    ----------
    sim_input : dict
        Dictionary of simulation parameters. Must include the following:
        - "Code" (StabilizerCode or List) = The QEC code (class, not object) to use in the simulation.
        - "error_model" (str or List) = The name of the error model to use. "pauli" for PauliErrorModel
        and "gauss" for GaussPauliErrorModel.
        - "decoder" (str or List) = The name of the decoder to use. "bposd" for OSD- and "bplsd" for LSD-
        belief propagation. Will use regular or Gaussian depending on the specified error model.
        - "r_xyz" (list) = List containing the r_x, r_y and r_z values to give as input to the
        error model. These are the distribution of how often the differen type of errors will
        occur, and must therefore sum to one.
        - "p_vals" (ndarray) = Array of error rates to simulate for.
        - "L_vals" (list) = List of code sizes to consider.
        - "n_trials" (int or List) = Number of Monte Carlo simulations to run the simulation for.
    
    Returns
    -------
    List[dict]
        List of dictionaries, each containing the input parameters for a single simulation.
    """
    varying_keys = []
    varying_values = []
    fixed_keys = []
    fixed_values = []
    for k, v in sim_inputs.items():
        if isinstance(v, list) and not (k in ["L_vals", "r_xyz"]):
            # Check if one of the inputs except L_vals and r_xyz is a list
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
    

def run_simulations(sim_inputs: dict) -> List[str]:
    """
    Runs simulation(s), and stores the output from each simulation into a separate
    JSON file. If one or more of the values in sim_inputs are lists (not including
    the code sizes "L_vals" or the error distribution "r_xyz", which are lists also
    for single simulations), this function considers all possible combinations of
    them and runs a separate simulation for each combination.

    Parameters
    ----------
    sim_input : dict
        Dictionary of simulation parameters. Must include the following:
        - "Code" (StabilizerCode or List) = The QEC code (class, not object) to use in the simulation.
        - "error_model" (str or List) = The name of the error model to use. "pauli" for PauliErrorModel
        and "gauss" for GaussPauliErrorModel.
        - "decoder" (str or List) = The name of the decoder to use. "bposd" for OSD- and "bplsd" for LSD-
        belief propagation. Will use regular or Gaussian depending on the specified error model.
        - "r_xyz" (list) = List containing the r_x, r_y and r_z values to give as input to the
        error model. These are the distribution of how often the differen type of errors will
        occur, and must therefore sum to one.
        - "p_vals" (ndarray) = Array of error rates to simulate for.
        - "L_vals" (list) = List of code sizes to consider.
        - "n_trials" (int or List) = Number of Monte Carlo simulations to run the simulation for.
    
    Returns
    -------
    List[str]
        List of paths to the JSON files where the output from each simulation is stored.
    """
    sim_input_list = split_inputs(sim_inputs)
    n_sims = len(sim_input_list)

    json_paths = []
    for i, sim_input in enumerate(sim_input_list):
        print(f"Running simulation {i+1}/{n_sims}:")
        json_path = run_single_simulation(sim_input)
        json_paths.append(json_path)
    
    return json_paths


def extract_data(json_paths: Union[str, List[str]]) -> pd.DataFrame:
    """
    Extracts the data from several JSON files, and returns them as a single Pandas
    DataFrame. The resulting DataFrame also includes a column labeled "L" for the
    code sizes at each data point.

    Parameters
    ----------
    json_paths : str or List[str]
        The JSON path(s) to extract data from.
    
    Returns
    -------
    DataFrame
        The Data from the JSON paths as a Pandas DataFrame.
    """
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
) -> str:
    """
    Analyze output from a simulation, and create the threshold plots. The resulting
    plot is saved in the plot directory.

    Parameters
    ----------
    data : DataFrame
        The data to plot from. This would be the DataFrame acquired by running
        extract_data() on the JSON files produced by the simulation.
    filename : str
        Name of the pdf file to where the produced plot is saved.
    style : str
        Column label of the data that should decide the style of the lines in
        the plot. If None: the plot is not divided into styles.
        See the Seaborn documentation for more information.
    hue : str
        Column label of the data that should decide the hue (color) of the lines
        in the plot. If None: the plot is not divided into colors.
        See the Seaborn documentation for more information.
    col : str
        Column label of the data that should decide the columns of the subplots
        in the plot. If None: the plot is not divided into subplots.
        See the Seaborn documentation for more information.
    
    Returns
    -------
    str
        The path to the saved plot.
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
    p_vals = np.linspace(0.001, 0.5, 5)
    sim_inputs = {
        "Code": Codes,
        "error_model": error_models,
        "decoder": decoders,
        "r_xyz": [0.5, 0.0, 0.5],
        "p_vals": p_vals,
        "L_vals": [8, 16],
        "n_trials": 10
    }

    # Run and analyze simulation
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", help="Run new simulations, overwriting any existing JSON files without asking.", action="store_true")
    parser.add_argument("-k", "--keep", help="Don't run new simulations for already existing JSON files.", action="store_true")
    args = parser.parse_args()
    if args.keep and args.overwrite:
        print("WARNING: Cannot both overwrite and keep JSON files. Reverting back to asking at each JSON file.")

    json_paths = run_simulations(sim_inputs)
    
    data = extract_data(json_paths)
    
    filename = "gauss_test"
    plot_path = plot_results(data, filename, style="L", hue="error_model", col="code")

    print(f"Results plotted in {plot_path}")