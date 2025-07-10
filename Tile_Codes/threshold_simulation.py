from tile_codes import TileCodes

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm # Progress bar

from panqec.error_models import PauliErrorModel
from panqec.decoders import BeliefPropagationOSDDecoder
from panqec.simulation import DirectSimulation, BatchSimulation
from panqec.analysis import Analysis

OUTPUT_DIR = "Simulation_Outputs/" # Directory to store JSON files containing the output from simulations.
PLOT_DIR = "Threshold_Plots/" # Directory to store threshold plots created from the simulations.

def get_sim_name(sim_input: dict):
    """
    Returns a name that is used for the output-data and plot files, such as "thresholds_error_0.5_0.0_0.5__L_8_12_16_20__p_0.001_0.3_20__trials_1000".

    ## Parameters:
        - sim_input = Dictionary of simulation parameters. Must include the following:
            - "E_X", "E_Y", "E_Z" = Distribution of X,Y and Z-errors. Must sum to 1.
            - "p_min", "p_max" = Min- and max-values for the error probability p of physical qubits.
            - "n_p" = Number of p-values to include in the simulation.
            - "L_vals" = List of code-sizes to consider.
            - "n_trials" = Number of Monte Carlo iterations to run the simulation for.
    """
    ## Extract and "stringify" the simulation parameters.
    E_X = sim_input["E_X"]
    L_text = f"L"
    for L in sim_input["L_vals"]:
        L_text = L_text + "_" + str(L)
    E_X, E_Y, E_Z = (sim_input["E_X"], sim_input["E_Y"], sim_input["E_Z"])
    error_text = f"error_{E_X:.2}_{E_Y:.2}_{E_Z:.2}"
    p_text = f"p_{sim_input["p_min"]:.2}_{sim_input["p_max"]:.2}_{sim_input["n_p"]}"
    trials_text = f"trials_{sim_input["n_trials"]}"

    ## Combine parameter strings into one string.
    sim_name = f"thresholds_{error_text}__{L_text}__{p_text}__{trials_text}"

    return sim_name


def run_threshold_simulation(sim_input: dict):
    """
    Runs a threshold simulation, and stores it into a JSON file.

    ## Parameters:
        - sim_input = Dictionary of simulation parameters. Must include the following:
            - "E_X", "E_Y", "E_Z" = Distribution of X,Y and Z-errors. Must sum to 1.
            - "p_min", "p_max" = Min- and max-values for the error probability p of physical qubits.
            - "n_p" = Number of p-values to include in the simulation.
            - "L_vals" = List of code-sizes to consider.
            - "n_trials" = Number of Monte Carlo iterations to run the simulation for.
    """
    ## Extract input parameters
    E_X, E_Y, E_Z = (sim_input["E_X"], sim_input["E_Y"], sim_input["E_Z"])
    p_min = sim_input["p_min"]
    p_max = sim_input["p_max"]
    n_p = sim_input["n_p"]
    L_vals = sim_input["L_vals"]
    n_trials = sim_input["n_trials"]

    ## Set remaining parameters
    error_model = PauliErrorModel(E_X, E_Y, E_Z)
    p_vals = np.linspace(p_min, p_max, n_p)

    sim_name = get_sim_name(sim_input)
    output_path = OUTPUT_DIR + sim_name + ".json"

    ## Run simulation
    batch_sim = BatchSimulation(output_path)

    for L in L_vals:
        code = TileCodes(L)
        for p in p_vals:
            p = float(p)
            decoder = BeliefPropagationOSDDecoder(code, error_model, p)
            dir_sim = DirectSimulation(code, error_model, decoder, p)
            batch_sim.append(dir_sim)
    
    batch_sim.run(n_trials, progress=tqdm)

def analyze_and_plot_threshold(sim_name):
    """
    Analyze output from a simulation, and create the threshold plots. The resulting plot is saved in the plot directory.

    ## Parameters:
        sim_name: Name of output and plot file (not including folder or file type). Should be the same as acquired from get_sim_name().
    """
    sim_data_path = OUTPUT_DIR + sim_name + ".json" # Path to output data from simulation.
    plot_path = PLOT_DIR + sim_name + ".pdf" # Path for saving the threshold plot.
    
    analysis = Analysis(sim_data_path)

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    plt.sca(ax[0])
    analysis.plot_thresholds()
    plt.sca(ax[1])
    analysis.plot_thresholds(sector='X')
    plt.sca(ax[2])
    analysis.plot_thresholds(sector='Z')

    fig.savefig(plot_path, bbox_inches="tight")


## Set simulation parameters
sim_input = {
    "E_X": 0.5,
    "E_Y": 0.0,
    "E_Z": 0.5,
    "p_min": 0.001,
    "p_max": 0.3,
    "n_p": 20,
    "L_vals": [8, 12, 16],
    "n_trials": 2000
}

## Run and analyze simulation
run_threshold_simulation(sim_input)
sim_name = get_sim_name(sim_input)
analyze_and_plot_threshold(sim_name)

print()
print(f"Name of simulation: {sim_name}")