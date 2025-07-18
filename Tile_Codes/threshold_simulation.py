import tile_codes as tc
from decoders import BeliefPropagationLSDDecoder

import numpy as np
import matplotlib.pyplot as plt

from typing import Type
from tqdm import tqdm # Progress bar

from panqec.codes import StabilizerCode
from panqec.error_models import PauliErrorModel
from panqec.decoders import BaseDecoder, BeliefPropagationOSDDecoder
from panqec.simulation import DirectSimulation, BatchSimulation
from panqec.analysis import Analysis

OUTPUT_DIR = "Simulation_Outputs/" # Directory to store JSON files containing the output from simulations.
PLOT_DIR = "Threshold_Plots/" # Directory to store threshold plots created from the simulations.

def get_sim_name(
        Code: Type[StabilizerCode],
        sim_input: dict,
        Decoder: Type[BaseDecoder]
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
    ## Extract and "stringify" the simulation parameters.
    code_name = Code.__name__
    decoder_name = Decoder.__name__
    if decoder_name == "BeliefPropagationOSDDecoder":
        decoder_name = "bposd"
    elif decoder_name == "BeliefPropagationLSDDecoder":
        decoder_name = "bplsd"
    E_X = sim_input["E_X"]
    L_text = f"L"
    for L in sim_input["L_vals"]:
        L_text = L_text + "_" + str(L)
    E_X, E_Y, E_Z = (sim_input["E_X"], sim_input["E_Y"], sim_input["E_Z"])
    error_text = f"error_{E_X:.2}_{E_Y:.2}_{E_Z:.2}"
    p_text = f"p_{sim_input["p_min"]:.2}_{sim_input["p_max"]:.2}_{sim_input["n_p"]}"
    trials_text = f"trials_{sim_input["n_trials"]}"

    ## Combine parameter strings into one string.
    sim_name = f"th_{code_name}__{decoder_name}__{error_text}__{L_text}__{p_text}__{trials_text}"

    return sim_name


def run_threshold_simulation(
        Code: Type[StabilizerCode],
        sim_input: dict,
        Decoder: Type[BaseDecoder] = BeliefPropagationOSDDecoder,
        p_logarithmic: bool = False
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
    error_model = PauliErrorModel(E_X, E_Y, E_Z)
    if p_logarithmic:
        p_vals = np.logspace(np.log10(p_min), np.log10(p_max), n_p)
    else:
        p_vals = np.linspace(p_min, p_max, n_p)

    sim_name = get_sim_name(Code, sim_input, Decoder)
    output_path = OUTPUT_DIR + sim_name + ".json"

    ## Run simulation
    batch_sim = BatchSimulation(output_path)

    for L in L_vals:
        code = Code(L)
        for p in p_vals:
            p = float(p)
            # decoder = BeliefPropagationOSDDecoder(code, error_model, p)
            # decoder = BeliefPropagationLSDDecoder(code, error_model, p)
            decoder = Decoder(code, error_model, p)
            dir_sim = DirectSimulation(code, error_model, decoder, p)
            batch_sim.append(dir_sim)
    
    batch_sim.run(n_trials, progress=tqdm)


def analyze_and_plot_threshold(
        Code: Type[StabilizerCode],
        sim_input: dict,
        Decoder: Type[BaseDecoder] = BeliefPropagationOSDDecoder,
        xscale: str = "linear",
        yscale: str = "linear",
        include_threshold_estimate: bool = True
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
    sim_name = get_sim_name(Code, sim_input, Decoder)
    sim_data_path = OUTPUT_DIR + sim_name + ".json" # Path to output data from simulation.
    plot_path = PLOT_DIR + sim_name + ".pdf" # Path for saving the threshold plot.
    
    analysis = Analysis(sim_data_path)
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    plt.sca(ax[0])
    analysis.plot_thresholds(include_threshold_estimate=include_threshold_estimate)
    plt.sca(ax[1])
    analysis.plot_thresholds(sector='X', include_threshold_estimate=include_threshold_estimate)
    plt.sca(ax[2])
    analysis.plot_thresholds(sector='Z', include_threshold_estimate=include_threshold_estimate)

    for ax_ in ax:
        ax_.set_xscale(xscale)
        ax_.set_yscale(yscale)

    fig.savefig(plot_path, bbox_inches="tight")


if __name__ == "__main__":
    ## Set simulation parameters
    Code = tc.TileCode_B3_W6
    sim_input = {
        "E_X": 1/3,
        "E_Y": 1/3,
        "E_Z": 1/3,
        "p_min": 1e-6,
        "p_max": 1e-1,
        "n_p": 5,
        "L_vals": [8, 12, 16],
        "n_trials": 10000
    }
    Decoder = BeliefPropagationLSDDecoder
    
    # Run and analyze simulation
    # run_threshold_simulation(Code, sim_input, Decoder, p_logarithmic=True)
    analyze_and_plot_threshold(Code, sim_input, Decoder, include_threshold_estimate=False, xscale="log", yscale="linear")
    print()
    print(f"Name of simulation: {get_sim_name(Code, sim_input, Decoder)}")