from src import tile_codes as tc
from src.simulation import Simulation

import argparse

import numpy as np

from panqec.codes import Toric2DCode

## Set output and plot directories
OUTPUT_DIR = "Simulation_Outputs/"
PLOT_DIR = "Plots/"

## Handle command-line options
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--overwrite", help="Run new simulations, overwriting any existing JSON files without asking.", action="store_true")
parser.add_argument("-k", "--keep", help="Don't run new simulations for already existing JSON files.", action="store_true")
args = parser.parse_args()

existing_json_handling = "ask"
if args.overwrite and not args.keep:
    existing_json_handling = "overwrite"
elif args.keep and not args.overwrite:
    existing_json_handling = "keep"
elif args.keep and args.overwrite:
    print("WARNING: Cannot both overwrite and keep JSON files. Reverting back to asking at each JSON file.")
            

## Set simulation parameters
Codes = [Toric2DCode, tc.TileCode_B3_W6]
error_models = ["pauli", "gauss"]
decoders = "bplsd"
r_xyz = [0.5, 0.0, 0.5]
p_vals = np.linspace(0.001, 0.5, 20)
L_vals = [8, 12, 16]
n_trials = 1000
sim_inputs = {
    "Codes": Codes,
    "error_models": error_models,
    "decoders": decoders,
    "r_xyz": r_xyz,
    "p_vals": p_vals,
    "L_vals": L_vals,
    "n_trials": n_trials
}

plot_filename = "pauli_gauss_comparison_4torics"

# Run and analyze simulation
sim = Simulation(
    **sim_inputs,
    output_dir=OUTPUT_DIR,
    plot_dir=PLOT_DIR,
    existing_json_handling=existing_json_handling
)

sim.run_simulations()
plot_path = sim.plot_results(
    filename=plot_filename,
    style="L",
    hue="code",
    col="error_model",
    # hue="error_model",
    # col="code",
    multiply_k=4
)

print(f"Results plotted in {plot_path}")