from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from panqec.simulation import read_input_dict
from panqec.analysis import Analysis

input_data = {
    'ranges': {
        'label': 'Toric 2D Experiment',  # Can be any name you want
        'code': {
            'name': 'Toric2DCode',  # Class name of the code
            'parameters': [
                {'L_x': 6},
                {'L_x': 12},
                {'L_x': 18},
            ]
        },
        'error_model': {
            'name': 'PauliErrorModel',  # Class name of the error model
            'parameters': [
                {'r_x': 1/3, 'r_y': 1/3, 'r_z': 1/3}  # Ratios of X, Y and Z errors
            ],
        },
        'decoder': {
            'name': 'MatchingDecoder',  # Class name of the decoder
            'parameters': [{}]
        },
        'error_rate': np.linspace(0.1, 0.2, 8).tolist()  # List of physical error rates
    }
}

plot_frequency = 20  # Frequency of plot update
save_frequency = 10  # Frequency of saving to file
n_trials = 1000  # Target number of Monte Carlo runs

# We create a BatchSimulation by reading the input dictionary
batch_sim = read_input_dict(
    input_data,
    output_file='toric-2d-results.json',  # Where to store the simulation results
    update_frequency=plot_frequency,
    save_frequency=save_frequency
)

# # Live update of the plot during the simulation
# # (only works in Jupyter notebooks)
# batch_sim.activate_live_update()

batch_sim.run(n_trials, progress=tqdm)

analysis = Analysis("toric-2d-results.json")
# analysis.plot_thresholds(pdf="test.pdf")

fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

plt.sca(ax[0])
analysis.plot_thresholds()
plt.sca(ax[1])
analysis.plot_thresholds(sector='X')
plt.sca(ax[2])
analysis.plot_thresholds(sector='Z')
fig.savefig("thresholds.pdf", bbox_inches="tight")

# analysis.make_collapse_plots("collapse.pdf")

results = analysis.get_results()
# print(results[['code', 'decoder', 'd', 'error_rate', 'p_est', 'p_se', 'n_fail', 'n_trials']].head(10))

selected_columns = ['code', 'error_model', 'bias', 'p_th_fss', 'p_th_fss_se', 'p_th_fss_left', 'p_th_fss_right']
print(analysis.thresholds[selected_columns])

print(analysis.sector_thresholds['X'][selected_columns])

plt.figure()
for d in results['d'].unique():
    plt.errorbar(results[results['d'] == d]['error_rate'],
                 results[results['d'] == d]['p_est'],
                 results[results['d'] == d]['p_se'],
                 label=f'd={d}')

plt.axvline(analysis.thresholds.iloc[0]['p_th_fss'], color='red', linestyle='--')
plt.axvspan(analysis.thresholds.iloc[0]['p_th_fss_left'], analysis.thresholds.iloc[0]['p_th_fss_right'],
            alpha=0.5, color='pink')

plt.xlabel('Physical error rate')
plt.ylabel('Logical error rate')
plt.legend()
plt.savefig("custom_plot.pdf")