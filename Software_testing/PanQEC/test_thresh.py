import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from panqec.codes import Toric2DCode
from panqec.decoders import MatchingDecoder
from panqec.error_models import PauliErrorModel
from panqec.simulation import DirectSimulation, BatchSimulation
from panqec.analysis import Analysis

error_model = PauliErrorModel(1/3, 1/3, 1/3)

p_vals = np.linspace(0.1, 0.2, 8).tolist()
L_vals = [6, 12, 18]

batch_sim = BatchSimulation("test_output.json")

for L in L_vals:
    code = Toric2DCode(L)
    for p in p_vals:
        decoder = MatchingDecoder(code, error_model, p)
        dir_sim = DirectSimulation(code, error_model, decoder, p)
        batch_sim.append(dir_sim)

n_trials = 1000
batch_sim.run(n_trials, progress=tqdm)

analysis = Analysis("test_output.json")
analysis.plot_thresholds(pdf="test.pdf")