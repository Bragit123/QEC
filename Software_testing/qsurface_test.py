# import matplotlib.blocking_input

import matplotlib.pyplot as plt
from qsurface.main import initialize, run
code, decoder = initialize((6, 6), "toric", "mwpm", enabled_errors=["pauli"], visualize=True)
run(code, decoder, iterations=10, error_rates={"p_bitflip": 0.1})

plt.show()