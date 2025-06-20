import qsurface.main as qsm

import matplotlib as mpl
from qsurface.plot import PlotParams
mpl.use("Qt5Agg")
my_params = PlotParams(scale_figure_length=7, scale_figure_height=7)


# code, decoder = qsm.initialize(
#     size=(5, 5), # Lattice size
#     Code="toric", # Surface topology
#     Decoder="mwpm", # Algorithm for decoding. "mwpm" = Minimal Weight Perfect Matching
#     enabled_errors=["pauli"],
#     # initial_states=(0,0),
#     # plotting=True,
#     # plot_params=my_params,
#     ### NB: Very prone to mistakes! No error message when including non-existent arguments
#     # dette_finnes_ikke_men_gir_ikke_feilmelding=0.5
# )

# benchmarker = qsm.BenchmarkDecoder({"decode": "duration"}, decoder=decoder)
# code.random_errors(p_bitflip=0.1)
# decoder.decode()
# benchmarker.lists

code, decoder = qsm.initialize((6,6), "toric", "mwpm", enabled_errors=["pauli"])
benchmarker = qsm.BenchmarkDecoder(
    {"decode": "duration"},
    decoder=decoder
)
code.random_errors(p_bitflip=0.1)
decoder.decode()
benchmarker.lists

# qsm.run(
#     code=code,
#     decoder=decoder,
#     iterations=5, # Number of times to add and decode errors
#     error_rates={
#         "p_bitflip": 0.1,
#         "p_phaseflip": 0.1
#     }, # Probability of errors occuring
#     decode_initial=False,
#     benchmark=benchmarker,
#     seed=100,
#     ### NB: Very prone to mistakes! No error message when including non-existent arguments
#     dette_finnes_ikke_men_gir_ikke_feilmelding=0.5
# )