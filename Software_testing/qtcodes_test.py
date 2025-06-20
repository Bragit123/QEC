# from qtcodes import RepetitionQubit
# qubit = RepetitionQubit({"d":3},"t")
# qubit.reset_z()
# qubit.stabilize()
# qubit.x()
# qubit.stabilize()
# qubit.readout_z()
# qubit.draw(output='mpl', fold=150)

# from qtcodes import TopologicalRegister, TopologicalCircuit, REPETITION
# treg = TopologicalRegister(ctypes=[REPETITION, REPETITION], params=[{"d": 3}, {"d": 3}])
# circ = TopologicalCircuit(treg)
# circ.x(treg[0])
# circ.stabilize(treg[1])
# circ.x(1)
# circ.draw(output='mpl', fold=500)

from qtcodes import RotatedDecoder
#d: surface code side length, T: number of rounds
decoder = RotatedDecoder({"d":5,"T":1})
all_syndromes = {"X": [(0,1.5,.5),(0,.5,1.5)], "Z": [(0,0.5,0.5),(0,1.5,1.5),(0,1.5,3.5), (0,3.5,3.5)]}
matches = {}

for syndrome_key, syndromes in all_syndromes.items():
    print(f"{syndrome_key} Syndrome Graph")
    error_graph = decoder._make_error_graph(syndromes,syndrome_key)
    print("Error Graph")
    decoder.draw(error_graph)
    matches[syndrome_key] = decoder._run_mwpm(error_graph)
    matched_graph = decoder._run_mwpm_graph(error_graph)
    print("Matched Graph")
    decoder.draw(matched_graph)
    print(f"Matches: {matches[syndrome_key]}")
    print("\n===\n")
