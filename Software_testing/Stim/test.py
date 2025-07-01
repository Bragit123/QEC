import stim

circuit = stim.Circuit("""
    H 0
    CNOT 0 1
    M 0 1
""")

# circuit = stim.Circuit()
# circuit.append("H", 0)
# circuit.append("CNOT", [0, 1])
# circuit.append("M", [0, 1])

# print(circuit.diagram())

sampler = circuit.compile_sampler()
# print(sampler.sample(shots=10))

circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
# print(circuit.diagram())

sampler = circuit.compile_detector_sampler()
# print(sampler.sample(shots=10))


circuit = stim.Circuit()
circuit.append("H", 0)
circuit.append("TICK")
circuit.append("CNOT", [0, 1])
circuit.append("X_ERROR", [0, 1], 0.2)
circuit.append("TICK")
circuit.append("M", [0, 1])
circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])

print(repr(circuit))
print(circuit.diagram())