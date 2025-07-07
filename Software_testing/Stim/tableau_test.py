# import stim
# import numpy as np
# import matplotlib.pyplot as plt

# n_qubits = 5
# circuit = stim.Circuit()
# for i in range(n_qubits-1):
#     circuit.append("H", i)
#     circuit.append("CNOT", [i, i+1])
#     circuit.append("M", [i, i+1])

# print(circuit.diagram())
# sampler = circuit.compile_sampler()
# sample = sampler.sample(10)
# summed = np.sum(sample[:,::2], axis=1)

# print(summed)

# plt.hist(summed, bins=5)
# plt.savefig("sample_test.pdf")

# import stim

# circuit = stim.Circuit()

# n = 9
# circuit.append("R", range(n))
# circuit.append("H", [1, 2, 5, 6])

# # Use stim.PauliString
# pauli_str = stim.PauliString("X1*X2*X5*X6")
# circuit.append("MPP", [pauli_str])
# circuit.append("DETECTOR", [stim.target_rec(-1)])

# print(circuit.diagram())

# samples = circuit.compile_detector_sampler().sample(shots=5)
# print(samples)


# import stim
# import numpy as np

# L = 3
# n_qubits = 2*L*L
# n_X_stabilizers = L*L

# def add_X_vertex(ind):
#     sq = n_qubits + ind # Stabilizer (ancilla) qubit
#     x, y = (ind // L, ind % L) # Get coordinate of vertex-index
#     qxs = np.array([(x-1)%L, x, x, x]) # Find qubit x-coordinates
#     qys = np.array([y, y, (y-1)%L, y]) # Find qubit y-coordinates
#     qubits = qxs*L + qys # Find qubit indices
#     qubits[2:] = qubits[2:] + L*L # Vertical qubit indices are shifted by L^2
#     q0, q1, q2, q3 = qubits
#     circuit.append("H", sq)
#     circuit.append("CZ", [sq, q0])
#     circuit.append("CZ", [sq, q1])
#     circuit.append("CZ", [sq, q2])
#     circuit.append("CZ", [sq, q3])

# def add_Z_plaquette(ind):
#     sq = n_qubits + n_X_stabilizers + ind # Stabilizer (ancilla) qubit
#     x, y = (ind // L, ind % L) # Get coordinate of plaquette-index
#     qxs = np.array([x, x, x, (x+1)%L]) # Find qubit x-coordinates
#     qys = np.array([y, (y+1)%L, y, y]) # Find qubit y-coordinates
#     qubits = qxs*L + qys # Find qubit indices
#     qubits[2:] = qubits[2:] + L*L # Vertical qubit indices are shifted by L^2
#     q0, q1, q2, q3 = qubits
#     circuit.append("H", sq)
#     circuit.append("CX", [sq, q0])
#     circuit.append("CX", [sq, q1])
#     circuit.append("CX", [sq, q2])
#     circuit.append("CX", [sq, q3])

# circuit = stim.Circuit()
# for i in range(n_X_stabilizers):
#     add_X_vertex(i)
#     add_Z_plaquette(i)

# pauli_str = stim.PauliString("X4*X5*X13*X16")
# circuit.append("MPP", [pauli_str])
# print(circuit.diagram())