# # import stim

# # circuit = stim.Circuit("""
# #     H 0
# #     CNOT 0 1
# #     M 0 1
# # """)

# # circuit = stim.Circuit()
# # circuit.append("H", 0)
# # circuit.append("CNOT", [0, 1])
# # circuit.append("M", [0, 1])

# # print(circuit.diagram())

# sampler = circuit.compile_sampler()
# # print(sampler.sample(shots=10))

# circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
# # print(circuit.diagram())

# sampler = circuit.compile_detector_sampler()
# # print(sampler.sample(shots=10))


# circuit = stim.Circuit()
# circuit.append("H", 0)
# circuit.append("TICK")
# circuit.append("CNOT", [0, 1])
# circuit.append("X_ERROR", [0, 1], 0.2)
# circuit.append("TICK")
# circuit.append("M", [0, 1])
# circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])

# print(repr(circuit))
# print(circuit.diagram())

import stim
import numpy as np

L = 3
n_qubits = 2*L*L
n_X_stabilizers = L*L

def add_stabilizer(ind):
    if ind < n_X_stabilizers:
        add_X_vertex(ind)
    else:
        ind = ind - n_X_stabilizers
        add_Z_plaquette(ind)

def add_X_vertex(ind):
    x, y = (ind // L, ind % L) # Get coordinate of vertex-index
    qxs = np.array([(x-1)%L, x, x, x]) # Find qubit x-coordinates
    qys = np.array([y, y, (y-1)%L, y]) # Find qubit y-coordinates
    qubits = qxs*L + qys # Find qubit indices
    qubits[2:] = qubits[2:] + L*L # Vertical qubit indices are shifted by L^2
    q0, q1, q2, q3 = qubits
    circuit.append("H", q0)
    circuit.append("CZ", [q0, q1])
    circuit.append("CZ", [q0, q2])
    circuit.append("CZ", [q0, q3])

def add_Z_plaquette(ind):
    x, y = (ind // L, ind % L) # Get coordinate of plaquette-index
    qxs = np.array([x, x, x, (x+1)%L]) # Find qubit x-coordinates
    qys = np.array([y, (y+1)%L, y, y]) # Find qubit y-coordinates
    qubits = qxs*L + qys # Find qubit indices
    qubits[2:] = qubits[2:] + L*L # Vertical qubit indices are shifted by L^2
    q0, q1, q2, q3 = qubits
    circuit.append("H", q2)
    circuit.append("CX", [q2, q0])
    circuit.append("CX", [q2, q1])
    circuit.append("CX", [q2, q3])
    # circuit.append("H", q0)
    # circuit.append("CX", [q0, q1])
    # circuit.append("CX", [q0, q2])
    # circuit.append("CX", [q0, q3])


# for i in range(n_qubits):
#     circuit.append("R", i)
# for x in range(L):
#     for y in range(L):
#         ind = x*L + y
#         circuit.append("QUBIT_COORDS", [ind, x, y])
#         circuit.append("QUBIT_COORDS", [ind+L*L, x, y])
#         # circuit.append("QUBIT_COORDS", [x, y, ind])
#         # circuit.append("QUBIT_COORDS", [x, y, ind+L*L])
#         # print(circuit.diagram())
#         # print()

# print(circuit.get_final_qubit_coordinates())



pauli_strs = []
pauli_strs.append(stim.PauliString("X1*X4*X12*X13"))
pauli_strs.append(stim.PauliString("Z1*Z4*Z12*Z13"))
pauli_strs.append(stim.PauliString("X0*X1*X9*X12"))
pauli_strs.append(stim.PauliString("Z0*Z1*Z9*Z12"))
pauli_strs.append(stim.PauliString("Z0*Z4*Z2*Z17"))
# for pauli_str in [pauli_strs[0]]:
for pauli_str in pauli_strs:
    circuit = stim.Circuit()
    for i in range(n_qubits):
        add_stabilizer(i)
    circuit.append("MPP", pauli_str)
    sampler = circuit.compile_sampler()
    # print(circuit.diagram())
    # print(sampler.sample(shots=1000))
    print(np.sum(sampler.sample(shots=1000)))
    # print(np.all(sampler.sample(shots=1000)==0))