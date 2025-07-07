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

X_ind_start = n_qubits
Z_ind_start = n_qubits + n_X_stabilizers

def get_coord(ind):
    x = ind // L
    y = ind % L
    return (x, y)

def add_stabilizer(ind, circ):
    if ind < n_X_stabilizers:
        add_X_vertex(ind, circ)
    else:
        ind = ind - n_X_stabilizers
        add_Z_plaquette(ind, circ)

def add_X_vertex(ind, circ):
    sq = X_ind_start + ind # Stabilizer index
    x, y = get_coord(ind)
    # x, y = (ind // L, ind % L) # Get coordinate of vertex-index
    qxs = np.array([(x-1)%L, x, x, x]) # Find qubit x-coordinates
    qys = np.array([y, y, (y-1)%L, y]) # Find qubit y-coordinates
    qubits = qxs*L + qys # Find qubit indices
    qubits[2:] = qubits[2:] + L*L # Vertical qubit indices are shifted by L^2
    q0, q1, q2, q3 = qubits
    # circ.append("H", q0)
    
    # circ.append("CNOT", [sq, q0])
    # circ.append("CNOT", [sq, q1])
    # circ.append("CNOT", [sq, q2])
    # circ.append("CNOT", [sq, q3])
    
    circ.append("CX", [sq, q0])
    circ.append("CX", [sq, q1])
    circ.append("CX", [sq, q2])
    circ.append("CX", [sq, q3])

def add_Z_plaquette(ind, circ):
    sq = Z_ind_start + ind
    x, y = get_coord(ind)
    # x, y = (ind // L, ind % L) # Get coordinate of plaquette-index
    qxs = np.array([x, x, x, (x+1)%L]) # Find qubit x-coordinates
    qys = np.array([y, (y+1)%L, y, y]) # Find qubit y-coordinates
    qubits = qxs*L + qys # Find qubit indices
    qubits[2:] = qubits[2:] + L*L # Vertical qubit indices are shifted by L^2
    q0, q1, q2, q3 = qubits

    # circ.append("H", sq)
    # circ.append("CNOT", [sq, q0])
    # circ.append("CNOT", [sq, q1])
    # circ.append("CNOT", [sq, q2])
    # circ.append("CNOT", [sq, q3])
    # circ.append("H", sq)

    circ.append("CZ", [sq, q0])
    circ.append("CZ", [sq, q1])
    circ.append("CZ", [sq, q2])
    circ.append("CZ", [sq, q3])

    # circ.append("H", q0)
    # circ.append("CX", [q0, q1])
    # circ.append("CX", [q0, q2])
    # circ.append("CX", [q0, q3])


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

circuit = stim.Circuit() # Initialize circuit

circuit.append("X_ERROR", [4, 5, 13, 16], 0.5)

# circuit.append("X", 4)
# circuit.append("Z_ERROR", 4, 0.5) # Add error

## Add stabilizers
for i in range(n_qubits):
    add_stabilizer(i, circuit)

# pauli_str = stim.PauliString("X4*X5*X13*X16")
pauli_str = stim.PauliString("Z1*Z4*Z12*Z13")
circuit.append("MPP", [pauli_str])

# ## Add measurements of stabilizers
# for i in range(n_qubits, n_qubits+2*n_X_stabilizers):
#     circuit.append("M", i)


# print(circuit.diagram())
samples = circuit.compile_sampler().sample(shots=100)
print(samples.shape)
# print(samples)
print(np.sum(samples))
# print(np.argwhere(samples != 0))



# print(samples[:,:])
# print(samples[:,:samples.shape[1]//2])
# print(samples[:,samples.shape[1]//2:])

# pauli_strs = []
# pauli_strs.append(stim.PauliString("X1*X4*X12*X13"))
# pauli_strs.append(stim.PauliString("Z1*Z4*Z12*Z13"))
# pauli_strs.append(stim.PauliString("X0*X1*X9*X12"))
# pauli_strs.append(stim.PauliString("Z0*Z1*Z9*Z12"))
# pauli_strs.append(stim.PauliString("Z0*Z4*Z2*Z17"))
# # for pauli_str in [pauli_strs[0]]:
# for pauli_str in pauli_strs:
#     circuit = stim.Circuit()
#     for i in range(n_qubits):
#         add_stabilizer(i)
#     circuit.append("MPP", pauli_str)
#     sampler = circuit.compile_sampler()
#     # print(circuit.diagram())
#     # print(sampler.sample(shots=1000))
#     print(np.sum(sampler.sample(shots=1000)))
#     # print(np.all(sampler.sample(shots=1000)==0))