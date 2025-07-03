import numpy as np

L = 6
B = 3
dB = B-1 # Number of "extra" stabilizers on each side

L_small = L - dB # Length of the short side of red/blue dots
L_big = L + dB # Length of the long side of red/blue dots

n_qubits = L*L
n_stabilizers = L_small*L_big # Number of X (or Z) stabilizers

X_tile = np.array([
    [[0,0],[2,1],[2,2]],
    [[2,0],[0,2],[1,2]]
], dtype=int)

def get_Z_tile(X_tile):
    Z_tile = np.zeros(X_tile.shape, dtype=int)
    Z_tile[[0,1]] = (dB - X_tile)[[1,0]] # Index swapping since horizontal X errors give vertical Z errors and vice versa
    return Z_tile

def get_tile_indices(ind):
    """
    Input: Index of stabilizer (row in Hamming matrix, with X tiles first then Z tiles)
    Output: List of indices of qubit errors (columns in Hamming matrix)
    """
    if ind < n_stabilizers:
        # X stabilizer
        x = ind % L_small
        y = ind // L_small - dB # Subtract dB to get coordinates relative to qubits
        tile = X_tile
    else:
        # Z stabilizer
        ind = ind - n_stabilizers
        x = ind % L_big - dB # Subtract dB to get coordinates relative to qubits
        y = ind // L_big
        tile = get_Z_tile(X_tile)

    # loc = np.zeros(tile.shape)
    # loc[:,:,:] = np.array([x,y])
    
    # q_loc = loc + tile
    # for is_vert, delta in enumerate(tile):
    #     print(is_vert, delta)
    #     q_loc = loc + delta

    indices = []
    for is_vert, delta_i in enumerate(tile):
        for d in delta_i:
            dx, dy = (d[0], d[1])
            qx, qy = (x + dx, y + dy)
            valid_x = qx >= 0 and qx < L
            valid_y = qy >= 0 and qy < L
            is_qubit = valid_x and valid_y
            if is_qubit:
                ind = qx + qy*L + L*L*is_vert # Add L^2 if the qubits are vertical
                indices.append(ind)

    return indices

# get_tile_indices(1)


## Test:
errors = []
test_indices = [1, 17, 31, 35, 43, 56, 62]
expected = [
    [3, 37, 38],
    [13, 21, 27, 51, 61, 62],
    [33, 71],
    [13, 3, 2, 51, 43, 37],
    [19, 9, 8, 57, 49, 43],
    [18, 66],
    [34, 23, 64, 58]
]
for i in range(len(test_indices)):
    ind = test_indices[i]
    exp = expected[i]
    res = get_tile_indices(ind)
    if not res == exp:
        errors.append((ind, res, exp))

if len(errors) == 0:
    print("Test passed!")
else:
    print("Got the following error(s):")
    for err in errors:
        print(f"  Index: {err[0]}: Got {err[1]}, expected {err[2]}.")    

# ## Test:
# test_indices = [3, 11, 24, 30]
# expected = [
#     [2, 3, 13, 37, 43, 51],
#     [8, 9, 19, 43, 49, 57],
#     [18, 66],
#     [23, 34, 58, 64]
# ]
# errors = []
# for i in range(len(test_indices)):
#     ind = test_indices[i]
#     exp = expected[i]
#     res = get_Z_tile(ind)
#     if not res == exp:
#         errors.append((ind, res, exp))

# if len(errors) == 0:
#     print("Test passed for Z tile!")
# else:
#     print("Got the following error(s) for the Z tile:")
#     for err in errors:
#         print(f"  Index: {err[0]}: Got {err[1]}, expected {err[2]}.")



# def get_X_tile(ind):
#     """
#     Input: Index of X_stabilizer (row in Hamming matrix)
#     Output: List of indices of qubit errors (columns in Hamming matrix)
#     """

#     x = ind % L_small
#     y = ind // L_small - dB # Subtract dB to get coordinates relative to qubits

#     ## Define the Tile by its errors (coordinates given as left/bottom vertex of the horizontal/vertical qubits)
#     delta_h = [(0,0), (2,1), (2,2)] # Horizontal qubits
#     delta_v = [(2,0), (0,2), (1,2)] # Vertical qubits
#     delta = [delta_h, delta_v]

#     ## Find indices
#     indices = []

#     for is_vert, delta_i in enumerate(delta):
#         for d in delta_i:
#             dx, dy = (d[0], d[1])
#             qx, qy = (x + dx, y + dy)
#             valid_x = qx >= 0 and qx < L
#             valid_y = qy >= 0 and qy < L
#             is_qubit = valid_x and valid_y
#             if is_qubit:
#                 ind = qx + qy*L + L*L*is_vert # Add L^2 if the qubits are vertical
#                 indices.append(ind)
    
#     return indices

# def get_Z_tile(ind):
#     """
#     Input: Index of X_stabilizer (row in Hamming matrix)
#     Output: List of indices of qubit errors (columns in Hamming matrix)
#     """

#     x = ind % L_big - dB # Subtract dB to get coordinates relative to qubits
#     y = ind // L_big

#     ## Define the Tile by its errors (coordinates given as left/bottom vertex of the horizontal/vertical qubits)
#     delta_h = [(1,0), (2,0), (0,2)] # Horizontal qubits
#     delta_v = [(0,0), (0,1), (2,2)] # Vertical qubits
#     delta = [delta_h, delta_v]

#     ## Find indices
#     indices = []

#     for is_vert, delta_i in enumerate(delta):
#         for d in delta_i:
#             dx, dy = (d[0], d[1])
#             qx, qy = (x + dx, y + dy)
#             valid_x = qx >= 0 and qx < L
#             valid_y = qy >= 0 and qy < L
#             is_qubit = valid_x and valid_y
#             if is_qubit:
#                 ind = qx + qy*L + L*L*is_vert # Add L^2 if the qubits are vertical
#                 indices.append(ind)
    
#     return indices


# ## Test:
# errors = []
# test_indices = [1, 17, 31]
# expected = [
#     [3, 37, 38],
#     [13, 21, 27, 51, 61, 62],
#     [33, 71]
# ]
# for i in range(len(test_indices)):
#     ind = test_indices[i]
#     exp = expected[i]
#     res = get_X_tile(ind)
#     if not res == exp:
#         errors.append((ind, res, exp))

# if len(errors) == 0:
#     print("Test passed for X tile!")
# else:
#     print("Got the following error(s) for the X tile:")
#     for err in errors:
#         print(f"  Index: {err[0]}: Got {err[1]}, expected {err[2]}.")


# ## Test:
# test_indices = [3, 11, 24, 30]
# expected = [
#     [2, 3, 13, 37, 43, 51],
#     [8, 9, 19, 43, 49, 57],
#     [18, 66],
#     [23, 34, 58, 64]
# ]
# errors = []
# for i in range(len(test_indices)):
#     ind = test_indices[i]
#     exp = expected[i]
#     res = get_Z_tile(ind)
#     if not res == exp:
#         errors.append((ind, res, exp))

# if len(errors) == 0:
#     print("Test passed for Z tile!")
# else:
#     print("Got the following error(s) for the Z tile:")
#     for err in errors:
#         print(f"  Index: {err[0]}: Got {err[1]}, expected {err[2]}.")