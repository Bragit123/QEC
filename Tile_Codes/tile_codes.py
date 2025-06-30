from panqec.codes import StabilizerCode
from panqec.gui import GUI


class TileCodes(StabilizerCode):
    dimension = 2

    @property
    def label(self):
        return f"Tile Code {self.size[0]}x{self.size[1]}"
    
    def get_qubit_coordinates(self):
        B = 3
        coordinates = []
        Lx, Ly = self.size
        
        # Qubits oriented along x-axis
        for x in range(2*(B-1)+1, 2*(Lx+B-1), 2):
            for y in range(2*(B-1), 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        # Qubits oriented along y-axis
        for x in range(2*(B-1), 2*(Lx+B-1), 2):
            for y in range(2*(B-1)+1, 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        return coordinates
    
    def qubit_axis(self, location):
        x, y = location

        if (y % 2 == 0) and (not x % 2 == 0):
            axis = "x"
        
        elif (x % 2 == 0) and (not y % 2 == 0):
            axis = "y"
        
        else:
            raise ValueError("Location in qubit_axis() must be an edge.")

        return axis
    
    def get_stabilizer_coordinates(self):
        coordinates = []
        Lx, Ly = self.size
        B = 3
        
        # X errors (red/black dots)
        for x in range(2*(B-1), 2*Lx, 2):
            for y in range(0, 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        # Z errors (blue/black dots, but placed on faces)
        for x in range(1, 2*(Lx+B-1), 2):
            for y in range(2*(B-1)+1, 2*Ly, 2):
                coordinates.append((x, y))
        
        return coordinates

    def stabilizer_type(self, location):
        x, y = location
        if (x % 2 == 0) and (y % 2 == 0):
            return "vertex"
        elif (x % 2 == 1) and (y % 2 == 1):
            return "face"
        else:
            raise ValueError("stabilizer_type() must return either 'vertex' or 'face'")

    def get_stabilizer(self, location):
        if self.stabilizer_type(location) == "vertex":
            pauli = "X"
        elif self.stabilizer_type(location) == "face":
            pauli = "Z"
        else:
            raise ValueError("stabilizer_type() must return either 'vertex' or 'face'")
        
        x, y = location

        # delta specifies the positions of the qubits involved in the stabilizer
        # relative to the stabilizer position
        if self.stabilizer_type(location) == "vertex":
            # X type
            delta = [
                (1, 0),
                (4, 1),
                (5, 2),
                (5, 4),
                (0, 5),
                (2, 5)
            ]
        else:
            # Z type
            # Note: The stabilizer is defined on the face. Thus the delta gets an extra
            # minus one in both the x- and y-coordinates compared to the X stabilizer.
            delta = [
                (2, -1),
                (4, -1),
                (-1, 0),
                (-1, 2),
                (0, 3),
                (3, 4)
            ]
        
        B = 3
        operator = dict()
        for d in delta:
            qubit_location = (x + d[0], y + d[1])

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli
        
        return operator

    def get_logicals_x(self):
        Lx, Ly = self.size
        logicals = []
        B = 3

        operator = dict()
        for x in range(2*(B-1)+1, 2*(Lx+B-1), 2):
            operator[(x, 2*(B-1))] = "X"
        logicals.append(operator)
        
        operator = dict()
        for y in range(2*(B-1)+1, 2*(Ly+B-1), 2):
            operator[(2*(B-1), y)] = "X"
        logicals.append(operator)

        return logicals

    def get_logicals_z(self):
        Lx, Ly = self.size
        logicals = []
        B = 3

        operator = dict()
        for x in range(2*(B-1), 2*(Lx+B-1), 2):
            operator[(x, 2*(B-1)+1)] = "Z"
        logicals.append(operator)
        
        operator = dict()
        for y in range(2*(B-1), 2*(Ly+B-1), 2):
            operator[(2*(B-1)+1, y)] = "Z"
        logicals.append(operator)

        return logicals
    
    def qubit_representation(self, location, rotated_picture=False):
        return super().qubit_representation(location, rotated_picture, json_file="tile_code.json")
    
    def stabilizer_representation(self, location, rotated_picture=False):
        res_dict = super().stabilizer_representation(location, rotated_picture, json_file="tile_code.json")
        if self.stabilizer_type(location) == "face":
            x, y = location
            location = [x-1, y-1]
            res_dict["location"] = location
        else:
            x, y = location
            location = [x, y, 1]
            res_dict["location"] = location
        return res_dict


# gui = GUI()
# gui.add_code(TileCodes, "Tile Code")
# gui.run(port=5000)

code = TileCodes(12)

import numpy as np
from panqec.error_models import PauliErrorModel
error_model = PauliErrorModel(0.5, 0.0, 0.5)
errors = error_model.generate(code, 0.1)

n_err = len(errors)
n = n_err // 2
x_err = errors[:n]
z_err = errors[n:]

log_x = np.zeros(n_err)
qubit_coords = code.get_qubit_coordinates()
for ind, coord in enumerate(qubit_coords):
    if coord[1] == 4:
        print(f"X: {ind}: {coord}")
        log_x[ind] = 1

print(code.in_codespace(log_x))
print(code.is_logical_error(log_x))