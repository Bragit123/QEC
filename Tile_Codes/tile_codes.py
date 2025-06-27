from panqec.codes import StabilizerCode
from panqec.gui import GUI

class TileCode(StabilizerCode):
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
        
        # Red dots at the bottom
        for x in range(2*(B-1), 2*Lx, 2):
            for y in range(0, 2*(B-1), 2):
                coordinates.append((x, y))
        
        # Red dots at the top
        for x in range(2*(B-1), 2*Lx, 2):
            for y in range(2*Ly, 2*(Ly+B-1), 2):
                coordinates.append((x, y))
        
        # Blue and black dots
        for x in range(0, 2*(Lx+B-1), 2):
            for y in range(2*(B-1), 2*Ly, 2):
                coordinates.append((x, y))
        
        return coordinates

    def stabilizer_type(self, location):
        return "vertex"

    def get_stabilizer(self, location):
        if self.stabilizer_type(location) == "vertex":
            pauli = "Z"
        else:
            pauli = "X"
        
        x, y = location

        # delta specifies the positions of the qubits involved in the stabilizer
        # relative to the stabilizer position
        if self.stabilizer_type(location) == "vertex":
            # Z type
            delta = [
                (3, 0),
                (5, 0),
                (0, 1),
                (0, 3),
                (1, 4),
                (4, 5)
            ]
        else:
            delta = []
        
        B = 3
        operator = dict()
        for d in delta:
            Lx, Ly = self.size
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
    
    def stabilizer_representation(self, location, rotated_picture=False):
        return super().stabilizer_representation(location, rotated_picture, json_file="tile_code.json")
    
    def qubit_representation(self, location, rotated_picture=False):
        return super().qubit_representation(location, rotated_picture, json_file="tile_code.json")


gui = GUI()
gui.add_code(TileCode, "Tile Code")
gui.run(port=5000)

# code = TileCode(12)

# for coord in code.get_qubit_coordinates():
#     print(coord, code.qubit_axis(coord))

# print()
# for coord in code.get_stabilizer_coordinates():
#     print(coord)

# print(code.get_qubit_coordinates())
# print()
# print(code.get_stabilizer_coordinates())

# print(code.get_logicals_x())
# print()
# print(code.get_logicals_z())