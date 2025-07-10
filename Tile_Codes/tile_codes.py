import numpy as np
from panqec.codes import StabilizerCode
from panqec.gui import GUI
from bposd.css import css_code


class TileCodes(StabilizerCode):
    dimension = 2

    # def __init__(L_x, L_y=None, L_z=None, B: int = 3, x_tile: List = None):
    #     """ tile = list of coordinates for 'delta' in get_stabilizer """
    #     super().__init__(L_x, L_y, L_z)
        
    #     delta_X = [
    #         (1,0),
    #         (4,1),
    #         (5,2),
    #         (5,4),
    #         (0,5),
    #         (2,5),
    #     ]

    #     delta_Z = []
    #     for i in range(len(delta_X)):
    #         x = delta_X[i][0]
    #         y = delta_X[i][1]
    #         delta_Z.append((2*(B-1)-x, 2*(B-1)-y))
    # def __init__(L_x, L_y=None, L_z=None, is_css=None):
    #     super().__init__(L_x, L_y, L_z, is_css=is_css)
            

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

        ## delta specifies the positions of the qubits involved in the stabilizer
        ## relative to the stabilizer position
        if self.stabilizer_type(location) == "vertex":
            ## X type
            delta = [
                (1, 0),
                (4, 1),
                (5, 2),
                (5, 4),
                (0, 5),
                (2, 5)
            ]
        else:
            ## Z type
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
        Hx = self.Hx
        Hz = self.Hz
        bposd_code = css_code(Hx, Hz)
        Lx = bposd_code.lx

        logicals = []
        for i in range(Lx.shape[0]):
            operator = dict()
            indices = np.argwhere(Lx[i,:] == 1)[:,1]
            for ind in indices:
                coord = self.get_qubit_coordinates()[ind]
                operator[coord] = "X"
            logicals.append(operator)
        
        return logicals

    def get_logicals_z(self):
        Hx = self.Hx
        Hz = self.Hz
        bposd_code = css_code(Hx, Hz)
        Lz = bposd_code.lz

        logicals = []
        for i in range(Lz.shape[0]):
            operator = dict()
            indices = np.argwhere(Lz[i,:] == 1)[:,1]
            for ind in indices:
                coord = self.get_qubit_coordinates()[ind]
                operator[coord] = "Z"
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


if __name__ == "__main__":
    gui = GUI()
    gui.add_code(TileCodes, "Tile Code")
    gui.run(port=5000)