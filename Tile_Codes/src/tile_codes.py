import numpy as np
import json
from abc import abstractmethod
from panqec.codes import StabilizerCode
from bposd.css import css_code
from pathlib import Path

type Coordinate = tuple[int]

##### Abstract class #####
class TileCode(StabilizerCode):
    """
    Abstract class for Tile Codes, a way of constructing quantum codes presented by
    Steffan, Choe, Breuckmann, Pereira and Eberhardt in the 2025 paper titled:
    "Tile Codes: High-Efficiency Quantum Codes on a Lattice with Boundary"

    https://arxiv.org/abs/2504.09171

    ## Usage
    The TileCode class is abstract. In order to make a valid subclass you must include the
    following properties:
        - B = Dimension of the stabilizer tiles.
        - delta_X = List of relative coordinates for the X-tile stabilizer.
    """
    dimension = 2

    @property
    @abstractmethod
    def B(self) -> int:
        """ Dimension of the stabilizer tiles """

    @property
    @abstractmethod
    def delta_X(self) -> list[Coordinate]:
        """
        List of relative coordinates for the X-tile stabilizer.
        
        Each coordinate specifies the position of a qubit involved in the
        stabilizer, relative to the vertex where the stabilizer is defined.
        """
    
    @property
    def delta_Z(self) -> list[Coordinate]:
        """
        List of relative coordinates for the Z-tile stabilizer.
        
        NOTE: The Z stabilizer is defined on the face, not the vertex. Thus
        the delta gets an extra minus one in both the x- and y-coordinates
        compared to the X stabilizer.
        """
        B = self.B
        delta_X = self.delta_X

        delta_Z = []
        for i in range(len(delta_X)):
            x = delta_X[i][0]
            y = delta_X[i][1]
            delta_Z.append((2*(B-1)-x, 2*(B-1)-y))
        
        return delta_Z

    @property
    def json_file(self) -> str:
        """ Path to JSON file for GUI visualization. """
        src_path = Path(__file__).parent.absolute()
        gui_config_path = src_path / "gui_config_tile_codes.json"
        return gui_config_path

    @property
    def label(self) -> str:
        return f"Tile Code {self.size[0]}x{self.size[1]}"
    
    def get_qubit_coordinates(self) -> list[Coordinate]:
        """ Returns a list of the qubit coordinates. """
        B = self.B
        Lx, Ly = self.size
        coordinates = []
        
        # Horizontal qubits
        for x in range(2*(B-1)+1, 2*(Lx+B-1), 2):
            for y in range(2*(B-1), 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        # Vertical qubits
        for x in range(2*(B-1), 2*(Lx+B-1), 2):
            for y in range(2*(B-1)+1, 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        return coordinates
    
    def qubit_axis(self, location: Coordinate) -> str:
        """
        Returns the axis of alignment for a qubit.
        "x" for horizontal qubits, and "y" for vertical qubits.
        """
        x, y = location

        if (y % 2 == 0) and (not x % 2 == 0):
            axis = "x"
        
        elif (x % 2 == 0) and (not y % 2 == 0):
            axis = "y"
        
        else:
            raise ValueError("Location in qubit_axis() must be an edge.")

        return axis
    
    def get_stabilizer_coordinates(self) -> list[Coordinate]:
        """ Returns a list of the qubit coordinates. """
        coordinates = []
        Lx, Ly = self.size
        B = self.B
        
        # X errors (red/black dots)
        for x in range(2*(B-1), 2*Lx, 2):
            for y in range(0, 2*(Ly+B-1), 2):
                coordinates.append((x, y))

        # Z errors (blue/black dots, but placed on faces)
        for x in range(1, 2*(Lx+B-1), 2):
            for y in range(2*(B-1)+1, 2*Ly, 2):
                coordinates.append((x, y))
        
        return coordinates

    def stabilizer_type(self, location: Coordinate) -> str:
        """
        Returns the type of a stabilizer.
        "vertex" for X-tiles, and "face" for Z-tiles.
        """
        x, y = location
        if (x % 2 == 0) and (y % 2 == 0):
            return "vertex"
        elif (x % 2 == 1) and (y % 2 == 1):
            return "face"
        else:
            raise ValueError("stabilizer_type() must return either 'vertex' or 'face'")

    def get_stabilizer(self, location: Coordinate) -> dict[Coordinate, str]:
        """
        Returns a dictionary containing the Pauli operator as a string ("X" or "Z")
        for all qubits in the stabilizer.
        """
        if self.stabilizer_type(location) == "vertex":
            ## X type
            pauli = "X"
            delta = self.delta_X
        elif self.stabilizer_type(location) == "face":
            ## Z type
            pauli = "Z"
            delta = self.delta_Z
        else:
            raise ValueError("stabilizer_type() must return either 'vertex' or 'face'")
        
        x, y = location
        
        operator = dict()
        
        for d in delta:
            qubit_location = (x + d[0], y + d[1]) # Get qubit coordinate from relative coordinate in delta

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli # Add qubit to stabilizer
        
        return operator

    def get_logicals_x(self) -> list[dict[Coordinate, str]]:
        """ Returns a list of all the logical X operators of the code. """
        Hx = self.Hx
        Hz = self.Hz
        bposd_code = css_code(Hx, Hz)
        Lx = bposd_code.lx # Compute the logical X operator.

        logicals = []
        for i in range(Lx.shape[0]):
            ## Translate Hamming matrix indices to PanQEC coordinates for each operator
            operator = dict()
            indices = np.argwhere(Lx[i,:] == 1)[:,1]
            for ind in indices:
                coord = self.get_qubit_coordinates()[ind]
                operator[coord] = "X"
            logicals.append(operator)
        
        return logicals

    def get_logicals_z(self) -> list[dict[Coordinate, str]]:
        """ Returns a list of all the logical Z operators of the code. """
        Hx = self.Hx
        Hz = self.Hz
        bposd_code = css_code(Hx, Hz)
        Lz = bposd_code.lz # Compute the logical Z operator.

        logicals = []
        for i in range(Lz.shape[0]):
            ## Translate Hamming matrix indices to PanQEC coordinates for each operator
            operator = dict()
            indices = np.argwhere(Lz[i,:] == 1)[:,1]
            for ind in indices:
                coord = self.get_qubit_coordinates()[ind]
                operator[coord] = "Z"
            logicals.append(operator)
        
        return logicals
    
    def qubit_representation(self, location: Coordinate, rotated_picture: bool = False) -> dict:
        """
        Finds how qubits are represented for the GUI visualization.
        This method is mostly copied directly from PanQECs qubit_representation(),
        but so all subclasses of TileCode can use the same JSON entry.
        """
        json_file = self.json_file # JSON file containing visualization parameters.
        
        with open(json_file, 'r') as f:
            data = json.load(f)

        code_name = "TileCode" # Name of the code in the JSON file.

        picture = 'rotated' if rotated_picture else 'kitaev'

        representation = data[code_name]['qubits'][picture]

        representation['params']['axis'] = self.qubit_axis(location)
        representation['location'] = location

        for pauli in ['I', 'X', 'Y', 'Z']:
            color_name = representation['color'][pauli]
            representation['color'][pauli] = self.colormap[color_name]

        return representation
    
    def stabilizer_representation(self, location: Coordinate, rotated_picture=False) -> dict:
        """
        Finds how stabilizers are represented for the GUI visualization.
        This method is mostly copied directly from PanQECs stabilizer_representation(),
        but so all subclasses of TileCode can use the same JSON entry.
        """
        json_file = self.json_file # JSON file containing visualization parameters.
        stab_type = self.stabilizer_type(location)

        with open(json_file, 'r') as f:
            data = json.load(f)

        code_name = "TileCode" # Name of the code in the JSON file.

        picture = 'rotated' if rotated_picture else 'kitaev'

        representation = data[code_name]['stabilizers'][picture][stab_type]
        representation['type'] = stab_type
        representation['location'] = location

        for activation in ['activated', 'deactivated']:
            color_name = representation['color'][activation]
            representation['color'][activation] = self.colormap[color_name]

        if self.stabilizer_type(location) == "face":
            x, y = location
            location = [x-1, y-1]
            representation["location"] = location
        else:
            x, y = location
            location = [x, y, 1]
            representation["location"] = location
        return representation



##### Specific Tile Codes #####

class TileCode_Planar(TileCode):
    """
    The 2D Planar code in the language of Tile Codes.
    The tiles are 2x2, and give the "regular" X-vertex and Z-plaquette as
    familiar from the CSS toric or planar codes.
    """
    B = 2
    delta_X = [
        (2,1),
        (1,2),
        (3,2),
        (2,3)
    ]


class TileCode_B3_W6(TileCode):
    """
    Tile Code with 3x3 tiles of weight 6.
    Specifically, the code specified by the X- and Z-tiles in the first row of
    Table 1 in https://arxiv.org/abs/2504.09171
    (the code labeled by [[288,8,12]]).
    """
    B = 3
    delta_X = [
        (1,0),
        (4,1),
        (5,2),
        (5,4),
        (0,5),
        (2,5),
    ]

class TileCode_B3_W8(TileCode):
    """
    Tile Code with 3x3 tiles of weight 8.
    Specifically, the code specified by the X- and Z-tiles in the second row of
    Table 1 in https://arxiv.org/abs/2504.09171
    (the code labeled by [[288,8,14]]).
    """
    B = 3
    delta_X = [
        (1,0),
        (5,0),
        (0,1),
        (1,2),
        (2,3),
        (1,4),
        (0,5),
        (4,5)
    ]


class TileCode_B4_W8(TileCode):
    """
    Tile Code with 4x4 tiles of weight 8.
    Specifically, the code specified by the X- and Z-tiles in the third row of
    Table 1 in https://arxiv.org/abs/2504.09171
    (the code labeled by [[288,18,13]]).
    """
    B = 4
    delta_X = [
        (1,0),
        (7,0),
        (2,1),
        (0,3),
        (2,3),
        (5,4),
        (1,6),
        (6,7)
    ]
