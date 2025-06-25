from panqec.codes import StabilizerCode
from panqec.gui import GUI

class MyToric3DCode(StabilizerCode):
    dimension = 3

    @property
    def label(self):
        return 'My Toric {}x{}x{}'.format(*self.size)
    
    def get_qubit_coordinates(self):
        coordinates = []
        Lx, Ly, Lz = self.size

        # Qubits along e_x
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_y
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Qubits along e_z
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def get_stabilizer_coordinates(self):
        coordinates = []
        Lx, Ly, Lz = self.size

        # Vertices
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in xy plane
        for x in range(1, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(0, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in yz plane
        for x in range(0, 2*Lx, 2):
            for y in range(1, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        # Face in xz plane
        for x in range(1, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                for z in range(1, 2*Lz, 2):
                    coordinates.append((x, y, z))

        return coordinates

    def stabilizer_type(self, location):
        x, y, z = location
        if x % 2 == 0 and y % 2 == 0:
            return 'vertex'
        else:
            return 'face'

    def get_stabilizer(self, location):
        if self.stabilizer_type(location) == 'vertex':
            pauli = 'Z'
        else:
            pauli = 'X'

        x, y, z = location

        # delta specifies the positions of the qubits involved in the stabilizer
        # relative to the stabilizer position
        if self.stabilizer_type(location) == 'vertex':
            delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        else:
            # Face in xy-plane.
            if z % 2 == 0:
                delta = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
            # Face in yz-plane.
            elif (x % 2 == 0):
                delta = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
            # Face in zx-plane.
            elif (y % 2 == 0):
                delta = [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)]

        operator = dict()
        for d in delta:
            Lx, Ly, Lz = self.size
            qubit_location = ((x + d[0]) % (2*Lx), (y + d[1]) % (2*Ly), (z + d[2]) % (2*Lz))

            if self.is_qubit(qubit_location):
                operator[qubit_location] = pauli

        return operator

    def qubit_axis(self, location):
        x, y, z = location

        if (z % 2 == 0) and (x % 2 == 1) and (y % 2 == 0):
            axis = 'x'
        elif (z % 2 == 0) and (x % 2 == 0) and (y % 2 == 1):
            axis = 'y'
        else:
            axis = 'z'

        return axis

    def get_logicals_x(self):
        """The 3 logical X operators."""

        Lx, Ly, Lz = self.size
        logicals = []

        # X operators along x edges in x direction.
        operator = dict()
        for x in range(1, 2*Lx, 2):
            operator[(x, 0, 0)] = 'X'
        logicals.append(operator)

        # X operators along y edges in y direction.
        operator = dict()
        for y in range(1, 2*Ly, 2):
            operator[(0, y, 0)] = 'X'
        logicals.append(operator)

        # X operators along z edges in z direction
        operator = dict()
        for z in range(1, 2*Lz, 2):
            operator[(0, 0, z)] = 'X'
        logicals.append(operator)

        return logicals

    def get_logicals_z(self):
        """Get the 3 logical Z operators."""
        Lx, Ly, Lz = self.size
        logicals = []

        # Z operators on x edges forming surface normal to x (yz plane).
        operator = dict()
        for y in range(0, 2*Ly, 2):
            for z in range(0, 2*Lz, 2):
                operator[(1, y, z)] = 'Z'
        logicals.append(operator)

        # Z operators on y edges forming surface normal to y (zx plane).
        operator = dict()
        for z in range(0, 2*Lz, 2):
            for x in range(0, 2*Lx, 2):
                operator[(x, 1, z)] = 'Z'
        logicals.append(operator)

        # Z operators on z edges forming surface normal to z (xy plane).
        operator = dict()
        for x in range(0, 2*Lx, 2):
            for y in range(0, 2*Ly, 2):
                operator[(x, y, 1)] = 'Z'
        logicals.append(operator)

        return logicals
        
    def get_deformation(self, location, deformation_name, deformation_axis='y'):
        if deformation_name == 'XZZX':
            undeformed_dict = {'X': 'X', 'Y': 'Y', 'Z': 'Z'}
            deformed_dict = {'X': 'Z', 'Y': 'Y', 'Z': 'X'}

            if self.qubit_axis(location) == deformation_axis:
                deformation = deformed_dict
            else:
                deformation = undeformed_dict

        else:
            raise ValueError(f"The deformation {deformation_name}"
                            "does not exist")

        return deformation

    def stabilizer_representation(self, location, rotated_picture=False):
        representation = super().stabilizer_representation(location, rotated_picture, json_file='toric3d.json')

        x, y, z = location
        if not rotated_picture and self.stabilizer_type(location) == 'face':
            if z % 2 == 0:  # xy plane
                representation['params']['normal'] = [0, 0, 1]
            elif x % 2 == 0:  # yz plane
                representation['params']['normal'] = [1, 0, 0]
            else:  # xz plane
                representation['params']['normal'] = [0, 1, 0]

        if rotated_picture and self.stabilizer_type(location) == 'face':
            if z % 2 == 0:
                representation['params']['normal'] = [0, 0, 1]
            elif x % 2 == 0:
                representation['params']['normal'] = [1, 0, 0]
            else:
                representation['params']['normal'] = [0, 1, 0]

        return representation

    def qubit_representation(self, location, rotated_picture=False):
        representation = super().qubit_representation(location, rotated_picture, json_file="toric3d.json")

        return representation

gui = GUI()
gui.add_code(MyToric3DCode, 'My Toric 3D')
gui.run(port=5000)