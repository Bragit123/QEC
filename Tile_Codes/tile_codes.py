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
        
        # Qubits along x-axis
        for x in range(2*(B-1)+1, 2*(Lx+B-1)):
            for y in range():
                pass

        return coordinates
    
    def get_stabilizer_coordinates(self):
        pass

    def stabilizer_type(self, location):
        pass

    def get_stabilizer(self, location):
        pass

    def qubit_axis(self, location):
        pass

    def get_logicals_x(self):
        pass

    def get_logicals_z(self):
        pass


code = TileCode(10)

print(code.get_qubit_coordinates())
print(code.label)