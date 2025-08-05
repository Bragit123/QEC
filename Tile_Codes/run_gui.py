from panqec.gui import GUI
from src.tile_codes import TileCode_Planar, TileCode_B3_W6, TileCode_B3_W8, TileCode_B4_W8

gui = GUI()
gui.add_code(TileCode_Planar, "Tile Code: 'Planar'")
gui.add_code(TileCode_B3_W6, "Tile Code: B=3, W=6")
gui.add_code(TileCode_B3_W8, "Tile Code: B=3, W=8")
gui.add_code(TileCode_B4_W8, "Tile Code: B=4, W=8")
gui.run(port=5000)