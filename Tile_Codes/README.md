# Tile_Codes
My implementation of [Tile Codes](https://arxiv.org/abs/2504.09171) using [PanQEC](https://github.com/panqec/panqec/tree/main).

To see an example on how to run simulations, see *main.py*. If there is already a JSON file with results for the given parameters you will be asked whether you want to keep the existing JSON file, or run a new simulation overwriting the existing data. You can also add the -k (--keep) or -o (--overwrite) arguments when running the script to stop the code from asking at each simulation. More info about these is given by running `python3 main.py --help`.

To run the PanQEC GUI for Tile Codes, run the *run_gui.py* script.

The necessary code for running simulations is found in the *src* folder. It contains the following python scripts:
- *tile_codes.py*: Contains classes for using Tile Codes with PanQEC. The abstract class *TileCode* inherits from PanQEC's *StabilizerCode*, and contains what is common for all Tile Codes, independent of layout. It also contains a few instances of specific layouts.
- *error_models.py*: Contains the class *GaussPauliErrorModel*, which inherits from PanQEC's *PauliErrorModel*, but uses Gaussian sampling to draw generate errors (see the following paper on [analog QEC](https://arxiv.org/abs/1706.03011)).
- *decoders.py*: PanQEC includes a belief propagation decoder *BeliefPropagationOSDDecoder*, using [LDPC](https://github.com/quantumgizmos/ldpc)'s *BpOsdDecoder* class. In *decoders.py* I have implemented a similar version using LDPC's BpLsdDecoder, as this gave slightly better results. This implementation is contained in the class *BeliefPropagationLSDDecoder*. In addition I have implemented the class *GaussBeliefPropagationLSDDecoder*, which is similar to *BeliefPropagationLSDDecoder*, but takes into account the measurements from Gaussian error generation in *GaussPauliErrorModel*.
- *simulation.py*: Contains classes for running simulations, storing the results in JSON files and producing plots.

The *Simulation_Outputs* folder contains all the JSON files produced when running simulations.

The *Plots* folder contains plots created when analyzing simulation results.