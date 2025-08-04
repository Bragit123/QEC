from . import tile_codes as tc
from .error_models import GaussPauliErrorModel
from .decoders import BeliefPropagationLSDDecoder, GaussBeliefPropagationLSDDecoder

import itertools
import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns

from typing import List, Optional, Union, Type
from tqdm import tqdm # Progress bar

from panqec.codes import StabilizerCode, Toric2DCode
from panqec.error_models import PauliErrorModel
from panqec.decoders import BeliefPropagationOSDDecoder
from panqec.simulation import DirectSimulation, BatchSimulation
from panqec.analysis import Analysis

class SingleSimulation:
    """
    Class for running a single simulation of quantum error correction.

    Attributes
    ----------
    Code : StabilizerCode
        The QEC code (class, not object) to use in the simulation.
    error_model : str
        The name of the error model to use. "pauli" for PauliErrorModel
        and "gauss" for GaussPauliErrorModel.
    decoder : str
        The name of the decoder to use. "bposd" for OSD- and "bplsd" for LSD-
        belief propagation. Will use regular or Gaussian depending on the specified error model.
    r_xyz : list of float
        List containing the r_x, r_y and r_z values to give as input to the
        error model. These are the distribution of how often the differen type of errors will
        occur, and must therefore sum to one.
    p_vals : ndarray
        Array of error rates to simulate for.
    L_vals : list of int
        List of code sizes to consider.
    n_trials : int
        Number of Monte Carlo simulations to run the simulation for.
    output_dir : str
        Path to directory for storing JSON files containing simulation outputs.
    plot_dir : str
        Path to directory for storing the produced plots.
    existing_json_handling : str
        Decides how to handle existing JSON files from the same simulations. Takes one of three
        values:
        - "overwrite": Overwrite existing JSON files and run simulation anew.
        - "keep": Keep the existing JSON files, and use data from these to produce plots instead
        of running new simulations.
        - None: Ask for each file whether they should be overwritten.
    """
    def __init__(
            self,
            Code: Type[StabilizerCode],
            error_model: str,
            decoder: str,
            r_xyz: List[float],
            p_vals: np.ndarray,
            L_vals: List[int],
            n_trials: int,
            output_dir: str,
            existing_json_handling: str = "ask"
    ):
        """
        Constructor for the SingleSimulation class.

        Parameters
        ----------
        Code : StabilizerCode
            The QEC code (class, not object) to use in the simulation.
        error_model : str
            The name of the error model to use. "pauli" for PauliErrorModel
            and "gauss" for GaussPauliErrorModel.
        decoder : str
            The name of the decoder to use. "bposd" for OSD- and "bplsd" for LSD-
            belief propagation. Will use regular or Gaussian depending on the specified error model.
        r_xyz : list of float
            List containing the r_x, r_y and r_z values to give as input to the
            error model. These are the distribution of how often the differen type of errors will
            occur, and must therefore sum to one.
        p_vals : ndarray
            Array of error rates to simulate for.
        L_vals : list of int
            List of code sizes to consider.
        n_trials : int
            Number of Monte Carlo simulations to run the simulation for.
        output_dir : str
            Path to directory for storing JSON files containing simulation outputs.
        plot_dir : str
            Path to directory for storing the produced plots.
        existing_json_handling : str
            Decides how to handle existing JSON files from the same simulations. Takes one of three
            values:
            - "overwrite": Overwrite existing JSON files and run simulation anew.
            - "keep": Keep the existing JSON files, and use data from these to produce plots instead
            of running new simulations.
            - "ask": Ask for each file whether they should be overwritten.
        """
        self.Code = Code
        self.error_model = error_model
        self.decoder = decoder
        self.r_xyz = r_xyz
        self.p_vals = p_vals
        self.L_vals = L_vals
        self.n_trials = n_trials
        self.output_dir = output_dir
        self.existing_json_handling = existing_json_handling
    
    def run_simulation(self) -> str:
        """
        Runs a single simulation, and saves the results into a JSON file.

        Returns
        -------
        str
            Path to the JSON file where the output from the simulation is stored.
        """
        ## Extract and check validity of code
        Code = self.Code
        try:
            assert issubclass(Code, StabilizerCode)
        except:
            raise TypeError(f"Code must be a subclass of PanQEC's StabilizerCode. Got {Code}")

        ## Extract and check validity of error model and decoder
        error_model_name = self.error_model
        decoder_name = self.decoder
        
        if error_model_name == "pauli":
            Error_Model = PauliErrorModel
            if decoder_name == "bposd":
                Decoder = BeliefPropagationOSDDecoder
            elif decoder_name == "bplsd":
                Decoder = BeliefPropagationLSDDecoder
            else:
                raise ValueError(f"sim_input['decoder'] must be either 'bposd' or 'bplsd'. Got '{decoder_name}'.")
        elif error_model_name == "gauss":
            Error_Model = GaussPauliErrorModel
            if decoder_name == "bposd":
                raise NotImplementedError("Gaussian BPOSD not implemented.")
            elif decoder_name == "bplsd":
                Decoder = GaussBeliefPropagationLSDDecoder
            else:
                raise ValueError(f"sim_input['decoder'] must be either 'bposd' or 'bplsd'. Got '{decoder_name}'.")
        else:
            raise ValueError(f"sim_input['error_model'] must be either 'pauli' or 'gauss'. Got '{error_model_name}'.")

        ## Extract input parameters
        r_x, r_y, r_z = self.r_xyz
        p_vals = self.p_vals
        L_vals = self.L_vals
        n_trials = self.n_trials

        ## Set remaining parameters
        error_model = Error_Model(r_x, r_y, r_z)

        sim_name = self._get_sim_name()
        output_path = self.output_dir + sim_name + ".json"

        json_handling = self.existing_json_handling.lower()
        if os.path.exists(output_path):
            if json_handling == "overwrite":
                delete_existing_json = True
            elif json_handling == "keep":
                delete_existing_json = False
            elif json_handling == "ask":
                user_input = input(f"FILE ALREADY EXISTS: {output_path} | Would you like to delete it and run a new simulation? (y/n): ")
                delete_existing_json = True if user_input.lower() == "y" else False
            else:
                raise ValueError(f"Not recognized: existing_json_handling must be one of 'overwrite', 'keep' or 'ask'. Got {repr(json_handling)}.")
            
            if delete_existing_json:
                os.remove(output_path)

        ## Run simulation
        batch_sim = BatchSimulation(output_path)

        for L in L_vals:
            code = Code(L)
            for p in p_vals:
                p = float(p)
                decoder = Decoder(code, error_model, p)
                
                dir_sim = DirectSimulation(code, error_model, decoder, p)
                batch_sim.append(dir_sim)
        
        batch_sim.run(n_trials, progress=tqdm)

        return output_path
    

    def _get_sim_name(self) -> str:
        """
        Returns a name that is used for the JSON files containing the simulation output.

        Returns
        -------
        str
            Name of the simulation, containing information about all the different input
            parameters. This is used to name the JSON files produced from the simulation.
            
            Example name:
            toric__pauli__bplsd__r_0.5_0.0_0.5__L_8_12_16__p_0.001_0.5_20__t_500.json
            
            This name tells us that the simulation is run on the Toric2DCode, using the
            PauliErrorModel and BeliefPropagationLSDDecoder with r_x=0.5, r_y=0.0,
            r_z=0.5; going through the L values 8, 12 and 16; for 20 error rates p from
            0.001 to 0.5; and running 500 Monte Carlo simulations.
        """
        ## Get code name
        code = self.Code.__name__
        tc_len = len("TileCode")
        if code == "Toric2DCode":
            code = "toric"
        elif code[:tc_len] == "TileCode":
            code = code.replace("_", "") # Remove underscores for shorter name
            code = "tc" + code[tc_len:] # Shorten TileCode to tc. Now: TileCode_B3_W6 -> tcB3W6
        
        ## Get error model and decoder names
        error_model = self.error_model
        decoder = self.decoder
        
        ## Get text for remaining parameters
        r_x, r_y, r_z = self.r_xyz
        error_text = f"e_{r_x:.2}_{r_y:.2}_{r_z:.2}"
        
        L_text = f"L"
        for L in self.L_vals:
            L_text = L_text + "_" + str(L)

        p_vals = self.p_vals
        p_min = np.min(p_vals)
        p_max = np.max(p_vals)
        n_p = len(p_vals)
        p_text = f"r_{p_min:.2}_{p_max:.2}_{n_p}"
        
        trials_text = f"t_{self.n_trials}"

        ## Combine parameter strings into one string.
        sim_name = f"{code}__{error_model}__{decoder}__{error_text}__{L_text}__{p_text}__{trials_text}"

        return sim_name

class Simulation:
    """
    Class for simulating quantum error correction and plotting results.

    Attributes
    ----------
    simulation_list : list of SingleSimulation
        List of SingleSimulation objects for the different simulations.
    plot_dir : str
        Path to directory where produced plots should be saved.
    """
    def __init__(
            self,
            Codes: Union[Type[StabilizerCode], List[Type[StabilizerCode]]],
            error_models: Union[str, List[str]],
            decoders: Union[str, List[str]],
            r_xyz: List[float],
            p_vals: np.ndarray,
            L_vals: List[int],
            n_trials: int,
            output_dir: str,
            plot_dir: str,
            existing_json_handling: str = "ask"
    ):
        """
        Constructor for the Simulation class.
        Parameters
        ----------
        Codes : StabilizerCode or list of StabilizerCode
            List of the QEC codes (classes, not objects) to use in the simulation.
        error_models : str or list of str
            The name of the error models to use. "pauli" for PauliErrorModel
            and "gauss" for GaussPauliErrorModel.
        decoders : str or list of str
            The name of the decoders to use. "bposd" for OSD- and "bplsd" for LSD-
            belief propagation. Will use regular or Gaussian depending on the specified error model.
        r_xyz : list of float
            List containing the r_x, r_y and r_z values to give as input to the
            error model. These are the distribution of how often the differen type of errors will
            occur, and must therefore sum to one.
        p_vals : ndarray
            Array of error rates to simulate for.
        L_vals : list of int
            List of code sizes to consider.
        n_trials : int
            Number of Monte Carlo simulations to run the simulation for.
        output_dir : str
            Path to directory for storing JSON files containing simulation outputs.
        plot_dir : str
            Path to directory for storing the produced plots.
        existing_json_handling : str
            Decides how to handle existing JSON files from the same simulations. Takes one of three
            values:
            - "overwrite": Overwrite existing JSON files and run simulation anew.
            - "keep": Keep the existing JSON files, and use data from these to produce plots instead
            of running new simulations.
            - "ask": Ask for each file whether they should be overwritten.
        """
        if isinstance(Codes, type):
            Codes = [Codes]
        if isinstance(error_models, str):
            error_models = [error_models]
        if isinstance(decoders, str):
            decoders = [decoders]
        if isinstance(L_vals, int):
            L_vals = [L_vals]

        sim_input_dict = {
            "Code": Codes,
            "error_model": error_models,
            "decoder": decoders,
            "r_xyz": r_xyz,
            "p_vals": p_vals,
            "L_vals": L_vals,
            "n_trials": n_trials
        }
        sim_input_list = self._split_inputs(sim_input_dict)

        self.simulation_list = []
        for sim_input in sim_input_list:
            single_sim = SingleSimulation(
                **sim_input,
                output_dir=output_dir,
                existing_json_handling=existing_json_handling
            )
            self.simulation_list.append(single_sim)
        
        self.plot_dir = plot_dir

    
    def run_simulations(self) -> List[str]:
        """
        Runs simulations as defined by the SingleSimulation objects in simulation_list,
        and stores the output from each simulation into a separate JSON file.

        Returns
        -------
        List[str]
            List of paths to the JSON files where the output from each simulation is stored.
        """
        n_sims = len(self.simulation_list)
        json_paths = []
        for i, single_sim in enumerate(self.simulation_list):
            print(f"Running simulation {i+1}/{n_sims}:")
            json_path = single_sim.run_simulation()
            json_paths.append(json_path)
        
        self.json_paths = json_paths
        return json_paths

    
    def plot_results(
            self,
            filename: str,
            style: Optional[str] = None,
            hue: Optional[str] = None,
            col: Optional[str] = None
    ) -> str:
        """
        Analyze output from a simulation, and create the threshold plots. The resulting
        plot is saved in the plot directory.

        Parameters
        ----------
        filename : str
            Name of the pdf file where the produced plot is saved.
        style : str
            Column label of the data that should decide the style of the lines in
            the plot. If None: the plot is not divided into styles.
            See the Seaborn documentation for more information.
        hue : str
            Column label of the data that should decide the hue (color) of the lines
            in the plot. If None: the plot is not divided into colors.
            See the Seaborn documentation for more information.
        col : str
            Column label of the data that should decide the columns of the subplots
            in the plot. If None: the plot is not divided into subplots.
            See the Seaborn documentation for more information.
        
        Returns
        -------
        str
            The path to the saved plot.
        """
        data = self._extract_data()

        if not filename.endswith(".pdf"):
            filename = filename + ".pdf"

        plot_path = self.plot_dir + filename # Path for saving the threshold plot.

        sns.set_theme()

        ## Plot data
        g = sns.relplot(data, kind="line", x="error_rate", y="p_est", hue=hue, style=style, col=col)

        cols = data[col].unique()

        for i in range(len(cols)):
            ## Plot pseudo threshold
            ax = g.axes.flat[i]
            filtered_data = data.loc[data[col] == cols[i]]
            k = filtered_data["k"].iloc[0]
            p_vals = filtered_data["error_rate"].unique()
            ps_th = 1.0 - (1.0 - p_vals)**k # Pseudo threshold
            ax.plot(p_vals, ps_th, color="gray", linestyle="dashed")

        suptitle = self.get_suptitle()
        g.figure.suptitle(suptitle)
        g.set(
            xlabel="Physical error rate $p$",
            ylabel="Logical error rate $p_L$"
        )
        g.figure.tight_layout()


        g.savefig(plot_path, bbox_inches="tight")

        return plot_path
    

    def _split_inputs(self, sim_input_dict: dict) -> List[dict]:
        """
        Takes a dictionary containing simulation inputs for multiple simulations, and returns
        a list of dictionaries containing the simulation inputs for a single simulation.

        If one or more of the values in sim_input_dict are lists (not including the code sizes
        "L_vals" or the error distribution "r_xyz", which are lists also for single
        simulations), this function considers all possible combinations of them and creates
        input dictionaries for each simulation.

        For instance: If the value of "Code" is [Code1, Code2, Code3] and the value of
        "error_model" is ["pauli", "gauss"], but all other entries are as usual for singular
        simulations, the function will return a list of six dictionaries, where "Code" and
        "error_model" takes the following combinations:
        
        - Code1, "pauli"
        - Code1, "gauss"
        - Code2, "pauli"
        - Code2, "gauss"
        - Code3, "pauli"
        - Code3, "gauss"

        and the remaining parameters are equal for all six dictionaries.

        Parameters
        ----------
        sim_input : dict
            Dictionary of simulation parameters for one or multiple simulations.
            It must contain the following keys: Code, error_model, decoder, r_xyz,
            p_vals, L_vals, n_trials. See the constructor for more information
            about their types.
        
        Returns
        -------
        List[dict]
            List of dictionaries, each containing the input parameters for a
            single simulation.
        """
        varying_keys = []
        varying_values = []
        fixed_keys = []
        fixed_values = []
        for k, v in sim_input_dict.items():
            if isinstance(v, list) and not (k in ["L_vals", "r_xyz"]):
                # Check if one of the inputs except L_vals and r_xyz is a list
                varying_keys.append(k)
                varying_values.append(v)
            else:
                fixed_keys.append(k)
                fixed_values.append(v)
        
        sim_input_list = []
        if len(varying_keys) == 0:
            # No varying inputs, so no splitting needed.
            sim_input_list.append(sim_input_dict)
        else:
            # Get a list of all combinations of the varying parameters
            varying_product = list(itertools.product(*varying_values))
            for k_values in varying_product:
                sim_input = dict()
                for k, v in zip(varying_keys, k_values):
                    sim_input[k] = v
                for k, v in zip(fixed_keys, fixed_values):
                    sim_input[k] = v
                sim_input_list.append(sim_input)
        
        return sim_input_list
    


    def _extract_data(self) -> pd.DataFrame:
        """
        Goes through the JSON files in json_paths (see attributes), and stores the
        data in a Pandas DataFrame.

        Returns
        -------
        DataFrame
            Extracted data from the Simulation output JSON files.
        """
        json_paths = self.json_paths
        results_list = []
        for json_path in json_paths:
            analysis = Analysis(json_path)
            analysis.calculate_sector_thresholds() # Necessary for p_est_X and p_est_Z to be available
            results_i = analysis.get_results()
            results_list.append(results_i)
        
        results = pd.concat(results_list, ignore_index=True)
        L_x = results["code_params"].apply(lambda d: d.get("L_x"))
        L_y = results["code_params"].apply(lambda d: d.get("L_y"))
        if np.all(L_x == L_y):
            results["L"] = L_x
        else:
            raise NotImplementedError("Only the case when L_x = L_y is implemented.")

        return results

    def get_suptitle(self) -> str:
        simulation_list = self.simulation_list
        n_sims = len(simulation_list)

        param_dict = {
            "Code": [],
            "error_model": [],
            "decoder": [],
            "r_xyz": [],
            "n_p": [],
            "n_trials": []
        }
        for i in range(n_sims):
            single_sim = simulation_list[i]
            param_dict["Code"].append(single_sim.Code)
            param_dict["error_model"].append(single_sim.error_model)
            param_dict["decoder"].append(single_sim.decoder)
            param_dict["r_xyz"].append(single_sim.r_xyz)
            param_dict["n_p"].append(len(single_sim.p_vals))
            param_dict["n_trials"].append(single_sim.n_trials)
        L_vals = single_sim.L_vals # L_vals are always equal for all single-simulations, so only need one.

        identical_param = {}
        if isinstance(L_vals, int):
            identical_param["L"] = L_vals
        else:
            if len(L_vals) == 1:
                identical_param["L"] = L_vals[0]
            
        for key, value in param_dict.items():
            print(key, value)
            if key == "r_xyz":
                # Check if all lists are the same (cannot use set()) on nested list, so handled separately
                same = [value[i]==value[i+1] for i in range(len(value)-1)]
                if all(same):
                    identical_param[key] = value[0]
            elif len(set(value)) == 1:
                # Check if all values are the same
                if key == "Code":
                    identical_param[key] = value[0].__name__
                else:
                    identical_param[key] = value[0]
        
        suptitle = "QEC Simulation"
        for key, value in identical_param.items():
            suptitle += f" | {key}={value}"
        suptitle += " | (Pseudo threshold in gray)"
        
        return suptitle
    

if __name__ == "__main__":
    ## Set simulation parameters
    Codes = [Toric2DCode, tc.TileCode_B3_W6]
    error_models = [
        "pauli",
        "gauss"
    ]
    decoders = ["bplsd"]
    p_vals = np.linspace(0.001, 0.5, 5)
    sim_inputs = {
        "Codes": Codes,
        "error_models": error_models,
        "decoders": decoders,
        "r_xyz": [0.5, 0.0, 0.5],
        "p_vals": p_vals,
        "L_vals": [8, 16],
        "n_trials": 10
    }

    # Run and analyze simulation
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", help="Run new simulations, overwriting any existing JSON files without asking.", action="store_true")
    parser.add_argument("-k", "--keep", help="Don't run new simulations for already existing JSON files.", action="store_true")
    args = parser.parse_args()

    if args.overwrite and not args.keep:
        existing_json_handling = "overwrite"
    elif args.keep and not args.overwrite:
        existing_json_handling = "keep"
    else:
        if args.keep and args.overwrite:
            print("WARNING: Cannot both overwrite and keep JSON files. Reverting back to asking at each JSON file.")
        existing_json_handling = "ask"

    sim = Simulation(
        **sim_inputs,
        output_dir="./",
        plot_dir="./",
        existing_json_handling=existing_json_handling
    )
    
    json_paths = sim.run_simulations()
    
    filename = "gauss_test"
    plot_path = sim.plot_results(filename, style="L", hue="error_model", col="code")

    print(f"Results plotted in {plot_path}")