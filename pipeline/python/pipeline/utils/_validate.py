import BioSimSpace as BSS
from BioSimSpace._SireWrappers import System as _System
from BioSimSpace._SireWrappers import Molecule as _Molecule
import os
import warnings
import networkx as nx
import logging

from typing import Union, Optional

import pipeline


class validate:
    """class of staticmethods to return validated input"""

    def __init__(self):
        pass

    @staticmethod
    def string(string: str) -> str:
        """validates the string

        Args:
            string (str): text

        Raises:
            TypeError: must be of type 'str'

        Returns:
            str: text
        """
        if not isinstance(string, str):
            raise TypeError(f"{string} / 'string' must be of type 'str'.")

        return string

    @staticmethod
    def dictionary(dictionary: dict) -> dict:
        """validates the dictionary

        Args:
            dictionary (dict): the dictionary

        Raises:
            TypeError: must be of type 'dict'

        Returns:
            dict: the dictionary
        """
        if not isinstance(dictionary, dict):
            raise TypeError(f"{dictionary} / 'dictionary' must be of type 'dict'.")

        return dictionary

    @staticmethod
    def nxgraph(nxgraph):
        """validates the nxgraph

        Args:
            nxgraph (networkx.Graph): the nxgraph

        Raises:
            TypeError: must be of type 'networkx.Graph'

        Returns:
            networkx.Graph: the nxgraph
        """
        if not isinstance(nxgraph, nx.Graph):
            raise TypeError(f"{nxgraph} / 'nxgraph' must be of type 'networkx.Graph'.")

        return nxgraph

    @staticmethod
    def is_list(a_list: list, make_list: bool = False) -> list:
        """validates the list

        Args:
            a_list (list): the list.
            make_list (boolean): if to make into a list if a single value.

        Raises:
            TypeError: must be of type 'list'

        Returns:
            a_list: the list
        """
        make_list = validate.boolean(make_list)

        if not isinstance(a_list, list):
            if make_list:
                a_list = [a_list]
            else:
                raise TypeError(f"{a_list} / 'a_list' must be of type 'list'.")

        return a_list

    @staticmethod
    def file_path(file_path: str) -> str:
        """validates the provided file path

        Args:
            file_path (str): path to file

        Raises:
            TypeError: must be of type 'str'
            ValueError: path must exist!

        Returns:
            str: file path
        """
        if not isinstance(file_path, str):
            raise TypeError("'file_path' must be of type 'str'.")

        if not os.path.exists(file_path):
            raise ValueError(f"{file_path} does not exist!")

        return file_path

    @staticmethod
    def folder_path(folder_path: str, create: bool = False) -> str:
        """validates the provided file path

        Args:
            folder_path (str): path to file

        Raises:
            TypeError: must be of type 'str'
            ValueError: path must exist!

        Returns:
            str: file path
        """
        if not isinstance(folder_path, str):
            raise TypeError("'folder_path' must be of type 'str'.")

        if create:
            if not os.path.exists(folder_path):
                logging.info(f"creating {folder_path} as it could not be found...")
                os.makedirs(folder_path)
            else:
                pass
        else:
            if not os.path.exists(folder_path):
                raise ValueError(f"{folder_path} does not exist!")

        return folder_path

    @staticmethod
    def engine(engine: str) -> str:
        """validates the provided engine

        Args:
            engine (str): engine to be used for the MD / FEP run

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be one of BSS.FreeEnergy.engines().

        Returns:
            str: engine.upper()
        """
        if not isinstance(engine, str):
            raise TypeError("'engine' must be of type 'str'.")

        if engine.upper() not in BSS.FreeEnergy.engines():
            raise ValueError(f"'engine' must be one of {BSS.FreeEnergy.engines()}.")

        return engine.upper()

    @staticmethod
    def engines(engines: Union[str, list] = None) -> list:
        """_summary_

        Args:
            engines (list or string, optional): engines to use. Defaults to None.

        Returns:
            list: a list of engines
        """

        # get engines for analysis
        if not engines:
            engines = BSS.FreeEnergy.engines()
        else:
            try:
                try:
                    engines = validate.is_list(engines)
                    val_engines = []
                    for engine in engines:
                        engine_val = validate.engine(engine)
                        val_engines.append(engine_val)
                    engines = val_engines
                # if single engine string, put into list
                except:
                    engines = validate.string(engines)
                    if engines.upper() == "ALL":
                        engines = BSS.FreeEnergy.engines()
                    elif "," in engines:
                        engines_list = engines.split(",")
                        val_engines = []
                        for engine in engines_list:
                            engine_val = validate.engine(engine)
                            val_engines.append(engine_val)
                        engines = val_engines
                    else:
                        engines = validate.engine(engines)
                        engines = [engines]
            except:
                logging.error("engine input not recognised. Will use all engines.")
                engines = BSS.FreeEnergy.engines()

        return engines

    @staticmethod
    def lig_ff(lig_ff: str) -> str:
        """validates the provided ligand forcefield

        Args:
            lig_ff (str): name of the ligand forcefield

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be either sage, parsely, gaff or gaff2.

        Returns:
            str: forcefield that can be used with BSS.Parameters.parameterise()
        """

        if not isinstance(lig_ff, str):
            raise TypeError("'lig_ff' must be of type 'str'.")
        else:
            lig_ff = lig_ff.lower()

        lig_ff_list = [
            "sage",
            "parsely",
            "gaff",
            "gaff2",
            "openff_unconstrained-2.2.1",
            "openff_unconstrained-2.0.0",
            "openff_unconstrained-1.3.0",
            "openff",
        ]
        if lig_ff not in lig_ff_list:
            raise ValueError(f"'lig_ff' must be one of {lig_ff_list}.")

        if lig_ff == "sage":
            lig_ff = "openff_unconstrained-2.2.1"
        elif lig_ff == "parsely":
            lig_ff = "openff_unconstrained-1.3.0"
        elif lig_ff == "openff":
            lig_ff = "openff_unconstrained-2.2.1"

        return lig_ff

    @staticmethod
    def prot_ff(prot_ff: str) -> str:
        """validates the provided protein forcefield

        Args:
            prot_ff (str): name of the protein forcefield

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be one of the BSS.Parameters.forcefields() protein forcefields.

        Returns:
            str: forcefield that can be used with BSS.Parameters.parameterise()
        """

        if not isinstance(prot_ff, str):
            raise TypeError("'prot_ff' must be of type 'str'.")

        prot_ff_list = ["ff03", "ff14SB", "ff99", "ff99SB", "ff99SBildn"]
        if prot_ff not in prot_ff_list:
            raise ValueError(f"'prot_ff' must be one of {prot_ff_list}.")

        return prot_ff

    @staticmethod
    def solvent_ff(solvent_ff: str) -> str:
        """validates the provided solvent forcefield / water model

        Args:
            solvent_ff (str): name of the water model

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be in BSS.Solvent.waterModels()

        Returns:
            str: water model that can be used with BSS.Solvent.solvate()
        """

        if not isinstance(solvent_ff, str):
            raise TypeError("'solvent_ff' must be of type 'str'.")

        if solvent_ff.lower() not in BSS.Solvent.waterModels():
            raise ValueError(
                f"'solvent_ff' must be one of {BSS.Solvent.waterModels()}."
            )

        return solvent_ff.lower()

    @staticmethod
    def integer(integer: int) -> int:
        """validates the provided integer so that it is an integer

        Args:
            integer (str or int or float): an integer / number to be converted into an integer

        Raises:
            TypeError: must be of type 'str' or 'int' or 'float'
            ValueError: 'str' could not be converted into an integer

        Returns:
            int: integer in integer format
        """

        if not isinstance(integer, int):
            if not isinstance(integer, float):
                if not isinstance(integer, str):
                    raise TypeError(
                        f"{integer} must be of type 'int', 'float' or 'str'"
                    )
                else:
                    try:
                        if "." in integer:
                            integer = float(integer)
                        else:
                            integer = int(integer)
                    except:
                        raise ValueError(
                            f"{integer} could not be converted into an integer"
                        )

        if isinstance(integer, float):
            logging.info(f"{integer} will be turned into {int(integer)}")
            integer = round(integer)

        return integer

    @staticmethod
    def is_float(value: float) -> float:
        """validates the provided value so that it is a float

        Args:
            number (str or int or float): an integer / number to be converted into a float

        Raises:
            TypeError: must be of type 'str' or 'int' or 'float'
            ValueError: 'str' could not be converted into a float

        Returns:
            float: float in float format
        """

        if not isinstance(value, float):
            if not isinstance(value, int):
                if not isinstance(value, str):
                    raise TypeError(f"{value} must be of type 'int', 'float' or 'str'")

        try:
            value = float(value)
        except:
            raise ValueError(f"{value} could not be converted into an float")

        return value

    @staticmethod
    def time_unit(time_unit: BSS.Types.Time) -> BSS.Types.Time:
        """validates the sampling unit

        Args:
            time_unit (str or BSS.Types.Time): unit of sampling

        Raises:
            TypeError: must be of type 'str' or 'BSS.Types.Time'
            ValueError: if not of BSS.Types.Time, must be 'ns' or 'ps'.

        Returns:
            BSS.Types.Time: in BSS Time
        """

        if not isinstance(time_unit, str):
            if not isinstance(time_unit, BSS.Types.Time):
                raise TypeError(
                    "'time_unit' must be of type 'str' or 'BSS.Types.Time'."
                )

        else:
            time_unit = time_unit.strip().split(" ")[-1].lower()
            time_unit_list = ["ns", "ps", "fs"]
            if time_unit not in time_unit_list:
                raise ValueError(f"'time_unit' must be one of {time_unit_list}.")

            if time_unit == "ns":
                time_unit = BSS.Units.Time.nanosecond
            elif time_unit == "ps":
                time_unit = BSS.Units.Time.picosecond
            elif time_unit == "fs":
                time_unit = BSS.Units.Time.femtosecond

        return time_unit

    @staticmethod
    def box_edges_unit(box_edges_unit: BSS.Types.Length) -> BSS.Types.Length:
        """validates the box edges unit

        Args:
            box_edges_unit (str or BSS.Types.Length): unit of length

        Raises:
            TypeError: must be of type 'str' or 'BSS.Types.Length'
            ValueError: if not of BSS.Types.Length, must be 'nm' or 'angstrom'.

        Returns:
            BSS.Types.Length: in BSS Length
        """

        if not isinstance(box_edges_unit, str):
            if not isinstance(box_edges_unit, BSS.Types.Length):
                raise TypeError(
                    "'box_edges_unit' must be of type 'str' or 'BSS.Types.Length'."
                )

        else:
            box_edges_unit = box_edges_unit.strip().split(" ")[-1].lower()
            box_edges_unit_list = ["angstrom", "a", "nanometer", "nm"]
            if box_edges_unit not in box_edges_unit_list:
                raise ValueError(
                    f"'box_edges_unit' must be one of {box_edges_unit_list}."
                )

            if box_edges_unit == "angstrom" or box_edges_unit == "a":
                box_edges_unit = BSS.Units.Length.angstrom
            elif box_edges_unit == "nm" or box_edges_unit == "nanometer":
                box_edges_unit = BSS.Units.Length.nanometer

        return box_edges_unit

    @staticmethod
    def temperature_unit(
        temperature_unit: BSS.Types.Temperature,
    ) -> BSS.Types.Temperature:
        """validates the temperature unit

        Args:
            temperature_unit (str or BSS.Types.Temperature): unit of temperature

        Raises:
            TypeError: must be of type 'str' or 'BSS.Types.Temperature'
            ValueError: if not of BSS.Types.Temperature
        Returns:
            BSS.Types.Temperature: in BSS Temperature
        """

        if not isinstance(temperature_unit, str):
            if not isinstance(temperature_unit, BSS.Types.Temperature):
                raise TypeError(
                    "'temperature_unit' must be of type 'str' or 'BSS.Types.Temperature'."
                )

        else:
            temperature_unit = temperature_unit.strip().split(" ")[-1].lower()
            temperature_unit_list = ["kelvin", "k", "celsius", "c"]
            if temperature_unit not in temperature_unit_list:
                raise ValueError(
                    f"'temperature_unit' must be one of {temperature_unit_list}."
                )

            if temperature_unit == "k" or temperature_unit == "kelvin":
                temperature_unit = BSS.Units.Temperature.kelvin
            elif temperature_unit == "c" or temperature_unit == "celsius":
                temperature_unit = BSS.Units.Temperature.celsius

        return temperature_unit

    @staticmethod
    def pressure_unit(pressure_unit: BSS.Types.Pressure) -> BSS.Types.Pressure:
        """validates the pressure unit

        Args:
            pressure_unit (str or BSS.Types.Pressure): unit of pressure

        Raises:
            TypeError: must be of type 'str' or 'BSS.Types.Pressure'
            ValueError: if not of BSS.Types.Pressure
        Returns:
            BSS.Types.Pressure: in BSS Pressure
        """

        if not isinstance(pressure_unit, str):
            if not isinstance(pressure_unit, BSS.Types.Pressure):
                raise TypeError(
                    "'pressure_unit' must be of type 'str' or 'BSS.Types.Pressure'."
                )

        else:
            pressure_unit = pressure_unit.strip().split(" ")[-1].lower()
            pressure_unit_list = ["atm", "atmosphere", "bar"]
            if pressure_unit not in pressure_unit_list:
                raise ValueError(
                    f"'pressure_unit' must be one of {pressure_unit_list}."
                )

            if pressure_unit == "atm" or pressure_unit == "atmosphere":
                pressure_unit = BSS.Units.Pressure.atm
            elif pressure_unit == "bar":
                pressure_unit = BSS.Units.Pressure.bar

        return pressure_unit

    @staticmethod
    def box_type(box_type: str) -> str:
        """validates the box type

        Args:
            box_type (str): type of box to be used for solvation.

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be one of the box types accepted by minimum_solvation()

        Returns:
            str: box type
        """

        if not isinstance(box_type, str):
            raise TypeError("'box_type' must be of type 'str'.")

        box_type_list = ["cubic", "truncatedOctahedron", "octahedral"]
        if box_type not in box_type_list:
            raise ValueError(f"'box_type' must be one of {box_type_list}.")

        return box_type

    @staticmethod
    def boolean(boolean: bool) -> bool:
        """validates boolean input

        Args:
            boolean (str or bool): boolean

        Raises:
            TypeError: must be of type 'str' or 'bool'
            ValueError: if 'str' , must be 'True'/'1' or 'False'/'0'

        Returns:
            bool: boolean
        """

        if boolean != 1 and boolean != 0:
            if not isinstance(boolean, bool):
                if not isinstance(boolean, str):
                    raise TypeError(
                        f"{boolean} must be of type 'str' (True or False) or 'bool'."
                    )
                else:
                    boolean_list = ["TRUE", "FALSE", "1", "0"]
                    if boolean.upper() not in boolean_list:
                        raise ValueError(f"{boolean} must be one of {boolean_list}.")
        else:
            boolean = str(boolean)

        boolean = boolean.upper()

        if boolean == "TRUE" or boolean == "1":
            boolean = True
        elif boolean == "FALSE" or boolean == "0":
            boolean = False

        return boolean

    @staticmethod
    def trajectories(trajectories: str) -> str:
        """validate input for trajectories being saved or not

        Args:
            trajectories (str): whether to save the trajectory or not

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be in trajectories_list so can be recognised by future scripts

        Returns:
           str: how many of the trajectories will be saved
        """

        if not isinstance(trajectories, str):
            raise TypeError("'trajectories' must be of type 'str'.")

        trajectories_list = ["None", "0,0.5,1", "0,1", "All"]
        if trajectories not in trajectories_list:
            raise ValueError(f"'trajectories' must be one of {trajectories_list}.")

        return trajectories

    @staticmethod
    def pert_val(pert_val: str) -> str:
        """validates if pert or val

        Args:
            pert_val (str): whether pert for perturbations or val for values, ie per ligand

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be either 'pert' or 'val' so can be recognised by future scripts

        Returns:
           str: either 'pert' or 'val'
        """

        pert_val = validate.string(pert_val).lower()
        if pert_val not in ["pert", "val", "bound", "free"]:
            raise ValueError("pert_val must be either 'pert' or 'val'")

        return pert_val

    @staticmethod
    def estimator(estimator: str) -> str:
        """validates the estimator provided

        Args:
            estimator (str): whether MBAR or TI. If None, will use MBAR.

        Raises:
            TypeError: must be of type 'str'.
            ValueError: must be either 'MBAR' or 'TI' so can be recognised by future scripts

        Returns:
           str: either 'MBAR' or 'TI'
        """

        if estimator:
            estimator = validate.string(estimator).upper()
        estimator_list = ["MBAR", "TI", None]
        if estimator not in estimator_list:
            raise ValueError(f"estimator must be in {estimator_list}")
        if not estimator:
            estimator = "MBAR"

        return estimator

    @staticmethod
    def analysis_method(analysis_method: str) -> str:
        """validates method to be used for analysis

        Args:
            analysis_method (str): whether alchemlyb or native

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be either 'alchemlyb' or 'native' so can be recognised by future scripts

        Returns:
           str: either 'alchemlyb' or 'native'
        """

        if analysis_method:
            analysis_method = validate.string(analysis_method).lower()
        method_list = ["alchemlyb", "native", None]
        if analysis_method not in method_list:
            raise ValueError(f"analysis_method must be in {method_list}")

        if not analysis_method:
            analysis_method = "alchemlyb"

        return analysis_method

    @staticmethod
    def mbar_method(mbar_method: str) -> str:
        """validates if the mbar method is accepted

        Args:
            mbar_method (str): None or robust or default

        Raises:
            TypeError: must be of type 'str'
            ValueError: must be either 'robust' or 'default' or None so can be recognised by future scripts

        Returns:
           str: either 'robust' or 'default' or 'None'
        """

        if mbar_method:
            mbar_method = validate.string(mbar_method).lower()
        method_list = ["robust", "default", None, "none"]
        if mbar_method not in method_list:
            raise ValueError(f"mbar_method must be in {method_list}")

        if not mbar_method:
            mbar_method = None
        if mbar_method == "none":
            mbar_method = None

        return mbar_method

    @staticmethod
    def num_lambda(num_lambda: int) -> int:
        """validate number of lambdas to be run

        Args:
            num_lambda (str or int): number of lambdas

        Raises:
            TypeError: must be of type 'str' or 'int'
            ValueError: must be accepted number of lambda windows.

        Returns:
            int: number of lambdas
        """

        validate.integer(num_lambda)

        num_lambda_list = [11]
        if num_lambda not in num_lambda_list:
            raise ValueError(f"'num_lambda' must be one of {num_lambda}.")

        return num_lambda

    @staticmethod
    def pipeline_protocol(
        protocol,
        auto_validate: bool = True,
        fepprep: bool = False,
    ):
        """check if the passed protocol is a correct pipeline_protocol

        Args:
            protocol (pipeline.prep.pipeline_protocol): a previously
            read and validated pipeline protocol
            auto_validate (boolean): whether to validate the protocol. default is 'True'.
            fepprep (bool) : whether it is needed for fep prep stage or not

        Returns:
            pipeline.prep.pipeline_protocol: the protocol if it is okay
        """

        if not isinstance(protocol, pipeline.prep.pipeline_protocol):
            raise TypeError(
                "'protocol' must be of type 'pipeline.prep.pipeline_protocol'."
            )

        if not isinstance(fepprep, bool):
            raise TypeError("'fepprep' must be of type 'bool'.")

        if auto_validate:
            # validate incase it wasnt
            protocol.validate()
        else:
            pass

        # if fepprep, check that an engine and no of lam was specified
        if fepprep:
            if not hasattr(protocol, "num_lambda"):
                warnings.warn(
                    "the provided protocol does not have attribute 'num_lambda'.\n 11 lambda windows will be used..."
                )
                protocol.num_lambda(11)
            if not hasattr(protocol, "engine"):
                raise TypeError(
                    "protocol must have an engine to be used for fep. This is usually defined in the network file. \
                                This is different from protocol.engines in the protocol file. "
                )

        return protocol

    @staticmethod
    def analysis_protocol(protocol, auto_validate: bool = True):
        """check if the passed protocol is a correct analysis_protocol

        Args:
            protocol (pipeline.prep.analysis_protocol): a previously
            read and validated pipeline protocol
            auto_validate (boolean): whether to validate the protocol. default is 'True'.

        Returns:
            pipeline.prep.analysis_protocol: the protocol if it is okay
        """

        if not isinstance(protocol, pipeline.prep.analysis_protocol):
            raise TypeError(
                "'protocol' must be of type 'pipeline.prep.analysis_protocol'."
            )

        if auto_validate:
            # validate incase it wasnt
            protocol.validate()
        else:
            pass

        return protocol

    @staticmethod
    def system(system: _System) -> _System:
        """checks if it is a BSS system

        Args:
            system (BioSimSpace._SireWrappers._system.System): the system

        Raises:
            TypeError: if not BioSimSpace._SireWrappers._system.System

        Returns:
            BioSimSpace._SireWrappers._system.System: the passed system
        """

        if not isinstance(system, _System):
            raise TypeError("'system' must be a BSS system!.")

        return system

    @staticmethod
    def bss_protocol(protocol):
        """checks if it is a BSS protocol

        Args:
            system (BioSimSpace.Protocol): the protocol

        Raises:
            TypeError: if not in BioSimSpace.Protocol.protocols()

        Returns:
            BioSimSpace.Protocol: the BSS protocol.
        """

        # this is a very hacky way and not ideal
        ty = type(protocol)

        if "BioSimSpace.Protocol" not in str(ty):
            raise TypeError("'protocol' must be a BSS protocol!.")

        return protocol

    @staticmethod
    def molecule(molecule: _Molecule) -> _Molecule:
        """Check if BSS molecule

        Args:
            molecule (BioSimSpace._SireWrappers._molecule.Molecule): the molecule

        Raises:
            TypeError: if not BioSimSpace._SireWrappers._system.System or if not BioSimSpace._SireWrappers._molecule.Molecule

        Returns:
            BioSimSpace._SireWrappers._molecule.Molecule: a single molecule
        """

        if not isinstance(molecule, _Molecule):
            if not isinstance(molecule, _System):
                raise TypeError("'molecule' must be a BSS molecule (or system)!.")
            else:  # if it is a system, convert to a molecule
                warnings.warn(
                    "a BSS system was passed. The first molecule in the system will be taken."
                )
                system = molecule
                molecule = system[0]

        return molecule

    @staticmethod
    def analysis(analysis, analysed: bool = True):
        """checks if it is an analysed pipeline.analysis.analyse

        Args:
            system (pipeline.analysis.analyse): an analysed pipeline analysis

        Raises:
            TypeError: if not pipeline.analysis.analyse

        Returns:
            pipeline.analysis.analyse: the passed system
        """

        if not isinstance(analysis, pipeline.analysis.analyse):
            raise TypeError("'analysis' must be a pipeline.analysis.analyse!.")

        if analysed == True:
            if not analysis.is_analysed:
                raise ValueError("'analysis' must have already been analysed!.")

        return analysis
