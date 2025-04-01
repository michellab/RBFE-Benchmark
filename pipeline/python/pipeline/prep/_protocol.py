import logging

from typing import Union, Optional

from ..utils import *


class protocol:
    def __init__(
        self,
        file: str = None,
        auto_validate: bool = False,
        protocol_type: str = None,
    ):
        """class for storing and validating protocol options from files or dictionary.

        Args:
            file (file path (or dictionary)): file with the options for the protocol. Alternatively a dictionary.
            auto_validate (bool, optional): Whether to automatically validate the given input. Defaults to False.
        """

        if file:
            try:
                # instantiate the class with the protocol file
                self._prot_file = validate.file_path(file)
                # turn into a query dict as needed
                self._query_dict = self._read_protocol()
                is_file = True
            except Exception as e:
                logging.error(e)
                logging.error("Not recognised as file, trying to read as dictionary...")
                is_file = False

            if not is_file:
                try:
                    self._query_dict = validate.dictionary(file)
                    self._prot_file = None
                    is_dict = True
                except Exception as e:
                    logging.error(e)
                    logging.error(
                        "dictionary wasn't recognised either. trying to read as pipeline protocol..."
                    )
                    is_dict = False

            if not is_file and not is_dict:
                try:
                    if protocol_type == "pipeline":
                        prot = validate.pipeline_protocol(file, auto_validate=False)
                    elif protocol_type == "analysis":
                        prot = validate.analysis_protocol(file, auto_validate=False)

                    self._query_dict = prot.dictionary()

                    try:
                        self._prot_file = validate.file_path(prot._prot_file)
                    except:
                        self._prot_file = None

                except Exception as e:
                    logging.error(e)
                    logging.critical(
                        f"input was not recognised as a file/dictionary/protocol object."
                    )
                    self._query_dict = {}

        else:
            logging.info("no file or dict passed, using entirely default values...")
            # set the query dict as the default dict
            self._query_dict = self.default_dict()

        # update query dict if anything is missing
        self._query_dict = self._check_query()

        # validated if needed
        auto_validate = validate.boolean(auto_validate)
        if auto_validate:
            self.validate()
            self._is_validated = True
        else:
            self._is_validated = False

    def default_dict(self) -> dict:
        default_dict = {"name": None, "kwargs": {}}

        return default_dict

    def dictionary(self) -> dict:
        """calls the dictionary made from the file, validated or not.

        Returns:
            dict: key value pairs for the given protocol
        """
        return self._query_dict

    def validate(self):
        """validates all the input from the current query dict (taken from the protocol file or dict)."""

        query_dict = self._query_dict

        try:
            # validate all the input dict and replace in the query dict, eg:
            self.kwargs(query_dict["kwargs"])
            self.name(query_dict["name"])
            self._is_validated = True

        except ValueError as e:
            logging.error(
                f"There is a problem with the input provided in {self._prot_file}.\n Error is:\n {e}"
            )

    def _read_protocol(self, file: str = None) -> dict:
        """reads the protocol file into a dictionary

        Raises:
            ValueError: if the protocol file does not contain "="

        Returns:
            dict: dictionary of the entries in the protocol file,
            seperated into key:value pairs by the "=" and units by "*".
        """

        query_dict = {}

        if file:
            file = validate.file_path(file)
        else:
            file = self._prot_file

        # get value regardless of the order of the protocol.dat
        with open(f"{file}", "r") as file:
            for line in file:
                if "=" not in line:
                    raise ValueError(
                        "protocol file may only contain lines with '=', eg 'ligand forcefield = GAFF2'"
                    )
                elif "*" in line:  # if unit, get as seperate dict entry
                    query_dict[f"{line.split('=')[0].strip().lower()}"] = (
                        line.split("=")[-1].strip().split("*")[0]
                    )
                    query_dict[f"{line.split('=')[0].strip().lower()} unit"] = (
                        line.split("=")[-1].strip().split("*")[-1]
                    )
                else:
                    query_dict[f"{line.split('=')[0].strip().lower()}"] = line.split(
                        "="
                    )[-1].strip()

        return query_dict

    def _check_query(self) -> dict:
        """fills in any gaps of the dict from the protocol file

        Returns:
            dict: the filled in query dictionary
        """

        query_dict = self._query_dict
        default_dict = self.default_dict()
        kwarg_dict = {}

        # remove any unrecognised dictionary entries
        for query in list(query_dict):
            if query not in default_dict.keys():
                kwarg_dict[query] = query_dict[query]
                del query_dict[query]
                logging.info(
                    f"{query} removed from the protocol as not recognised.\n please use only:\n {default_dict.keys()}\n added as a kwarg argument instead."
                )

        query_dict["kwargs"] = kwarg_dict

        # add any missing protocol entries
        for query in default_dict.keys():
            if query not in query_dict.keys():
                query_dict[query] = default_dict[query]
                logging.info(
                    f"{query} not found in protocol. {default_dict[query]} will be used."
                )

        return query_dict

    def rewrite_protocol(self, file_path: str = None):
        """rewrite the protocol to the same file after validation

        Args:
            file_path (str, optional): path to a new protocol file location. Defaults to None.

        Raises:
            ValueError: if file_path is None and there is no file path to overwrite.
        """
        if file_path:
            new_file = validate.string(file_path)
        else:
            if self._prot_file:
                new_file = self._prot_file
            else:
                raise ValueError(
                    "there is no file to overwrite, please provide a 'file_path'"
                )

        pipeline.utils.write_protocol(self._query_dict, new_file)

    def print_protocol(self):
        """prints the protocol."""

        if not self._is_validated:
            logging.error("please validate the protocol first.")

        else:
            query_dict = self._query_dict

            for query in query_dict.keys():
                print(f"{query} : {query_dict[query]}")

    def kwargs(self, value: Optional[dict] = None) -> dict:
        if value:
            value = validate.dictionary(value)
            self._query_dict["kwargs"] = value
            self._kwargs = value
        else:
            try:
                value = self._kwargs
            except:
                value = {}
                self._kwargs = value

        return value

    def name(self, value: str = None) -> str:
        if value:
            value = validate.string(value)
            self._query_dict["name"] = value
            self._name = value
        else:
            try:
                value = self._name
            except:
                self._name = value

        return value


class pipeline_protocol(protocol):
    def __init__(self, file: str = None, auto_validate: bool = False):
        # inherit the init from other protocol too
        super().__init__(file, auto_validate, "pipeline")

    def default_dict(self) -> dict:
        """the default dictionary for the protocol

        Returns:
            dict: the default dictionary for the protocol
        """

        default_dict = {
            "ligand forcefield": "gaff2",
            "solvent": "TIP3P",
            "box edges": "15",
            "box edges unit": "angstrom",
            "box type": "cubic",
            "sampling": "2",
            "sampling unit": "ns",
            "hmr": "False",
            "hmr factor": "auto",
            "timestep overwrite": "True",
            "timestep": "2",
            "equilibration timestep": "2",
            "timestep unit": "fs",
            "repeats": "1",
            "trajectories": "None",
            "protein forcefield": "ff14SB",
            "start temperature": "10",
            "end temperature": "300",
            "temperature": "300",
            "temperature unit": "kelvin",
            "pressure": "1",
            "pressure unit": "bar",
            "minimisation steps": "10000",
            "equilibration runtime": "100",
            "equilibration runtime unit": "ps",
            "engines": "ALL",
            "fepprep": "both",
            "number of lambda windows": "12",
            "rerun": "False",
            "reverse": "False",
            "config options": None,
            "config options file": None,
            "name": None,
            "kwargs": {},
        }

        return default_dict

    def _config_dict(self) -> dict:
        legs = ["min", "eq", "heat", "prod", "all"]
        query_dict = {}
        for leg in legs:
            query_dict[leg] = {}

        return query_dict

    def _read_config_file(self, file: str = None) -> dict:
        """reads the config file into a dictionary

        Raises:
            ValueError: if the protocol file does not contain "=" or ";"

        Returns:
            dict: dictionary of the entries in the protocol file,
            seperated into key:value pairs by the "=" and into sections by ";"
        """

        query_dict = self._config_dict()

        file = validate.file_path(file)
        leg = None
        # get value regardless of the order of the protocol.dat
        with open(f"{file}", "r") as file:
            for line in file:
                if ";" in line:
                    leg = f"{line.split(';')[-1].strip().lower()}"
                    if leg not in query_dict.keys():
                        raise ValueError(
                            f"please seperate config file options using ; and {query_dict.keys()}"
                        )
                elif "=" not in line:
                    raise ValueError(
                        "protocol file may only contain lines with '=', eg 'ligand forcefield = GAFF2'"
                    )
                elif not leg:
                    query_dict["all"][
                        f"{line.split('=')[0].strip().lower()}"
                    ] = line.split("=")[-1].strip()
                else:
                    query_dict[leg][
                        f"{line.split('=')[0].strip().lower()}"
                    ] = line.split("=")[-1].strip()

        return query_dict

    def validate(self):
        """validates all the input from the current query dict (taken from the protocol file or dict)."""

        query_dict = self._query_dict

        # validate all the input dict and replace in the query dict
        # any failed validations use default values
        self.ligand_forcefield(query_dict["ligand forcefield"])
        self.protein_forcefield(query_dict["protein forcefield"])
        self.solvent(query_dict["solvent"])
        self.box_edges(query_dict["box edges"])
        self.box_edges_unit(query_dict["box edges unit"])
        self.box_type(query_dict["box type"])
        self.sampling(query_dict["sampling"])
        self.sampling_unit(query_dict["sampling unit"])
        self.repeats(query_dict["repeats"])
        self.trajectories(query_dict["trajectories"])
        self.start_temperature(query_dict["start temperature"])
        self.end_temperature(query_dict["end temperature"])
        self.temperature(query_dict["temperature"])
        self.temperature_unit(query_dict["temperature unit"])
        self.pressure(query_dict["pressure"])
        self.pressure_unit(query_dict["pressure unit"])
        self.min_steps(query_dict["minimisation steps"])
        self.eq_runtime(query_dict["equilibration runtime"])
        self.eq_runtime_unit(query_dict["equilibration runtime unit"])
        self.engines(query_dict["engines"])
        self.fepprep(query_dict["fepprep"])
        self.num_lambda(query_dict["number of lambda windows"])
        self.config_options_file(query_dict["config options file"])
        if query_dict["config options"]:
            self.config_options(query_dict["config options"])
        else:
            self.config_options(query_dict["config options file"])
        self.kwargs(query_dict["kwargs"])
        self.name(query_dict["name"])
        self.rerun(query_dict["rerun"])
        self.reverse(query_dict["reverse"])

        # choose timestep based on whether HMR is applied or not
        # this is important as BSS hmr mixin considers the timestep for the default auto
        self.hmr(query_dict["hmr"])
        self.hmr_factor(query_dict["hmr factor"])

        self.timestep(query_dict["timestep"])
        self.eq_timestep(query_dict["equilibration timestep"])
        self.timestep_unit(query_dict["timestep unit"])
        self.timestep_overwrite(query_dict["timestep overwrite"])

        # is now validated
        self._is_validated = True

    # this is not part of the default protocol and is found in the network file
    # it needs to be allocated before fepprep
    def fepprep(self, value: Optional[str] = None) -> str:
        """set the fepprep in the protocol or return its value.

        Args:
            value (str, optional): fepprep value. Defaults to None.

        Raises:
            ValueError:if not 'start', 'middle', or 'end'.

        Returns:
            str: fepprep value.
        """

        if value:
            value = value.lower()
            options_list = ["start", "both", "end"]
            if value not in options_list:
                raise ValueError(f"fepprep option must be in {options_list}")
            self._query_dict["fepprep"] = value
            self._fepprep = value
        else:
            value = self._fepprep

        return value

    def num_lambda(self, value: Optional[int] = None) -> int:
        """set the number of lambda windows in the protocol or return its value.

        Args:
            value (int, optional): number of lambda windows value. Defaults to None.

        Returns:
            int: number of lambda windows
        """

        if value:
            value = validate.integer(value)
            self._query_dict["number of lambda windows"] = value
            self._num_lambda = value
        else:
            value = self._num_lambda

        return value

    def engine(self, value: Optional[str] = None) -> str:
        """set the engine in the protocol or return its value.

        Args:
            value (str, optional): the engine. Defaults to None.

        Returns:
            str: engine
        """

        if value:
            value = validate.engine(value)
            # self._query_dict["engine"] = value # dont need as only for fepprep
            self._engine = value
        else:
            value = self._engine

        return value

    def engines(self, value: Optional[list] = None) -> list:
        """set the engines in the protocol or return its value.

        Args:
            value (list, optional): list of engines. Defaults to None.

        Returns:
            list: engines
        """

        if value:
            value = validate.engines(value)
            self._query_dict["engines"] = value
            self._engines = value
        else:
            value = self._engines

        return value

    def ligand_forcefield(self, value: Optional[str] = None) -> str:
        """set the ligand forcefield in the protocol or return its value.

        Args:
            value (str, optional): ligand forcefield. Defaults to None.

        Returns:
            str: ligand forcefield
        """

        if value:
            value = validate.lig_ff(value)
            self._query_dict["ligand forcefield"] = value
            self._ligand_forcefield = value
        else:
            value = self._ligand_forcefield

        return value

    def protein_forcefield(self, value: Optional[str] = None) -> str:
        """set the protein forcefield in the protocol or return its value.

        Args:
            value (str, optional): protein forcefield. Defaults to None.

        Returns:
            str: protein forcefield
        """

        if value:
            value = validate.prot_ff(value)
            self._query_dict["protein forcefield"] = value
            self._protein_forcefield = value
        else:
            value = self._protein_forcefield

        return value

    def solvent(self, value: Optional[str] = None) -> str:
        """set the solvent forcefield in the protocol or return its value.

        Args:
            value (str, optional): solvent forcefield. Defaults to None.

        Returns:
            str: solvent forcefield
        """

        if value:
            value = validate.solvent_ff(value)
            self._query_dict["solvent"] = value
            self._solvent = value
        else:
            value = self._solvent

        return value

    def box_edges(self, value: Optional[int] = None) -> int:
        """set the box edges length in the protocol or return its value.

        Args:
            value (int, optional): box edges length. Defaults to None.

        Returns:
            int: box edges length
        """

        if value:
            value = validate.integer(value)
            self._query_dict["box edges"] = value
            self._box_edges = value
        else:
            value = self._box_edges

        return value

    def box_edges_unit(self, value: Optional[str] = None):
        """set the box edges unit in the protocol or return its value.

        Args:
            value (str, optional): box edges unit. Defaults to None.

        Returns:
            BSS.Units.Length: box edges unit
        """

        if value:
            value = validate.box_edges_unit(value)
            self._query_dict["box edges unit"] = value
            self._box_edges_unit = value
        else:
            value = self._box_edges_unit

        return value

    def box_type(self, value: Optional[str] = None) -> str:
        """set the box type in the protocol or return its value.

        Args:
            value (str, optional): box type. Defaults to None.

        Returns:
            str: box edges type
        """

        if value:
            value = validate.box_type(value)
            self._query_dict["box type"] = value
            self._box_type = value
        else:
            value = self._box_type

        return value

    def sampling(self, value: Optional[int] = None) -> int:
        """set the sampling value in the protocol or return its value.

        Args:
            value (int, optional): sampling value. Defaults to None.

        Returns:
            int: sampling value
        """

        if value:
            value = validate.is_float(value)
            self._query_dict["sampling"] = value
            self._sampling = value
        else:
            value = self._sampling

        return value

    def sampling_unit(self, value: Optional[str] = None):
        """set the sampling unit in the protocol or return its value.

        Args:
            value (str, optional): sampling unit. Defaults to None.

        Returns:
            str: sampling unit
        """

        if value:
            value = validate.time_unit(value)
            self._query_dict["sampling unit"] = value
            self._sampling_unit = value
        else:
            value = self._sampling_unit

        return value

    def repeats(self, value: Optional[int] = None) -> int:
        """set the number of repeats in the protocol or return its value.

        Args:
            value (int, optional): number of repeats. Defaults to None.

        Returns:
            int: number of repeats
        """

        if value:
            value = validate.integer(value)
            self._query_dict["repeats"] = value
            self._repeats = value
        else:
            value = self._repeats

        return value

    def trajectories(self, value: Optional[str] = None) -> str:
        """set the trajectories to keep in the protocol or return its value.

        Args:
            value (str, optional): trajectories to keep. Defaults to None.

        Returns:
            str: trajectories to keep
        """

        if value:
            value = validate.trajectories(value)
            self._query_dict["trajectories"] = value
            self._trajectories = value
        else:
            value = self._trajectories

        return value

    def start_temperature(self, value: Optional[float] = None) -> float:
        """set the start temperature at equilibration in the protocol or return its value.

        Args:
            value (float, optional): the start temperature. Defaults to None.

        Returns:
            float: start temperature
        """

        if value:
            value = validate.is_float(value)
            self._query_dict["start temperature"] = value
            self._start_temperature = value
        else:
            try:
                value = self._start_temperature
            except:
                # if it is the default start temperature, it is ten
                value = int(10)
                self._start_temperature = value

        return value

    def end_temperature(self, value: Optional[float] = None) -> float:
        """set the end temperature at equilibration in the protocol or return its value.

        Args:
            value (float, optional): the end temperature. Defaults to None.

        Returns:
            float: end temperature
        """

        if value:
            value = validate.is_float(value)
            self._query_dict["end temperature"] = value
            self._end_temperature = value
        else:
            value = self._end_temperature

        return value

    def temperature(self, value: Optional[float] = None) -> float:
        """set the temperature of the simulation in the protocol or return its value.

        Args:
            value (float, optional): the temperature. Defaults to None.

        Returns:
            float: temperature
        """

        if value:
            value = validate.is_float(value)
            self._query_dict["temperature"] = value
            self._temperature = value
        else:
            value = self._temperature

        return value

    def temperature_unit(self, value: Optional[str] = None) -> str:
        """set the temperature unit of the simulation in the protocol or return its value.

        Args:
            value (str, optional): the temperature unit. Defaults to None.

        Returns:
            str: temperature unit
        """

        if value:
            value = validate.temperature_unit(value)
            self._query_dict["temperature unit"] = value
            self._temperature_unit = value
        else:
            value = self._temperature_unit

        return value

    def pressure(self, value: Optional[float] = None) -> float:
        """set the pressure of the simulation in the protocol or return its value.

        Args:
            value (int, optional): the pressure. Defaults to None.

        Returns:
            int: the pressure
        """

        if value:
            value = validate.is_float(value)
            self._query_dict["pressure"] = value
            self._pressure = value
        else:
            value = self._pressure

        return value

    def pressure_unit(self, value: Optional[str] = None) -> str:
        """set the pressure unit of the simulation in the protocol or return its value.

        Args:
            value (str, optional): the pressure unit. Defaults to None.

        Returns:
            str: pressure unit
        """

        if value:
            value = validate.pressure_unit(value)
            self._query_dict["pressure unit"] = value
            self._pressure_unit = value
        else:
            value = self._pressure_unit

        return value

    def min_steps(self, value: Optional[int] = None) -> int:
        """set the number of minimisation steps in the protocol or return its value.

        Args:
            value (int, optional): the number of minimisation steps. Defaults to None.

        Returns:
            int: the number of minimisation steps
        """

        if value:
            value = validate.integer(value)
            self._query_dict["minimisation steps"] = value
            self._min_steps = value
        else:
            value = self._min_steps

        return value

    def eq_runtime(self, value: Optional[int] = None) -> int:
        """set the equilibration runtime in the protocol or return its value.

        Args:
            value (int, optional): the equilibration runtime. Defaults to None.

        Returns:
            int: the equilibration runtime
        """

        if value:
            value = validate.integer(value)
            self._query_dict["equilibration runtime"] = value
            self._eq_runtime = value
        else:
            value = self._eq_runtime

        return value

    def eq_runtime_unit(self, value: Optional[str] = None) -> str:
        """set the equilibration runtime unit of the simulation in the protocol or return its value.

        Args:
            value (str, optional): the equilibration runtime unit. Defaults to None.

        Returns:
            str: equilibration runtime unit
        """

        if value:
            value = validate.time_unit(value)
            self._query_dict["equilibration runtime unit"] = value
            self._eq_runtime_unit = value
        else:
            value = self._eq_runtime_unit

        return value

    def hmr(self, value: Optional[bool] = None) -> bool:
        """set whether HMR is applied to the system in the protocol or return its value.

        Args:
            value (boolean, optional): if HMR is applied. Defaults to None.

        Returns:
            boolean: if HMR is applied
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["hmr"] = value
            self._hmr = value
        else:
            try:
                value = self._hmr
            except:
                self._hmr = value

        return value

    def hmr_factor(self, value: Optional[int] = None) -> int:
        """set the hmr factor in the protocol or return its value.

        Args:
            value (int, optional): the hmr factor. Defaults to None.

        Returns:
            int: the hmr factor
        """

        if value:
            if self._hmr:
                pass
            else:
                logging.info(
                    f"'hmr' must be set to True for a hmr factor to be applied. It will still be set as {value}."
                )

            try_float = True

            if isinstance(value, str):
                if value.lower() == "auto":
                    value = "auto"
                    try_float = False
                else:
                    try_float = True

            if try_float:
                try:
                    value = validate.is_float(value)
                except:
                    raise ValueError("hmr_factor must be 'auto' or a integer/float")

            self._hmr_factor = value
            self._query_dict["hmr factor"] = value

        else:
            value = self._hmr_factor

        return value

    def timestep_overwrite(self, value: Optional[bool] = None) -> bool:
        """set whether the timestep in the protocol should be overwritten based on the HMR or return its value.
        If True, overwrite the timestep to 4 fs if HMR is applied or 2 fs if not.
        If False, uses the provided timestep value (or the default for this, which is currently 2 fs)

        Args:
            value (boolean, optional): whether to overwrite the timestep. Defaults to None.

        Returns:
            boolean: whether to overwrite the timestep
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["timestep overwrite"] = value
            self._timestep_overwrite = value
        else:
            try:
                value = self._timestep_overwrite
            except:
                self._timestep_overwrite = value

        # will overwrite the provided timestep based on default values
        if self._timestep_overwrite:
            self.timestep_unit("fs")
            if self._hmr:
                self.timestep(4)
            else:
                self.timestep(2)

        return value

    def timestep_unit(self, value: Optional[str] = None) -> str:
        """set the timestep unit of the simulation in the protocol or return its value.

        Args:
            value (str, optional): the timestep unit. Defaults to None.

        Returns:
            str: timestep unit
        """

        if value:
            value = validate.time_unit(value)
            self._query_dict["timestep unit"] = value
            self._timestep_unit = value
        else:
            value = self._timestep_unit

        return value

    def timestep(self, value: Optional[float] = None) -> float:
        """set the timestep in the protocol or return its value.

        Args:
            value (float, optional): the timestep (ideally in femtoseconds). Defaults to None.

        Returns:
            float: the timestep
        """

        if value:
            value = validate.is_float(value)
            self._query_dict["timestep"] = value
            self._timestep = value
        else:
            value = self._timestep

        return value

    def eq_timestep(self, value: Optional[float] = None) -> float:
        """set the equilibration timestep in the protocol or return its value.

        Args:
            value (float, optional): the timestep (ideally in femtoseconds). Defaults to None.

        Returns:
            float: the timestep
        """

        if value:
            value = validate.is_float(value)
            self._query_dict["equilibration timestep"] = value
            self._eq_timestep = value
        else:
            value = self._eq_timestep

        return value

    def config_options(self, value: Optional[dict] = None) -> dict:
        if value:
            try:
                value = validate.file_path(value)
                value_dict = self._read_config_file(file=value)
                self._query_dict["config options"] = value_dict
                self._query_dict["config options file"] = value
                self._config_options = value_dict
                self._config_options_file = value
            except:
                logging.error(
                    "config options not a file, trying to read in as dictionary..."
                )
                try:
                    value_dict = validate.dictionary(value)
                    self._query_dict["config options"] = value_dict
                    self._config_options = value_dict
                except:
                    logging.error(f"could not validate config options, {value}")
                    value_dict = self._config_dict()
                    self._config_options = value_dict
        else:
            try:
                value_dict = self._config_options
            except:
                value_dict = self._config_dict()
                self._config_options = value_dict

        return value_dict

    def config_options_file(self, value: Optional[str] = None) -> str:
        if value:
            try:
                value = validate.file_path(value)
                self._query_dict["config options file"] = value
                self._config_options_file = value
            except:
                logging.error(f"could not validate config options file path, {value}")
                value = None
                self._config_options_file = value
        else:
            try:
                value = self._config_options_file
            except:
                value = None
                self._config_options_file = value

        return value

    def rerun(self, value: Optional[bool] = None) -> bool:
        """set whether its reruns/additional runs or return its value.
        This runs so that all the repeats are done.

        Args:
            value (boolean, optional): if rerun. Defaults to None.

        Returns:
            boolean: if it is a rerun
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["rerun"] = value
            self._rerun = value
        else:
            try:
                value = self._rerun
            except:
                self._rerun = value

        return value

    def rerepeat(self, value: Optional[int] = None) -> int:
        if value:
            value = validate.integer(value)
            self._query_dict["rerun start repeat"] = value
            self._rerepeat = value
        else:
            value = self._rerepeat

        return value

    def reverse(self, value: Optional[bool] = None) -> bool:
        """set whether to also run perturbations in the reverse direction.
        This will write a network file containing the reverse perturbations during the setup stages.

        Args:
            value (boolean, optional): if rerun. Defaults to None.

        Returns:
            boolean: if it is a rerun
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["reverse"] = value
            self._reverse = value
        else:
            try:
                value = self._reverse
            except:
                self._reverse = value

        return value


class analysis_protocol(protocol):
    def __init__(self, file=None, auto_validate=False):
        # inherit the init from other protocol too
        super().__init__(file, auto_validate, "analysis")

    # check query also inherited but with new default dict
    def default_dict(self) -> dict:
        default_dict = {
            "estimator": "MBAR",
            "method": "alchemlyb",
            "check overlap": "True",
            "try pickle": "True",
            "save pickle": "True",
            "auto equilibration": "False",
            "statistical inefficiency": "False",
            "truncate lower": "0",
            "truncate upper": "100",
            "mbar method": "None",  # robust or default
            "name": None,
            "kwargs": {},
        }

        return default_dict

    def validate(self):
        """validates all the input from the dict from the protocol file."""

        query_dict = self._query_dict

        # validate all the input dict
        self.estimator(query_dict["estimator"])
        self.analysis_method(query_dict["method"])
        self.check_overlap(query_dict["check overlap"])
        self.try_pickle(query_dict["try pickle"])
        self.save_pickle(query_dict["save pickle"])
        self.auto_equilibration(query_dict["auto equilibration"])
        self.statistical_inefficiency(query_dict["statistical inefficiency"])
        self.truncate_lower(query_dict["truncate lower"])
        self.truncate_upper(query_dict["truncate upper"])
        self.mbar_method(query_dict["mbar method"])
        self.kwargs(query_dict["kwargs"])
        self.name(query_dict["name"])

        # is now validated
        self._is_validated = True

    def estimator(self, value: Optional[str] = None) -> str:
        """set the estimator in the analysis protocol or return its value.

        Args:
            value (str, optional): the estimator. Defaults to None.

        Returns:
            str: estimator
        """

        if value:
            value = validate.estimator(value)
            self._query_dict["estimator"] = value
            self._estimator = value
        else:
            value = self._estimator

        return value

    def analysis_method(self, value: Optional[str] = None) -> str:
        """set the analysis method (eg native, alchemlyb) in the analysis protocol or return its value.

        Args:
            value (str, optional): the analysis method. Defaults to None.

        Returns:
            str: analysis method
        """

        if value:
            value = validate.analysis_method(value)
            self._query_dict["method"] = value
            self._method = value
        else:
            value = self._method

        return value

    def check_overlap(self, value: Optional[bool] = None) -> bool:
        """set whether to check the overlap in the analysis protocol or return its value.

        Args:
            value (boolean, optional): whether to check the overlap. Defaults to None.

        Returns:
            boolean: if to check the overlap
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["check overlap"] = value
            self._check_overlap = value
        else:
            try:
                value = self._check_overlap
            except:
                self._check_overlap = value

    def try_pickle(self, value: Optional[bool] = None) -> bool:
        """set whether to try to find/use pickle files in the analysis protocol or return its value.

        Args:
            value (boolean, optional): whether to use pickle files. Defaults to None.

        Returns:
            boolean: if to use pickle files
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["try pickle"] = value
            self._try_pickle = value
        else:
            try:
                value = self._try_pickle
            except:
                self._try_pickle = value

        return value

    def save_pickle(self, value: Optional[bool] = None) -> bool:
        """set whether to try to save pickle files in the analysis protocol or return its value.

        Args:
            value (boolean, optional): whether to save pickle files. Defaults to None.

        Returns:
            boolean: if to save pickle files
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["save pickle"] = value
            self._save_pickle = value
        else:
            try:
                value = self._save_pickle
            except:
                self._save_pickle = value

        return value

    def auto_equilibration(self, value: Optional[bool] = None) -> bool:
        """set whether to use auto equilibration in the analysis protocol or return its value.

        Args:
            value (boolean, optional): whether to use auto equilibration. Defaults to None.

        Returns:
            boolean: if to use auto equilibration
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["auto equilibration"] = value
            self._auto_equilibration = value
        else:
            try:
                value = self._auto_equilibration
            except:
                self._auto_equilibration = value

        return value

    def statistical_inefficiency(self, value: Optional[bool] = None) -> bool:
        """set whether to use statistical inefficiency in the analysis protocol or return its value.

        Args:
            value (boolean, optional): whether to use statistical inefficiency. Defaults to None.

        Returns:
            boolean: if to use statistical inefficiency
        """

        if value:
            value = validate.boolean(value)
            self._query_dict["statistical inefficiency"] = value
            self._statistical_inefficiency = value
        else:
            try:
                value = self._statistical_inefficiency
            except:
                self._statistical_inefficiency = value

        return value

    def mbar_method(self, value: Optional[str] = None) -> str:
        """set the mbar method for use with pymbar in the analysis protocol or return its value.

        Args:
            value (str, optional): the mbar method. Defaults to None.

        Returns:
            str: mbar method
        """
        if value:
            value = validate.mbar_method(value)
            self._query_dict["mbar method"] = value
            self._mbar_method = value
        else:
            try:
                value = self._mbar_method
            except:
                value = None
                self._mbar_method = value

        return value

    def truncate_lower(self, value: Optional[str] = None) -> str:
        """lower percentage of the data to truncate.

        Args:
            value (str, optional): truncate lower. Defaults to None.

        Returns:
            str: truncate lower
        """

        if value:
            value = validate.integer(value)
            self._query_dict["truncate lower"] = value
            self._truncate_lower = value
        else:
            try:
                value = self._truncate_lower
            except:
                value = "0"
                self._truncate_lower = value

        return value

    def truncate_upper(self, value: Optional[str] = None) -> str:
        """lower percentage of the data to truncate.

        Args:
            value (str, optional): truncate upper. Defaults to None.

        Returns:
            str: truncate upper
        """

        if value:
            value = validate.integer(value)
            self._query_dict["truncate upper"] = value
            self._truncate_upper = value
        else:
            try:
                value = self._truncate_upper
            except:
                value = "100"
                self._truncate_upper = value

        return value
