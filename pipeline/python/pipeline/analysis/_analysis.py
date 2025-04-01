# import libraries
import warnings
import BioSimSpace as BSS
from BioSimSpace import Units as _Units
import pickle
import numpy as np
import math as math
import pandas as pd
from scipy.stats import sem
import scipy.stats as _stats
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import re
import pathlib
import logging
from statistics import stdev
import alchemlyb as _alchemlyb
import shutil

from typing import Union, Optional, Tuple
from typing import List as _List

from alchemlyb.postprocessors.units import to_kcalmol, to_kJmol, to_kT
from alchemlyb.visualisation import plot_mbar_overlap_matrix, plot_ti_dhdl
from alchemlyb.visualisation import plot_convergence as _plot_convergence
from alchemlyb.preprocessing import (
    decorrelate_dhdl,
    dhdl2series,
    decorrelate_u_nk,
    u_nk2series,
    slicing,
)
from pymbar.timeseries import detect_equilibration as _detect_equilibration
from alchemlyb.convergence import fwdrev_cumavg_Rc, A_c, forward_backward_convergence

from ..utils import *
from ..prep import analysis_protocol


class analyse:
    """class to analyse a work dir"""

    def __init__(
        self,
        work_dir: str,
        pert: Optional[str] = None,
        engine: Optional[str] = None,
        analysis_prot: Optional[str] = None,
    ):
        """class for analysing results in a directory for a single perturbation

        Args:
            work_dir (str): directory of the perturbation results
            pert (str, optional): Name of the perturbation. lig0~lig1 . Defaults to None.
            engine (str, optional): engine to analyse for. Defaults to None.
            analysis_protocol (pipeline.prep.analysis_protocol, optional): the analysis protocol. Defaults to None.

        Raises:
            ValueError: need to have perturbation written as 'ligand_0~ligand_1'
        """
        # instantiate the class with the work directory

        self._work_dir = validate.folder_path(work_dir)
        self._pickle_dir = validate.folder_path(f"{self._work_dir}/pickle", create=True)
        self._graph_dir = validate.folder_path(self._work_dir + "/graphs", create=True)
        self._edgembar_dir = validate.folder_path(
            self._work_dir + "/edgembar_dats", create=True
        )

        if pert:
            self.perturbation = validate.string(pert)
            try:
                self.ligand_0 = self.perturbation.split("~")[0]
                self.ligand_1 = self.perturbation.split("~")[1]
            except:
                logging.critical(
                    "please seperate the ligands in the pert using '~' so ligand 0 and ligand 1 can be identified."
                )

        else:
            try:
                self.perturbation = self._work_dir.split("/")[-1]
                self.ligand_0 = self.perturbation.split("~")[0]
                self.ligand_1 = self.perturbation.split("~")[1]
            except Exception as e:
                logging.exception(e)
                logging.critical(
                    "was unable to get the perturbation name and ligands from the file path.\n please add these when initialising using pert='lig_0~lig_1'."
                )

        if engine:
            self.engine = validate.engine(engine)
        else:
            try:
                try:
                    # first try to find in second position
                    self.engine = validate.engine(
                        self._work_dir.split("/")[-2].replace("_extracted", "")
                    )
                except:
                    # then try to find anywhere in the folder path
                    for eng in BSS.FreeEnergy.engines():
                        if eng.upper() in self._work_dir.upper():
                            self.engine = validate.engine(eng)
                            logging.info(f"found {eng.upper()} as engine in work_dir")
            except Exception as e:
                logging.exception(e)
                logging.critical(
                    "was unable to get the engine from the file path.\n please add when initialising using engine='ENGINE'."
                )

        # initialise dictionaries
        self._bound_pmf_dict = {}  # for the intial results
        self._free_pmf_dict = {}
        self._bound_matrix_dict = {}  # for the energy matrix
        self._free_matrix_dict = {}
        self._bound_val_dict = {}  # for the actual value for all the windows
        self._free_val_dict = {}
        self._bound_err_dict = {}  # for the final error defined
        self._free_err_dict = {}
        self.repeats_tuple_list = []

        # for convergence plotting
        self.spert_results_dict = {}
        self.spert_bound_dict = {}
        self.spert_free_dict = {}
        self.epert_results_dict = {}
        self.epert_bound_dict = {}
        self.epert_free_dict = {}

        # intialise other things
        self._get_repeat_folders()
        self.options_dict = None
        self._set_default_options()  # will set default options and file extension
        self.is_analysed = False

        # initialise values
        self.free_val = None
        self.free_err = None
        self.bound_val = None
        self.bound_err = None
        self.freenrg_val = None
        self.freenrg_err = None
        self.freenrg_sd = None
        self.upper_ci = None
        self.lower_ci = None

        # list for the successful calculations
        self.bound_calculated = []
        self.free_calculated = []

        if analysis_prot:
            analysis_prot = analysis_protocol(analysis_prot, auto_validate=True)
            self.set_options(analysis_prot)

    @staticmethod
    def _default_analysis_options_dict() -> dict:
        """the default analysis options dictionary

        Returns:
            dict: default dictionary
        """

        options_dict = {
            "estimator": "MBAR",
            "method": "alchemlyb",
            "check overlap": True,
            "try pickle": True,
            "save pickle": True,
            "auto equilibration": False,
            "statistical inefficiency": False,
            "truncate lower": 0,
            "truncate upper": 100,
            "mbar method": None,  # robust or default
            "name": None,
        }

        return options_dict

    def _set_default_options(self):
        """set the default options in the class"""

        default_options = analyse._default_analysis_options_dict()
        self.set_options(default_options)

    def _file_ext(self):
        """set the file extension for the analusis files based on the options dict

        Returns:
            str: the file extension
        """

        file_ext = analyse.file_ext(self.options_dict)
        self.file_extension = file_ext

        return file_ext

    @staticmethod
    def file_ext(options_dict: dict) -> str:
        """write a file extension based on a protocol style options dictionary

        Args:
            options_dict (dict): analysis protocol dictionary

        Returns:
            str: the file extension
        """

        # validate any inputs in the dictionary
        options_dict = analyse._update_options_dict(options_dict)

        file_ext = str(
            f"{options_dict['estimator']}_{options_dict['method']}_{options_dict['mbar method']}_"
            + f"eq{str(options_dict['auto equilibration']).lower()}_"
            + f"stats{str(options_dict['statistical inefficiency']).lower()}_"
            + f"truncate{str(options_dict['truncate lower'])}_{options_dict['truncate upper']}"
        )

        return file_ext

    def _pickle_ext(self) -> str:
        """the extension for the pickle files based on the file extension, engine, and perturbation

        Returns:
            str: pickle extension
        """

        pickle_ext = analyse.pickle_ext(
            self.options_dict, self.perturbation, self.engine
        )
        self.pickle_extension = pickle_ext

        return pickle_ext

    @staticmethod
    def pickle_ext(options_dict: dict, perturbation: str, engine: str) -> str:
        # validate any inputs in the dictionary
        options_dict = analyse._update_options_dict(options_dict)

        file_ext = analyse.file_ext(options_dict)

        pickle_ext = f"{perturbation}_{engine}_" + f"{file_ext}"

        return pickle_ext

    @staticmethod
    def _update_options_dict(
        options_dict: dict, current_options: Optional[dict] = None
    ) -> dict:
        """update the options dict, if no current one the default will be used.

        Args:
            options_dict (dict or pipeline.prep.analysis_protocol): dictionary of options or the analysis protocol
            current_options (dict, optional): current options. Defaults to None.

        Returns:
            dict: the new options dict that will be used
        """
        # returns a dict with the considered options, and fills in any not provided with default or previous options.

        # if analysis protocol is supplied, make sure to get the dictionary form
        if isinstance(options_dict, analysis_protocol):
            options_dict = options_dict.dictionary()

        options_dict = validate.dictionary(options_dict)

        # use the default dict as the default options
        if not current_options:
            current_options = analyse._default_analysis_options_dict()
        else:
            current_options = validate.dictionary(current_options)

        new_options_dict = {}

        # when initialising, a default dict is set
        # replace any values in this by the new options dict

        # replace any default values by those passed
        for key, value in current_options.items():
            if key in options_dict:
                new_options_dict[key] = options_dict[key]
            else:
                new_options_dict[key] = value

        # validate to make sure all inputs are acceptable format
        val_options_dict = analyse._validate_options_dict(new_options_dict)

        return val_options_dict

    @staticmethod
    def _validate_options_dict(options_dict: dict) -> dict:
        """validate the passed dictionary

        Args:
            options_dict (dict): analysis options dict

        Returns:
            dict: validated options dictionary.
        """
        # validate all the new inputs
        # replace as needed in the options dict

        if "estimator" in options_dict:
            estimator = validate.estimator(options_dict["estimator"])
            options_dict["estimator"] = estimator

        if "mbar method" in options_dict:
            mbar_method = validate.mbar_method(options_dict["mbar method"])
            options_dict["mbar method"] = mbar_method
            if options_dict["estimator"] == "TI":
                options_dict["mbar method"] = None

        if "check overlap" in options_dict:
            check_overlap = validate.boolean(options_dict["check overlap"])
            if check_overlap == True and estimator != "MBAR":
                check_overlap = False
            options_dict["check overlap"] = check_overlap

        if "method" in options_dict:
            method = validate.analysis_method(options_dict["method"])
            options_dict["method"] = method

        if "save pickle" in options_dict:
            save_pickle = validate.boolean(options_dict["save pickle"])
            options_dict["save pickle"] = save_pickle

        if "try pickle" in options_dict:
            try_pickle = validate.boolean(options_dict["try pickle"])
            options_dict["try pickle"] = try_pickle

        if "auto equilibration" in options_dict:
            auto_equilibration = validate.boolean(options_dict["auto equilibration"])
            options_dict["auto equilibration"] = auto_equilibration

        if "statistical inefficiency" in options_dict:
            statistical_inefficiency = validate.boolean(
                options_dict["statistical inefficiency"]
            )
            options_dict["statistical inefficiency"] = statistical_inefficiency

        if "truncate upper" in options_dict:
            truncate_upper = validate.integer(options_dict["truncate upper"])
            options_dict["truncate upper"] = truncate_upper

        if "truncate lower" in options_dict:
            truncate_lower = validate.integer(options_dict["truncate lower"])
            options_dict["truncate lower"] = truncate_lower

        if "name" in options_dict:
            if options_dict["name"]:
                name = validate.string(options_dict["name"])
                options_dict["name"] = name
            else:
                pass

        return options_dict

    def set_options(self, options_dict: dict):
        """set the analysis options for this object

        Args:
            options_dict (dict or pipeline.prep.analysis_protocol): options for analysis
        """

        # first use staticmethod to get a new options dict
        # if already have an options dict for this object, want to use that
        options_dict = analyse._update_options_dict(
            options_dict, current_options=self.options_dict
        )

        # update self.options_dict for use w the file extension
        self.options_dict = options_dict

        # then set all of these to self options
        self.estimator = options_dict["estimator"]
        self._mbar_method = options_dict["mbar method"]
        self._check_overlap = options_dict["check overlap"]
        self.method = options_dict["method"]
        self._save_pickle = options_dict["save pickle"]
        self._try_pickle = options_dict["try pickle"]
        self._auto_equilibration = options_dict["auto equilibration"]
        self._statistical_inefficiency = options_dict["statistical inefficiency"]
        self._truncate_lower = options_dict["truncate lower"]
        self._truncate_upper = options_dict["truncate upper"]
        self.name = options_dict["name"]

        # set the file extensions
        self._file_ext()
        self._pickle_ext()

    def _get_repeat_folders(self):
        """how many of each the free and bound repeat folders there are."""

        self._b_folders, self._f_folders = get_repeat_folders(self._work_dir)

        self._no_of_b_repeats = len(self._b_folders)
        self._no_of_f_repeats = len(self._f_folders)
        self._b_repeats = list(range(self._no_of_b_repeats))
        self._f_repeats = list(range(self._no_of_f_repeats))

        if self._no_of_b_repeats != self._no_of_f_repeats:
            logging.warning(
                f"There are a different number of repeats for bound ({self._no_of_b_repeats}) and free ({self._no_of_f_repeats}) for {self._work_dir}."
                + f" These are {self._b_folders} and {self._f_folders}."
            )
        else:
            logging.info(
                f"There are {self._no_of_b_repeats} repeats for each the bound and the free for {self._work_dir}."
                + f" These are {self._b_folders} and {self._f_folders}."
            )

    def _check_pickle(self) -> bool:
        """check if there are all the pickle files present in the pickle folder.

        Returns:
            boolean: whether to try pickle when analysing
        """

        try_pickle = True
        pickle_ext = self.pickle_extension

        # try loading in if previously calculated
        if try_pickle:
            try:
                logging.info(
                    f"trying to locate pickles in default pickle folder, {self._pickle_dir} for {pickle_ext}..."
                )
                with open(
                    f"{self._pickle_dir}/bound_pmf_{pickle_ext}.pickle", "rb"
                ) as file:
                    self._bound_pmf_dict = pickle.load(file)
                with open(
                    f"{self._pickle_dir}/free_pmf_{pickle_ext}.pickle", "rb"
                ) as file:
                    self._free_pmf_dict = pickle.load(file)
                with open(
                    f"{self._pickle_dir}/bound_matrix_{pickle_ext}.pickle", "rb"
                ) as file:
                    self._bound_matrix_dict = pickle.load(file)
                with open(
                    f"{self._pickle_dir}/free_matrix_{pickle_ext}.pickle", "rb"
                ) as file:
                    self._free_matrix_dict = pickle.load(file)
                with open(
                    f"{self._pickle_dir}/bound_val_{pickle_ext}.pickle", "rb"
                ) as file:
                    self._bound_val_dict = pickle.load(file)
                with open(
                    f"{self._pickle_dir}/free_val_{pickle_ext}.pickle", "rb"
                ) as file:
                    self._free_val_dict = pickle.load(file)
                with open(
                    f"{self._pickle_dir}/bound_err_{pickle_ext}.pickle", "rb"
                ) as file:
                    self._bound_err_dict = pickle.load(file)
                with open(
                    f"{self._pickle_dir}/free_err_{pickle_ext}.pickle", "rb"
                ) as file:
                    self._free_err_dict = pickle.load(file)
                logging.info("pickles found!")
            except:
                logging.info("loading pickle failed. Calculating normally.")
                try_pickle = False

        return try_pickle

    def analyse_all_repeats(self) -> tuple:
        """Analyse all existing free-energy data from a simulation working directory.

        Returns:
            tuple: tuple of result, error, repeats tuple list
            (freenrg_rel[0], freenrg_rel[1], repeats_tuple_list)
        """

        if self._try_pickle:
            do_pickle = self._check_pickle()
        else:
            do_pickle = False

        logging.info(f"analysing using {self.options_dict}.")

        if do_pickle:
            freenrg_rel, repeats_tuple_list = self._analyse_all_repeats_pickle()
            if freenrg_rel[0] is np.nan:
                logging.info("loaded pickle is None. Will try calculating again...")
                do_pickle = False

        if not do_pickle:
            freenrg_rel, repeats_tuple_list = self._analyse_all_repeats_normal()

        # set the average and error
        self.freenrg = freenrg_rel[0]
        self.error = freenrg_rel[1]
        self.repeats_tuple_list = repeats_tuple_list
        self.is_analysed = True
        try:
            self.confidence_interval()
        except:
            logging.error("cant calc CI!")

        if self._check_overlap:
            self.check_overlap()

        if self._save_pickle:
            if do_pickle:
                logging.info("already using pickles, will not be saving again.")
            else:
                logging.info(
                    "saving the pmf dictionaries for bound and free as pickles."
                )
            self.save_pickle()

        logging.info(
            f"The freenerg is {self.freenrg} +/- {self.error}. The {self._confidence_interval*100} CI is [{self.lower_ci}, {self.upper_ci}]"
        )

        return (freenrg_rel[0], freenrg_rel[1], repeats_tuple_list)

    def _analyse_all_repeats_normal(self):
        """Analyse all existing free-energy data from a simulation working directory.

        Returns:
            tuple: (freenrg_rel, repeats_tuple_list)
        """

        # remake list for the successful calculations
        self.bound_calculated = []
        self.free_calculated = []

        process_dict = {
            "auto equilibration": self._auto_equilibration,
            "statistical inefficiency": self._statistical_inefficiency,
            "truncate upper": self._truncate_upper,
            "truncate lower": self._truncate_lower,
            "mbar method": self._mbar_method,
        }

        if os.path.exists(f"{self._work_dir}/{self._b_folders[0]}/prod"):
            # check with just the first bound folder
            pass
        else:
            for leg in self._b_folders + self._f_folders:
                validate.folder_path(f"{self._work_dir}/{leg}/prod", create=True)
                list_dir = sorted(
                    [
                        folder_name
                        for folder_name in os.listdir(f"{self._work_dir}/{leg}")
                        if "lambda" in folder_name
                    ]
                )
                for lambda_folder in list_dir:
                    shutil.move(
                        f"{self._work_dir}/{leg}/{lambda_folder}",
                        f"{self._work_dir}/{leg}/prod/{lambda_folder}",
                    )

        # Analyse the results for each leg of the transformation.
        for b in self._b_repeats:
            try:
                name = str(b) + "_bound"
                pmf_bound, overlap_matrix_bound = BSS.FreeEnergy.Relative.analyse(
                    f"{self._work_dir}/{self._b_folders[b]}/prod",
                    estimator=self.estimator,
                    method=self.method,
                    **process_dict,
                )
                self._bound_pmf_dict.update({name: pmf_bound})
                self._bound_matrix_dict.update({name: overlap_matrix_bound})
                self.bound_calculated.append(name)
            except Exception as e:
                logging.exception(e)
                logging.error(
                    f"Unable to analyse values for {name}, which is repeat {self._b_folders[b]} in {self._work_dir}."
                )

        for f in self._f_repeats:
            try:
                name = str(f) + "_free"
                pmf_free, overlap_matrix_free = BSS.FreeEnergy.Relative.analyse(
                    f"{self._work_dir}/{self._f_folders[f]}/prod",
                    estimator=self.estimator,
                    method=self.method,
                    **process_dict,
                )
                self._free_pmf_dict.update({name: pmf_free})
                self._free_matrix_dict.update({name: overlap_matrix_free})
                self.free_calculated.append(name)
            except Exception as e:
                logging.exception(e)
                logging.error(
                    f"Unable to analyse values for {name}, which is repeat {self._f_folders[f]} in {self._work_dir}."
                )

        # create a dictionary of the calculated values, for each bound and free
        for r in self._b_repeats:
            try:
                bound_name = str(r) + "_bound"
                bound_val = (
                    self._bound_pmf_dict[bound_name][-1][1]
                    - self._bound_pmf_dict[bound_name][0][1]
                )
                bound_err = BSS.Types.Energy(
                    math.sqrt(
                        math.pow(self._bound_pmf_dict[bound_name][-1][2].value(), 2)
                        + math.pow(self._bound_pmf_dict[bound_name][0][2].value(), 2)
                    ),
                    self._bound_pmf_dict[bound_name][-1][2].unit(),
                )
                self._bound_val_dict.update({bound_name: bound_val})
                self._bound_err_dict.update({bound_name: bound_err})
            except Exception as e:
                logging.exception(e)
                logging.error(
                    f"""Unable to compute values for {bound_name} in {self._work_dir}.\
                    Check earlier error message if these values could be analysed."""
                )
        for r in self._f_repeats:
            try:
                free_name = str(r) + "_free"
                free_val = (
                    self._free_pmf_dict[free_name][-1][1]
                    - self._free_pmf_dict[free_name][0][1]
                )
                free_err = BSS.Types.Energy(
                    math.sqrt(
                        math.pow(self._free_pmf_dict[free_name][-1][2].value(), 2)
                        + math.pow(self._free_pmf_dict[free_name][0][2].value(), 2)
                    ),
                    self._free_pmf_dict[free_name][-1][2].unit(),
                )
                self._free_val_dict.update({free_name: free_val})
                self._free_err_dict.update({free_name: free_err})
            except Exception as e:
                logging.exception(e)
                logging.error(
                    f"""Unable to compute values for {free_name} in {self._work_dir}.\
                    Check earlier error message if these values could be analysed."""
                )

        freenrg_rel, repeats_tuple_list = self._calculate_freenrg()

        return (freenrg_rel, repeats_tuple_list)

    def _analyse_all_repeats_pickle(self) -> tuple:
        """Analyse all existing free-energy data from a simulation working directory in the pickle directory.

        Returns:
            tuple: (freenrg_rel, repeats_tuple_list)
        """

        # remake list for the successful calculations
        self.bound_calculated = []
        self.free_calculated = []

        # Analyse the results for each leg of the transformation.
        for b in self._b_repeats:
            try:
                name = str(b) + "_bound"
                if name in self._bound_pmf_dict.keys():
                    self.bound_calculated.append(name)
            except Exception as e:
                logging.exception(e)
                logging.error(
                    f"Unable to analyse values for {name} from pickle, which should be repeat {self._b_folders[b]} in {self._work_dir}."
                )

        for f in self._f_repeats:
            try:
                name = str(f) + "_free"
                if name in self._free_pmf_dict.keys():
                    self.free_calculated.append(name)
            except Exception as e:
                logging.exception(e)
                logging.error(
                    f"Unable to analyse values for {name} from pickle, which should be repeat {self._f_folders[f]} in {self._work_dir}."
                )

        freenrg_rel, repeats_tuple_list = self._calculate_freenrg()

        return (freenrg_rel, repeats_tuple_list)

    def _calculate_freenrg(self) -> tuple:
        """calculate the free energy.

        Returns:
            tuple: ((freenrg, freenrg_err), repeats_tuple_list)
        """

        # list for all the repeats
        repeats_tuple_list = []

        # first check if there are calculated values for each.
        if len(self.bound_calculated) == 0:
            logging.error("there are no results for the bound leg.")
            missing_results = True
        elif len(self.free_calculated) == 0:
            logging.error("there are no results for the free leg.")
            missing_results = True
        else:
            missing_results = False

        if missing_results:
            return (np.nan, np.nan), repeats_tuple_list

        # list of values for each leg
        free_vals = [val.value() for val in self._free_val_dict.values()]
        bound_vals = [val.value() for val in self._bound_val_dict.values()]
        free_avg = np.mean(free_vals)
        bound_avg = np.mean(bound_vals)
        # errors depending on how many results
        if len(self.free_calculated) == 1:
            free_err = self._free_err_dict[self.free_calculated[0]].value()
        else:
            free_err = sem(free_vals)
        if len(self.bound_calculated) == 1:
            bound_err = self._bound_err_dict[self.bound_calculated[0]].value()
        else:
            bound_err = sem(bound_vals)
        # freenerg values
        freenrg_val = bound_avg - free_avg
        freenrg_err = math.sqrt(math.pow(bound_err, 2) + math.pow(free_err, 2))
        freenrg_rel = (
            freenrg_val * _Units.Energy.kcal_per_mol,
            freenrg_err * _Units.Energy.kcal_per_mol,
        )

        # set the average bound and free values and their error
        self.free_val = free_avg
        self.free_err = free_err
        self.bound_val = bound_avg
        self.bound_err = bound_err
        self.freenrg_val = freenrg_val
        self.freenrg_err = freenrg_err

        # create tuple list of each repeat that was calculated
        # first check the length of the calculated values and check if this is also the length of the folders found
        if len(self.bound_calculated) != self._no_of_b_repeats:
            logging.warning(
                "the number of calculated values for bound does not match the number of bound folders.\n maybe try reanalysing/check errors?"
            )
        if len(self.free_calculated) != self._no_of_f_repeats:
            logging.warning(
                "the number of calculated values for free does not match the number of free folders.\n maybe try reanalysing/check errors?"
            )

        # if the numebr of calculated values is the same, match these evenly
        if len(self.bound_calculated) == len(self.free_calculated):
            logging.info(
                f"There are {len(self.bound_calculated)} calculated values for each the bound and the free leg for the folders in {self._work_dir}."
            )
            self.no_of_repeats = len(self.bound_calculated)

        elif len(self.bound_calculated) != len(self.free_calculated):
            logging.warning(
                f"There are {len(self.bound_calculated)} calculated values for the bound and {len(self.free_calculated)} calculated values for the free leg for the folders in {self._work_dir}."
            )
            # use the shorter calculated values as the number of complete repeats
            if len(self.bound_calculated) < len(self.free_calculated):
                self.no_of_repeats = len(self.bound_calculated)
            else:
                self.no_of_repeats = len(self.free_calculated)

        # calculate
        logging.info(f"{self.no_of_repeats} repeats will be calculated.")
        r = 0
        for b, f in zip(self.bound_calculated, self.free_calculated):
            logging.info(f"calculating repeat {r} as {b} and {f}.")
            freenrg_rel_rep = BSS.FreeEnergy.Relative.difference(
                self._bound_pmf_dict[b], self._free_pmf_dict[f]
            )
            freenrg_val = freenrg_rel_rep[0]
            freenrg_err = freenrg_rel_rep[1]
            repeats_tuple_list.append((f"{str(r)}_repeat", freenrg_val, freenrg_err))
            r += 1

        freenrg_sd = np.std(np.array([r[1].value() for r in repeats_tuple_list]))
        self.freenrg_sd = freenrg_sd

        return freenrg_rel, repeats_tuple_list

    def save_pickle(self):
        """save the analysis as a pickle."""

        self._pickle_dir = validate.folder_path(self._pickle_dir, create=True)
        pickle_ext = self.pickle_extension

        if not self.is_analysed:
            warnings.warn(
                "can't save pickle, not all repeats have been analysed. please self.analyse_all_repeats() first!"
            )
        else:
            try:
                # write the pmf as a pickle
                with open(
                    f"{self._pickle_dir}/bound_pmf_{pickle_ext}.pickle", "wb"
                ) as handle:
                    pickle.dump(self._bound_pmf_dict, handle)
                with open(
                    f"{self._pickle_dir}/free_pmf_{pickle_ext}.pickle", "wb"
                ) as handle:
                    pickle.dump(self._free_pmf_dict, handle)
                with open(
                    f"{self._pickle_dir}/bound_matrix_{pickle_ext}.pickle", "wb"
                ) as handle:
                    pickle.dump(self._bound_matrix_dict, handle)
                with open(
                    f"{self._pickle_dir}/free_matrix_{pickle_ext}.pickle", "wb"
                ) as handle:
                    pickle.dump(self._free_matrix_dict, handle)
                with open(
                    f"{self._pickle_dir}/bound_val_{pickle_ext}.pickle", "wb"
                ) as handle:
                    pickle.dump(self._bound_val_dict, handle)
                with open(
                    f"{self._pickle_dir}/free_val_{pickle_ext}.pickle", "wb"
                ) as handle:
                    pickle.dump(self._free_val_dict, handle)
                with open(
                    f"{self._pickle_dir}/bound_err_{pickle_ext}.pickle", "wb"
                ) as handle:
                    pickle.dump(self._bound_err_dict, handle)
                with open(
                    f"{self._pickle_dir}/free_err_{pickle_ext}.pickle", "wb"
                ) as handle:
                    pickle.dump(self._free_err_dict, handle)
            except Exception as e:
                logging.exception(e)
                logging.warning("could not save pickle :( ")

    def check_overlap(self) -> tuple:
        """check the overlap of the analysed object."""

        if not self.is_analysed:
            warnings.warn(
                "can't check overlap, not all repeats have been analysed. please self.analyse_all_repeats() first!"
            )

            return None

        else:
            # check overlap matrix if okay
            okay = 0
            no_overlaps = 0
            too_smalls = 0
            for b in self._b_repeats:
                try:
                    name = str(b) + "_bound"
                    overlap = self._bound_matrix_dict[name]
                    overlap_okay, too_small = BSS.FreeEnergy.Relative.checkOverlap(
                        overlap
                    )
                    no_overlaps += 1
                    too_smalls += too_small
                    if overlap_okay:
                        okay += 1
                except Exception as e:
                    logging.exception(e)
                    logging.error(f"could not check overlap matrix for {name}")

            for f in self._f_repeats:
                try:
                    name = str(f) + "_free"
                    overlap = self._free_matrix_dict[name]
                    overlap_okay, too_small = BSS.FreeEnergy.Relative.checkOverlap(
                        overlap
                    )
                    no_overlaps += 1
                    too_smalls += too_small
                    if overlap_okay:
                        okay += 1
                except Exception as e:
                    logging.exception(e)
                    logging.error(f"could not check overlap matrix for {name}")

            if no_overlaps:
                percen_okay = (okay / no_overlaps) * 100
                too_smalls_avg = too_smalls / no_overlaps
            else:
                percen_okay = None
                too_smalls_avg = None

            return percen_okay, too_smalls_avg

    def plot_graphs(self):
        """plot (overlap matrix or dHdl) for the analysed run."""

        if not self.is_analysed:
            warnings.warn(
                "can't plot, not all repeats have been analysed. please self.analyse_all_repeats() first!"
            )

        else:
            pass

            if self.estimator == "MBAR":
                for b in self._b_repeats:
                    try:
                        name = str(b) + "_bound"
                        overlap = self._bound_matrix_dict[name]
                        ax = plot_mbar_overlap_matrix(overlap)
                        ax.figure.savefig(
                            f"{self._graph_dir}/{name}_overlap_MBAR_{self.pickle_extension}",
                        )
                    except Exception as e:
                        logging.exception(e)
                        logging.error(f"could not plt overlap matrix for {name}")

                for f in self._f_repeats:
                    try:
                        name = str(f) + "_free"
                        overlap = self._free_matrix_dict[name]
                        ax = plot_mbar_overlap_matrix(overlap)
                        ax.figure.savefig(
                            f"{self._graph_dir}/{name}_overlap_MBAR_{self.pickle_extension}",
                        )
                    except Exception as e:
                        logging.exception(e)
                        logging.error(f"could not plt overlap matrix for {name}")

            elif self.estimator == "TI":
                for b in self._b_repeats:
                    name = str(b) + "_bound"
                    overlap = self._bound_matrix_dict[name]
                    try:
                        ax = plot_ti_dhdl(overlap)
                        ax.figure.savefig(
                            f"{self._graph_dir}/{name}_dHdl_TI_{self.pickle_extension}",
                        )
                    except Exception as e:
                        logging.exception(e)
                        logging.error(f"could not plt dhdl for {name}")

                for f in self._f_repeats:
                    name = str(f) + "_free"
                    overlap = self._free_matrix_dict[name]
                    try:
                        ax = plot_ti_dhdl(overlap)
                        ax.figure.savefig(
                            f"{self._graph_dir}/{name}_dHdl_TI_{self.pickle_extension}",
                        )
                    except Exception as e:
                        logging.exception(e)
                        logging.error(f"could not plt dhdl for {name}")

    def confidence_interval(self, confidence_interval: float = 0.95):
        """calculate the confidence of the edge. This is very large as there are not that many replicas generally.

        Args:
            confidence_interval (float, optional): the confidence interval to use. Defaults to 0.95.
        """

        self._confidence_interval = confidence_interval

        if not self.is_analysed:
            warnings.warn(
                "can't calculate CI, not all repeats have been analysed. please self.analyse_all_repeats() first!"
            )

        # check more than one run has been carried out
        if self.no_of_repeats < 2:
            warnings.warn(
                "will not calculate CI as there is only one repeat calculated."
            )

        # create 95% confidence interval
        # small sample so use t interval
        # if larger sample, use norm.interval() as assume sample mean is normally distributed due to central limit theorem
        lower_ci, upper_ci = _stats.t.interval(
            confidence_interval,
            df=self.no_of_repeats - 1,
            loc=self.freenrg_val,  # mean
            scale=self.freenrg_err,  # SEM
        )

        self.lower_ci = lower_ci
        self.upper_ci = upper_ci

    def check_repeat_convergence(self):
        """check if the standard deviation of the repeats is less than 1.5 kcal/mol"""
        # ((f"{str(r)}_repeat", freenrg_val, freenrg_err))

        try:
            repeats_values = [repeat_val[1] for repeat_val in self.repeats_tuple_list]
            standard_deviation_of_vals = stdev(repeats_values)

            threshold = 1.5
            if standard_deviation_of_vals < threshold:
                logging.info(
                    f"Standard deviation of the freenerg repeats is {standard_deviation_of_vals}, which is less than {threshold}. The freenrg is considered converged."
                )
            else:
                logging.error(
                    f"Standard deviation of the freenerg repeats is {standard_deviation_of_vals}, which is more than {threshold}. The freenrg is NOT considered converged."
                )
        except:
            logging.error(
                "could not check convergence / calculate the standard deviation of repeats."
            )
            standard_deviation_of_vals = None

        return standard_deviation_of_vals

    def check_convergence(self):
        """using alchemlyb functions."""
        converged_percen_dict = {}
        eq_dict = {}
        for leg in self._b_folders + self._f_folders:
            try:
                files, temperatures, lambdas = self.get_files_temperatures_lambdas(
                    f"{self._work_dir}/{leg}/prod"
                )

                u_nk = BSS.FreeEnergy.Relative._get_data(
                    files, temperatures, self.engine, self.estimator
                )

                df = forward_backward_convergence(u_nk, self.estimator)

                ac = A_c([u_nk2series(_u_nk) for _u_nk in u_nk], precision=0.01, tol=2)
                eq_dict[leg] = ac

                # define convergence by checking if forward and backward have the same free energy estimate within error
                converged_no = 0
                for row in range(0, len(df)):
                    # error added
                    if abs(df["Forward"][row] - df["Backward"][row]) < [
                        df["Forward_Error"][row] + df["Backward_Error"][row]
                    ]:
                        converged_no += 1
                converged_percen = converged_no / len(df)

            except Exception as e:
                logging.error(e)
                converged_percen = None
                eq_dict[leg] = None

            converged_percen_dict[leg] = converged_percen

        if self._save_pickle:
            with open(
                f"{self._pickle_dir}/converged_percen_dict_{self.pickle_extension}.pickle",
                "wb",
            ) as handle:
                print(handle)
                pickle.dump(converged_percen_dict, handle)

            with open(
                f"{self._pickle_dir}/ac_{self.pickle_extension}.pickle", "wb"
            ) as handle:
                print(handle)
                pickle.dump(eq_dict, handle)

        return converged_percen_dict

    def calculate_convergence(self, recalculate=False):
        """calculate the convergence for average of the results."""

        if self._try_pickle:
            do_pickle = self._check_convergence_pickle()
        else:
            do_pickle = False

        if not do_pickle or recalculate:
            # calculate all the truncated data
            sresults_dict, sbound_dict, sfree_dict = analyse._calculate_truncated(
                self._work_dir,
                self.estimator,
                "start",
                self._statistical_inefficiency,
                self._auto_equilibration,
                do_pickle,
            )
            eresults_dict, ebound_dict, efree_dict = analyse._calculate_truncated(
                self._work_dir,
                self.estimator,
                "end",
                self._statistical_inefficiency,
                self._auto_equilibration,
                do_pickle,
            )

            self.spert_results_dict = sresults_dict
            self.spert_bound_dict = sbound_dict
            self.spert_free_dict = sfree_dict
            self.epert_results_dict = eresults_dict
            self.epert_bound_dict = ebound_dict
            self.epert_free_dict = efree_dict

        self._save_convergence_pickle()

    def _check_convergence_pickle(self) -> bool:
        """check if there are all the pickle files present in the pickle folder.

        Returns:
            boolean: whether to try pickle when analysing
        """

        do_pickle = False
        pickle_ext = self.pickle_extension.split("truncate")[0]

        # try loading in if previously calculated
        try:
            logging.info(
                f"trying to locate pickles for convergence in default pickle folder, {self._pickle_dir} for {pickle_ext}..."
            )
            with open(
                f"{self._pickle_dir}/spert_results_dict_{pickle_ext}.pickle", "rb"
            ) as file:
                self.spert_results_dict = pickle.load(file)
            with open(
                f"{self._pickle_dir}/epert_results_dict_{pickle_ext}.pickle", "rb"
            ) as file:
                self.epert_results_dict = pickle.load(file)
            with open(
                f"{self._pickle_dir}/spert_bound_dict_{pickle_ext}.pickle", "rb"
            ) as file:
                self.spert_bound_dict = pickle.load(file)
            with open(
                f"{self._pickle_dir}/epert_bound_dict_{pickle_ext}.pickle", "rb"
            ) as file:
                self.epert_bound_dict = pickle.load(file)
            with open(
                f"{self._pickle_dir}/spert_free_dict_{pickle_ext}.pickle", "rb"
            ) as file:
                self.spert_free_dict = pickle.load(file)
            with open(
                f"{self._pickle_dir}/epert_free_dict_{pickle_ext}.pickle", "rb"
            ) as file:
                self.epert_free_dict = pickle.load(file)
            logging.info("pickles found!")
            do_pickle = True
        except:
            logging.info("loading pickle failed. Calculating normally.")

        return do_pickle

    def _save_convergence_pickle(self):
        """save the convergence calculation pickle."""

        self._pickle_dir = validate.folder_path(self._pickle_dir, create=True)
        pickle_ext = self.pickle_extension.split("truncate")[0]

        # save pickle
        if self._save_pickle:
            logging.info(
                f"trying to save pickles for convergence in default pickle folder, {self._pickle_dir} for {pickle_ext}..."
            )
            # write the pmf as a pickle
            with open(
                f"{self._pickle_dir}/spert_results_dict_{pickle_ext}.pickle", "wb"
            ) as handle:
                pickle.dump(self.spert_results_dict, handle)
            with open(
                f"{self._pickle_dir}/epert_results_dict_{pickle_ext}.pickle", "wb"
            ) as handle:
                pickle.dump(self.epert_results_dict, handle)
            with open(
                f"{self._pickle_dir}/spert_bound_dict_{pickle_ext}.pickle", "wb"
            ) as handle:
                pickle.dump(self.spert_bound_dict, handle)
            with open(
                f"{self._pickle_dir}/epert_bound_dict_{pickle_ext}.pickle", "wb"
            ) as handle:
                pickle.dump(self.epert_bound_dict, handle)
            with open(
                f"{self._pickle_dir}/spert_free_dict_{pickle_ext}.pickle", "wb"
            ) as handle:
                pickle.dump(self.spert_free_dict, handle)
            with open(
                f"{self._pickle_dir}/epert_free_dict_{pickle_ext}.pickle", "wb"
            ) as handle:
                pickle.dump(self.epert_free_dict, handle)
            logging.info("saved pickles!")

    def plot_convergence(self, use_alchemlyb=False):
        """plot the convergence."""

        if use_alchemlyb:
            for leg in self._b_folders + self._f_folders:
                files, temperatures, lambdas = self.get_files_temperatures_lambdas(
                    f"{self._work_dir}/{leg}/prod"
                )

                u_nk = BSS.FreeEnergy.Relative._get_data(
                    files, temperatures, self.engine, self.estimator
                )

                # plot convergence
                df = forward_backward_convergence(u_nk, self.estimator)
                ax = _plot_convergence(df)
                ax.figure.savefig(
                    f"{self._graph_dir}/forward_reverse_{leg}_{self.perturbation}_{self.file_extension.split('truncate')[0]}_alchemlyb.png"
                )

        else:
            try:
                for from_start, from_end, leg in zip(
                    [
                        self.spert_results_dict,
                        self.spert_bound_dict,
                        self.spert_free_dict,
                    ],
                    [
                        self.epert_results_dict,
                        self.epert_bound_dict,
                        self.epert_free_dict,
                    ],
                    ["freenerg", "bound", "free"],
                ):
                    sdf = analyse.single_pert_dict_into_df(from_start)
                    edf = analyse.single_pert_dict_into_df(from_end)
                    # plot individually for perts
                    logging.info(
                        f"plotting for {leg}, {self.perturbation}, {self.engine} in {self._graph_dir}..."
                    )
                    analyse.plot_truncated(
                        sdf,
                        edf,
                        file_path=f"{self._graph_dir}/forward_reverse_{leg}_{self.perturbation}_{self.file_extension.split('truncate')[0]}.png",
                        plot_difference=False,
                    )

            except Exception as e:
                logging.exception(e)
                logging.error(
                    "failed to plot convergence, please check Exception message."
                )

    # not in use
    # def plot_equilibration(self, eq_lines=[], eq_line_labels=[]):

    #     eq_lines = []
    #     eq_line_labels = []

    #     try:
    #         for from_start, from_end, leg in zip(
    #             [self.spert_results_dict, self.spert_bound_dict, self.spert_free_dict],
    #             [self.epert_results_dict, self.epert_bound_dict, self.epert_free_dict],
    #             ["freenerg", "bound", "free"],
    #         ):
    #             sdf = analyse.single_pert_dict_into_df(from_start)
    #             edf = analyse.single_pert_dict_into_df(from_end)
    #             # plot individually for perts
    #             logging.info(
    #                 f"plotting for {leg}, {self.perturbation}, {self.engine} in {self._graph_dir}..."
    #             )
    #             analyse.plot_truncated(
    #                 sdf,
    #                 None,
    #                 file_path=f"{self._graph_dir}/equilibrated_region_{('_').join(eq_line_labels)}_{leg}_{self.perturbation}_{self.file_extension.split('truncate')[0]}.png",
    #                 plot_difference=False,
    #                 eq_lines=eq_lines,
    #                 eq_line_labels=eq_line_labels,
    #             )

    #     except Exception as e:
    #         logging.exception(e)
    #         logging.error(
    #             "failed to plot convergence, please check Exception message.")

    @staticmethod
    def _calculate_truncated(
        path_to_dir: str,
        estimator: str,
        start_end: str,
        statsineff: bool = False,
        eq: bool = False,
        try_pickle: bool = True,
        save_pickle: bool = True,
        truncate_percentage: list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100],
    ) -> tuple:
        truncate_percentage = validate.is_list(truncate_percentage)
        for val in truncate_percentage:
            validate.integer(val)

        # analyse the work dir
        analysed_pert = analyse(path_to_dir)

        results_dict = {}
        bound_dict = {}
        free_dict = {}

        for trunc_per in truncate_percentage:
            if start_end == "start":
                trunc_lower = 0
                trunc_upper = trunc_per
            elif start_end == "end":
                trunc_lower = 100 - trunc_per
                trunc_upper = 100
            else:
                raise ValueError("'start_end' must be 'start' or 'end'.")
            analysed_pert.set_options(
                {
                    "estimator": estimator,
                    "truncate upper": trunc_per,
                    "try pickle": try_pickle,
                    "save_pickle": save_pickle,
                    "truncate lower": trunc_lower,
                    "truncate upper": trunc_upper,
                    "statistical inefficiency": statsineff,
                    "auto equilibration": eq,
                }
            )

            analysed_pert.analyse_all_repeats()

            results_dict[trunc_per] = (
                analysed_pert.freenrg_val,
                analysed_pert.freenrg_err,
            )
            bound_dict[trunc_per] = (analysed_pert.bound_val, analysed_pert.bound_err)
            free_dict[trunc_per] = (analysed_pert.free_val, analysed_pert.free_err)

        return results_dict, bound_dict, free_dict

    @staticmethod
    def single_pert_dict_into_df(pert_dict: dict):
        df = pd.DataFrame.from_dict(pert_dict)
        df = df.transpose()
        df.columns = ["avg", "max"]
        df["min"] = df.loc[:, "max"] * -1

        return df

    @staticmethod
    def plot_truncated(
        sdf: pd.DataFrame,
        edf: pd.DataFrame,
        file_path: Optional[str] = None,
        plot_error: Optional[bool] = False,
        plot_difference: Optional[bool] = True,
        eq_lines: Optional[list] = [],
        eq_line_labels: Optional[list] = [],
    ):
        include_key = True

        assert len(eq_lines) == len(eq_line_labels)

        plt.rc("font", size=12)
        plt.rcParams["axes.xmargin"] = 0  # plt.margins(x=0)
        fig, ax = plt.subplots(figsize=(10, 10))
        lines = []

        # fill in final value and its error
        plt.axhline(y=sdf["avg"].iloc[-1], color="c")
        plt.fill_between(
            sdf.index,
            sdf["avg"].iloc[-1] + sdf["min"].iloc[-1],
            sdf["avg"].iloc[-1] + sdf["max"].iloc[-1],
            color="paleturquoise",
            alpha=0.8,
        )
        lines += plt.plot(0, 0, c="c", label="final estimate")

        scatterplot = [plt.plot(sdf.index, sdf["avg"], c="lightcoral")]
        plt.fill_between(
            sdf.index,
            sdf["avg"] + sdf["min"],
            sdf["avg"] + sdf["max"],
            color="mistyrose",
            alpha=0.4,
        )
        lines += plt.plot(0, 0, c="lightcoral", label="forward")

        if edf is not None:
            scatterplot = [plt.plot(edf.index, edf["avg"], c="cornflowerblue")]
            plt.fill_between(
                edf.index,
                edf["avg"] + edf["min"],
                edf["avg"] + edf["max"],
                color="lightskyblue",
                alpha=0.3,
            )
            lines += plt.plot(0, 0, c="cornflowerblue", label="reverse")

        if eq_lines:
            # Add vertical lines for equilibration
            line_colors = [cm.viridis(i / len(eq_lines)) for i in range(len(eq_lines))]

            for x_val, color, label in zip(eq_lines, line_colors, eq_line_labels):
                lines += plt.axvline(x=x_val, color=color, linestyle="--", label=label)

        if include_key:
            labels = [l.get_label() for l in lines]
            plt.legend(lines, labels, loc="upper right")

        plt.xlabel("Percentage of run used")
        if file_path:
            plt.title(f"{file_path.split('/')[-1].split('.')[0].replace('_',' ')}")
        else:
            pass

        if plot_error:
            if plot_difference:
                plt.ylabel("Computed Error difference to final (kcal/mol)")
            else:
                plt.ylabel("Computed Error (kcal/mol)")
        else:
            if plot_difference:
                plt.ylabel(
                    "difference to final result for computed $\Delta\Delta$G$_{bind}$ (kcal/mol)"
                )
            else:
                plt.ylabel("Computed $\Delta\Delta$G$_{bind}$ (kcal/mol)")

        if file_path:
            plt.savefig(file_path)

    def plot_across_lambda(self):
        """plot the repeats across lambdas"""

        if not self.is_analysed:
            warnings.warn(
                "can't check overlap, not all repeats have been analysed. please self.analyse_all_repeats() first!"
            )

            return None

        try:
            # get the lambda values
            lambda_vals = [
                entry[0]
                for entry in self._free_pmf_dict[list(self._free_pmf_dict.keys())[0]]
            ]

            # make the dicts
            freenrg_dict = {}
            freenrg_err_dict = {}
            bound_dict = {}
            bound_err_dict = {}
            free_dict = {}
            free_err_dict = {}
            for lam in lambda_vals:
                freenrg_dict[lam] = []
                freenrg_err_dict[lam] = []
                bound_dict[lam] = []
                bound_err_dict[lam] = []
                free_dict[lam] = []
                free_err_dict[lam] = []

            for repf, repb in zip(self._free_pmf_dict, self._bound_pmf_dict):
                bound_pmf = self._bound_pmf_dict[repb]
                free_pmf = self._free_pmf_dict[repf]

                for pb, pf in zip(bound_pmf, free_pmf):
                    freenrg_dict[pb[0]].append((pb[1].value()) - (pf[1].value()))
                    freenrg_err_dict[pb[0]].append((pb[2].value()) + (pf[2].value()))
                    bound_dict[pb[0]].append(pb[1].value())
                    bound_err_dict[pb[0]].append(pb[2].value())
                    free_dict[pb[0]].append(pf[1].value())
                    free_err_dict[pb[0]].append(pf[2].value())

            lines = []

            # currently only for freenrg
            for val_dict in [freenrg_dict]:
                index_dict = {}

                for x in lambda_vals:
                    val_list = [x for x in val_dict[x] if pd.notna(x)]
                    avg = np.mean(val_list)
                    min_val = min(val_list)
                    max_val = max(val_list)

                    index_dict[x] = (avg, min_val, max_val)

                df = pd.DataFrame.from_dict(
                    index_dict, orient="index", columns=["avg", "min", "max"]
                )

                plt.plot(df.index, df["avg"], c="cornflowerblue")
                plt.fill_between(
                    df.index,
                    df["min"],
                    df["max"],
                    color="lightskyblue",
                    alpha=0.3,
                )
                lines += plt.plot(0, 0, c="cornflowerblue", label="freenrg")

                plt.xlim(xmin=0, xmax=1)
                plt.ylabel("Computed $\Delta$$\Delta$G$_{perturbation}$ (kcal/mol)")
                plt.xlabel("Lambda")
                labels = [l.get_label() for l in lines]
                plt.legend(lines, labels)
                plt.title(f"Energy across lambda windows for {self.perturbation}")
                plt.savefig(
                    f"{self._graph_dir}/{self.perturbation}_across_lambda_windows_freenrg.png"
                )
        except Exception as e:
            logging.error("could not plot, is the data okay? the error is:")
            logging.error(f"{e}")

    def format_for_edgembar(self, dats_folder=None, overwrite=False):
        # need to get the data into the format of
        # filename efep_tlam_elam.dat
        # tlam is the window at which it is generated ie the name of the folder lambda
        # elam is the energy lambda ie the different column headings
        # timeseries kcal/mol

        if dats_folder:
            dats_folder = validate.folder_path(dats_folder, create=True)
        else:
            dats_folder = self._edgembar_dir

        if overwrite:
            try_pickle = False
            for f in os.listdir(dats_folder):
                try:
                    os.remove(os.path.join(dats_folder, f))
                except:
                    shutil.rmtree(os.path.join(dats_folder, f))
        else:
            try_pickle = self._try_pickle
        save_pickle = self._save_pickle

        for leg in self._b_folders + self._f_folders:
            files, temperatures, lambdas = self.get_files_temperatures_lambdas(
                f"{self._work_dir}/{leg}/prod"
            )

            u_nk = None

            # extract u_nk
            if try_pickle:
                try:
                    logging.info("opening pickle")
                    with open(
                        f"{dats_folder}/u_nk_{leg}_{self.pickle_extension}.pickle", "rb"
                    ) as file:
                        u_nk = pickle.load(file)
                    save_pickle = False
                except:
                    logging.error("there is no pickle")
                    u_nk = None

            if u_nk is None:
                try:
                    logging.info("getting the unk")
                    u_nk = BSS.FreeEnergy.Relative._get_data(
                        files, temperatures, self.engine, self.estimator
                    )
                    # logging.info(type(u_nk))
                    # for df in u_nk:
                    # logging.info(type(df))
                    try:
                        logging.info("preprocessing the data")
                        u_nk = BSS.FreeEnergy.Relative._preprocess_data(
                            u_nk,
                            estimator=self.estimator,
                            **{
                                "truncate upper": self._truncate_upper,
                                "truncate lower": self._truncate_lower,
                                "autoequilibration": self._auto_equilibration,
                                "statistical inefficiency": self._statistical_inefficiency,
                            },
                        )
                        # logging.info(type(u_nk))
                        # for df in u_nk:
                        # logging.info(type(df))
                        logging.info("finished preprocessing data")
                    except Exception as e:
                        logging.error(e)
                        logging.error("error w preprocessing the data")
                except Exception as e:
                    logging.error(e)
                    logging.error("somethingwent wrong w the unk")
            else:
                pass

            if save_pickle:
                logging.info("saving the pickle.")
                with open(
                    f"{dats_folder}/u_nk_{leg}_{self.pickle_extension}.pickle", "wb"
                ) as handle:
                    pickle.dump(u_nk, handle)

            # for the amber output format, this is not the val - ref but just the val.
            # as alchemlyb uses the val - ref, this needs to be added back to have the correct format

            try:
                kcal_u_nk = []

                logging.info("converting to kcal mol")
                for df in u_nk:
                    new_df = to_kcalmol(df)
                    kcal_u_nk.append(new_df)

                logging.info("doing for the df")
                for df in kcal_u_nk:
                    if self.engine == "AMBER":
                        lam_name = "lambdas"
                    elif self.engine == "SOMD" or self.engine == "GROMACS":
                        lam_name = "fep-lambda"

                    # get the trajectory lambda values (tlam)
                    tlam = df.index.get_level_values(lam_name)[0]

                    # for each of the energy lambda (elam)
                    for elam in df.columns:
                        new_df = df[elam]  # the df is the energy lambda
                        ref_df = df[tlam]  # the ref data is the tlam
                        # as alchemlyb uses df(alchemlyb) = val - ref , and amber output is just the val, the val = df(alchemlyb) + ref
                        val_df = new_df + ref_df
                        newer_df = val_df.droplevel(
                            lam_name
                        )  # the tlams are dropped from the row names
                        final_df = newer_df.reset_index()
                        final_df["time"] = final_df["time"].round(1)
                        # need to replace any infinity values - this is the value used for infinity in the pymaberdats
                        final_df.replace([np.inf, -np.inf], 100000.000000, inplace=True)
                        final_df.replace(np.inf, 100000.000000, inplace=True)

                        folder = validate.folder_path(
                            f"{dats_folder}/{leg.split('_')[0]}/{leg.split('_')[1]}_{self.file_extension}",
                            create=True,
                        )
                        final_df.to_csv(
                            f"{folder}/efep_{tlam}_{elam}.dat",
                            sep=" ",
                            index=False,
                            header=False,
                        )
            except Exception as e:
                logging.error(e)
                logging.error(f"failed to extract the edge to df for folder {leg}.")

            # # for the amber output format, this is not the val - ref but just the val.
            # # as alchemlyb uses the val - ref, this needs to be added back to have the correct format
            # kcal_u_nk = []

            # for df in u_nk:
            #     logging.info("converting to kcal mol")
            #     new_df = to_kcalmol(df)
            #     kcal_u_nk.append(new_df)

            # for df in kcal_u_nk:

            #     try:
            #         logging.info("doing for the df")
            #         if self.engine == "AMBER":
            #             lam_name = "lambdas"
            #         elif self.engine == "SOMD" or self.engine == "GROMACS":
            #             lam_name = "fep-lambda"

            #         tlam = df.index.get_level_values(lam_name)[0]

            #         for elam in df.columns:
            #             new_df = df[elam]
            #             ref_df = df[tlam]
            #             val_df = ref_df + new_df
            #             newer_df = val_df.droplevel(lam_name)
            #             final_df = newer_df.reset_index()
            #             final_df["time"] = final_df["time"].round(1)
            #             # need to replace any infinity values - this is the value used for infinity in the pymaberdats
            #             final_df.replace([np.inf, -np.inf],
            #                              100000.000000, inplace=True)
            #             final_df.replace(np.inf,
            #                              100000.000000, inplace=True)

            #             folder = validate.folder_path(
            #                 f"{dats_folder}/{leg.split('_')[0]}/{leg.split('_')[1]}",
            #                 create=True,
            #             )
            #             final_df.to_csv(
            #                 f"{folder}/efep_{tlam}_{elam}_{self.file_extension}.dat",
            #                 sep=" ",
            #                 index=False,
            #                 header=False,
            #             )
            #     except:
            #         logging.error(
            #             f"failed to extract the edge to df after folder {leg}, {tlam}, {elam}.")

            # for df in dhdl:
            #     # first column is simulation time, second column is the potential energy in kcal/mol of the elam state
            #     unit_df = to_kcalmol(df)

            #     try:
            #         new_df = unit_df["dHdl"]
            #         newer_df = new_df.droplevel("lambdas")
            #         final_df = newer_df.reset_index()

            #         folder = validate.folder_path(
            #             f"{dats_folder}/{leg.split('_')[0]}/{leg.split('_')[1]}",
            #             create=True,
            #         )
            #         final_df.to_csv(
            #             f"{folder}/dvdl_{tlam}.dat",
            #             sep=" ",
            #             index=False,
            #             header=False,
            #         )
            #     except:
            #         logging.error(f"failed to extract the dhdl edge to df after folder {leg}, {tlam}, {elam}.")

    def get_files_temperatures_lambdas(self, work_dir: str) -> tuple:
        function_glob_dict = {
            "SOMD": "**/simfile.dat",
            "GROMACS": "**/[!bar]*.xvg",
            "AMBER": "**/*.out",
        }

        mask = function_glob_dict[self.engine]
        glob_path = pathlib.Path(work_dir)
        files = sorted(glob_path.glob(mask))

        lambdas = []
        for file in files:
            for part in file.parts:
                if "lambda" in part:
                    lambdas.append(float(part.split("_")[-1]))

        if self.engine == "AMBER":
            # Find the temperature for each lambda window.
            temperatures = []
            for file, lambda_ in zip(files, lambdas):
                found_temperature = False
                with open(file) as f:
                    for line in f.readlines():
                        if not found_temperature:
                            match = re.search(r"temp0=([\d.]+)", line)
                            if match is not None:
                                temperatures += [float(match.group(1))]
                                found_temperature = True
                            elif found_temperature == True:
                                pass

                    if not found_temperature:
                        raise ValueError(
                            "The temperature was not detected in the AMBER output file."
                        )

        elif self.engine == "SOMD":
            temperatures = []
            for file in files:
                found_temperature = False
                with open(file, "r") as f:
                    for line in f.readlines():
                        temp = None
                        start = "#Generating temperature is"
                        if start in line:
                            split_line = (line.split(start)[1]).strip().split(" ")
                            temp = split_line[0]
                            unit = split_line[-1]
                            if unit.upper() == "C":
                                temp = float(temp) + 273.15  # Convert to K
                            else:
                                temp = float(temp)
                            temperatures.append(temp)
                            if temp is not None:
                                found_temperature = True
                                break

                if not found_temperature:
                    raise ValueError(
                        f"The temperature was not detected in the SOMD output file, {file}"
                    )

        elif self.engine == "GROMACS":
            # find the temperature at each lambda window
            temperatures = []
            for file in files:
                found_temperature = False
                with open(file, "r") as f:
                    for line in f.readlines():
                        t = None
                        start = "T ="
                        end = "(K)"
                        if start and end in line:
                            t = int(((line.split(start)[1]).split(end)[0]).strip())
                            temperatures.append(float(t))
                            if t is not None:
                                found_temperature = True
                                break

                if not found_temperature:
                    raise ValueError(
                        f"The temperature was not detected in the GROMACS output file, {file}"
                    )

        return files, temperatures, lambdas

    def get_eq_times(self):
        eq_dict = {}

        for leg in self._b_folders + self._f_folders:
            try:
                files, temperatures, lambdas = self.get_files_temperatures_lambdas(
                    f"{self._work_dir}/{leg}/prod"
                )
            except:
                logging.error(
                    f"could not extract files and temperatures for {self.perturbation}, {self.engine}, {leg} for get eq times."
                )

            try:
                # extract u_nk
                u_nk = BSS.FreeEnergy.Relative._get_u_nk(
                    files, temperatures, self.engine
                )

                eq_time_frac = [self._get_eq_time(i) for i in u_nk]
                mean = np.mean(eq_time_frac)
            except:
                logging.error(
                    f"could not get eq times for {self.perturbation}, {self.engine}, {leg}"
                )
                eq_time_frac = []
                mean = None

            logging.info(f"fractional equilibration times for {leg} was {eq_time_frac}")
            logging.info(f"the average frac eq time for {leg} was {mean}")

            eq_dict[leg] = {"times": eq_time_frac, "mean": mean}

        with open(
            f"{self._pickle_dir}/eq_times_{self.pickle_extension}.pickle", "wb"
        ) as handle:
            print(handle)
            pickle.dump(eq_dict, handle)

        df = pd.DataFrame.from_dict(eq_dict)
        validate.folder_path(f"{self._work_dir}/eq_dats", create=True)
        df.to_csv(f"{self._work_dir}/eq_dats/equlibration_times.dat")

        return eq_dict

    def check_Ac(self):
        """
        Generate the ensemble convergence criteria Ac for a set of simulations. For each leg, average.
        1 is equilibrated, 0 is not.
        """

        # from the docs, can compute for just one window
        # from the ref, can plot as a cdf
        # u_nk = BSS.FreeEnergy.Relative._get_data(
        #     files, temperatures, ana_obj.engine, ana_obj.estimator)
        # Rc_list = []
        # for i,_u_nk in enumerate(u_nk):
        #     # decorrelated = decorrelate_dhdl(_dhdl, remove_burnin=True)
        #     R_c, running_average = fwdrev_cumavg_Rc(u_nk2series(_u_nk), tol=2)
        #     Rc_list.append(R_c)

        #     # ax = plot_convergence(running_average, final_error=2)
        #     # ax.set_ylabel("$\partial H/\partial\lambda$ (in kT)")

        # Rc_lambda = np.array(Rc_list)
        # print(Rc_lambda)

        # Rc_lambda_sorted = Rc_lambda # np.sort(Rc_lambda)

        # # Compute CDF values
        # cdf = np.arange(1, len(Rc_lambda_sorted) + 1) / len(Rc_lambda_sorted)

        # plt.figure(figsize=(8, 5))
        # plt.plot(Rc_lambda_sorted, cdf, marker='o', linestyle='-',
        #          color='b', label=r'$P(R_c(\lambda) < R_c)$')
        # plt.xlabel(r'$R_c$')
        # plt.ylabel('Cumulative Probability')
        # plt.title('Cumulative Distribution Function')
        # plt.legend()
        # plt.grid()
        # plt.show()

        eq_dict = {}

        for leg in self._b_folders + self._f_folders:
            try:
                files, temperatures, lambdas = self.get_files_temperatures_lambdas(
                    f"{self._work_dir}/{leg}/prod"
                )
            except:
                logging.error(
                    f"could not extract files and temperatures for {self.perturbation}, {self.engine}, {leg} for get Ac."
                )

            try:
                # extract u_nk
                u_nk = BSS.FreeEnergy.Relative._get_u_nk(
                    files, temperatures, self.engine
                )

                ac = A_c([u_nk2series(_u_nk) for _u_nk in u_nk], precision=0.01, tol=2)
                eq_dict[leg] = ac

            except:
                logging.error(
                    f"could not get Ac for {self.perturbation}, {self.engine}, {leg}"
                )
                eq_dict[leg] = None

            logging.info(f"Ac for {leg} was {ac}")

        with open(
            f"{self._pickle_dir}/ac_{self.pickle_extension}.pickle", "wb"
        ) as handle:
            print(handle)
            pickle.dump(eq_dict, handle)

        return eq_dict

    @staticmethod
    def _get_eq_time(df):
        lower = None
        upper = None
        step = None

        series = u_nk2series(df)
        series = slicing(series, lower=lower, upper=upper, step=step)
        df = slicing(df, lower=lower, upper=upper, step=step)

        # calculate statistical inefficiency of series, with equilibrium detection
        t, statinef, Neff_max = _detect_equilibration(series.values)

        equil_time_frac = 1 - len(df[t:]) / len(df)

        return equil_time_frac

    def equil_paired_t(self, update_results=True):
        update_results = validate.boolean(update_results)

        validate.folder_path(f"{self._work_dir}/t-test", create=True)

        fractional_equil_time = self.check_equil_multiwindow_paired_t(
            output_dir=f"{self._work_dir}/t-test",
        )

        if fractional_equil_time and update_results:
            logging.info(
                f"Reanalysing results with equilibration time set to {fractional_equil_time*100}%..."
            )
            self.set_options({"truncate lower": fractional_equil_time * 100})
            self.analyse_all_repeats()

        return fractional_equil_time
