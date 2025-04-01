# import libraries
import BioSimSpace as BSS
import csv
import numpy as np
import math
import pandas as pd
from scipy.stats import sem as sem
import csv
import numpy as np
import pandas as pd
import logging

from typing import Union, Optional

from ..utils import *
from ._analysis import *


class make_dict:
    """class of static methods for making dicts to analyse the results"""

    def __init__():
        pass

    @staticmethod
    def comp_results(
        results_files: Optional[list] = None,
        perturbations: Optional[list] = None,
        engine: Optional[str] = None,
        name: Optional[str] = None,
        method: Optional[str] = None,
        ana_protocol: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> dict:
        """write results files into different file or dictionary for certain perturbations.

        Args:
            results_files (list, optional): list of results files (for the repeats). Defaults to None.
            perturbations (list, optional): list of perturbations to use. Defaults to None.
            engine (str, optional): engine to use. Defaults to None. Else will get average result for all entries regardless of engine.
                                    Can also be used for the other results methods.
            output_file (str, optional): output file to write. Defaults to None.

        Returns:
            dict: dictionary of the computed differences (RBFE)
        """

        # check if list, if not make a list
        results_files = validate.is_list(results_files, make_list=True)
        for file in results_files:
            validate.file_path(file)

        if perturbations:
            perturbations = validate.is_list(perturbations)
            make_pert_list = False
        else:
            make_pert_list = True
            perturbations = []

        if method:
            method = validate.string(method)

        if name:
            name = validate.string(name)

        if engine:
            try:
                engine = validate.engine(engine)
            except:
                engine = validate.string(engine)

        if ana_protocol:
            ana_protocol = validate.analysis_protocol(ana_protocol)
            ana_string = analyse.file_ext(ana_protocol.dictionary())
        else:
            ana_string = "unspecified"

        if engine and name:
            raise ValueError(
                "can only have engine or name currently for nameing")

        # make a dictionary with the results of the files
        comp_dict_list = {}
        comp_err_dict_list = {}

        if make_pert_list:
            perturbations, ligs = get_info_network(results_files=results_files)

        for pert in perturbations:
            comp_dict_list[pert] = []
            comp_err_dict_list[pert] = []

        # append for results file
        for res_file in results_files:
            res_df = pd.read_csv(res_file)
            # drop any none values in freenrg
            res_df = res_df[res_df["freenrg"].notna()]
            try:
                res_df = res_df[res_df["freenrg"].str.contains("nan") == False]
            except:
                pass
            for index, row in res_df.iterrows():
                if method:
                    if method.lower() == str(row["method"]).strip().lower():
                        pass
                    else:
                        continue

                if engine:
                    if engine == row["engine"].strip():
                        pass
                    else:
                        continue

                lig_0 = row["lig_0"]
                lig_1 = row["lig_1"]
                pert = f"{lig_0}~{lig_1}"

                if pert in perturbations:
                    if not isinstance(row["freenrg"], float):
                        # to convert as it will be a string
                        ddG = BSS.Types.Energy(
                            float(row["freenrg"].split()[0]
                                  ), row["freenrg"].split()[-1]
                        ).value()
                        ddG_unit = row["freenrg"].split()[-1]
                    else:
                        ddG = row["freenrg"]
                    if not isinstance(row["error"], float):
                        ddG_err = BSS.Types.Energy(
                            float(row["error"].split()[0]
                                  ), row["error"].split()[-1]
                        ).value()
                        ddG_err_unit = row["error"].split()[-1]
                    else:
                        ddG_err = row["error"]

                    # Append the value in list if not zero
                    if ddG:
                        comp_dict_list[pert].append(ddG)
                        comp_err_dict_list[pert].append(ddG_err)

                else:
                    pass

        # now calculate all the avg and SEM for the network perturbations
        # put these into a dictionary
        comp_diff_dict = {}

        for pert in perturbations:
            lig_0 = pert.split("~")[0]
            lig_1 = pert.split("~")[1]

            # check if the perturbations calculated are also those in the network file and if any are missing
            try:
                # find the values in the dictionary
                ddGs = comp_dict_list[pert]
                ddGs_error = comp_err_dict_list[pert]
                # calculate the average and the error
                comp_ddG = np.average(ddGs)
                # comp_ddG = np.average([ddG.value() for ddG in ddGs])
                if len(ddGs) == 1:
                    comp_err = ddGs_error[0]
                    # comp_err = ddGs_error.value()
                else:
                    comp_err = sem(ddGs)
                    # comp_err = sem([ddG.value() for ddG in ddGs])

            # if unable to calculate one of the perturbations, this is a None value.
            except:
                comp_ddG = None
                comp_err = None

            # update the dictionary for plotting later
            comp_diff_dict.update({pert: (comp_ddG, comp_err)})

        if output_file:
            # write these to a csv file
            with open(f"{output_file}.csv", "w+") as comp_pert_file:
                writer = csv.writer(comp_pert_file, delimiter=",")
                writer.writerow(
                    [
                        "lig_0",
                        "lig_1",
                        "freenrg",
                        "error",
                        "engine",
                        "analysis",
                        "method",
                    ]
                )

                for key, value in comp_diff_dict.items():
                    lig_0 = key.split("~")[0]
                    lig_1 = key.split("~")[1]
                    comp_ddG = value[0]
                    comp_err = value[1]

                    if engine:
                        writer.writerow(
                            [
                                lig_0,
                                lig_1,
                                comp_ddG,
                                comp_err,
                                engine,
                                ana_string,
                                str(method),
                            ]
                        )
                    elif name:
                        writer.writerow(
                            [
                                lig_0,
                                lig_1,
                                comp_ddG,
                                comp_err,
                                name,
                                ana_string,
                                str(method),
                            ]
                        )

        return comp_diff_dict

    @staticmethod
    def value_list_from_files(
        results_files: list, header: Optional[str] = "error"
    ) -> list:
        """get list of errors from the files. Eg used for error histogram plotting.

        Args:
            results_files (list): list of results files
            header (str): name of header in file to make list of. Defaults to error.

        Returns:
            list: list of values.
        """

        results_files = validate.is_list(results_files)
        for file in results_files:
            validate.file_path(file)

        error_list = []

        # append for results file
        for res_file in results_files:
            res_df = pd.read_csv(res_file)
            for index, row in res_df.iterrows():
                # assume here as this is normal format of files
                if not isinstance(row[header], float):
                    ddG_err = BSS.Types.Energy(
                        float(row[header].split()[0]), row[header].split()[-1]
                    ).value()
                else:
                    ddG_err = row[header]

                error_list.append(ddG_err)

        return error_list

    @staticmethod
    def experimental_from_freenrgworkflows(
        experimental_DGs, ligands: list, perturbations: list
    ) -> tuple:
        """get the experimental dicts from the freenergworkflows

        Args:
            experimental_DGs (freenergworkflows experimental): from frreenergworkflows
            ligands (list): list of ligands
            perturbations (list): list of perturbations

        Returns:
            tuple: (experimental difference dict, experimental value dict)
        """

        ligands = validate.is_list(ligands)
        perturbations = validate.is_list(perturbations)

        exper_val_dict = make_dict._from_freenrgworkflows_experimental_val(
            experimental_DGs, ligands
        )
        exper_diff_dict = make_dict._from_freenrgworkflows_experimental_diff(
            exper_val_dict, perturbations
        )

        return exper_val_dict, exper_diff_dict

    @staticmethod
    def _from_freenrgworkflows_experimental_val(
        experimental_DGs, ligands: list
    ) -> dict:
        """get the experimental value dict from the freenergworkflows

        Args:
            experimental_DGs (freenergworkflows experimental): from frreenergworkflows
            ligands (list): list of ligands

        Returns:
            dict: dictionary of values.
        """

        # create a dictionary for the experimental values
        exper_val_dict = {}

        # convert the list of dicitonaries from freenrgworkflows into a single dictionary
        for lig_dict in experimental_DGs:
            lig_name = list(lig_dict.keys())[0]
            exper = lig_dict[lig_name]
            exper_err = lig_dict["error"]
            exper_val_dict.update({lig_name: (exper, exper_err)})

        # add any ligands that are in the ligands file but dont have experimental values for
        for lig_name in ligands:
            if lig_name in exper_val_dict:
                pass
            else:
                exper_val_dict.update({lig_name: (None, None)})

        return exper_val_dict

    @staticmethod
    def _from_freenrgworkflows_experimental_diff(
        exper_val_dict, perturbations: list
    ) -> dict:
        """get the experimental difference dict from the freenergworkflows

        Args:
            experimental_DGs (freenergworkflows experimental): from frreenergworkflows
            perturbations (list): list of perturbations

        Returns:
            dict: dictionary of diff based on provided perturbations.
        """

        # we can also create a dictionary with all the experimental values for the perturbations
        exper_diff_dict = {}

        # calculate the experimental RBFEs

        for pert in perturbations:
            lig_0 = pert.split("~")[0]
            lig_1 = pert.split("~")[1]
            # exclude from calculating if one of the ligands is not available
            if exper_val_dict[lig_0][0] is None or exper_val_dict[lig_1][0] is None:
                exper_ddG = None
                exper_err = None
                exper_diff_dict.update({pert: (None, None)})
            # if experimental data is available, calculate experimental perturbation and propagate
            else:
                exper_ddG = exper_val_dict[lig_1][0] - exper_val_dict[lig_0][0]
                exper_err = math.sqrt(
                    math.pow(exper_val_dict[lig_0][1], 2)
                    + math.pow(exper_val_dict[lig_1][1], 2)
                )
                exper_diff_dict.update({pert: (exper_ddG, exper_err)})

        return exper_diff_dict

    @staticmethod
    def from_freenrgworkflows_network_analyser(computed_relative_DGs) -> dict:
        """convert freenergworkflows into a dictionary

        Args:
            computed_relative_DGs (freenergworkflows computed): from freenergworkflows

        Returns:
            dict: dictionary of freenerg values per ligand
        """

        freenrg_dict = {}

        # append computed freenrg and error.
        for item in computed_relative_DGs:
            ligand = list(item.keys())[0]
            freenrg = list(item.values())[0]
            error = list(item.values())[1]

            freenrg_dict.update({ligand: (freenrg, error)})

        return freenrg_dict

    @staticmethod
    def from_cinnabar_network_edges(
        network, calc_exp: str, perturbations: list
    ) -> dict:
        """get a dictionary of results from a cinnabar network

        Args:
            network (): A cinnabar network for the perturbation network.
            calc_exp (str): whether to get the calculated 'calc' or experimental 'exp' data.
            perturbations (list): perturbations to consider

        Raises:
            ValueError: calc_exp must be either 'calc' or 'exp'

        Returns:
            dict: freenreg dict of the {pert:(value, error)}
        """

        if calc_exp not in ["calc", "exp"]:
            raise ValueError("calc_exp must be either 'calc' or 'exp'")

        freenrg_dict = {}

        name_dict = {}
        for node in network.graph.nodes(data=True):
            name_dict.update({node[0]: node[1]["name"]})

        for edge in network.graph.edges(data=True):
            lig_0 = name_dict[edge[0]]
            lig_1 = name_dict[edge[1]]
            pert = f"{lig_0}~{lig_1}"
            anti_pert = f"{lig_1}~{lig_0}"
            if pert in perturbations:
                pert_name = pert
                add_dict = True
            elif anti_pert in perturbations:
                pert_name = anti_pert
                add_dict = True
            else:
                add_dict = False

            if add_dict:
                try:
                    freenrg_dict.update(
                        {
                            pert_name: (
                                edge[2][f"{calc_exp}_DDG"],
                                edge[2][f"{calc_exp}_dDDG"],
                            )
                        }
                    )
                except Exception as e:
                    logging.exception(e)
                    logging.error(
                        f"{pert_name} value not computed. cannot add to dictionary"
                    )

        return freenrg_dict

    @staticmethod
    def from_cinnabar_network_node(
        network, calc_exp: str, normalise: Optional[bool] = False
    ) -> dict:
        """get value per ligand (node) from a cinnabar network as a dictionary.

        Args:
            network (): A cinnabar network for the perturbation network.
            calc_exp (str): whether to get the calculated 'calc' or experimental 'exp' data.
            normalise (bool, optional): whether to normalise the data. Defaults to False.

        Raises:
            ValueError: calc_exp must be either 'calc' or 'exp'

        Returns:
            dict: freenreg dict of the {ligand:(value, error)}
        """

        if calc_exp not in ["calc", "exp"]:
            raise ValueError("calc_exp must be either 'calc' or 'exp'")

        freenrg_dict = {}

        for node in network.graph.nodes(data=True):
            try:
                freenrg_dict.update(
                    {
                        node[1]["name"]: (
                            node[1][f"{calc_exp}_DG"],
                            node[1][f"{calc_exp}_dDG"],
                        )
                    }
                )
            except:
                logging.warning(
                    f"{node[1]['name']} value not computed. cannot add to dictionary"
                )

        if normalise:
            normalised_freenrg_dict = make_dict._normalise_data(freenrg_dict)
            return normalised_freenrg_dict
        else:
            return freenrg_dict

    @staticmethod
    def exper_from_ligands(
        exper_val_dict: dict, ligands: list, normalise: Optional[bool] = False
    ) -> dict:
        """make a new dict of experimental values, can normalise.

        Args:
            exper_dict (dict): dictionary of experimental values
            ligands (list): list of ligands
            normalise (bool, optional): whether to normalise the values. Defaults to False.

        Returns:
            dict: experimental value dict
        """
        exper_val_dict = validate.dictionary(exper_val_dict)
        ligands = validate.is_list(ligands)
        normalise = validate.boolean(normalise)

        new_exper_val_dict = {}

        for lig in ligands:
            if lig not in exper_val_dict.keys():
                logging.warning(
                    f"{lig} not found in experimental values. Please add to the experimental file."
                )
            else:
                exper_dG = exper_val_dict[lig][0]
                exper_err = exper_val_dict[lig][1]
                new_exper_val_dict.update({lig: (exper_dG, exper_err)})

        if normalise:
            normalised_exper_val_dict = make_dict._normalise_data(
                new_exper_val_dict)
            return normalised_exper_val_dict
        else:
            return new_exper_val_dict

    @staticmethod
    def exper_from_perturbations(exper_val_dict: dict, perturbations: list) -> dict:
        """make experimental dict based on and perturbations

        Args:
            exper_dict (dict): dictionary of experimental values
            perturbations (list): list of perturbations

        Returns:
            dict: experimental differences dict
        """

        exper_diff_dict = {}

        # calculate the experimental RBFEs
        for pert in perturbations:
            lig_0 = pert.split("~")[0]
            lig_1 = pert.split("~")[1]
            try:
                exper_ddG = exper_val_dict[lig_1][0] - exper_val_dict[lig_0][0]
                exper_err = math.sqrt(
                    math.pow(exper_val_dict[lig_0][1], 2)
                    + math.pow(exper_val_dict[lig_1][1], 2)
                )
            except Exception as e:
                logging.error(e)
                logging.error(
                    f"the experimental pert value for {pert} could not be computed and will be left empty. Is the ligand in the experimental file?")
                exper_ddG = None
                exper_err = None

            exper_diff_dict.update({pert: (exper_ddG, exper_err)})

        return exper_diff_dict

    @staticmethod
    def _normalise_data(data: Union[list, dict]) -> Union[list, dict]:
        """normalise the data

        Args:
            data (list or dict): data to normalise

        Returns:
            list or dict: normalised data
        """
        # normalise either a list / np / dict of data

        try:
            data_dict = validate.dictionary(data)
            is_dict = True
        except:
            data = validate.is_list(data)
            is_dict = False

        if is_dict:
            normalised_dict = {}
            values = []
            for val in data_dict.values():
                if isinstance(val, tuple):
                    if val[0]:
                        values.append(float(val[0]))
                else:
                    if val:
                        values.append(float(val))
            avg_val = np.mean(values)
            for key in data_dict:
                try:
                    normalised_dict.update(
                        {key: (float(data_dict[key][0]) -
                            avg_val, data_dict[key][1])}
                    )  # also incl error
                except:
                    pass # if there is no value, can't update so skip

            return normalised_dict

        else:
            avg_val = np.mean(data)
            normalised_data = []
            for val in data:
                normalised_data.append(val - avg_val)

            return normalised_data
