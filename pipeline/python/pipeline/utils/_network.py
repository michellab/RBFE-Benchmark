import os
import logging

from ._validate import *


def get_ligands_from_perts(perturbations: list) -> list:
    perturbations = validate.is_list(perturbations)
    ligands = []

    for pert in perturbations:
        lig_0 = pert.split("~")[0]
        lig_1 = pert.split("~")[1]
        if lig_0 not in ligands:
            ligands.append(lig_0)
        if lig_1 not in ligands:
            ligands.append(lig_1)

    return ligands


def get_info_network(
    net_file: Optional[str] = None,
    results_files: Optional[list] = None,
    extra_options: Optional[dict] = None,
) -> tuple:
    """get information about the network from the network file

    Args:
        net_file (str, optional): network file. Defaults to None.
        results_files (list, optional): list of results files. Defaults to None.
        extra_options (dict, optional): extra options (engine or engines). Defaults to None.

    Returns:
        tuple: (perturbations, ligands)
    """
    # get info from a network file for engine
    # For the network that we are considering,
    # we want to get results files with these perturbations from the overall file that contains the large network results (if this is the case).
    # This is just good to have a consistent format of results to analyse.

    use_net_file = False

    if net_file:
        try:
            net_file = validate.file_path(net_file)
            use_net_file = True
        except Exception as e:
            logging.error(e)
            logging.info("can't use net_file, will use results files instead.")

    if not use_net_file:
        try:
            results_files = validate.is_list(results_files, make_list=True)
            for file in results_files:
                validate.file_path(file)
        except Exception as e:
            logging.error(e)
            logging.error(
                "cant use network or results files, please provide one or the other."
            )
            return (None, None)

    # set extra_options variables as defaults
    engines = None

    if extra_options:
        extra_options = validate.dictionary(extra_options)

        if "engine" in extra_options.keys():
            try:
                engines = [validate.engine(extra_options["engine"])]
            except Exception as e:
                logging.error(e)
        if "engines" in extra_options.keys():
            try:
                engines = validate.engines(extra_options["engines"])
            except Exception as e:
                logging.error(e)

    # We also want to create a list of the perturbations in our network.
    perturbations = []

    if use_net_file:
        # use the network file to find the ligands and perturbations
        with open(f"{net_file}", "r") as file:
            lines = (
                line.rstrip() for line in file
            )  # All lines including the blank ones
            lines = (line for line in lines if line)  # Non-blank lines
            for line in lines:
                if engines:
                    use_line = False
                    for engine in engines:
                        if line.split()[-1] == engine:
                            use_line = True
                else:
                    use_line = True

                if "lig0" in line or "lig_0" in line:
                    use_line = False

                if use_line:
                    lig_0 = line.split()[0]
                    lig_1 = line.split()[1]
                    pert = f"{lig_0}~{lig_1}"
                    if pert not in perturbations:
                        perturbations.append(pert)

    else:
        for res_file in results_files:
            # use the network file to find the ligands and perturbations
            with open(f"{res_file}", "r") as file:
                lines = (
                    line.rstrip() for line in file
                )  # All lines including the blank ones
                lines = (line for line in lines if line)  # Non-blank lines
                for line in lines:
                    if engines:
                        use_line = False
                        for engine in engines:
                            if line.split(",")[4] == engine:
                                use_line = True

                    else:
                        use_line = True

                    if "lig0" in line or "lig_0" in line:
                        use_line = False

                    if use_line:
                        lig_0 = line.split(",")[0]
                        lig_1 = line.split(",")[1]
                        pert = f"{lig_0}~{lig_1}"
                        if pert not in perturbations:
                            perturbations.append(pert)

    ligands = get_ligands_from_perts(perturbations)

    return (perturbations, ligands)


def get_info_network_from_dict(res_dict: dict) -> tuple:
    """get list of perturbations and ligands from a dictionary

    Args:
        res_dict (dict): dictionary of free energy results

    Returns:
        tuple: (perturbations, ligands)
    """
    # get info for the network from a perturbation results dictionary

    res_dict = validate.dictionary(res_dict)

    perturbations = []

    for key in res_dict.keys():
        if key not in perturbations:
            perturbations.append(key)

    ligands = get_ligands_from_perts(perturbations)

    return (perturbations, ligands)
