import os
import logging

from ._validate import *


def engine_network(engine: str, file_path: str):
    """Generate a network file for only the engine specified.

    Args:
        engine (str): engine to generate the network for
        file_path (str): network.dat file containing multiple engines
    """
    engine = validate.engine(engine)
    file_path = validate.file_path(file_path)

    try:
        logging.info(
            "using the folder of the file as the location to write the output file..."
        )
        output_file = f"{file_path.rsplit('.',1)[0]}_{engine.lower()}.dat"

    except:
        logging.info(
            "assuming the file path is just network_combined.dat as not provided..."
        )
        file_path = "network_combined.dat"
        if not os.path.exists(file_path):
            raise ValueError(f"{file_path} does not exist in the current folder.")
        output_file = f"network_combined_{engine.lower()}.dat"

    with open(file_path, "r") as file:
        with open(output_file, "w") as f:
            for line in file:
                if engine in line:
                    f.write(f"{line}")


def set_colours(
    colour_dict: Optional[dict] = None, other_results_names: Optional[list] = None
) -> dict:
    """set the colours of the bars or scatter plots.

    Args:
        colour_dict (dict, optional): dicitonary of names and their colours. Defaults to None.

    Returns:
        dict: dictionary of new colours
    """

    if colour_dict:
        colour_dict = validate.dictionary(colour_dict)
    else:
        colour_dict = {}

    if other_results_names:
        other_results_names = validate.is_list(other_results_names, make_list=True)

    other_colours = [
        "limegreen",
        "gold",
        "mediumpurple",
        "darkred",
        "grey",
        "lightsteelblue",
        "peru",
        "plum",
        "papayawhip",
        "honeydew",
    ]

    if other_results_names:
        other_res_list = []
        for res in other_results_names:
            if res not in colour_dict.keys():
                other_res_list.append(res)

        for res, col in zip(other_res_list, other_colours):
            colour_dict[res] = col

    default_colour_dict = {
        "AMBER": "orange",
        "SOMD": "darkturquoise",
        "GROMACS": "orchid",
        "experimental": "midnightblue",
    }

    for key in colour_dict:
        # replace default colour dict keys with those in the passed dictionary
        default_colour_dict[key] = colour_dict[key]

    return default_colour_dict
