# import libraries
import BioSimSpace as BSS

import yaml
import csv
import numpy as np
import math

from ..utils import *
from ._dictionaries import *

from typing import Union, Optional

# conversion of constants
from scipy.constants import R, calorie

kJ2kcal = 1 / calorie
R_kJmol = R / 1000
R_kcalmol = R_kJmol * kJ2kcal


class convert:
    """class of static methods for converting data"""

    def __init__(self):
        pass

    @staticmethod
    def convert_M_kcal(
        value: float, err: float, temperature: Optional[Union[int, float]] = 300
    ) -> tuple:
        """convert value into kcal/mol

        Args:
            value (float): value, in M
            err (float): error, in M
            temperature (int, optional): temperature of the simulation. Defaults to 300.

        Returns:
            string: value in kcal/mol
        """

        # gas constant in kcal per Kelvin per mol, exp val is Ki converted into M
        # value is /1 as the concentration of substrate, log result has no units, units from RT
        kcal_val = R_kcalmol * temperature * np.log(value)

        # propagate the error
        # if x = ln(a)
        # errx = erra/a
        error = err / value
        # A is RT
        kcal_err = abs(R_kcalmol * temperature * error)

        return kcal_val, kcal_err

    @staticmethod
    def _read_yml_kcal(
        exp_file,
    ) -> dict:
        """convert yml file into experimental dictionary of values.

        Args:
            exp_file (str): yml file. Each key is a ligand, with a 'measurement' that has 'unit', 'value', 'error'. Unit in uM or nM.
            temperature (int, optional): Temperature to use during the conversion. Defaults to 300.
            format_type(str): "Ki" or "IC50". Format of the raw data dict.

        Returns:
            dict: kcal/mol value for each ligand
        """

        exp_file = validate.file_path(exp_file)

        with open(exp_file, "r") as file:
            data = yaml.safe_load(file)  # loads as dictionary

        exper_dict = {}
        for key in data.keys():  # write for each ligand that was in yaml file
            value = None
            error = None
            # check what type of data
            if data[key]["measurement"]["type"].lower().strip() == "dg":
                if data[key]["measurement"]["unit"] == "kcal/mol":
                    value = data[key]["measurement"]["value"]
                    error = data[key]["measurement"]["error"]

            if not error or error == -1:
                # assume error of 0.64 kcal/mol, assuming it was a pKi measurement, https://livecomsjournal.org/index.php/livecoms/article/view/v4i1e1497/1399
                # the log is base 10 so for antilog error propagation: https://sites.science.oregonstate.edu/~gablek/CH361/Propagation.htm
                error = 0.64

            exper_dict[key] = (
                value,
                error,
            )

        return exper_dict

    @staticmethod
    def _yml_into_exper_raw_dict(
        exp_file,
        temperature: Optional[Union[int, float]] = 300,
        format_type="Ki",
    ) -> dict:
        """convert yml file into experimental dictionary of values.

        Args:
            exp_file (str): yml file. Each key is a ligand, with a 'measurement' that has 'unit', 'value', 'error'. Unit in uM or nM.
            temperature (int, optional): Temperature to use during the conversion. Defaults to 300.
            format_type(str): "Ki" or "IC50". Format of the raw data dict.

        Returns:
            dict: kcal/mol value for each ligand
        """

        exp_file = validate.file_path(exp_file)
        temperature = validate.is_float(temperature)

        with open(exp_file, "r") as file:
            data = yaml.safe_load(file)  # loads as dictionary

        exper_raw_dict = {}
        for key in data.keys():  # write for each ligand that was in yaml file
            # check what type of data
            if data[key]["measurement"]["type"].lower().strip() == "ki":
                if format_type == "Ki":
                    # assuming concentration of substrate in the assay is 1, IC50 is approx Ki*2
                    factor = 1
                elif format_type == "IC50":
                    factor = 2
                else:
                    raise ValueError("format_type must be Ki or IC50")

                pki_err_factor = 0.44

            elif data[key]["measurement"]["type"].lower().strip() == "ic50":
                if format_type == "Ki":
                    # assuming concentration of substrate in the assay is 1, IC50 is approx Ki*2
                    factor = 0.5
                elif format_type == "IC50":
                    factor = 1
                else:
                    raise ValueError("format_type must be Ki or IC50")

                pki_err_factor = 0.55
            else:
                raise Exception

            if data[key]["measurement"]["unit"] == "uM":
                magnitude = 10**-6
            elif data[key]["measurement"]["unit"] == "nM":
                magnitude = 10**-9
            else:
                logging.critical("only nM and uM recognised as units!")
                magnitude = 1

            value = data[key]["measurement"]["value"]
            error = data[key]["measurement"]["error"]

            # find the value in M
            value = value * factor * magnitude

            if not error or error == -1:
                # assume error of 0.44 log units for Ki and 0.55 log units for ic50, https://livecomsjournal.org/index.php/livecoms/article/view/v4i1e1497/1399
                # as pKi = -logKi
                # Ki = 10^-pKi
                # the log is base 10 so for antilog error propagation: https://sites.science.oregonstate.edu/~gablek/CH361/Propagation.htm
                # error is errKi = 2.303 * Ki * errpKi(the error factor)
                error = 2.303 * value * pki_err_factor
                # error = 2.303 pki_err_factor * np.log(10) * value
            else:
                error = error * factor * magnitude

            exper_raw_dict[key] = (
                value,
                error,
            )

        return exper_raw_dict

    @staticmethod
    def yml_into_exper_dict(
        exp_file, temperature: Optional[Union[int, float]] = 300
    ) -> dict:
        """convert yml file into experimental dictionary of values.

        Args:
            exp_file (str): yml file. Each key is a ligand, with a 'measurement' that has 'unit', 'value', 'error'. Unit in uM or nM.
            temperature (int, optional): Temperature to use during the conversion. Defaults to 300.

        Returns:
            dict: kcal/mol value for each ligand
        """

        exper_raw_dict = convert._yml_into_exper_raw_dict(exp_file, temperature)

        exper_val_dict = convert.exper_raw_dict_into_val_dict(
            exper_raw_dict, temperature
        )

        return exper_val_dict

    @staticmethod
    def csv_into_exper_dict(
        exp_file: str, temperature: Optional[Union[int, float]] = 300
    ) -> dict:
        """convert csv file into experimental dictionary of values.

        Args:
            exp_file (str): csv file. Has columns 'ligand', 'unit', 'value', 'error'. Unit in uM or nM.
            temperature (int, optional): Temperature to use during the conversion. Defaults to 300.

        Returns:
            dict: kcal/mol value for each ligand
        """

        exp_file = validate.file_path(exp_file)
        temperature = validate.is_float(temperature)

        res_df = pd.read_csv(exp_file)

        exper_raw_dict = {}

        for index, row in res_df.iterrows():
            if row["type"].lower().strip() == "ki":
                # assuming concentration of substrate in the assay is 1, IC50 is approx Ki*2
                factor = 1
            elif row["type"].lower().strip() == "ic50":
                factor = 0.5
            else:
                factor = 0.5
                logging.error(
                    "the type of data was not recognised. Must be IC50 or Ki. Assuming IC50 ..."
                )

            if row["unit"].strip() == "uM":
                magnitude = 10**-6
            elif row["unit"].strip() == "nM":
                magnitude = 10**-9
            else:
                logging.critical("only nM and uM recognised as units!")
                magnitude = 1

            # get this value as Ki in M
            value = float(row["value"].strip()) * factor * magnitude

            try:
                err = float(row["error"].strip()) * factor * magnitude
            except:
                err = None

                exper_raw_dict[row["ligand"]] = (value, err)

        exper_val_dict = convert.exper_raw_dict_into_val_dict(
            exper_raw_dict, temperature
        )

        return exper_val_dict

    @staticmethod
    def exper_raw_dict_into_val_dict(
        exper_raw_dict: dict, temperature: Optional[Union[int, float]] = 300
    ) -> dict:
        """convert raw exp data dict in uM to kcal/mol

        Args:
            exper_raw_dict (dict): the experimental data in uM with {ligand:(value, error)}
            temperature (int, optional): temperature for conversion. Defaults to 300.

        Returns:
            _type_: _description_
        """

        exper_raw_dict = validate.dictionary(exper_raw_dict)
        temperature = validate.is_float(temperature)

        exper_val_dict = {}

        for key in exper_raw_dict.keys():
            lig = str(key)

            exp_val = float(
                exper_raw_dict[key][0]
            )  # this is in IC50 from reading in the files
            try:
                err_val = float(exper_raw_dict[key][1])
            except:
                err_val = None
            # if there is no error, assign based on https://doi.org/10.1016/j.drudis.2009.01.012
            # ie that assay error is usually 0.3 log units standard deviation pIC50 or a factor of 2 in IC50
            if not err_val:  # ie there is no provided error
                # median standard deviation is 0.3
                err_val = 0.3 + np.log(10) * exp_val

            # convert into kcal mol
            kcal_val, kcal_err = convert.convert_M_kcal(
                exp_val, err_val, temperature=temperature
            )

            # add to dict
            exper_val_dict.update({lig: (kcal_val, kcal_err)})

        return exper_val_dict

    @staticmethod
    def cinnabar_file(
        results_files: list,
        exper_val: Union[dict, str],
        output_file: str,
        perturbations: Optional[list] = None,
        method: Optional[str] = None,
    ):
        """convert results files into format needed for cinnabar. If multiple results files, uses the average of a perturbation.

        Args:
            results_files (list): list of results files
            exper_val (dict or str): dict of experimental values or yml file of experimental values.
            output_file (str): output file path
            perturbations (list, optional): list of perturbations to include. Defaults to None.
            method (str, optional): name of the method to consider.
        """
        # files is a list of files
        results_files = validate.is_list(results_files, make_list=True)
        # output file

        if exper_val:
            # first check if the experimental values are a dict or a file
            try:
                exper_val_dict = validate.dictionary(exper_val)
            except:
                validate.file_path(exper_val)
                logging.info("input is a file, will convert this into a dict...")
                logging.info("please check that the conversion of values is okay.")
                exper_val_dict = convert.yml_into_exper_dict(exper_val)

            add_exper_values = True

        else:
            add_exper_values = False

        # write to a csv file
        with open(f"{output_file}.csv", "w") as cinnabar_data_file:
            writer = csv.writer(cinnabar_data_file, delimiter=",")

            if add_exper_values:
                # first, write the experimental data
                writer.writerow(["# Experimental block"])
                writer.writerow(["# Ligand", "expt_DDG", "expt_dDDG"])

                # write in kcal/mol
                for lig in exper_val_dict.keys():
                    writer.writerow(
                        [lig, f"{exper_val_dict[lig][0]}", f"{exper_val_dict[lig][1]}"]
                    )

            # second write the perturbation data
            writer.writerow([" "])
            writer.writerow(["# Calculated block"])
            writer.writerow(
                [
                    "# Ligand1",
                    "Ligand2",
                    "calc_DDG",
                    "calc_dDDG(MBAR)",
                    "calc_dDDG(additional)",
                ]
            )

            # need to write the average of the data, otherwise cinnabar just uses the last entry
            comp_diff_dict = make_dict.comp_results(
                results_files,
                perturbations=perturbations,
                method=method,
            )

            # write to file
            for key in comp_diff_dict:
                lig_0 = key.split("~")[0]
                lig_1 = key.split("~")[1]
                comp_ddG = comp_diff_dict[key][0]
                comp_err = comp_diff_dict[key][1]

                if comp_err == 0:
                    logging.error(
                        f"when converting to the cinnanar file, the error for {key} in {output_file} is 0. This can create issues later for network wide analysis, so setting as 0.001."
                    )
                    comp_err = 0.0001

                if not comp_ddG:
                    pass
                elif math.isnan(comp_ddG):
                    pass
                else:
                    if perturbations:
                        pert = f"{lig_0}~{lig_1}"
                        if pert in perturbations:
                            writer.writerow([lig_0, lig_1, comp_ddG, comp_err, "0.0"])
                        else:
                            pass

                    else:
                        writer.writerow([lig_0, lig_1, comp_ddG, comp_err, "0.0"])

    @staticmethod
    def _convert_html_mbarnet_into_dict(file):
        try:
            from bs4 import BeautifulSoup
            import re
            import json
            import execjs
        except Exception as e:
            logging.critical(f"{e}")
            logging.critical("cannot import for converting html mbarnet to dictionary.")

        file = validate.file_path(file)

        with open(file, "r") as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Define a regular expression pattern to extract the data array from the JavaScript code
        pattern = re.compile(r"var idata = (.*?);", re.DOTALL)

        # Search for the pattern in the HTML content
        match = pattern.search(html_content)

        if match:
            data_js = match.group(1)
            # Replace single quotes with double quotes in the JSON data
            data_js = data_js.replace("\n", "")
            data_js = data_js.replace("'", '"')

            javascript_code = f"""
            var gdata = {data_js};
            var jsonString = JSON.stringify(gdata);
            jsonString;
            """

            # Run JavaScript code
            try:
                context = execjs.compile(javascript_code)
                json_string = context.eval("jsonString")
            except Exception as e:
                print(f"Error executing JavaScript: {e}")

            try:
                # Parse the JSON data into a dictionary
                data = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
        else:
            print("Data pattern not found.")

        ligands_dict = {}

        for lig in data:
            # from end as different order if experimental data file
            # (Expt, CFE_Expt), CFE, dCFE, LMI, AvgCC, name
            ligands_dict[lig[-1]] = (lig[-5], lig[-4])

        # UFE, dUFE, CFE, dCFE, Shift, LME, OFC2, AvgCC, MaxCC, ErrMsgs, Outliers

        # # the below is if the search pattern is dgata, however this has no errors
        # # seperate into perturbations and ligands
        # perturbations_dict = {}
        # ligands_dict = {}

        # # CONFE : Constrained net free energy (complexed-solvated) in kcal/mol.
        # # AVGCC : Average CC for all cycles traversing this node or edge.
        # # LMI : Lagrange multiplier index (unitless).
        # for key in gdata_dict.keys():
        #     if "~" in key:
        #         use_dict = perturbations_dict
        #     else:
        #         use_dict = ligands_dict

        #     use_dict[key] = {}
        #     use_dict[key]["CONFE"] = gdata_dict[key]["CONFE"]["size"]
        #     use_dict[key]["AVGCC"] = gdata_dict[key]["AVGCC"]["size"]
        #     use_dict[key]["LMI"] = gdata_dict[key]["LMI"]["size"]

        return ligands_dict
