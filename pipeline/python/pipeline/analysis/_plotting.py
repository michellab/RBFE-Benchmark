# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

from scipy.stats import sem as sem
from scipy.stats import bootstrap, norm

from typing import Union, Optional

from ..utils import *
from ._network import *
from ._analysis import *
from ._dictionaries import *


class plotting_engines:
    def __init__(
        self,
        analysis_object=None,
        results_folder: str = None,
    ):
        """for plotting analysis network results.

        Args:
            analysis_object (pipeline.analysis.analysis_network, optional): analysis object that is to be plotted for. Defaults to None.
            results_folder (str, optional): output folder for generated csv files and graph images. Defaults to None.

        Raises:
            ValueError: must provide an analysis network object.
        """

        if analysis_object:
            self._analysis_object = analysis_object
            # get info about things for plotting from analysis
            self._analysis_obj_into_format()
            self._analysis_obj_dicts_into_format()
        else:
            raise ValueError(
                "please provide an analysis object to be plotted for.")

        # place to write results to
        if not results_folder:
            # want to write to the graph directory
            self.results_folder = self._analysis_object.results_folder
            self.graph_folder = self._analysis_object.graph_dir
        else:
            self.results_folder = validate.folder_path(
                results_folder, create=True)
            self.graph_folder = validate.folder_path(
                f"{results_folder}/graphs", create=True
            )

        # set the colours
        self.colours = set_colours(
            other_results_names=self.other_results_names)

        # convert the dictionaries into dataframes for plotting
        self._analysis_dicts_to_df()

    def _analysis_obj_into_format(self):
        """turn the passed pipeline.analysis.analysis_network object into format for this class"""

        ana_obj = self._analysis_object

        # analysis information
        self.engines = sorted(ana_obj.engines)
        self.ligands = ana_obj.ligands
        self.perturbations = ana_obj.perturbations

        # for other results
        self.other_results_names = ana_obj.other_results_names

        # name of all options
        self._eng_other_list()

        # file and network extensions
        self.file_ext = None
        self.net_ext = None
        self.file_extension()
        self.network_extension()

    def _analysis_obj_dicts_into_format(self):
        ana_obj = self._analysis_object

        # dictionaries of engines for plotting from cinnabar
        self.calc_val_dict = ana_obj.cinnabar_calc_val_dict
        self.exper_val_dict = ana_obj.cinnabar_exper_val_dict
        # if these have failed to compute the dict value will be empty for that engine.

        # make pert dict from the self computed pert values
        self.calc_pert_dict = ana_obj.calc_pert_dict

        self.calc_bound_dict = ana_obj.calc_bound_dict
        self.calc_free_dict = ana_obj.calc_free_dict

        # experimental calculated directly from exp values (for bar)
        self.all_exper_pert_dict = ana_obj.exper_pert_dict
        self.all_exper_val_dict = ana_obj.normalised_exper_val_dict

        # for convergence
        self.spert_results_dict = ana_obj.spert_results_dict
        self.spert_bound_dict = ana_obj.spert_bound_dict
        self.spert_free_dict = ana_obj.spert_free_dict
        self.epert_results_dict = ana_obj.epert_results_dict
        self.epert_bound_dict = ana_obj.epert_bound_dict
        self.epert_free_dict = ana_obj.epert_free_dict

    def file_extension(self, file_ext: str = None) -> str:
        """set or return the file extension

        Args:
            file_ext (str, optional): file extension to set. Defaults to None.

        Returns:
            str: the file extension
        """

        if file_ext:
            self.file_ext = validate.string(file_ext)

        elif not self.file_ext:
            file_ext = self._analysis_object.file_ext

            if file_ext == ".+":
                file_ext = "na"

            self.file_ext = file_ext

        else:
            pass

        return self.file_ext

    def network_extension(self, net_ext: str = None) -> str:
        """set or return the network extension

        Args:
            file_ext (str, optional): network extension to set. Defaults to None.

        Returns:
            str: the network extension
        """

        if net_ext:
            self.net_ext = validate.string(net_ext)
        elif not self.net_ext:
            net_ext = self._analysis_object.net_ext
            self.net_ext = net_ext
        else:
            pass

        return self.net_ext

    def _eng_other_list(self) -> list:
        """list of engines and any other results and the experimental.

        Returns:
            list: names list
        """

        names_list = []

        # all possible engines
        for eng in self.engines:
            names_list.append(eng)
        # all other results
        for name in self.other_results_names:
            names_list.append(name)
        # the experimental
        names_list.append("experimental")

        self.names_list = names_list

        return names_list

    def _validate_in_names_list(self, name: str, make_list: bool = False) -> list:
        """validate if the name is in the names list

        Args:
            name (str): the name to validate
            make_list (bool, optional): whether to make into a list, default is False.

        Raises:
            ValueError: if not in names list

        Returns:
            str: the validated name or names list.
        """

        names = validate.is_list(name, make_list=True)

        for name in names:
            name = validate.string(name)
            if name not in (self.names_list):
                raise ValueError(f"{name} must be in {self.names_list}")

        if not make_list:
            names = names[0]

        return names

    def _analysis_dicts_to_df(self):
        """turn the dicts from the analysis network to ones for the plotting object"""

        # create the overall dict from all the passed results
        self._overall_dict()
        self.freenrg_df_dict = {}
        for name in self.names_list:
            self.freenrg_df_dict.update({name: None})
        self._calc_all_dict_to_df()

    def _overall_dict(self) -> dict:
        """create an overall dict with the information passed from the analysis network object.

        Returns:
            dict: values dict for all the engines, other results, and experimental. Each dict in the dict contains the 'perts', 'ligs', 'pert_results', and 'val_results'.
        """

        values_dict = {}
        for name in self.names_list:
            values_dict.update({name: {}})

        # run for all engines with selected network and populate the dictionary for plotting
        for eng in self.engines + self.other_results_names:
            try:
                # get perts and ligands for each engine
                pert_lig = get_info_network_from_dict(self.calc_pert_dict[eng])
                values_dict[eng]["perts"] = pert_lig[0]
                values_dict[eng]["ligs"] = pert_lig[1]
                # put results into values dict
                values_dict[eng]["pert_results"] = self.calc_pert_dict[eng]
                values_dict[eng]["val_results"] = self.calc_val_dict[eng]
                
                try:
                    values_dict[eng]["bound_results"] = self.calc_bound_dict[eng]
                    values_dict[eng]["free_results"] = self.calc_free_dict[eng]
                except: # if added later there may not be bound and free results
                    logging.error("didnt free bound")
                    values_dict[eng]["bound_results"] = {
                        x: (None, None) for x in pert_lig[0]
                    }
                    values_dict[eng]["free_results"] = {
                        x: (None, None) for x in pert_lig[0]
                    }

            except Exception as e:
                values_dict[eng]["perts"] = [None]
                values_dict[eng]["ligs"] = [None]
                values_dict[eng]["pert_results"] = [None]
                values_dict[eng]["val_results"] = [None]
                values_dict[eng]["bound_results"] = [None]
                values_dict[eng]["free_results"] = [None]
                logging.error(e)
                logging.error(
                    f"could not convert {eng} values for plotting. None will be used. Was earlier analysis okay?"
                )

        values_dict["experimental"]["perts"] = self.perturbations
        values_dict["experimental"]["ligs"] = self.ligands
        values_dict["experimental"]["pert_results"] = self.all_exper_pert_dict
        values_dict["experimental"][
            "val_results"
        ] = self.all_exper_val_dict  # normalised data
        # all bound and free values as None for matching df later.
        values_dict["experimental"]["bound_results"] = {
            x: (None, None) for x in self.all_exper_pert_dict
        }
        values_dict["experimental"]["free_results"] = {
            x: (None, None) for x in self.all_exper_pert_dict
        }

        self.values_dict = values_dict

        return values_dict

    def _calc_all_dict_to_df(self):
        """for all identified engines, other results, and experimental, convert the dictionary into a dataframe."""

        for name in self.names_list:
            self._dict_to_df(x_name=name)

    def _dict_to_df(self, x_name: Optional[str] = "experimental") -> dict:
        """turn a dictionary of results into a dataframe

        Args:
            x_name (str, optional): name of the dictionary to convert. Defaults to "experimental".

        Returns:
            dict: dictionary of pandas dataframe, which is a dictionary (other names) of the 'pv' (pert or val) to match the values found in the x_name dictionary.
        """

        # calculate df w respect to each other value

        x_name = self._validate_in_names_list(x_name)

        freenrg_df_dict = {}

        to_convert_list = [x for x in self.names_list]
        # to_convert_list.remove(x_name)
        for name in to_convert_list:
            freenrg_df_dict.update({name: {}})

        # construct dict with experimental freenrg and error and computed
        for name in to_convert_list:  # will do this for engines and other results
            for pv in ["pert", "val", "bound", "free"]:
                if pv == "pert":
                    which_list = "perts"
                elif pv == "val":
                    which_list = "ligs"
                else:
                    which_list = "perts"

                freenrg_df = self.match_dicts_to_df(
                    self.values_dict[x_name][f"{pv}_results"],
                    self.values_dict[name][f"{pv}_results"],
                    x_name,
                    "calc",
                    self.values_dict[x_name][which_list],
                )

                freenrg_df_dict[name][pv] = freenrg_df

                # save our results to a file that can be opened in e.g. Excel.
                freenrg_df.to_csv(
                    f"{self.results_folder}/{name}_vs_{x_name}_{pv}_results_table_{self.file_ext}_{self.net_ext}.csv"
                )

        self.freenrg_df_dict[x_name] = freenrg_df_dict

        return freenrg_df_dict

    @staticmethod
    def match_dicts_to_df(
        dict_x: dict,
        dict_y: dict,
        x_name: str,
        y_name: str,
        values: Optional[list] = None,
    ):
        """match two dictionaries into one dataframe (to be used for plotting)

        Args:
            dict_x (dict): dictionary of values for x, in format value : (freenerg, err)
            dict_y (dict): dictionary of values for y, in format value : (freenerg, err)
            x_name (str): name of the x values ( eg experimental )
            y_name (str): name of the y values
            values (list, optional): list of dict values to convert. Defaults to None.

        Returns:
            _type_: _description_
        """

        freenrg_dict = {}

        if values:
            values = validate.is_list(values)
        else:
            values = dict_x.keys()

        for value in values:
            try:
                x_ddG = dict_x[value][0]
                x_err = dict_x[value][1]
                y_ddG = dict_y[value][0]
                y_err = dict_y[value][1]
                freenrg_dict[value] = [x_ddG, x_err, y_ddG, y_err]
            except Exception as e:
                logging.error(e)
                logging.error(
                    f"{value} not in both dicts, {x_name} and {y_name}")

        freenrg_df = pd.DataFrame(
            freenrg_dict,
            index=[
                f"freenrg_{x_name}",
                f"err_{x_name}",
                f"freenrg_{y_name}",
                f"err_{y_name}",
            ],
        ).transpose()

        return freenrg_df

    @staticmethod
    def _prune_perturbations(
        df: pd.DataFrame, perturbations: list, remove: bool = False
    ) -> pd.DataFrame:
        """keep or remove perturbations in list from the dataframe

        Args:
            df (pandas.dataframe): dataframe of results
            perturbations (list): list of perturbations that want to keep.
            remove (boolean): whether to keep or remove the perturbations in the list

        Returns:
            df: pruned dataframe
        """

        remove = validate.boolean(remove)
        perturbations = validate.is_list(perturbations, make_list=True)

        to_del = []
        to_keep = []

        if remove:
            # delete specified perturbations from the dataframe
            for pert in df.index:
                if pert in perturbations:
                    to_del.append(pert)
                else:
                    to_keep.append(pert)
        else:
            # keep only specified perturbations in the dataframe
            for pert in df.index:
                if pert not in perturbations:
                    to_del.append(pert)
            for pert in perturbations:
                if pert not in df.index:
                    to_keep.append(pert)

        for pert in to_del:
            df = df.drop(index=[pert])

        # fill in with none values for consistent plotting if to keep
        for pert in to_keep:
            if pert not in df.index:
                df.loc[pert] = [0, 0, 0, 0]

        return df

    @staticmethod
    def get_bar_spacing(names: list = None) -> tuple:
        names = validate.is_list(names)

        # so experimental is the last thing plotted
        if "experimental" in names:
            names.remove("experimental")
            names.append("experimental")

        placement_dict = {}

        if len(names) == 10:
            width = 0.10  # set bar width
            placement = [
                -width * (9 / 2),
                -width * (7 / 2),
                -width * (5 / 2),
                -width * (3 / 2),
                -width * (1 / 2),
                width * (1 / 2),
                width * (3 / 2),
                width * (5 / 2),
                width * (7 / 2),
                width * (9 / 2),
            ]
        elif len(names) == 9:
            width = 0.10  # set bar width
            placement = [
                -width * (8 / 2),
                -width * (6 / 2),
                -width * (4 / 2),
                -width * (2 / 2),
                0,
                width * (2 / 2),
                width * (4 / 2),
                width * (6 / 2),
                width * (8 / 2),
            ]
        elif len(names) == 8:
            width = 0.10  # set bar width
            placement = [
                -width * (7 / 2),
                -width * (5 / 2),
                -width * (3 / 2),
                -width * (1 / 2),
                width * (1 / 2),
                width * (3 / 2),
                width * (5 / 2),
                width * (7 / 2),
            ]
        elif len(names) == 7:
            width = 0.12  # set bar width
            placement = [
                -width * (6 / 2),
                -width * (4 / 2),
                -width * (2 / 2),
                0,
                width * (2 / 2),
                width * (4 / 2),
                width * (6 / 2),
            ]
        elif len(names) == 6:
            width = 0.12  # set bar width
            placement = [
                -width * (5 / 2),
                -width * (3 / 2),
                -width * (1 / 2),
                width * (1 / 2),
                width * (3 / 2),
                width * (5 / 2),
            ]
        elif len(names) == 5:
            width = 0.14  # set bar width
            placement = [
                -width * (4 / 2),
                -width * (2 / 2),
                0,
                width * (2 / 2),
                width * (4 / 2),
            ]
        elif len(names) == 4:
            width = 0.15  # set bar width
            placement = [
                -width * (3 / 2),
                -width * (1 / 2),
                width * (1 / 2),
                width * (3 / 2),
            ]
        elif len(names) == 3:
            width = 0.23  # set bar width
            placement = [-width * (2 / 2), 0, width * (2 / 2)]
        elif len(names) == 2:
            width = 0.4  # set bar width
            placement = [-width * (1 / 2), width * (1 / 2)]
        elif len(names) == 1:
            width = 0.6  # set bar width
            placement = [0]
        else:
            raise ValueError(
                f"length of the engine list + exp is {(len(names))}. it cannot exceed 10? must have atleast 1 engine/exp."
            )

        for eng, place in zip(names, placement):
            placement_dict.update({eng: place})  # for each engine

        return placement_dict, width

    def _parse_kwargs_graphs(self, graph: str = None, **kwargs):
        graph = validate.string(graph).upper()
        if graph not in ["BAR", "SCATTER", "OUTLIER"]:
            raise ValueError(
                f"graph argument must be bar, scatter or outlier.")

        # default
        y_label = None
        x_label = None
        suptitle = None
        title = None
        include_key = True
        save_fig_location = None
        include_x_error = True
        include_y_error = True

        # check kwargs incase there is plotting info
        for key, value in kwargs.items():
            key = key.upper().replace(" ", "").replace("_", "")
            if key == "YLABEL":
                y_label = value
            if key == "XLABEL":
                x_label = value
            if key == "SUPTITLE":
                suptitle = value
            if key == "TITLE":
                title = value
            if key == "KEY":
                include_key = validate.boolean(value)
            if key == "SAVE":
                save_fig_location = validate.string(f"{value}.png")
            if key == "XERROR":
                include_x_error = validate.boolean(value)
            if key == "YERROR":
                include_y_error = validate.boolean(value)

        return (
            y_label,
            x_label,
            suptitle,
            title,
            include_key,
            save_fig_location,
            include_x_error,
            include_y_error,
        )

    def bar(
        self,
        pert_val: str = None,
        names: Optional[list] = None,
        values: Optional[list] = None,
        **kwargs,
    ):
        """plot a bar plot of the results

        Args:
            pert_val (str, optional): whether plotting 'pert' ie perturbations or 'val' ie values (per ligand result). Defaults to None.
            names (list, optional): engines and other results and experimental to plot for. Defaults to None.
            values (list, optional): list of values (perturbations or ligands) to plot for. Defaults to None.
        """

        pert_val = validate.pert_val(pert_val)

        if names:
            names = validate.is_list(names, make_list=True)
            for eng in names:
                self._validate_in_names_list(eng)
        # if no engines provided, use the defaults that were set based on the analysis object
        else:
            names = self.engines

        bar_spacing, width = plotting_engines.get_bar_spacing(names=names)

        if not values:
            if pert_val == "pert":
                values = self.perturbations
            elif pert_val == "val":
                values = self.ligands
            else:
                values = self.perturbations
        else:
            values = validate.is_list(values)

        # other kwargs
        (
            y_label,
            x_label,
            suptitle,
            title,
            include_key,
            save_fig_location,
            include_x_error,
            include_y_error,
        ) = self._parse_kwargs_graphs(graph="bar", **kwargs)

        plt.rc("font", size=12)
        fig, ax = plt.subplots(figsize=(15, 8))

        # df_dict = freenrg_df_plotting
        exp_df_len = len(
            self._prune_perturbations(
                self.freenrg_df_dict["experimental"]["experimental"][pert_val].fillna(
                    0
                ),
                values,
            )
        )

        for eng in names:
            col = self.colours[eng]
            space = bar_spacing[eng]

            # just always compare to experimental for this
            freenrg_df_plotting = self.freenrg_df_dict["experimental"][eng][
                pert_val
            ].fillna(0)

            # prune df to only have perturbations considered
            # sort so they are all in the same order
            freenrg_df_plotting = self._prune_perturbations(
                freenrg_df_plotting, values
            ).sort_index()

            if len(freenrg_df_plotting) != exp_df_len:
                raise ValueError(
                    "for bar plotting, the length of the used dataframes must be the same. Please pass 'values',\
                                 ie the perts or ligs to plot for, as neccessary to ensure this."
                )

            # determine positions for X axis labels.
            x_locs = np.arange(len(freenrg_df_plotting))

            if include_y_error:
                yerr = freenrg_df_plotting["err_calc"]
            else:
                yerr = None

            # plot both our experimental and FEP free energies using an offset on the x position so bars don't overlap.
            ax.bar(
                x_locs + space,
                height=freenrg_df_plotting["freenrg_calc"],
                width=width,
                yerr=yerr,
                label=eng,
                color=col,
            )

        # plt.xlabel('ΔΔG for experimental (kcal/mol)')
        # plt.ylabel('ΔΔG for calculated (kcal/mol)')
        # format the plot further.
        plt.axhline(color="black")

        if title:
            title = title
        else:
            title = f"Freenrg for {self.file_ext.replace('_',',')}, {self.net_ext.replace('_',',')}"

        plt.suptitle(suptitle, fontsize=20)
        plt.title(title, fontsize=14)

        if y_label:
            y_label = y_label
        else:
            if pert_val == "pert":
                y_label = "$\Delta\Delta$G$_{bind}$ (kcal/mol)"
            elif pert_val == "val":
                y_label = "$\Delta$G$_{bind}$ (kcal/mol)"

        plt.ylabel(y_label)

        if x_label:
            x_label = x_label
        else:
            if pert_val == "pert":
                x_label = "perturbations"
            elif pert_val == "val":
                x_label = "ligands"

        plt.xlabel(x_label)

        plt.xticks(x_locs, freenrg_df_plotting.index, rotation=70, ha="right")

        if include_key:
            plt.legend()
        else:
            pass

        eng_name = self._get_y_name(names)
        if save_fig_location:
            save_fig_location = save_fig_location
        else:
            save_fig_location = f"{self.graph_folder}/barplot_{pert_val}_{self.file_ext}_{self.net_ext}_{eng_name}.png"

        plt.savefig(save_fig_location, dpi=300, bbox_inches="tight")

        return ax

    def scatter(
        self,
        pert_val: str = None,
        y_names: Optional[list] = None,
        x_name: str = "experimental",
        values: Optional[list] = None,
        outliers: bool = False,
        no_outliers: int = 3,
        **kwargs,
    ):
        """plot scatter plot.

        Args:
            pert_val (str, optional): whether plotting 'pert' ie perturbations or 'val' ie values (per ligand result). Defaults to None.
            y_names (list, optional): y_names to plot for. Defaults to None.
            x_name (str, optional): what to plot against. Defaults to "experimental".
            values (list, optional): list of values (perturbations or ligands) to plot for. Defaults to None.

        Raises:
            ValueError: the x_name must be available in the other names list ie have results assosciated with it.
        """

        pert_val = validate.pert_val(pert_val)
        x_name = self._validate_in_names_list(x_name)

        if y_names:
            y_names = validate.is_list(y_names, make_list=True)
            for y_name in y_names:
                y_name = self._validate_in_names_list(y_name)
        else:
            y_names = self.names_list.copy()
            y_names.remove(x_name)

        if not values:
            if pert_val == "pert":
                values = self.perturbations
            elif pert_val == "val":
                values = self.ligands
        else:
            values = validate.is_list(values)

        # if check outliers
        outliers = validate.boolean(outliers)
        no_outliers = validate.integer(no_outliers)

        # other kwargs
        (
            y_label,
            x_label,
            suptitle,
            title,
            include_key,
            save_fig_location,
            include_x_error,
            include_y_error,
        ) = self._parse_kwargs_graphs(graph="scatter", **kwargs)

        # plot a scatter plot
        plt.rc("font", size=20)
        fig, ax = plt.subplots(figsize=(7, 7))

        lines = []

        for y_name in y_names:
            col = self.colours[y_name]

            freenrg_df_plotting = self.freenrg_df_dict[x_name][y_name][
                pert_val
            ].dropna()

            # prune df to only have perturbations considered
            freenrg_df_plotting = self._prune_perturbations(
                freenrg_df_plotting, values)

            x = freenrg_df_plotting[f"freenrg_{x_name}"]
            y = freenrg_df_plotting["freenrg_calc"]

            if include_x_error:
                x_er = freenrg_df_plotting[f"err_{x_name}"]
            else:
                x_er = None
            if include_y_error:
                y_er = freenrg_df_plotting["err_calc"]
            else:
                y_er = None

            # outliers
            if outliers:
                # get an array of the MUE values comparing experimental and FEP values. Take the absolute values.
                mue_values = abs(x - y)

                # find the n ligand names that are outliers.
                outlier_names = mue_values.nlargest(
                    no_outliers).index.values.tolist()
                logging.info(f"outlier names for {y_name} are {outlier_names}")

                # construct a list of labels to annotate the scatterplot with.
                annot_labels = []
                colours = []
                for label in freenrg_df_plotting.index.values:
                    # if the ligand is an outlier, append the name to the annotation labels list.
                    if label in outlier_names:
                        if len(y_names) > 1:
                            annot_labels.append(f"{label}, {y_name}")
                        else:
                            annot_labels.append(f"{label}")
                        colours.append(self.colours["experimental"])
                    else:
                        # if the ligand is not an outlier, append an empty string to the annotation labels list.
                        annot_labels.append("")
                        colours.append(self.colours[y_name])

            scatterplot = [plt.scatter(x, y, zorder=10, c=col)]

            lines += plt.plot(0, 0, c=col, label=y_name)

            plt.errorbar(
                x,
                y,
                yerr=y_er,
                xerr=x_er,
                ls="none",
                lw=0.5,
                capsize=2,
                color="black",
                zorder=5,
            )

            if outliers:
                # then, after generating the figure, we can annotate:
                for i, txt in enumerate(annot_labels):
                    plt.annotate(
                        txt,
                        (
                            freenrg_df_plotting[f"freenrg_{x_name}"].values.tolist()[
                                i]
                            + 0.1,  # x coords
                            freenrg_df_plotting["freenrg_calc"].values.tolist()[
                                i]
                            + 0.1,
                        ),  # y coords
                        size=15,
                        color=self.colours["experimental"],
                    )

        # plot 1/2 kcal bounds:
        plt.fill_between(
            x=[-100, 100],
            y2=[-100.25, 99.75],
            y1=[-99.75, 100.25],
            lw=0,
            zorder=-10,
            alpha=0.3,
            color="grey",
        )
        # upper bound:
        plt.fill_between(
            x=[-100, 100],
            y2=[-99.5, 100.5],
            y1=[-99.75, 100.25],
            lw=0,
            zorder=-10,
            color="grey",
            alpha=0.2,
        )
        # lower bound:
        plt.fill_between(
            x=[-100, 100],
            y2=[-100.25, 99.75],
            y1=[-100.5, 99.5],
            lw=0,
            zorder=-10,
            color="grey",
            alpha=0.2,
        )

        min_lim, max_lim = self._get_bounds_scatter(
            y_names, self.freenrg_df_dict[x_name], pert_val, values, x_name
        )

        # for a scatterplot we want the axis ranges to be the same.
        plt.xlim(min_lim * 1.3, max_lim * 1.3)
        plt.ylim(min_lim * 1.3, max_lim * 1.3)

        plt.axhline(color="black", zorder=1)
        plt.axvline(color="black", zorder=1)

        if include_key:
            labels = [l.get_label() for l in lines]
            plt.legend(lines, labels, loc="upper left")

        if title:
            title = title
        else:
            if outliers:
                title = f"Computed vs {x_name} outliers\nfor {self.file_ext.replace('_',',')}, {self.net_ext.replace('_',',')}"
            else:
                title = f"Computed vs {x_name}\nfor {self.file_ext.replace('_',',')}, {self.net_ext.replace('_',',')}"

        plt.suptitle(suptitle, fontsize=20)
        plt.title(title, fontsize=14)

        if y_label:
            y_label = y_label
        else:
            if pert_val == "pert":
                y_label = "Computed $\Delta\Delta$G$_{bind}$ (kcal/mol)"
            elif pert_val == "val":
                y_label = "Computed $\Delta$G$_{bind}$ (kcal/mol)"

        plt.ylabel(y_label)

        if x_label:
            x_label = x_label
        else:
            if pert_val == "pert":
                x_label = f"{x_name} " + "$\Delta\Delta$G$_{bind}$ (kcal/mol)"
            elif pert_val == "val":
                x_label = f"{x_name} " + "$\Delta$G$_{bind}$ (kcal/mol)"

        plt.xlabel(x_label)

        eng_name = self._get_y_name(y_names)
        if save_fig_location:
            save_fig_location = save_fig_location
        else:
            if outliers:
                save_fig_location = f"{self.graph_folder}/calc_vs_{x_name}_outlierplot_{pert_val}_{self.file_ext}_{self.net_ext}_{eng_name}.png"
            else:
                save_fig_location = f"{self.graph_folder}/calc_vs_{x_name}_scatterplot_{pert_val}_{self.file_ext}_{self.net_ext}_{eng_name}.png"

        plt.savefig(save_fig_location, dpi=300, bbox_inches="tight")

        return ax

    # some functions so cleaner in plotting functions
    @staticmethod
    def _get_y_name(engines: list) -> str:
        """get the engine names for writing the titles and file names

        Args:
            engines (list): list of engines or other results names

        Returns:
            str: name for the plotting and file name
        """

        eng_name = "_".join(str(eng) for eng in engines)

        return eng_name

    @staticmethod
    def _get_bounds_scatter(
        engines: list, freenrg_df_dict: dict, pert_val: str, values: list, name: str
    ):
        """get the upper and lower bounds of the scatter plot based on the results plotted.

        Args:
            engines (list): list of engines and other results.
            freenrg_df_dict (dict): dictionary of dataframes of results
            pert_val (str, optional): whether plotting 'pert' ie perturbations or 'val' ie values (per ligand result).
            values (list): list of values (perturbations or ligands) to plot for. Defaults to None.
            name (str): what is being plotted against.

        Returns:
            tuple: min and max limit for plotting.
        """

        # get the bounds. This can be done with min/max or simply by hand.
        all_freenrg_values_pre = []
        for eng in engines:
            freenrg_df_plotting = freenrg_df_dict[eng][pert_val].dropna()
            freenrg_df_plotting = plotting_engines._prune_perturbations(
                freenrg_df_plotting, values
            )
            x = np.array(freenrg_df_plotting[f"freenrg_{name}"]).tolist()
            y = np.array(freenrg_df_plotting["freenrg_calc"]).tolist()
            all_freenrg_values_pre.append(x)
            all_freenrg_values_pre.append(y)

        print(x)
        print(y)
        all_freenrg_values = []
        for sublist in all_freenrg_values_pre:
            for item in sublist:
                all_freenrg_values.append(item)

        min_lim = min(all_freenrg_values)
        max_lim = max(all_freenrg_values)
        print(min_lim, max_lim)
        return min_lim, max_lim

    def plot_convergence(self, engines: Optional[str] = None):
        engines = validate.engines(engines)

        for engine in engines:
            logging.info(
                f"plotting diff to final result for all perturbations in {engine}..."
            )
            sdf = self.pert_dict_into_convergence_df(
                self.spert_results_dict[engine], plot_error=False, plot_difference=True
            )
            edf = self.pert_dict_into_convergence_df(
                self.epert_results_dict[engine], plot_error=False, plot_difference=True
            )
            analyse.plot_truncated(
                sdf,
                edf,
                f"{self.graph_folder}/plt_truncated_{engine}_difference_forward_reverse_{self.file_ext}.png",
                plot_difference=True,
            )
            logging.info(f"saved images in {self.graph_folder}.")

    @staticmethod
    def pert_dict_into_convergence_df(
        pert_dict: dict, plot_error: bool = False, plot_difference: bool = True
    ) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(pert_dict)
        perts = list(df.columns)
        df = df.reset_index().dropna()

        index_dict = {}
        for x in df["index"]:
            index_dict[x] = []

        for pert in perts:
            x_vals = []
            y_vals = []
            for a, x in zip(df[pert], df["index"]):
                if plot_error:
                    ind = 1
                else:
                    ind = 0
                try:
                    if plot_difference:
                        y_val = df.iloc[-1][pert][ind] - a[ind]
                    else:
                        y_val = a[ind]
                except:
                    y_val = None
                if not index_dict[x]:
                    index_dict[x] = [y_val]
                else:
                    index_dict[x].append(y_val)
                y_vals.append(y_val)
                x_vals.append(x)

        for x in df["index"]:
            try:
                val_list = [x for x in index_dict[x] if pd.notna(x)]
                avg = np.mean(val_list)
                # bc of how its plotted later (ie as inbetween a min and a max added to the avg), need this as the difference to the mean
                min_val = min(val_list) - avg  # is a negative value
                max_val = max(val_list) - avg  # is a positive value
            except:
                avg = None
                min_val = None
                max_val = None

            index_dict[x] = (avg, min_val, max_val)

        df = pd.DataFrame.from_dict(
            index_dict, orient="index", columns=["avg", "min", "max"]
        )
        df = df.dropna()

        return df


class plotting_histogram(plotting_engines):
    def __init__(
        self,
        analysis_object=None,
        results_folder: Optional[str] = None,
    ):
        """for plotting histograms.

        Args:
            analysis_object (pipeline.analysis.analysis_network, optional): analysis object that is to be plotted for. Defaults to None.
            results_folder (str, optional): output folder for generated csv files and graph images. Defaults to None.

        Raises:
            ValueError: must provide an analysis network object.
        """

        if analysis_object:
            self._analysis_object = analysis_object
            # get info about things for plotting from analysis
            self._analysis_obj_into_format()
            self._analysis_obj_files_into_format()
            # remove experimental as not needed for this
            try:
                self.names_list.remove("experimental")
            except:
                pass
        else:
            raise ValueError(
                "please provide an analysis object to be plotted for.")

        # place to write results to
        if not results_folder:
            # want to write to the graph directory
            self.results_folder = self._analysis_object.results_folder
            self.graph_folder = self._analysis_object.graph_dir
        else:
            self.results_folder = validate.folder_path(
                results_folder, create=True)
            self.graph_folder = validate.folder_path(
                f"{results_folder}/graphs", create=True
            )
        # set the colours
        self.colours = set_colours(
            other_results_names=self.other_results_names)

        # set the dictionary for histograms
        self._files_into_error_lists()

        # make empty dicts
        self.best_fit_dict = {}
        for err in self.error_type:
            self.best_fit_dict[err] = {}

    def _analysis_obj_files_into_format(self):
        ana_obj = self._analysis_object

        self._results_files = ana_obj._results_files
        self._results_repeat_files = ana_obj._results_repeat_files
        self._results_free_repeat_files = ana_obj._results_free_repeat_files
        self._results_bound_repeat_files = ana_obj._results_bound_repeat_files
        self._results_value_files = ana_obj._results_value_files

    def _files_into_error_lists(self):
        self.error_dict = {}

        self.error_type = ["SEM_pert", "per_lig", "repeat", "free", "bound"]
        dict_to_use = (
            self._results_files,
            self._results_value_files,
            self._results_repeat_files,
            self._results_free_repeat_files,
            self._results_bound_repeat_files,
        )

        for err, dtu in zip(self.error_type, dict_to_use):
            self.error_dict[err] = {}
            for name in self.names_list:
                try:
                    error_list = make_dict.value_list_from_files(dtu[name])
                    self.error_dict[err][name] = error_list
                except:
                    self.error_dict[err][name] = []

    def histogram(self, name: str = None, type_error: str = "SEM_pert"):
        if type_error not in self.error_type:
            raise ValueError(f"error name must be in {self.error_type}")
        name = self._validate_in_names_list(name)

        # set plot defaults
        plt.rc("font", size=12)
        fig, ax = plt.subplots(figsize=(8, 8))
        col = self.colours[name]
        x_list = self.error_dict[type_error][name]
        x = [x_val for x_val in x_list if str(x_val) != "nan"]
        no_bins = abs(round(math.sqrt(len(x))))

        if no_bins < 1:
            self.best_fit_dict[type_error][name] = (([0], [0]), 0, 0)
            logging.error(
                f"could not plot the histogram for {name} for {type_error}. can it find the results files with the error?"
            )
            return

        # Fit a normal distribution to the data, mean and standard deviation
        mu, std = norm.fit(x)
        se = std / np.sqrt(len(x))  # standard error
        ci = norm.interval(0.95, loc=mu, scale=se)  # confidence interval

        # plot histogram
        plt.hist(x, bins=no_bins, density=True,
                 alpha=0.7, color=col, edgecolor="grey")

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        y = norm.pdf(x, mu, std)

        plt.plot(x, y, "--", linewidth=2, color=self.colours["experimental"])

        self.best_fit_dict[type_error][name] = ((x, y), mu, std, ci)

        # plot
        plt.xlabel("Error (kcal/mol)")
        plt.ylabel("Frequency")
        plt.title(
            f"Distribution of {type_error} error for {name}, {self.net_ext.replace('_',', ')}\n mu = {mu:.3f} , std = {std:.3f}"
        )
        plt.savefig(
            f"{self.graph_folder}/histogram_{name}_{type_error}_{self.file_ext}_{self.net_ext}.png",
            dpi=300,
            bbox_inches="tight",
        )

        return ax

    def histogram_distribution(self, names: str = None, type_error: str = "SEM_pert"):
        for name in names:
            name = self._validate_in_names_list(name)
            if name in self.best_fit_dict[type_error].keys():
                pass
            else:
                self.histogram(name, type_error)

        # plot the distributions
        fig, ax = plt.subplots(figsize=(10, 10))

        lines = []
        mu_std_string = ""

        for name in names:
            col = self.colours[name]
            plt.plot(
                self.best_fit_dict[type_error][name][0][0],
                self.best_fit_dict[type_error][name][0][1],
                "k",
                linewidth=2,
                color=col,
            )
            lines += plt.plot(0, 0, c=col, label=name)
            mu_std_string += f"\n{name} : mu = {self.best_fit_dict[type_error][name][1]:.3f} , std = {self.best_fit_dict[type_error][name][2]:.3f}, 95% CI = ({self.best_fit_dict[type_error][name][3][0]:.3f}, {self.best_fit_dict[type_error][name][3][1]:.3f})"

        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc="upper right")

        plt.xlabel("Error")
        plt.ylabel("Frequency")
        eng_name = self._get_y_name(names)
        plt.title(
            f"Distribution of error for {type_error}, {eng_name}, {self.net_ext.replace('_',', ')}{mu_std_string}"
        )
        plt.savefig(
            f"{self.graph_folder}/normal_dist_{eng_name}_{type_error}_{self.file_ext}_{self.net_ext}.png",
            dpi=300,
            bbox_inches="tight",
        )

        return ax
