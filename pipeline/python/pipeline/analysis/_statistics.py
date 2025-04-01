import BioSimSpace as BSS
import numpy as np

from ..utils import *
from ._network import *
from ._analysis import *
from ._plotting import *
from ._dictionaries import *

from cinnabar import stats


class stats_engines(plotting_engines):
    """statistics"""

    def __init__(self, analysis_object=None, output_folder=None):
        # inherit the init from other protocol too
        super().__init__(analysis_object, output_folder)

        self._set_statistic_dicts()

    @staticmethod
    def available_statistics():
        """list of the available statistics.

        Returns:
            list: available statistics
        """
        available_statistics = ["RMSE", "MUE", "R2", "rho", "KTAU"]
        # RMSE = Root Mean Squared Error
        # MUE = Mean Unsigned Error
        # R2 = correlation coefficient
        # rho = Spearman's rank correlation
        # KTAU = Kendall's rank correlation

        return available_statistics

    def _set_statistic_dicts(self):
        self.statistics = stats_engines.available_statistics()

        # make stats dict for each name and each stat
        self.statistics_dict = {}
        for pert_val in ["pert", "val", "bound", "free"]:
            self.statistics_dict[pert_val] = {}
            for namex in self.names_list:
                self.statistics_dict[pert_val][namex] = {}
                for namey in self.names_list:
                    self.statistics_dict[pert_val][namex][namey] = {}
                    for stats in self.available_statistics():
                        self.statistics_dict[pert_val][namex][namey][stats] = None

    def _get_x_y(
        self,
        pert_val: str = None,
        data_x: str = None,
        data_y: str = None,
        x: Optional[list] = None,
        y: Optional[list] = None,
        xerr: Optional[list] = None,
        yerr: Optional[list] = None,
    ) -> tuple:
        """get the x and y data from the dataframes from the inherited plotting object.

        Args:
            pert_val (str, optional): whether for 'pert' or 'val'. Defaults to None.
            data_x (str, optional): name of x data. Defaults to None.
            data_y (str, optional): name of y data. Defaults to None.
            x (list, optional): list of x data if no name is provided. Defaults to None.
            y (list, optional): list of y data if no name is provided. Defaults to None.
            xerr (list, optional): list of xerr data if no name is provided. Defaults to None.
            yerr (list, optional): list of yerr data if no name is provided. Defaults to None.

        Returns:
            tuple: lists of each x,y,xerr,yerr
        """

        if data_x and data_y:
            pv = validate.pert_val(pert_val)

            data_x = self._validate_in_names_list(data_x)
            data_y = self._validate_in_names_list(data_y)
            df = self.freenrg_df_dict[data_x][data_y][pv]
            df = df.dropna()

            x = df[f"freenrg_{data_x}"]
            y = df[f"freenrg_calc"]
            xerr = df[f"err_{data_x}"]
            yerr = df[f"err_calc"]

        else:
            try:
                x = x
                y = y
                xerr = xerr
                yerr = yerr
            except:
                logging.error(
                    "if not providing data_x and data_y (which should be a name in the self.names_list),\
                      please provide x,y,xerr,yerr values"
                )

        return x, y, xerr, yerr

    @staticmethod
    def compute_stats(
        x: list = None,
        y: list = None,
        xerr: Optional[list] = None,
        yerr: Optional[list] = None,
        statistic: str = None,
        nbootstrap: int = 10000,
    ) -> tuple:
        """static method for computing various statistics.

        Args:
            x (list): ordered list of x data. Defaults to None.
            y (list): ordered list of y data. Defaults to None.
            xerr (list, optional): ordered list of xerr data. Defaults to None.
            yerr (list, optional): ordered list of yerr data. Defaults to None.
            statistic (str): name of statistic to use. Defaults to None.

        Raises:
            ValueError: statistic must be an available statistic

        Returns:
            tuple: (value, error)
        """

        if statistic not in stats_engines.available_statistics():
            raise ValueError(
                f"please use one of the statistics in {stats_engines.available_statistics()}, not {statistic}"
            )

        # using cinnabar function
        s = stats.bootstrap_statistic(
            x, y, xerr, yerr, nbootstrap=nbootstrap, statistic=statistic
        )
        values = (s["mle"], s["stderr"], [s["low"], s["high"]])
        # string = f"{statistic}:   {s['mle']:.2f} [95%: {s['low']:.2f}, {s['high']:.2f}] " + "\n"

        return values

    def _compute_stats(
        self,
        pert_val: str = None,
        data_x: str = None,
        data_y: str = None,
        x: Optional[list] = None,
        y: Optional[list] = None,
        xerr: Optional[list] = None,
        yerr: Optional[list] = None,
        statistic: str = None,
    ) -> tuple:
        """internal to get data from df and then pass to static method

        Args:
            pert_val (str, optional): whether for 'pert' or 'val'. Defaults to None.
            data_x (str, optional): name of x data. Defaults to None.
            data_y (str, optional): name of y data. Defaults to None.
            statistic (str, optional): name of statistic to use. Defaults to None.
            x (list, optional): list of x data if no name is provided. Defaults to None.
            y (list, optional): list of y data if no name is provided. Defaults to None.
            xerr (list, optional): list of xerr data if no name is provided. Defaults to None.
            yerr (list, optional): list of yerr data if no name is provided. Defaults to None.

        Returns:
            tuple: (value, error)
        """

        # get the x y values from the dictionaries, also validates pert val and engine
        x, y, xerr, yerr = self._get_x_y(pert_val, data_x, data_y, x, y, xerr, yerr)
        values = stats_engines.compute_stats(x, y, xerr, yerr, statistic)

        return values

    def compute_statistics(self, names: Optional[list] = None) -> dict:
        """compute all statistics compared to experimental values.

        Args:
            names (list, optional): list of names from names list to compute for. Defaults to None.

        Returns:
            dict: dictionary of results
        """

        if not names:
            names = self.names_list
        else:
            names = validate.is_list(names)
            for name in names:
                name = self._validate_in_names_list(name)

        for name in names:
            for pv in ["pert", "val"]:
                for stats in self.available_statistics():
                    try:
                        values = self._compute_base(
                            pert_val=pv, y=name, x="experimental", statistic=stats
                        )
                        self.statistics_dict[pv]["experimental"][name][stats] = values
                    except Exception as e:
                        logging.error(e)
                        logging.error(
                            f"could not compute {stats} for {pv}, 'experimental' and '{name}'"
                        )
                        self.statistics_dict[pv]["experimental"][name][stats] = np.nan

        return self.statistics_dict

    def _compute_base(
        self, pert_val: str = None, y: str = None, x: str = None, statistic: str = None
    ) -> tuple:
        """base function to pass data to cinnabar stats compute function.

        Args:
            pert_val (str): whether for 'pert' or 'val'. Defaults to None.
            y (str): name of y data. Defaults to None.
            x (str): name of x data. Defaults to None.
            statistic (str): statistic to calculate. Defaults to None.

        Raises:
            ValueError: must be one of the available statistics

        Returns:
            tuple: (value, error) error is from bootstrapping
        """

        # validate from other that it is in names list
        x = self._validate_in_names_list(x)
        y = self._validate_in_names_list(y)
        pert_val = validate.pert_val(pert_val)

        if statistic not in stats_engines.available_statistics():
            raise ValueError(
                f"please use one of the statistics in {stats_engines.available_statistics()}, not {statistic}"
            )

        values = self._compute_stats(pert_val, data_x=x, data_y=y, statistic=statistic)

        return values

    def compute_mue(
        self, pert_val: str = None, y: str = None, x: str = "experimental"
    ) -> tuple:
        """compute MUE for two names in names list.

        Args:
            pert_val (str): whether for 'pert' or 'val'. Defaults to None.
            y (str): name of y data. Defaults to None.
            x (str, optional): name of x data. Defaults to 'experimental'.

        Returns:
            tuple: (value, error) error is from bootstrapping
        """

        values = self._compute_base(pert_val=pert_val, y=y, x=x, statistic="MUE")
        self.statistics_dict[pert_val][x][y]["MUE"] = values
        return values

    def compute_rmse(
        self, pert_val: str = None, y: str = None, x: str = "experimental"
    ) -> tuple:
        """compute RMSE for two names in names list.

        Args:
            pert_val (str): whether for 'pert' or 'val'. Defaults to None.
            y (str): name of y data. Defaults to None.
            x (str, optional): name of x data. Defaults to 'experimental'.

        Returns:
            tuple: (value, error) error is from bootstrapping
        """

        values = self._compute_base(pert_val=pert_val, y=y, x=x, statistic="RMSE")
        self.statistics_dict[pert_val][x][y]["RMSE"] = values
        return values

    def compute_r2(
        self, pert_val: str = None, y: str = None, x: str = "experimental"
    ) -> tuple:
        """compute R2 for two names in names list.

        Args:
            pert_val (str): whether for 'pert' or 'val'. Defaults to None.
            y (str): name of y data. Defaults to None.
            x (str, optional): name of x data. Defaults to 'experimental'.

        Returns:
            tuple: (value, error) error is from bootstrapping
        """

        values = self._compute_base(pert_val=pert_val, y=y, x=x, statistic="R2")
        self.statistics_dict[pert_val][x][y]["R2"] = values
        return values

    def compute_rho(
        self, pert_val: str = None, y: str = None, x: str = "experimental"
    ) -> tuple:
        """compute rho for two names in names list.

        Args:
            pert_val (str): whether for 'pert' or 'val'. Defaults to None.
            y (str): name of y data. Defaults to None.
            x (str, optional): name of x data. Defaults to 'experimental'.

        Returns:
            tuple: (value, error) error is from bootstrapping
        """

        values = self._compute_base(pert_val=pert_val, y=y, x=x, statistic="rho")
        self.statistics_dict[pert_val][x][y]["rho"] = values
        return values

    def compute_rae(
        self, pert_val: str = None, y: str = None, x: str = "experimental"
    ) -> tuple:
        """compute RAE for two names in names list.

        Args:
            pert_val (str): whether for 'pert' or 'val'. Defaults to None.
            y (str): name of y data. Defaults to None.
            x (str, optional): name of x data. Defaults to 'experimental'.

        Returns:
            tuple: (value, error) error is from bootstrapping
        """

        values = self._compute_base(pert_val=pert_val, y=y, x=x, statistic="RAE")
        self.statistics_dict[pert_val][x][y]["RAE"] = values
        return values

    def compute_ktau(
        self, pert_val: str = None, y: str = None, x: str = "experimental"
    ) -> tuple:
        """compute KTAU for two names in names list.

        Args:
            pert_val (str): whether for 'pert' or 'val'. Defaults to None.
            y (str): name of y data. Defaults to None.
            x (str, optional): name of x data. Defaults to 'experimental'.

        Returns:
            tuple: (value, error) error is from bootstrapping
        """

        values = self._compute_base(pert_val=pert_val, y=y, x=x, statistic="KTAU")
        self.statistics_dict[pert_val][x][y]["KTAU"] = values
        return values
