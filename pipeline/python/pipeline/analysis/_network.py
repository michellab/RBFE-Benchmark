# import libraries
import BioSimSpace as BSS
import networkx as nx
from scipy.stats import sem as sem
from rdkit import Chem
import math
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image, ImageOps
from scipy.stats import norm

from typing import Union, Optional

from ..utils._validate import *

font = {"family": "normal", "weight": "bold", "size": 22}

matplotlib.rc("font", **{"size": 10})


class network_graph:
    def __init__(
        self,
        ligands: list,
        perturbations: list,
        calc_pert_dict: dict = None,
        file_dir: Optional[str] = None,
        ligands_folder: Optional[str] = None,
    ):
        """_summary_

        Args:
            ligands (list): list of ligands
            perturbations (list): list of perturbations
            file_dir (str, optional): folder path to save graph image in. Defaults to None.
        """

        self.ligands = sorted(validate.is_list(ligands))
        self.perturbations = validate.is_list(perturbations)

        if calc_pert_dict:
            self.calc_pert_dict = validate.dictionary(calc_pert_dict)
        else:
            self.calc_pert_dict = None

        if file_dir:
            self.file_dir = validate.folder_path(file_dir, create=True)
            self.ligand_image_dir = validate.folder_path(
                f"{file_dir}/ligand_images", create=True
            )
            self._save_image = True
        else:
            self._save_image = False
            self.ligand_image_dir = None

        if ligands_folder:
            self.ligands_folder = validate.folder_path(ligands_folder)
        else:
            self.ligands_folder = None

        network_graph._gen_graph(self)

    def _gen_graph(self):
        """generate a network x graph from the perturbations and ligands."""

        # Generate the graph.
        graph = nx.Graph()

        # Loop over the nligands and add as nodes to the graph.
        for lig in self.ligands:
            if self.ligands_folder:
                img = self._ligand_image(lig)
                img = f"{self.ligand_image_dir}/{lig}.png"
            else:
                img = None
            graph.add_node(lig, label=lig, labelloc="t", image=img)

        # Loop over the edges in the nxgraph and add to the graph.
        for edge in self.perturbations:
            lig_0 = edge.split("~")[0]
            lig_1 = edge.split("~")[1]
            if self.calc_pert_dict:
                val = self.calc_pert_dict[edge][0]
                err = self.calc_pert_dict[edge][1]
                graph.add_edge(lig_0, lig_1, value=val, error=err)
            else:
                graph.add_edge(lig_0, lig_1)

        self.graph = graph

    def draw_graph(self, file_dir: Optional[str] = None,
                   title=None,
                   edge_labels: str = True,
                   figsize: tuple = None,
                   networkx_layout_func=None,
                   layout_func_kwargs: dict ={},
                   dot_graph: bool = False,
                   ):
        """draw the network x graph.

        Args:
            file_dir (str, optional): where to save the image. Defaults to None, no image is saved.
        """

        graph = validate.nxgraph(self.graph)
        if title:
            title = validate.string(title)
        else:
            title = "Network"

        if dot_graph:
            # Convert to a dot graph.
            dot_graph = nx.drawing.nx_pydot.to_pydot(graph)

            # Write to a PNG.
            network_plot = f"{self.file_dir}/network_pydot.png"
            dot_graph.write_png(network_plot)

        else:

            # Plot the networkX graph.
            if networkx_layout_func:
                func = networkx_layout_func
            else:
                func = nx.kamada_kawai_layout
            pos = func(graph, **layout_func_kwargs)

            # default fig size based on network size
            if not figsize:
                figsize = (graph.number_of_nodes(),# * 2,
                        graph.number_of_nodes())# * 2)

            fig, ax = plt.subplots(
                figsize=figsize, dpi=100*(math.ceil(graph.number_of_nodes()/100))
            )

            if self.calc_pert_dict:
                cmap_col = plt.cm.magma
                edge_colours = [graph[u][v]["error"] for u, v in graph.edges]
                edges, weights = zip(*nx.get_edge_attributes(graph, "error").items())
                if edge_labels:
                    draw_edge_labels = True
                else:
                    draw_edge_labels = False
            else:
                cmap_col = None
                weights = "black"
                draw_edge_labels = False

            nx.draw(
                graph,
                pos,
                edge_color=weights,
                edge_cmap=cmap_col,
                width=1,
                linewidths=1,
                node_size=15000,
                node_color="navy",
                font_size=15,
                labels={node: node.split("lig_CHEMBL")[-1] for node in graph.nodes()},
                font_color="white",
            )
            if draw_edge_labels:
                nx.draw_networkx_edge_labels(
                    graph, pos,
                    edge_labels={(u, v): format(graph[u][v]['value'], ".2f") for u, v in graph.edges},
                    font_color='navy',
                    font_size=16,
                    label_pos=0.45
                )
                title = f"{title}\n ddG in kcal/mol"

            plt.title(
                f"{title}", fontdict={"fontsize": graph.number_of_nodes()}
            )

            if cmap_col:
                sm = plt.cm.ScalarMappable(
                    cmap=cmap_col, norm=plt.Normalize(
                        vmin=min(weights), vmax=max(weights))
                )
                cbar = plt.colorbar(sm, shrink=0.5, location="bottom", aspect=50)
                cbar.set_label(label="error (kcal/mol)", size=15)
                cbar.ax.tick_params(labelsize=10)

            trans = ax.transData.transform
            trans2 = fig.transFigure.inverted().transform
            piesize = 0.05  # this is the image size
            p2 = piesize / 2.0
            for n in graph:
                xx, yy = trans(pos[n])  # figure coordinates
                xa, ya = trans2((xx, yy))  # axes coordinates
                a = plt.axes([xa - p2, ya - p2, piesize, piesize])
                a.set_aspect("equal")
                a.imshow(ImageOps.expand(graph.nodes[n]["image"], border=2, fill="black"))
                a.set_title(f"{n}", y=0.85)
                a.axis("off")

            if self._save_image:
                plt.savefig(f"{self.file_dir}/network.png", dpi=300)
            if file_dir:
                file_dir = validate.folder_path(file_dir, create=True)
                plt.savefig(f"{file_dir}/network.png", dpi=300)

            return ax

    def _ligand_image(self, ligand: str):
        if not self.ligands_folder:
            raise ValueError(
                "please provide a ligands dir w the files inside.")

        m = Chem.SDMolSupplier(f"{self.ligands_folder}/{ligand}.sdf")[0]
        smi = Chem.MolToSmiles(m)
        m2 = Chem.MolFromSmiles(smi)
        img = Chem.Draw.MolToImage(m2)
        if self._save_image:
            Chem.Draw.MolToFile(m2, f"{self.ligand_image_dir}/{ligand}.png")
        else:
            # logging.info(
            #     "as there is no output folder to save the network image, the ligand images will not be written."
            # )
            pass

        return img

    def draw_ligand(self, ligand: str):
        img = self._ligand_image(ligand)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img)
        ax.axis("off")
        ax.title.set_text(f"{ligand}")

        return fig

    def draw_all_ligands(self, file_dir: Optional[str] = None, figsize: tuple = (10, 15)):
        """draw all the ligands in the network.

        Args:
            file_dir (Optional[str], optional): where to save the image. Defaults to None, no image saved, or in the object directory if given.
            figsize (tuple, optional): Size of the figure, to adjust based on the ligand. Defaults to (10,15).

        Raises:
            ValueError: need a ligands directory with the structures inside to draw them.

        Returns:
            fig: the figure.
        """

        if not self.ligands_folder:
            raise ValueError(
                "please provide a ligands dir w the files inside.")

        len_ligs = len(self.ligands)

        fig, axs = plt.subplots(
            math.ceil(len_ligs / 3), 3, figsize=figsize, dpi=300
        )  # 10,15

        for ax in axs.ravel():
            ax.axis("off")

        for lig, ax in zip(self.ligands, axs.ravel()):
            img = self._ligand_image(lig)
            ax.imshow(img)
            ax.title.set_text(f"{lig}")

        if self._save_image:
            plt.savefig(f"{self.file_dir}/ligands.png", dpi=300)
        if file_dir:
            file_dir = validate.folder_path(file_dir, create=True)
            plt.savefig(f"{file_dir}/ligands.png", dpi=300)

        return fig

    def draw_perturbation(self, pert: str):
        lig_0 = pert.split("~")[0]
        lig_1 = pert.split("~")[1]

        img = self._ligand_image(lig_0)
        img2 = self._ligand_image(lig_1)

        # plt.rcParams['figure.figsize'] = 11 ,8
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(img)
        ax[0].title.set_text(f"{lig_0}")
        ax[0].axis("off")
        ax[1].imshow(img2)
        ax[1].title.set_text(f"{lig_1}")
        ax[1].axis("off")

        return fig

    def disconnected_ligands(self) -> list:
        ligs = [lig for lig in nx.isolates(self.graph)]

        return ligs

    def cycle_closures(self) -> list:
        """get cycle closures in the network

        Returns:
            list: list of cycle closures
        """

        cycles = nx.cycle_basis(self.graph)

        # get list of all cycle closures
        cycle_closures = []

        for cycle in cycles:
            length = len(cycle)
            ligas = []
            ligbs = []
            for i in range(0, length - 1):
                ligas.append(cycle[i])
                ligbs.append(cycle[i + 1])
            # add final cycle closure
            ligas.append(cycle[-1])
            ligbs.append(cycle[0])
            # make list for cycle closure
            cycle_closure = []
            for liga, ligb in zip(ligas, ligbs):
                cycle_closure.append(f"{liga}~{ligb}")

            cycle_closures.append(cycle_closure)

            self.cycles_list = cycle_closures

        return cycle_closures

    def cycle_closure_dict(self):
        pert_dict = self.calc_pert_dict
        cycle_closures = self.cycle_closures()

        cycles_dict = {}

        for cycle in cycle_closures:
            cycle_dict = {}
            cycle_val = []
            cycle_val_err = []
            for pert in cycle:
                liga = pert.split("~")[0]
                ligb = pert.split("~")[1]
                rev_pert = f"{ligb}~{liga}"

                if pert in pert_dict:
                    if pert_dict[pert][0] is not None:
                        cycle_val.append(+pert_dict[pert][0])
                        cycle_val_err.append(pert_dict[pert][1])
                    else:
                        logging.warning(
                            f"{pert} or {rev_pert} does not exist in the results for {cycle}. This cycle is not included."
                        )
                        break
                elif rev_pert in pert_dict:
                    if pert_dict[rev_pert][0] is not None:
                        cycle_val.append(-pert_dict[rev_pert][0])
                        cycle_val_err.append(pert_dict[rev_pert][1])
                    else:
                        logging.warning(
                            f"{pert} or {rev_pert} does not exist in the results for {cycle}. This cycle is not included."
                        )
                        break

            cycles_dict.update(
                {"_".join(cycle): (sum(cycle_val), sum(cycle_val_err))})

        self.cycles_dict = cycles_dict

        return cycles_dict

    def cycle_vals(self):
        self.cycle_closure_dict()

        cycle_vals = []

        for key in self.cycles_dict.keys():
            val = self.cycles_dict[key]
            cycle_vals.append(val[0])

        return cycle_vals

    def average_cycle_closures(self):
        cycle_vals = self.cycle_vals()

        cycle_vals_not_nan = [abs(x) for x in cycle_vals if str(x) != "nan"]
        avg_cc = np.mean(cycle_vals_not_nan)
        std_cc = np.std(cycle_vals_not_nan)
        se = std_cc / np.sqrt(len(cycle_vals_not_nan))
        ci_cc = norm.interval(0.95, loc=avg_cc, scale=se)

        return avg_cc, std_cc, ci_cc

    def add_weight(self, input_data: Union[dict, str], name: str = "weights"):
        """add weights to the network x graph for the edges and each perturbation.

        Args:
            input_data (str or dict): file path or dict of the weights for the perturbation edges.
            name (str): name for the data, eg weight.

        Raises:
            TypeError: if dict, must be of format {(lig_0, lig_1): weight}
        """
        # these weights are for the lomap or rbfenn score
        name = validate.string(name)

        try:
            weight_dict = validate.dictionary(input_data)
            for key in weight_dict.keys():
                if not isinstance(key, tuple):
                    raise TypeError(
                        "dict entry must be of the format {(lig_0, lig_1): weight}"
                    )
            use_file = False
        except:
            use_file = True
            weight_dict = {}

        if use_file:
            scores_file = validate.file_path(input_data)

            with open(scores_file, "r") as file:
                for line in file:
                    lig_0 = line.split(",")[0]
                    lig_1 = line.split(",")[1]
                    weight = line.split(",")[-1].strip()
                    weight_dict[(lig_0, lig_1)] = float(weight)

        nx.set_edge_attributes(self.graph, weight_dict, name=name)

    # from alvaro
    def get_average_weighted_simple_paths(self) -> float:
        """Calculate the average number of connection between each pair of nodes."""

        G = self.graph

        paths_per_nodepair_combination = []
        for node_i in G.nodes:
            for node_j in G.nodes:
                if node_i == node_j:
                    break
                possible_paths = nx.all_simple_edge_paths(G, node_i, node_j)
                sum_of_weighted_averaged_paths = sum(
                    [
                        np.average([G.get_edge_data(*edge)["weight"]
                                   for edge in path])
                        for path in possible_paths
                    ]
                )
                paths_per_nodepair_combination.append(
                    sum_of_weighted_averaged_paths)

        # no of possible paths between each two nodes on average
        return np.average(paths_per_nodepair_combination)
