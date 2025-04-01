#!/usr/bin/python3

from argparse import ArgumentParser

from scipy.stats import sem as sem
import sys
import glob

from pipeline import *
from pipeline.utils import validate
from pipeline.analysis import *


def analyse_results(main_dir, experimental_file):
    """simple analysis of the entire pipeline."""

    # choose location for the files
    net_file = f"{main_dir}/execution_model/network.dat"
    ana_file = f"{main_dir}/execution_model/analysis_protocol.dat"
    exp_file = experimental_file
    output_folder = f"{main_dir}/outputs_extracted"

    prot_file = f"{main_dir}/execution_model/protocol.dat"
    pipeline_prot = pipeline_protocol(prot_file, auto_validate=True)

    all_analysis_object = analysis_network(
        output_folder=output_folder,
        exp_file=exp_file,
        net_file=net_file,
        analysis_prot=ana_file,
        # method = pipeline_prot.name(), # if the protocol had a name
        engines=pipeline_prot.engines(),
    )

    all_analysis_object.compute_results()

    all_analysis_object.compute_convergence(main_dir=main_dir)
    all_analysis_object.plot_convergence()

    # plotting all
    # bar
    all_analysis_object.plot_bar_dG()
    all_analysis_object.plot_bar_ddG()

    # scatter
    all_analysis_object.plot_scatter_dG()
    all_analysis_object.plot_scatter_ddG()
    all_analysis_object.plot_scatter_dG(use_cinnabar=True)
    all_analysis_object.plot_scatter_ddG(use_cinnabar=True)

    for eng in all_analysis_object.engines:
        all_analysis_object.plot_scatter_dG(engines=eng)
        all_analysis_object.plot_scatter_ddG(engines=eng)

        # outliers
        all_analysis_object.plot_outliers(engine=eng)
        all_analysis_object.plot_outliers(engine=eng, pert_val="val")

        for pv in ["pert", "val"]:
            stat_rank = all_analysis_object._stats_object.compute_rho(pv, y=eng)
            print(f"rank correlation for {pv}, {eng} is {stat_rank}")

    # all_analysis_object.calc_mad_engines(pert_val="pert")
    all_analysis_object.calc_mae_engines(pert_val="pert")
    # all_analysis_object.calc_mad_engines(pert_val="val")
    all_analysis_object.calc_mae_engines(pert_val="val")


def check_arguments(args):
    # pass the checks to the other check functions
    if args.main_folder:
        main_folder = validate.folder_path(args.main_folder)
    else:
        main_folder = validate.folder_path(
            str(input("what is the main folder of the runs? : ")).strip()
        )

    if args.experimental_file:
        experimental_file = validate.file_path(args.experimental_file)
    else:
        experimental_file = validate.file_path(
            str(input("what is the path to the experimental results?: ")).strip()
        )

    return main_folder, experimental_file


def main():
    print("analyse the pipeline!")

    # accept all options as arguments
    parser = ArgumentParser(description="analyse simulations")
    parser.add_argument(
        "-mf",
        "--main_folder",
        type=str,
        default=None,
        help="main folder for the pipeline.",
    )
    parser.add_argument(
        "-ef",
        "--experimental_file",
        type=str,
        default=None,
        help="Path to the experimental yml file.",
    )
    args = parser.parse_args()

    # check arguments
    print("checking the provided command line arguments...")
    main_folder, experimental_file = check_arguments(args)

    analyse_results(main_folder, experimental_file)

    print(f"finished analysing.")


if __name__ == "__main__":
    main()
