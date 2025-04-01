#!/usr/bin/python3

from pipeline.prep import *
from pipeline.utils import write_analysis_file
from pipeline.analysis import *
import BioSimSpace as BSS
import sys
import os as _os
from argparse import ArgumentParser
import logging

BSS.setVerbose = True


def run_all_analysis_methods(
    work_dir, pert, engine, final_results_folder=None, analysis_options_name=None
):
    """analyses all the iterations of estimators (TI, MBAR) and stats ineff and auto eq
    along with different truncated percentages, as it calculates the convergence.
    """

    pert_name = None

    # only want to calculate edgembar once
    edgembar_calc_already = False

    for estimator in ["MBAR", "TI"]:
        analysis_options = [
            {
                "estimator": estimator,
                "statistical inefficiency": False,
                "auto equilibration": False,
            },
            {
                "estimator": estimator,
                "statistical inefficiency": True,
                "auto equilibration": False,
            },
            {
                "estimator": estimator,
                "statistical inefficiency": True,
                "auto equilibration": True,
            },
            # for truncated sampling time
            {
                "estimator": estimator,
                "statistical inefficiency": True,
                "auto equilibration": False,
                "truncate upper": 25,  # 1/4 of the run, so 1 ns if 4 ns run
                "truncate lower": 0,
            },
            {
                "estimator": estimator,
                "statistical inefficiency": True,
                "auto equilibration": False,
                "truncate upper": 50,  # 1/2 of the run, so 2 ns if 4 ns run
                "truncate lower": 0,
            },
            {
                "estimator": estimator,
                "statistical inefficiency": True,
                "auto equilibration": False,
                "truncate upper": 75,  # 3/4 of the run, so 3 ns if 4 ns run
                "truncate lower": 0,
            },
        ]

        for ana_option in analysis_options:
            ana_option = analysis_protocol(ana_option, auto_validate=True)
            ana_option.name(analysis_options_name)

            logging.info(f"analysing results for {work_dir}")
            logging.info(f"using {ana_option.print_protocol()} for analysis")

            # using the pipeline module for analysis
            analysed_pert = analyse(work_dir, pert, engine, analysis_prot=ana_option)
            avg, error, repeats_tuple_list = analysed_pert.analyse_all_repeats()
            analysed_pert.check_convergence()
            analysed_pert.plot_graphs()
            analysed_pert.plot_across_lambda()

            # write the final result
            if final_results_folder:
                write_analysis_file(analysed_pert, final_results_folder)

            # plot the convergence
            # this only works if stats ineff and autoeq arent used as otherwise just truncates the data a lot
            if not ana_option.statistical_inefficiency():
                analysed_pert.calculate_convergence()
                analysed_pert.plot_convergence()
                # try:
                #     frac_eq_time = analysed_pert.equil_paired_t()
                #     if frac_eq_time:
                #         write_analysis_file(
                #             analysed_pert, final_results_folder, method="t-test")
                #         analysed_pert.plot_equilibration(eq_lines=[frac_eq_time], eq_line_labels=["t-test"])
                # except:
                #     logging.info("calculating frac eq time with t-test failed.")

            # write for edgembar
            if not edgembar_calc_already:
                # analysed_pert.format_for_edgembar()
                edgembar_calc_already = True

            # if ana_option.auto_equilibration():

            #     with open(f"{work_dir}/analysis_log.txt", "r") as file:
            #         start_idxs = []
            #         for f in file.readlines():
            #             if "Start index:" in f:
            #                 start_idxs.append(f.split(":")[1].split(" ")[0])

            #     logging.info(f"autoequilibration times were {start_idxs}")


def analysis_work_dir(
    work_dir,
    pert,
    engine,
    analysis_options,
    final_results_folder,
    analysis_options_name,
):
    if analysis_options:
        analysis_options = analysis_protocol(analysis_options, auto_validate=True)
    else:
        analysis_options = analysis_protocol(auto_validate=True)

    analysis_options.name(analysis_options_name)

    analysed_pert = analyse(work_dir, pert, engine, analysis_options)
    avg, error, repeats_tuple_list = analysed_pert.analyse_all_repeats()
    analysed_pert.check_convergence()
    analysed_pert.plot_graphs()

    # write the final result
    if final_results_folder:
        write_analysis_file(analysed_pert, final_results_folder)

    print(avg, error, repeats_tuple_list)


def check_arguments(args):
    # pass the checks to the other check functions

    if args.work_dir:
        work_dir = args.work_dir
        main_folder = None
    else:
        work_dir = None

        if args.main_folder:
            main_folder = args.main_folder
        else:
            main_folder = str(input("what is the main folder of the run?: ")).strip()

    if args.protocol_file:
        prot_file = args.protocol_file
    else:
        prot_file = None
        logging.error(
            "protocol file not provided. Will use auto settings for analysis."
        )

    if args.perturbation:
        perturbation = args.perturbation
    else:
        if not work_dir:
            perturbation = str(input("what is the perturbation?: ")).strip()
        else:
            perturbation = None

    if args.engine:
        engine = args.engine
    else:
        if not work_dir:
            engine = str(input("what is the engine?: ").strip())
        else:
            engine = None

    if args.analysis_file:
        analysis_file = args.analysis_file
    else:
        analysis_file = None
        logging.error(
            "analysis file not provided. Will use auto settings for analysis."
        )

    run_all_methods = args.run_all

    return (
        perturbation,
        engine,
        analysis_file,
        main_folder,
        prot_file,
        work_dir,
        run_all_methods,
    )


def main():
    # accept all options as arguments
    parser = ArgumentParser(description="run the fepprep")
    parser.add_argument(
        "-pert", "--perturbation", type=str, default=None, help="name of perturbation"
    )
    parser.add_argument(
        "-eng", "--engine", type=str, default=None, help="engine of the run"
    )
    parser.add_argument(
        "-mf",
        "--main_folder",
        type=str,
        default=None,
        help="main folder path for all the runs",
    )
    parser.add_argument(
        "-a",
        "--analysis_file",
        type=str,
        default=None,
        help="path to analysis protocol file",
    )
    parser.add_argument(
        "-p", "--protocol_file", type=str, default=None, help="path to protocol file"
    )
    parser.add_argument(
        "-wd",
        "--work_dir",
        type=str,
        default=None,
        help="work dir of run, will ignore mf and protocol and ana args.",
    )
    parser.add_argument(
        "-ra",
        "--run_all",
        action="store_true",
        help="run all analysis methods.",
    )
    args = parser.parse_args()

    # check arguments
    print("checking the provided command line arguments...")
    (
        pert,
        engine,
        ana_file,
        main_dir,
        prot_file,
        work_dir,
        run_all_methods,
    ) = check_arguments(args)

    if prot_file:
        # instantiate the protocol as an object
        protocol = pipeline_protocol(prot_file, auto_validate=True)
        analysis_options_name = protocol.name()
    else:
        analysis_options_name = None

    if ana_file:
        # options
        analysis_options = analysis_protocol(ana_file, auto_validate=True)
    else:
        analysis_options = None

    if not work_dir:
        pert_name = None

        if analysis_options_name:
            pert_name = f"{pert}_{protocol.name()}"
            analysis_options.name(protocol.name())

        if not pert_name:
            pert_name = pert

        # find correct path, use extracted if it exists
        if _os.path.exists(f"{main_dir}/outputs_extracted/{engine}/{pert_name}"):
            work_dir = f"{main_dir}/outputs_extracted/{engine}/{pert_name}"
            final_results_folder = f"{main_dir}/outputs_extracted/results"
        else:
            work_dir = f"{main_dir}/outputs/{engine}/{pert_name}"
            final_results_folder = f"{main_dir}/outputs/results"

        if not _os.path.exists(work_dir):
            raise OSError(f"{work_dir} does not exist.")

    else:
        final_results_folder = None

    logging.info(f"analysis for {work_dir} ...")

    if run_all_methods:
        # analyse all methods
        logging.info("running analysis for all methods...")
        run_all_analysis_methods(
            work_dir, pert, engine, final_results_folder, analysis_options_name
        )
    else:
        analysis_work_dir(
            work_dir,
            pert,
            engine,
            analysis_options,
            final_results_folder,
            analysis_options_name,
        )


if __name__ == "__main__":
    main()

# things to get from logging analysis txt
# "Start index: {}." for the start of the autoequilibrated data. This is the step, can get this as a percentage.

# open logging file after autoequilibration was run
# try to find the line
# get the step of the start index
# get the overall indexes ie the length of the data
# get the fractional equilibration time of this
# plot it as the ensequil is plotted

# # Write out data
# with open(f"{output_dir}/check_equil_autoequilibration_{leg}_{self.estimator}.txt", "w") as ofile:
#     ofile.write(f"Equilibrated: {equilibrated}\n")
#     ofile.write(f"p value: {p_value}\n")
#     ofile.write(f"p values and times: {p_vals_and_times}\n")
#     ofile.write(
#         f"Fractional equilibration time: {fractional_equil_time} \n")
#     ofile.write(f"Run numbers: {n_repeats}\n")

# def check_autoequilibration(self):

#     # get length of dataframe from auto eq?


#     from EnsEquil.analyse.plot import general_plot, p_plot

#     general_plot(
#         x_vals=overall_times[0],
#         y_vals=overall_dgs,
#         x_label="Total Simulation Time / ns",
#         y_label=r"$\Delta G$ / kcal mol$^{-1}$",
#         outfile=f"{output_dir}/check_equil_multiwindow_paired_t_{leg}.png",
#         vline_val=equil_time,
#         run_nos=list(range(0, n_repeats)),
#     )

#     # Create plot of p values
#     p_vals, times = zip(*p_vals_and_times)
#     p_plot(
#         times=np.array(times),
#         p_vals=np.array(p_vals),
#         outfile=f"{output_dir}/check_equil_multiwindow_paired_t_p_vals_{leg}.png",
#         p_cutoff=p_cutoff,
#     )
