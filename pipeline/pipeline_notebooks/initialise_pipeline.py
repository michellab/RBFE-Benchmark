#!/usr/bin/python3

from argparse import ArgumentParser
from pipeline.utils import *
import pandas as pd


def ask_things():
    lig_ff = str(
        input(
            "what is the Ligand forcefield? (allowed: GAFF2, Parsely, Sage. Default: GAFF2): "
        )
    ).strip()
    if not lig_ff:
        lig_ff = "GAFF2"

    sampling_time = str(
        input("what is the sampling time? (in ns. Default: 4 ns.): ")
    ).strip()
    if not sampling_time:
        sampling_time = 4

    engines = str(
        input("what is the engine? (allowed: SOMD, AMBER, GROMACS. Default: SOMD): ")
    ).strip()
    if not engines:
        engines = "SOMD"

    hmr = str(
        input("should HMR be applied? (allowed: True, False. Default: True): ")
    ).strip()
    if not hmr:
        hmr = True

    repeats = str(input("how many repeats? (Default: 3): ")).strip()
    if not repeats:
        repeats = 3

    trajectories = str(
        input(
            "save the trajectories? (allowed: 'None', '0,0.5,1', '0,1', 'All'. Default: All): "
        )
    ).strip()
    if not trajectories:
        trajectories = "All"

    # make these into a dictionary for the setup
    protocol_dict = {
        "ligand forcefield": lig_ff,
        "sampling": sampling_time,
        "hmr": hmr,
        "repeats": repeats,
        "trajectories": trajectories,
        "engines": engines,
    }

    return protocol_dict


def check_arguments(pl, args):
    # pass the checks to the other check functions
    if args.ligands_folder:
        pl.ligands_folder(args.ligands_folder)
    else:
        pl.ligands_folder(str(input("what is the ligands folder?: ")).strip())

    if args.protein_path:
        pl.protein_path(args.protein_path)
    else:
        pl.protein_path(str(input("what is the parameterised protein path?: ")).strip())

    if args.main_folder:
        pl.main_folder(args.main_folder)
    else:
        pl.main_folder(
            str(
                input("what is the main folder where all the files should go?: ")
            ).strip()
        )

    return pl


def main():
    print("setup the pipeline! first choose options")

    # accept all options as arguments
    parser = ArgumentParser(description="set up simulations")
    parser.add_argument(
        "-lf",
        "--ligands_folder",
        type=str,
        default=None,
        help="folder path to the ligand files",
    )
    parser.add_argument(
        "-pf",
        "--protein_path",
        type=str,
        default=None,
        help="path to parameterised protein *.prm7 and *.rst7",
    )
    parser.add_argument(
        "-mf",
        "--main_folder",
        type=str,
        default=None,
        help="main folder path to create for all the runs",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="benchmark",
        help="descriptor of what this run is for.",
    )
    args = parser.parse_args()

    # intialise setup class
    pl = initialise_pipeline()

    # check arguments
    print("checking the provided command line arguments...")
    pl = check_arguments(pl, args)

    print("please decide some basic protocol settings. Leave blank for default values:")
    protocol_dict = ask_things()
    # fill in rest with default and save the files
    pl.setup_protocols(protocol_dict)

    # make the files needed
    pl.setup_ligands()

    print("please edit the network as needed. Otherwise, leave blank.")

    rem_ligs = str(
        input(
            "do you want to remove any ligands? Please list all ligands to remove, seperated by a comma: "
        )
    ).strip()
    if rem_ligs:
        rem_ligs = [lig.strip() for lig in rem_ligs.split(",")]
        for lig in rem_ligs:
            pl.remove_ligand(lig)
    else:
        pass

    net_file_path = str(
        input(
            "do you want to generate a network or pass a network file? Please enter a file location, else leave blank: "
        )
    ).strip()
    if os.path.exists(net_file_path):
        df = pd.read_csv(net_file_path)
        perturbations = list(df["perturbation"])
        for pert in perturbations:
            pl.add_perturbation(pert)
    else:
        pl.setup_network()

        rem_perts = str(
            input(
                "do you want to remove any perturbations? Please list all perts to remove (name as above), seperated by a comma: "
            )
        ).strip()
        if rem_perts:
            rem_perts = [pert.strip() for pert in rem_perts.split(",")]
            for pert in rem_perts:
                pl.remove_perturbation(pert)
        else:
            pass

        add_perts = str(
            input(
                "do you want to add any perturbations? Please list all perts to add (lig0~lig1), seperated by a comma: "
            )
        ).strip()
        if add_perts:
            add_perts = [pert.strip() for pert in rem_perts.split(",")]
            for pert in add_perts:
                pl.remove_perturbation(pert)
        else:
            pass

    run_reverse = str(
        input(
            "do you want to also run the perturbations in reverse? (allowed: True, False. Default: False): "
        )
    ).strip()
    if run_reverse:
        pl.run_reverse(run_reverse)
    else:
        pass

    pl.add_source_file(
        str(
            input(
                "please state the bash file that includes all the source and module parameters: "
            )
        ).strip()
    )

    # make run_all_slurm to the main folder that was made.
    pl.write_run_all()

    print(
        f"made all files in {pl.main_folder()}. Run run_all_slurm.sh to run all pipeline components."
    )
    print(
        "carefully check the run all script first to make sure all paths and executables are correct and can be sourced correctly."
    )
    print("modify any of the input files as required.")


if __name__ == "__main__":
    main()
