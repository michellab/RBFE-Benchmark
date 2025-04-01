#!/usr/bin/python3

import sys
import csv
import BioSimSpace as BSS
import os

from argparse import ArgumentParser
import itertools as it

import pipeline
from pipeline.utils import validate


def no_perturbing_atoms(pert, prep):
    lig_0 = pert.split("~")[0]
    lig_1 = pert.split("~")[1]

    # Load equilibrated inputs for both ligands
    system0 = BSS.IO.readMolecules(
        [f"{prep}/{lig_0}_lig_equil_solv.rst7", f"{prep}/{lig_0}_lig_equil_solv.prm7"]
    )
    system1 = BSS.IO.readMolecules(
        [f"{prep}/{lig_1}_lig_equil_solv.rst7", f"{prep}/{lig_1}_lig_equil_solv.prm7"]
    )

    l0a, l1a, mapping = pipeline.prep.merge.atom_mappings(
        system0, system1, **({"prune perturbed constraints": True})
    )
    # print(len(l0a))
    # print(len(l1a))
    # print(len(mapping))
    # print(len(l0a) - len(mapping))
    # print(len(l1a) - len(mapping))
    no_atoms = (len(l0a) + len(l1a)) / 2 - len(mapping)

    return no_atoms


def get_inputs(net_file, prot):
    perts, ligs = pipeline.utils.get_info_network(net_file)
    engs = prot.engines()

    return perts, engs


def check_arguments(args):
    # pass the checks to the other check functions
    if args.main_folder:
        main_folder = validate.folder_path(args.main_folder)
    else:
        main_folder = validate.folder_path(
            str(input("what is the main folder of the run?: ")).strip()
        )

    if args.prep_folder:
        prep_folder = validate.folder_path(args.prep_folder)
    else:
        try:
            prep_folder = validate.folder_path(f"{main_folder}/prep")
        except:
            print("Cant find prep folder...")
            prep_folder = validate.folder_path(
                str(input("what is the prep folder of the run?: ")).strip()
            )

    if args.net_file:
        net_file = validate.file_path(args.net_file)
    else:
        net_file = validate.file_path(
            f"{main_folder}/execution_model/network_combined.dat"
        )

    if args.output_file:
        file = validate.string(args.output_file)
    else:
        validate.folder_path(f"{main_folder}/outputs_extracted/analysis", create=True)
        file = validate.string(
            f"{main_folder}/outputs_extracted/analysis/perturbing_overlap.dat"
        )

    if args.protocol_file:
        protocol_file = validate.file_path(args.protocol_file)
    else:
        protocol_file = validate.file_path(
            f"{main_folder}/execution_model/protocol.dat"
        )
    prot = pipeline.prep.pipeline_protocol(protocol_file, auto_validate=True)

    try:
        ana_protocol = validate.file_path(
            f"{main_folder}/execution_model/analysis_protocol.dat"
        )
    except Exception as e:
        print(e)
        print("ana protocol must be in exec model folder")
    ana_prot = pipeline.prep.analysis_protocol(ana_protocol, auto_validate=True)
    ana_prot.try_pickle(True)

    if args.exp_file:
        exp_file = validate.file_path(args.exp_file)
    else:  # f"/home/anna/Documents/benchmark/inputs/experimental/{protein}.yml"
        exp_file = validate.folder_path(
            str(input("what is the experimental file for this protein?: ")).strip()
        )

    return main_folder, net_file, prot, ana_prot, file, prep_folder, exp_file


def main():
    # accept all options as arguments
    parser = ArgumentParser(description="correlate perturbing atoms and bad overlap")
    parser.add_argument(
        "-mf",
        "--main_folder",
        type=str,
        default=None,
        help="main folder path where runs are",
    )
    parser.add_argument(
        "-prep",
        "--prep_folder",
        type=str,
        default=None,
        help="were the prepped files are",
    )
    parser.add_argument(
        "-net", "--net_file", type=str, default=None, help="network file"
    )
    parser.add_argument(
        "-prot", "--protocol_file", type=str, default=None, help="the protocol file"
    )
    parser.add_argument(
        "-out", "--output_file", type=str, default=None, help="the output file"
    )
    parser.add_argument(
        "-exp", "--exp_file", type=str, default=None, help="the experimental file"
    )
    args = parser.parse_args()

    mf, nf, prot, ana_prot, file, prep, exp_file = check_arguments(args)
    # get the perturbations
    perts, engs = get_inputs(nf, prot)

    all_analysis_object = pipeline.analysis.analysis_network(
        exp_file=exp_file, net_file=nf
    )

    all_analysis_object.get_experimental()
    pert_dict = all_analysis_object.exper_pert_dict

    if not os.path.exists(file):
        with open(file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "perturbation",
                    "engine",
                    "perturbing_atoms",
                    "percen_overlap_okay",
                    "too_small_avg",
                    "diff_to_exp",
                    "error",
                ]
            )

    with open(file, "r") as readfile:
        # then, grab all of the data that is already in the file.
        reader = csv.reader(readfile)
        data_entries = [(row[0], row[1]) for row in reader]

    print(engs)

    for pert, eng in it.product(perts, engs):
        print(f"running {pert}, {eng}....")
        folder = f"{mf}/outputs_extracted/{eng}/{pert}"

        # check if our data entry is not already in the results file. Raise an error if is.
        if (pert, eng) in data_entries:
            print(f"{pert} {eng} is already in {file}")
            pass

        else:
            try:
                pert_atoms = no_perturbing_atoms(pert, prep)
                ana_obj = pipeline.analysis.analyse(folder, analysis_prot=ana_prot)
                avg, error, repeats_tuple_list = ana_obj.analyse_all_repeats()
                percen_okay, too_smalls_avg = ana_obj.check_overlap()
                diff = abs(pert_dict[pert][0] - avg.value())
                err = error.value()
            except Exception as e:
                print(e)
                percen_okay = None
                too_smalls_avg = None
                diff = None
                err = None

            with open(file, "a") as f:
                writer = csv.writer(f)
                row = [pert, eng, pert_atoms, percen_okay, too_smalls_avg, diff, err]
                print(row)
                writer.writerow(row)


if __name__ == "__main__":
    main()
