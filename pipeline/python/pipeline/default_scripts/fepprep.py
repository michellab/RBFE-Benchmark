#!/usr/bin/python3

import BioSimSpace as BSS
from distutils.dir_util import remove_tree
from argparse import ArgumentParser
import pickle

from pipeline.prep import *
from pipeline.utils import *


def fep_prep(pert, prot_file, num_lambda_query, engine_query, main_dir, prep_dir):
    lig_1 = pert.split("~")[0]
    lig_2 = pert.split("~")[1]

    workdir = f"{main_dir}/outputs/{engine_query}/{lig_1}~{lig_2}"  # pert dir

    # parse protocol file
    print("reading in the protocol file...")
    protocol = pipeline_protocol(prot_file)  # instantiate the protocol as an object
    print("validating the protocol file...")
    protocol.validate()  # validate all the input
    # add the number of lambdas and engine to the protocol
    protocol.num_lambda(num_lambda_query)
    protocol.engine(engine_query)
    if protocol.name():
        workdir += f"_{protocol.name()}"

    print("the protocol is now:")
    protocol.print_protocol()

    # instantiate each system as a fepprep class with the protocol
    fepprep_obj = fepprep(protocol=protocol)

    for name, leg in zip(["lig", "sys"], ["free", "bound"]):
        # Load equilibrated inputs for both ligands
        system_1 = BSS.IO.readMolecules(
            [
                f"{prep_dir}/{lig_1}_{name}_equil_solv.rst7",
                f"{prep_dir}/{lig_1}_{name}_equil_solv.prm7",
            ]
        )
        system_2 = BSS.IO.readMolecules(
            [
                f"{prep_dir}/{lig_2}_{name}_equil_solv.rst7",
                f"{prep_dir}/{lig_2}_{name}_equil_solv.prm7",
            ]
        )

        fepprep_obj.add_system(system_1, free_bound=leg, start_end="start")
        fepprep_obj.add_system(system_2, free_bound=leg, start_end="end")

    # remove any existing files in the workdir if not rerun.
    if protocol.rerun():
        pass
    else:
        try:
            remove_tree(workdir)
        except:
            pass

    kwargs = {}

    # load in any predefined mapping
    if os.path.exists(f"{main_dir}/inputs/mapping/{lig_1}~{lig_2}_mapping_dict.pickle"):
        with open(
            f"{main_dir}/inputs/mapping/{lig_1}~{lig_2}_mapping_dict.pickle",
            "rb",
        ) as handle:
            mapping_dict = pickle.load(handle)
            kwargs = {"mapping": mapping_dict}
            print(f"using {mapping_dict} as the mapping...")

    # generate folder based on fepprep protocol (both or start)
    fepprep_obj.generate_folders(workdir, **kwargs)


def check_arguments(args):
    # pass the checks to the other check functions
    if args.perturbation:
        perturbation = args.perturbation
    else:
        perturbation = str(input("what is the perturbation?: ")).strip()

    if args.num_lambda:
        num_lambda = args.num_lambda
    else:
        num_lambda = str(input("what is the number of lambda windows?: ")).strip()

    if args.engine:
        engine = args.engine
    else:
        engine = str(input("what is the engine?: ").strip())

    if args.main_folder:
        main_folder = args.main_folder
    else:
        main_folder = str(input("what is the main folder of the run?: ")).strip()

    if args.protocol_file:
        protocol_file = args.protocol_file
    else:
        protocol_file = str(input("what is the path to the protocol file?: ").strip())

    if args.prep_folder:
        prep_folder = args.prep_folder
    else:
        prep_folder = f"{main_folder}/prep"

    return perturbation, protocol_file, num_lambda, engine, main_folder, prep_folder


def main():
    # accept all options as arguments
    parser = ArgumentParser(description="run the fepprep")
    parser.add_argument(
        "-pert", "--perturbation", type=str, default=None, help="name of perturbation"
    )
    parser.add_argument(
        "-lam", "--num_lambda", type=str, default=None, help="number of lambda windows"
    )
    parser.add_argument(
        "-eng", "--engine", type=str, default=None, help="engine to be used"
    )
    parser.add_argument(
        "-mf",
        "--main_folder",
        type=str,
        default=None,
        help="main folder path to create for all the runs",
    )
    parser.add_argument(
        "-p", "--protocol_file", type=str, default=None, help="path to protocol file"
    )
    parser.add_argument(
        "-prep",
        "--prep_folder",
        type=str,
        default=None,
        help="folder with the prepped ligands",
    )
    args = parser.parse_args()

    # check arguments
    print("checking the provided command line arguments...")
    (
        pert,
        prot_file,
        num_lambda_query,
        engine_query,
        main_folder,
        prep_dir,
    ) = check_arguments(args)

    print(f"fepprep for {pert, prot_file, num_lambda_query, engine_query, main_folder}")

    fep_prep(pert, prot_file, num_lambda_query, engine_query, main_folder, prep_dir)


if __name__ == "__main__":
    main()
