#!/usr/bin/python3

# import BioSimSpace as BSS
from argparse import ArgumentParser

from pipeline.utils import *
from pipeline.prep import *


def extract_output(folder, prot_file):
    # read in protocol
    protocol = pipeline_protocol(prot_file, auto_validate=True)

    # so can pass with name, but will also append if not there
    if protocol.name():
        if protocol.name() != str(folder).split("_")[-1]:
            folder += f"_{protocol.name()}"
            print(
                f"name of the protocol ({protocol.name()}) is not in the folder path, will use:\n"
                f"{folder} as folder path for this run..."
            )

    if protocol.trajectories() == "None":
        traj_lambdas = []
    if protocol.trajectories() == "0,0.5,1":
        traj_lambdas = ["0.0000", "0.5000", "1.0000"]
    if protocol.trajectories() == "0,1":
        traj_lambdas = ["0.0000", "1.0000"]
    if protocol.trajectories() == "All":
        free_folders = [
            name
            for name in os.listdir(f"{folder}/free_0")
            if os.path.isdir(os.path.join(f"{folder}/free_0", name))
        ]
        traj_lambdas = [
            name.replace("lambda", "").replace("_", "")
            for name in free_folders
            if "lambda" in name
        ]

    # simfile header
    if "SOMD" in folder:
        try:
            print("adding header to simfile for SOMD...")
            add_header_simfile(folder)
        except:
            print(f"could not add the header to simfile in {folder}")

    # extract to output folder
    # initialise
    extraction = extract(folder)

    # get the output from the folder to new folder
    print("extracting output files...")
    extraction.extract_output()
    # # get trajectory, will get rmsd by default
    # print("extracting trajectory files...")
    # extraction.extract_frames(traj_lambdas=traj_lambdas, overwrite=False)
    print("extracting sample config files...")
    extraction.extract_config()


def check_arguments(args):
    # pass the checks to the other check functions
    if args.folder:
        folder = args.folder
    else:
        folder = str(input("what is the folder to extract?: ")).strip()

    if args.protocol_file:
        protocol_file = args.protocol_file
    else:
        protocol_file = str(input("what is the path to the protocol file?: ").strip())

    return folder, protocol_file


def main():
    # accept all options as arguments
    parser = ArgumentParser(description="extract the output")
    parser.add_argument(
        "-f", "--folder", type=str, default=None, help="folder to extract"
    )
    parser.add_argument(
        "-p", "--protocol_file", type=str, default=None, help="path to protocol file"
    )
    args = parser.parse_args()

    # check arguments
    print("checking the provided command line arguments...")
    folder, protocol_file = check_arguments(args)

    extract_output(folder, protocol_file)


if __name__ == "__main__":
    main()
