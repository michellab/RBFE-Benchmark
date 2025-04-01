#!/usr/bin/python3

import BioSimSpace as BSS
from argparse import ArgumentParser
import sys
import os
import logging

BSS.setVerbose(True)

from pipeline.utils import *
from pipeline.prep import *


def lig_prep(
    main_dir,
    protocol_file,
    protein_file,
    ligands_folder,
    lig_name,
    engine,
):
    # Exit if prep files are already available for ligand and protein
    if os.path.exists(f"{main_dir}/prep/{lig_name}_sys_equil_solv.prm7"):
        if os.path.exists(f"{main_dir}/prep/{lig_name}_lig_equil_solv.prm7"):
            print(f"Prep files already generated for {lig_name}. Done.")
            sys.exit(0)

    # make other folders that need to be made for the prepped files
    folders = [
        f"{main_dir}/prep",
        f"{main_dir}/prep/pre_run",
        f"{main_dir}/prep/solvation",
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
        else:
            pass

    # parse protocol file
    protocol = pipeline_protocol(protocol_file)  # instantiate the protocol as an object
    protocol.validate()  # validate all the input into needed format.

    # load, solvate and run the systems.
    # load the protein, this was paramaterised during the setup stage.
    try:
        prot_wat = BSS.IO.readMolecules(
            [f"{protein_file}.rst7", f"{protein_file}.prm7"]
        )
    except:
        try:
            prot_wat = BSS.IO.readMolecules(
                [f"{protein_file}.top", f"{protein_file}.gro"]
            )
        except:
            logging.critical(
                "could not read in parameterised protein file. Use either rst7/top with prm7/gro."
            )

    # load ligand, these should already be in the correct position
    try:  # sdf first
        ligand = BSS.IO.readMolecules(f"{ligands_folder}/{lig_name}.sdf")[0]
    except:  # mol2 if sdf is not available
        ligand = BSS.IO.readMolecules(f"{ligands_folder}/{lig_name}.mol2")[0]

    # paramaterise the ligand
    print(f"Parameterising {lig_name}...")
    lig_p = ligprep.lig_paramaterise(ligand, protocol.ligand_forcefield())

    # Combine protein, ligand and crystallographic waters.
    system = lig_p + prot_wat

    # Solvate and run each the bound and the free system.
    legs_mols, legs = [lig_p, system], ["lig", "sys"]

    # zip together the molecules in that leg with the name for that leg
    for leg, leg_mol in zip(legs, legs_mols):
        # solvate
        print(f"Solvating {leg} for {lig_name}...")
        leg_mol_solvated = ligprep.minimum_solvation(
            leg_mol,
            protocol.solvent(),
            protocol.box_type(),
            protocol.box_edges(),
            protocol.box_edges_unit(),
            work_dir=f"{main_dir}/prep/solvation/{lig_name}",
        )

        # saving pre runs
        print(f"Saving solvated for {leg} and {lig_name}")
        BSS.IO.saveMolecules(
            f"{main_dir}/prep/pre_run/{lig_name}_{leg}_s",
            leg_mol_solvated,
            ["PRM7", "RST7", "PDB"],
        )

        # minimise and eqiulibrate these
        print(f"minimising and equilibrating for {leg} and {lig_name}")
        leg_equil_final = minimise_equilibrate_leg(
            leg_mol_solvated, engine, work_dir=f"{main_dir}/prep/mineq/{lig_name}"
        )

        # finally, save last snapshot
        print(f"Saving solvated/equilibrated for {leg} and {lig_name}")
        BSS.IO.saveMolecules(
            f"{main_dir}/prep/{lig_name}_{leg}_equil_solv",
            leg_equil_final,
            ["PRM7", "RST7", "PDB"],
        )
        print(f"Done for {lig_name}.")


def check_arguments(args):
    # pass the checks to the other check functions
    if args.ligand_name:
        lig_name = args.ligand_name
    else:
        lig_name = str(input("what is the ligand name?: ")).strip()

    if args.ligands_folder:
        ligands_folder = args.ligands_folder
    else:
        ligands_folder = str(input("what is the ligands folder?: ")).strip()

    if args.protein_path:
        protein_path = args.protein_path
    else:
        protein_path = str(input("what is the parameterised protein path?: ").strip())

    if args.main_folder:
        main_folder = args.main_folder
    else:
        main_folder = str(
            input("what is the main folder where all the files should go?: ")
        ).strip()

    if args.protocol_file:
        protocol_file = args.protocol_file
    else:
        protocol_file = str(input("what is the path to the protocol file?: ").strip())

    return main_folder, protocol_file, protein_path, ligands_folder, lig_name


def main():
    # accept all options as arguments
    parser = ArgumentParser(description="run the ligprep")
    parser.add_argument(
        "-lig", "--ligand_name", type=str, default=None, help="name of ligand"
    )
    parser.add_argument(
        "-lf",
        "--ligands_folder",
        type=str,
        default=None,
        help="folder path to the ligand files",
    )
    parser.add_argument(
        "-prot",
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
        "-p", "--protocol_file", type=str, default=None, help="path to protocol file"
    )

    args = parser.parse_args()

    # check arguments
    print("checking the provided command line arguments...")
    (
        main_dir,
        protocol_file,
        protein_file,
        ligands_folder,
        lig_name,
    ) = check_arguments(args)

    print(f"prep for {lig_name}")

    engine = "AMBER"

    lig_prep(main_dir, protocol_file, protein_file, ligands_folder, lig_name, engine)


if __name__ == "__main__":
    main()
