import os
import shutil
import subprocess

import MDAnalysis as mda
import MDAnalysis.transformations as trans

from ._validate import *


def get_repeat_folders(work_dir):
    """how many of each the free and bound repeat folders there are in work dir.

    Args:
        work_dir (string): path to work dir.
    Raises:
        ValueError: can't find bound folders
        ValueError: can't find free folders
    """

    work_dir = validate.folder_path(work_dir)

    # Read how many repeats are in the directory.
    folders = next(os.walk(work_dir))[1]
    b_folders, f_folders = [], []
    for f in folders:
        if "bound" in f:
            b_folders.append(f"{f}")
        elif "free" in f:
            f_folders.append(f"{f}")
        else:
            continue

    # sort the folders
    b_folders.sort()
    f_folders.sort()

    if not b_folders:
        raise ValueError(
            "Couldn't find any folders with 'bound' in the specified directory?"
        )
    elif not f_folders:
        raise ValueError(
            "Couldn't find any folders with 'free' in the specified directory?"
        )

    return b_folders, f_folders


def add_header_simfile(trans_dir):
    """Adds header to simfiles if failed to generate.

    Args:
        trans_dir (str): path to the trans_dir for which headers will be added.
    """
    trans_dir = validate.folder_path(trans_dir)

    items_in_folder = [leg for leg in sorted(os.listdir(trans_dir))]
    legs = []

    for item in items_in_folder:
        if not "pickle" in item:
            if "bound" in item:
                legs.append(item)
            elif "free" in item:
                legs.append(item)
        else:
            pass

    for leg in legs:
        lambdas = []
        if os.path.exists(f"{trans_dir}/{leg}/prod"):
            leg_prod_dir = f"{trans_dir}/{leg}/prod"
        else:
            leg_prod_dir = f"{trans_dir}/{leg}"

        for dir in sorted(os.listdir(f"{leg_prod_dir}")):
            if "lambda" in dir:
                lambdas.append(dir.split("_")[1])

        for lam in lambdas:
            direc = f"{leg_prod_dir}/lambda_{lam}"

            sim_okay = False
            hash_counter = 0
            line_counter = 0
            # first check if the simfile.dat in the folder needs a new header
            # only check the first few lines
            with open(f"{direc}/simfile.dat") as sfile:
                for line in sfile.readlines():
                    if line.startswith("#"):
                        hash_counter += 1
                        if hash_counter == 8:
                            sim_okay = True
                            break
                    if line_counter == 15:
                        break

            sim_okay = False
            if not sim_okay:
                print(f"will write header for simfiles in {direc}")
                try:
                    if not os.path.exists(f"{direc}/old_simfile.dat"):
                        os.rename(f"{direc}/simfile.dat",
                                  f"{direc}/old_simfile.dat")

                    if not os.path.exists(f"{direc}/old_old_simfile.dat"):
                        os.rename(
                            f"{direc}/old_simfile.dat", f"{direc}/old_old_simfile.dat"
                        )

                    file1 = open(f"{direc}/old_old_simfile.dat", "r")
                    file2 = open(f"{direc}/old_simfile.dat", "w")

                    # reading each line from original
                    for line in file1.readlines():
                        if not (line.startswith("#")):
                            if not (line.startswith("Simulation")):
                                if not (line.startswith("Generating")):
                                    file2.write(line)

                    # close and save the files
                    file2.close()
                    file1.close()

                except:
                    continue

                # get the timestep from the somd.cfg
                try:
                    with open(f"{direc}/somd.cfg", "r") as file:
                        for line in file.readlines():
                            if "timestep" in line:
                                timestep = float(
                                    line.split("=")[-1].strip().split(" ")[0]
                                )
                                break

                    last_line = (
                        subprocess.check_output(
                            ["tail", "-1", f"{direc}/old_old_simfile.dat"], timeout=3
                        )
                        .decode("utf-8")
                        .strip()
                    )
                    final_step = float(
                        last_line.strip().split(" ")[0].strip()) + 200
                    sim_time = final_step * timestep / 1000
                    extr_timestep = True
                except:
                    extr_timestep = False

                file = open(f"{direc}/simfile_header.dat", "w")

                file.write("#This file was generated to fix the header \n")
                file.write(
                    "#Using the somd command, of the molecular library Sire version <2022.3.0> \n"
                )
                file.write(
                    "#For more information visit: https://github.com/michellab/Sire \n"
                )
                file.write("# \n")
                file.write("#General information on simulation parameters: \n")
                if extr_timestep:
                    file.write(
                        f"#Simulation used {int(final_step)} moves, 1 cycles and {int(sim_time)} ps of simulation time \n"
                    )
                else:
                    file.write(
                        f"#Simulation used 250000 moves, 4 cycles and 4000 ps of simulation time \n"
                    )
                file.write(f"#Generating lambda is		 {lam}\n")
                file.write(f"#Alchemical array is		 {(', ').join(lambdas)}\n")
                file.write("#Generating temperature is 	300 t\n")
                file.write("#Energy was saved every 200 steps \n")
                file.write("#\n")
                file.write("#\n")
                file.write(
                    "#   [step]      [potential kcal/mol]       [gradient kcal/mol]      [forward Metropolis]     [backward Metropolis]                   [u_kl] \n"
                )
                file.close()

                # merge both files
                simfiles_tomerge = [
                    f"{direc}/simfile_header.dat",
                    f"{direc}/old_simfile.dat",
                ]

                with open(f"{direc}/simfile.dat", "w") as outfile:
                    for names in simfiles_tomerge:
                        with open(names) as infile:
                            outfile.write(infile.read())

            else:
                pass


class extract:
    """class for extracting output data"""

    def __init__(self, trans_dir, extract_dir=None):
        self.trans_dir = validate.folder_path(trans_dir)
        # need to check that the trans directory contains outputs
        if not "outputs" in self.trans_dir:
            raise ValueError(f"{trans_dir} must be in an 'outputs' folder.")
        # set the extract directory
        if extract_dir:
            self.extract_dir = validate.folder_path(extract_dir, create=True)
        else:
            extract._set_extract_dir(self)

    def _set_extract_dir(self):
        """set the directory for extracted data based on the passed trans_dir"""

        # rename so just output_extracted instead of output
        extract_dir = f"{self.trans_dir.split('outputs')[0]}outputs_extracted{self.trans_dir.split('outputs')[1]}"

        # make directory
        extract_dir = validate.folder_path(extract_dir, create=True)
        print(f"using {extract_dir} for target file location as none was set.")

        self.extract_dir = extract_dir

    def extract_config(self):
        """Extracts sample config files from the directory.

        Args:
            main_dir (str): Main directory, outputs directory.
        """
        extract._extract_config(self.trans_dir, self.extract_dir)

    @staticmethod
    def _extract_config(folder, extract_dir):
        """extract sample config files used for run at lambda=0.0000.

        Args:
            folder (str): folder path to the folder to extract data from
            extract_dir (str): folder path to the extracted dir
        """
        dir_list = [dirs[0] for dirs in os.walk(folder)]

        for dirs in dir_list:
            if "lambda_0.0000" in dirs:
                file_names = []
                new_dir = validate.folder_path(
                    f"{extract_dir}{dirs.split(f'{folder}')[1]}", create=True
                )
                if "min" in dirs:
                    if len([x for x in validate.engines() if x in dirs]) > 1:
                        print(
                            "There are too many engines found in the file path. will check for all output formats....")
                        file_names = ["amber.cfg", "amber.prm7", "initial_amber.rst7",
                                      "gromacs.mdp", "initial_gromacs.gro", "gromacs.top",]
                    elif "AMBER" in dirs:
                        file_names = ["amber.cfg",
                                      "amber.prm7", "initial_amber.rst7"]
                    elif "GROMACS" in dirs:
                        file_names = [
                            "gromacs.mdp",
                            "initial_gromacs.gro",
                            "gromacs.top",
                        ]
                elif "eq" in dirs:
                    if len([x for x in validate.engines() if x in dirs]) > 1:
                        print(
                            "There are too many engines found in the file path. will check for all output formats....")
                        file_names = ["amber.cfg", "gromacs.mdp",
                                      "somd.cfg", "somd.prm7", "initial_somd.rst7"]
                    elif "AMBER" in dirs:
                        file_names = ["amber.cfg"]
                    elif "GROMACS" in dirs:
                        file_names = ["gromacs.mdp"]
                    elif "SOMD" in dirs:
                        file_names = ["somd.cfg",
                                      "somd.prm7", "initial_somd.rst7"]
                else:
                    if len([x for x in validate.engines() if x in dirs]) > 1:
                        print(
                            "There are too many engines found in the file path. will check for all output formats....")
                        file_names = ["amber.cfg", "gromacs.mdp", "somd.cfg"]
                    elif "AMBER" in dirs:
                        file_names = ["amber.cfg"]
                    elif "GROMACS" in dirs:
                        file_names = ["gromacs.mdp"]
                    elif "SOMD" in dirs:
                        file_names = ["somd.cfg"]
                if not file_names:
                    print(
                        "no engine found in filepath, will try to extract each engine's input config file format..."
                    )
                    file_names = ["amber.cfg", "somd.cfg", "gromacs.mdp"]

                for file in file_names:
                    try:
                        shutil.copyfile(f"{dirs}/{file}", f"{new_dir}/{file}")
                    except:
                        print(
                            f"{dirs} does not have a recognised input file, {str(file)}."
                        )

    def extract_output(self):
        """Extracts the output files from the directory.

        Args:
            main_dir (str): Main directory, outputs directory.
        """
        extract._extract_output(self.trans_dir, self.extract_dir)

    @staticmethod
    def _extract_output(folder, extract_dir):
        """extract simulation output needed for the AFE analysis.

        Args:
            folder (str): folder path to the folder to extract data from
            extract_dir (str): folder path to the extracted dir
        """

        dir_list = [dirs[0] for dirs in os.walk(folder)]
        dirs_found = 0

        # exclude the min, heat, eq directories from extraction.
        for dirs in dir_list:
            if "prod" in dirs:
                extract = True
                dirs_found += 1
            elif not "min" in dirs:
                if not "heat" in dirs:
                    if not "eq" in dirs:
                        extract = True
                        dirs_found += 1
            else:
                extract=False
            
            if extract_dir:
                if "lambda" in dirs:
                    new_dir = validate.folder_path(
                        f"{extract_dir}{dirs.split(f'{folder}')[1]}",
                        create=True,
                    )
                    # check if too many engines in the file path
                    if len([x for x in validate.engines() if x in dirs]) > 1:
                        print(
                            "There are too many engines found in the file path. will check for all output formats....")
                        file_names = ["amber.out",
                                    "gromacs.xvg", "simfile.dat"]
                    elif "AMBER" in dirs:
                        file_names = ["amber.out"]
                    elif "GROMACS" in dirs:
                        file_names = ["gromacs.xvg"]
                    elif "SOMD" in dirs:
                        file_names = ["simfile.dat"]
                    # if cant find the engine in the file path try all
                    else:
                        print(
                            "no engine found in filepath, will try to extract each engine's output format..."
                        )
                        file_names = ["amber.out",
                                    "gromacs.xvg", "simfile.dat"]

                    for file in file_names:
                        try:
                            shutil.copyfile(
                                f"{dirs}/{file}", f"{new_dir}/{file}"
                            )
                            if os.path.getsize(f"{new_dir}/{file}") == 0:
                                print(
                                    f"File extracting to '{new_dir}/{file}' is empty!"
                                )
                        except:
                            print(
                                f"{dirs} does not have a recognised input file, {str(file)}."
                            )

        if dirs_found == 0:
            raise ValueError(
                "no directories were found to extract from! make sure there is not min/heat/eq in the main folder path as these will be excluded."
            )

    def extract_frames(self, traj_lambdas=None, rmsd=True, overwrite=False):
        """extract the rmsd and/or frames for certain lambda windows.

        Args:
            traj_lambdas (list, optional): lambda values at which the trajectory should be extracted. Defaults to None.
            rmsd (bool, optional): whether or not to calculate the rmsd of the ligand for the simulation. Defaults to True.
            overwrite (bool, optional): wether to overwrite any existing extracted data in the folder. Defaults to False. If True, will rextract.

        Raises:
            ValueError: _description_
        """

        trans_dir = self.trans_dir
        rmsd = validate.boolean(rmsd)
        overwrite = validate.boolean(overwrite)

        # get all lambda windows
        items_in_folder = [leg for leg in sorted(os.listdir(trans_dir))]
        legs = []

        if traj_lambdas:
            traj_lambdas = validate.is_list(traj_lambdas)
        else:
            print("no traj_lambdas provided, will not extract any frames.")
            traj_lambdas = []

        # exclude pickle folder from list, only do bound and free legs
        for item in items_in_folder:
            if not "pickle" in item:
                if "bound" in item:
                    legs.append(item)
                elif "free" in item:
                    legs.append(item)
            else:
                pass

        for leg in legs:
            lambdas = []

            if os.path.exists(f"{trans_dir}/{leg}/prod"):
                leg_prod_dir = f"{trans_dir}/{leg}/prod"
            else:
                leg_prod_dir = f"{trans_dir}/{leg}"

            for dir in sorted(os.listdir(f"{leg_prod_dir}")):
                if "lambda" in dir:
                    lambdas.append(dir.split("_")[1])

            for lam in lambdas:
                # only do it for considered lambdas
                if lam in traj_lambdas:
                    # get target directory for extraction and create
                    direc = f"{leg_prod_dir}/lambda_{lam}"
                    traj_extract_dir = (
                        f"{direc.replace(self.trans_dir, self.extract_dir)}"
                    )
                    traj_extract_dir = validate.folder_path(
                        traj_extract_dir, create=True
                    )

                    # check if the output files already exist in the target directory
                    if not overwrite:
                        file_exists = False
                        onlyfiles = [
                            f
                            for f in os.listdir(traj_extract_dir)
                            if os.path.isfile(f"{traj_extract_dir}/{f}")
                        ]
                        if "system_0.pdb" in onlyfiles:
                            file_exists = True
                        if rmsd:
                            if "rmsd_ligand.png" in onlyfiles:
                                file_exists = True
                            else:
                                file_exists = False

                        if file_exists:
                            print(
                                "traj extracted already exists, will NOT extract again."
                            )
                            return

                    # create mda universe based on file type
                    if "SOMD" in direc:
                        # must be parm7 for mda
                        shutil.copy(f"{direc}/somd.prm7",
                                    f"{direc}/somd.parm7")
                        coord_file = validate.file_path(f"{direc}/somd.parm7")

                        # combine traj files
                        traj_files = []
                        for file in os.listdir(direc):
                            if "dcd" in file:
                                traj_files.append(
                                    validate.file_path(f"{direc}/{file}"))
                        if not traj_files:
                            raise ValueError(
                                "there are no dcd trajectory files for somd in this folder."
                            )

                    elif "AMBER" in direc:
                        # must be parm7 for mda
                        shutil.copy(f"{direc}/amber.prm7",
                                    f"{direc}/amber.parm7")
                        coord_file = validate.file_path(f"{direc}/amber.parm7")

                        traj_files = validate.file_path(f"{direc}/amber.nc")

                    elif "GROMACS" in direc:
                        coord_file = validate.file_path(f"{direc}/gromacs.tpr")

                        traj_files = validate.file_path(f"{direc}/gromacs.trr")

                    u = mda.Universe(coord_file, traj_files)
                    if "free" in leg:
                        u = extract.centre_molecule(u, "resname LIG")
                    elif "bound" in leg:
                        u = extract.centre_molecule(u, "protein")

                    # want to write the frames for the
                    extract._write_traj_frames(u, traj_extract_dir)
                    if rmsd:
                        extract._rmsd_trajectory(u, traj_extract_dir)

                    del u

                else:
                    pass

    @staticmethod
    def centre_molecule(universe, selection):
        """centre the molecule (eg either 'resname LIG' or 'protein') in the mda universe/

        Args:
            universe (MDAnalysis.Universe): mda analysis universe
            selection (str): a valid mda selection criterion. (eg 'resname LIG' or 'protein')

        Returns:
            MDAnalysis.Universe: mda universe centred around the molecule selection
        """

        u = universe

        # want to centre the protein/ligand
        molecules = u.select_atoms(f"{selection}")
        not_molecules = u.select_atoms(f"not {selection}")
        transforms = [
            trans.unwrap(molecules),
            trans.center_in_box(molecules),  # , wrap=True, center="mass"
            trans.wrap(not_molecules),
        ]
        u.trajectory.add_transformations(*transforms)

        return u

    @staticmethod
    def _write_traj_frames(universe, traj_extract_dir):
        """write 5 evenly spaced trajectory frames from the universe. Start, three evenly spaced from the middle, end.

        Args:
            universe (MDAnalysis.Universe): mda universe, best if centred
            traj_extract_dir (str): folder path to where the frames should be written
        """

        write_dir = traj_extract_dir
        u = universe

        print("starting to write trajectory frames...")
        timesteps = [u.trajectory.ts for ts in u.trajectory]
        timestep_freq = int(len(timesteps) / 4)
        # five frames - if 4 ns, this is every ns
        timesteps_used = [
            0,
            timestep_freq - 1,
            (timestep_freq * 2) - 1,
            (timestep_freq * 3) - 1,
            int(len(timesteps) - 1),
        ]

        for time in timesteps_used:
            u.trajectory[time]
            with mda.Writer(f"{write_dir}/system_{str(time)}.pdb") as W:
                W.write(u)

        print("finished writing 5 frames to pdb.")

    @staticmethod
    def _rmsd_trajectory(universe, traj_extract_dir):
        """rmsd of the ligand (with name 'LIG') over the course of the trajectory

        Args:
            universe (MDAnalysis.Universe): mda universe
            traj_extract_dir (str): folder path to where the frames should be written
        """

        u = universe
        ref = universe
        write_dir = traj_extract_dir

        import MDAnalysis.analysis.rms

        R = MDAnalysis.analysis.rms.RMSD(
            u,
            ref,
            select="resname LIG",
            groupselections=["resname LIG and resname LIG",
                             "backbone and resname LIG"],
        )
        R.run()

        import matplotlib.pyplot as plt

        rmsd = R.rmsd.T  # transpose makes it easier for plotting
        time = rmsd[1]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.plot(time, rmsd[2], "k-", color="teal")
        ax.legend(loc="best")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel(r"RMSD of ligand ($\AA$)")
        fig.savefig(f"{write_dir}/rmsd_ligand.png")

        print(
            f"the maximum rmsd of the ligand is {max(rmsd[2])} and the minimum is {min(rmsd[2])}"
        )

    @staticmethod
    def extract_output_all(main_dir, extract_dir=None):
        """Extracts only the output files if an output directory is given.

        Args:
            main_dir (str): must be the outputs directory.
        """
        # this should be the outputs directory
        main_dir = validate.folder_path(main_dir)

        if not extract_dir:
            extract_dir = f"{main_dir}_extracted"

        # make directory
        extract_dir = validate.folder_path(extract_dir, create=True)

        extract._extract_output(main_dir, extract_dir)
        extract._extract_config(main_dir, extract_dir)
