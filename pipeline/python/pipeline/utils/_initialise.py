import glob
import warnings
import BioSimSpace as BSS
import shutil
from collections import OrderedDict

from typing import Union, Optional

from ._validate import *
from ._files import *
from ._files import write_network as _write_network
from ..prep._protocol import *
from ..analysis._network import *


class initialise_pipeline:
    """class to intialise the pipeline."""

    def __init__(self):
        # need a main folder
        # execuction folder automatically made in main folder
        # protein file path

        self.ligands_dict = None
        self.perturbations = None

        self._ligands_folder = None
        self._protein_path = None

        self._main_folder = None
        self._exec_folder = None

        self.protocol = None
        self.analysis_protocol = None

        self._is_ligands_setup = False

    def ligands_folder(self, folder_path: str = None) -> str:
        """set the folder that contains the ligand files if passed, else state the current ligands folder.

        Args:
            folder_path (str, optional): folder that contains ligands. Defaults to None.

        Raises:
            ValueError: folder must contain .sdf file types.

        Returns:
            str: ligands folder path.
        """

        if folder_path:
            folder_path = validate.folder_path(folder_path)

            ligand_files = glob.glob(f"{folder_path}/*.sdf")
            if len(ligand_files) < 2:
                raise ValueError(
                    "need atleast two *.sdf ligands in folder for setting up the pipeline."
                )

            self._ligands_folder = folder_path

        else:
            pass

        return self._ligands_folder

    def main_folder(
        self, folder_path: str = None, create_exec_folder: bool = True
    ) -> str:
        """create the main folder path for the pipeline, else return its location.

        Args:
            folder_path (str, optional): Path for the main folder. Defaults to None.
            create_exec_folder (bool, optional): whether to create and execution model folder in the main folder. Defaults to True.

        Returns:
            str: main folder path
        """

        if folder_path:
            # make main folder
            self._main_folder = validate.folder_path(folder_path, create=True)

            if create_exec_folder:
                self.exec_folder(f"{self._main_folder}/execution_model")

        else:
            pass

        return self._main_folder

    def exec_folder(self, folder_path: str = None) -> str:
        """create the execution model folder path for the pipeline, else return its location.

        Args:
            folder_path (str, optional): execution model folder path. Defaults to None.

        Returns:
            str: execution model folder path
        """

        if folder_path:
            self._exec_folder = validate.folder_path(folder_path, create=True)
        else:
            pass

        return self._exec_folder

    def protein_path(self, protein_path: str = None) -> str:
        """set the protein path for the parameterised protein used for the pipeline.

        Args:
            protein_path (str, optional): path name of the parameterised protein. Defaults to None.

        Raises:
            ValueError: must be the base name of where the prm7 and rst7 parameterised protein files are.

        Returns:
            str: protein file path
        """

        if protein_path:
            try:
                validate.file_path(f"{protein_path}.rst7")
                validate.file_path(f"{protein_path}.prm7")
                self._protein_path = protein_path
            except:
                try:
                    validate.file_path(f"{protein_path}.top")
                    validate.file_path(f"{protein_path}.gro")
                    self._protein_path = protein_path
                except:
                    raise ValueError(
                        "protein path must be that to rst7/top with prm7/gro file formats for the parameterised protein"
                    )
        else:
            pass

        return self._protein_path

    @staticmethod
    def _setup_ligands(path_to_ligands: str, file_name: str) -> dict:
        """setup the ligands based on the folder.

        Args:
            path_to_ligands (str): folder path to ligands folder.
            file_name (str): file name to write the ligands file to.

        Returns:
            ligands_dict: dicitonary of ligand names and the BSS molecule.
        """

        # generate transformation network based on ligands
        ligand_files = sorted(glob.glob(f"{path_to_ligands}/*.sdf"))

        ligands_dict = OrderedDict()

        for filepath in ligand_files:
            lig = BSS.IO.readMolecules(filepath)[0]
            lig_name = filepath.split("/")[-1].replace(".sdf", "")

            ligands_dict[lig_name] = lig

        print(
            f"there were {len(ligands_dict.keys())} ligands found. These are:\n")
        for lig in ligands_dict.keys():
            print(lig)

        # write ligands file.
        write_ligands(list(ligands_dict.keys()), file_name)

        return ligands_dict

    def setup_ligands(self, file_name: str = None):
        """setup the ligands

        Args:
            file_name (str): file name to write the ligands file to. Default is in the execution model folder as ligands.dat .
                             This setting should not be changed usually as then it's not compatible with later scripts.

        Raises:
            ValueError: need to have previously set a ligands folder
            ValueError: need to have previously set an execution model folder
        """

        if not self._ligands_folder:
            raise ValueError(
                "please provide a ligands folder first using .ligands_folder(path_to_ligands)"
            )
        if not self._exec_folder:
            raise ValueError(
                "please provide an execution model folder first using .exec_folder(path_to_exec) or create it using the .main_folder(path, create_exec_folder=True)"
            )

        if not file_name:
            file_name = f"{self._exec_folder}/ligands.dat"
        else:
            warnings.warn(
                "The file name should not be changed usually as then it's not compatible with later scripts."
            )
            file_name = validate.string(file_name)
            validate.folder_path(
                ("/").join(file_name.split("/")[:-1]), create=True)

        ligands_dict = initialise_pipeline._setup_ligands(
            self._ligands_folder, file_name
        )

        self.ligands_dict = ligands_dict

        self._is_ligands_setup = True

    def remove_ligand(self, lig: str):
        """remove a ligand from the ligands for the pipeline.

        Args:
            lig (str): name of the ligand.
        """

        # if have setup the ligands remove it from the names, ligands and the dict
        if self._is_ligands_setup:
            if lig in self.ligands_dict.keys():
                del self.ligands_dict[lig]

            # write ligands file again as updted
            write_ligands(
                list(self.ligands_dict.keys()
                     ), f"{self._exec_folder}/ligands.dat"
            )

        else:
            print(
                "please setup ligands first (using .setup_ligands() ) before removing any"
            )

    @staticmethod
    def _setup_network(
        ligands_dict: dict, folder: str, links_file: Optional[str] = None
    ) -> dict:
        """setup a network

        Args:
            ligands_dict: dicitonary of ligand names and the BSS molecule.
            folder (str): folder path of where to save the network setup files and visualisation
            links_file (str, optional): file path to a links file to use instead of LOMAP. Defaults to None.

        Returns:
            dict: dictionary of perturbations and their scores
        """

        ligand_names = list(ligands_dict.keys())

        try:
            transformations, lomap_scores = BSS.Align.generateNetwork(
                list(ligands_dict.values()),
                plot_network=True,
                names=ligand_names,
                work_dir=f"{folder}/visualise_network",
                links_file=links_file,
            )
        except:
            try:
                transformations, lomap_scores = BSS.Align.generateNetwork(
                    list(ligands_dict.values()),
                    plot_network=False,
                    names=ligand_names,
                    work_dir=f"{folder}/visualise_network",
                    links_file=links_file,
                )
            except Exception as e:
                print(e)
                print("failed to generate network")

        # dict of perts
        pert_network_dict = {}
        transformations_named = [
            (ligand_names[transf[0]], ligand_names[transf[1]])
            for transf in transformations
        ]
        for score, transf in sorted(zip(lomap_scores, transformations_named)):
            pert_network_dict[transf] = score
            print(transf, score)

        return pert_network_dict

    def setup_network(self, folder: str = "LOMAP", links_file: str = None):
        """setup the network for the ligands, write the scores used for each perturbation and get the perturbations.

        Args:
            folder (str, optional): name of folder in the exection model to setup the network in. Defaults to "LOMAP".
            links_file (str, optional): links file to use instead of LOMAP when generating the network. Defaults to None.
        """

        if not self._is_ligands_setup:
            print("please run setup_ligands before setting up the network")
            return

        pert_network_dict = initialise_pipeline._setup_network(
            self.ligands_dict, f"{self.exec_folder()}/{folder}", links_file=links_file
        )

        self.pert_network_dict = pert_network_dict
        self.perturbations = [
            f"{key[0]}~{key[1]}" for key in pert_network_dict.keys()]

        self.write_network()

    def remove_perturbation(self, pert: str):
        """remove a perturbation from the network for the pipeline.

        Args:
            pert (str): name of the perturbation.
        """

        if pert in self.perturbations:
            self.perturbations.remove(pert)
            del self.pert_network_dict[
                (f"{pert.split('~')[0]}", f"{pert.split('~')[1]}")
            ]

        self.write_network()


    def add_perturbation(self, pert: str, links_file: Optional[str] = None):
        """add a perturbation from the network for the pipeline. Can use a links file for the score.

        Args:
            pert (str): name of the perturbation.
            links_file (str, optional): links file to use instead of LOMAP when generating the network. Defaults to None.
        """

        lig0 = f"{pert.split('~')[0]}"
        lig1 = f"{pert.split('~')[1]}"

        # regenerate the network just for that perturbation
        single_transformation, single_lomap_score = BSS.Align.generateNetwork(
            [self.ligands_dict[lig0], self.ligands_dict[lig1]],
            names=[lig0, lig1],
            plot_network=False,
            links_file=links_file,
        )

        self.pert_network_dict[(lig0, lig1)] = single_lomap_score[0]
        self.perturbations.append(pert)

        self.write_network()


    def run_reverse(self, reverse: bool):
        """whether to also run the network in the reverse direction. Important for writing the network file.

        Args:
            reverse (bool): whether to write the reverse perturbations as well.
        """

        self.protocol.reverse(reverse)
        self.protocol.rewrite_protocol(
            file_path=f"{self.exec_folder()}/protocol.dat")

    def draw_network(self, folder: Optional[str] = None):
        """draw the network.

        Args:
            folder (str, optional): folder path if want to save the image. Defaults to None, image will not be saved.
        """

        if not folder:
            folder = self.exec_folder()

        graph = network_graph(
            list(self.ligands_dict.keys()), self.perturbations)
        graph.draw_graph(file_dir=folder)

    def draw_ligand(self, ligand: str = None):
        """draw the network.

        Args:
            folder (str, optional): folder path if want to save the image. Defaults to None, image will not be saved.
        """

        if not folder:
            folder = self.exec_folder()
        
        ligand = validate.string(ligand)

        graph = network_graph(
            list(self.ligands_dict.keys()), self.perturbations)
        graph.draw_ligand(ligand)

    def setup_protocols(
        self, protocol_dictionary: dict = None, ana_protocol_dictionary: dict = None
    ):
        """set default protocols for the pipeline, consider any passed dictionaries.

        Args:
            protocol_dictionary (dict, optional): dictionary for protocol options. Defaults to None.
            ana_protocol_dictionary (dict, optional): dictionary for analysis protocol options. Defaults to None.
        """

        # setup the protocol, write the file, and add to class object
        protocol = pipeline_protocol(protocol_dictionary, auto_validate=True)
        protocol.rewrite_protocol(
            file_path=f"{self.exec_folder()}/protocol.dat")
        self.protocol = protocol

        # create an analysis protocol
        ana_protocol = analysis_protocol(
            ana_protocol_dictionary, auto_validate=True)
        ana_protocol.rewrite_protocol(
            file_path=f"{self.exec_folder()}/analysis_protocol.dat"
        )
        self.analysis_protocol = ana_protocol

    def add_pipeline_protocol(self, protocol: pipeline_protocol):
        """add a pipeline protocol object.

        Args:
            protocol (pipeline.prep.pipeline_protocol): pipeline protocol to be used.
        """

        self.protocol = validate.pipeline_protocol(protocol)
        self.protocol.rewrite_protocol(
            file_path=f"{self.exec_folder()}/protocol.dat")

    def add_analysis_protocol(self, protocol: analysis_protocol):
        """add an analysis protocol object.

        Args:
            protocol (pipeline.prep.analysis_protocol): analysis protocol to be used.
        """

        self.analysis_protocol = validate.analysis_protocol(protocol)
        self.analysis_protocol.rewrite_protocol(
            file_path=f"{self.exec_folder()}/analysis_protocol.dat"
        )

    def write_network(self, file_path: str = None):
        """write the network file for the pipeline.

        Args:
            file_path (str, optional): file path to write the file to. Defaults to None, will then write to the execution model.
        """

        if not file_path:
            file_path = f"{self.exec_folder()}/network.dat"
        _write_network(self.pert_network_dict, self.protocol, file_path)
        write_lomap_scores(
            self.pert_network_dict, f"{self.exec_folder()}/network_scores.dat"
        )


    def add_source_file(self, file: str):
        """source file that has info for modules, conda paths, MD engines, etc.

        Args:
            file (str): file path
        """

        self.source_file = validate.file_path(file)

    def write_run_all(self):
        """write the run all file needed to run the pipeline. This file should be checked manually."""

        # copy over the files
        print("copying default scripts...")
        shutil.copytree(
            f"{pipeline.__file__.split('__init__.py')[0]}default_scripts",
            f"{self._main_folder}/scripts",
        )
        if self.source_file:
            shutil.copyfile(
                self.source_file, f"{self._main_folder}/scripts/source_file.sh"
            )
        else:
            print("there is no source file. this will create an issue.")

        with open(f"{self._main_folder}/run_all_slurm.sh", "w") as rsh:
            rsh.write(
                f"""\
#!/bin/bash 
#
# generated to run all the lig prep, FEP prep, and the production runs
# please check all file paths and MD engines are sourced to correct path between the hashtags

###########

# export important file locations
export MAINDIRECTORY="{self._main_folder}"
export protein_file="{self._protein_path}"
export ligands_folder="{self._ligands_folder}"

# export all execution model files for later scripts
export lig_file="$MAINDIRECTORY/execution_model/ligands.dat"
export net_file="$MAINDIRECTORY/execution_model/network.dat"
export prot_file="$MAINDIRECTORY/execution_model/protocol.dat"
export ana_file="$MAINDIRECTORY/execution_model/analysis_protocol.dat"

# this should be the location of in the pipeline module
export scripts_dir="$MAINDIRECTORY/scripts" # choose location of scripts

# if have a different, already prepped ligands folder, can change this here
export prep_folder="$MAINDIRECTORY/prep"

# replace trailing ^M
cp $prot_file $prot_file\_0
cp $lig_file $lig_file\_0
cp $net_file $net_file\_0
sed 's/\r$//' $prot_file\_0 > $prot_file
sed 's/\r$//' $lig_file\_0 > $lig_file
sed 's/\r$//' $net_file\_0 > $net_file
rm $prot_file\_0
rm $lig_file\_0
rm $net_file\_0

# sourcing
source $scripts_dir/source_file.sh
source $scripts_dir/extract_execution_model_bash.sh

# chmod all files so can be executed by sbatch.
# chmod u+x $scripts_dir/run_ligprep_slurm.sh
# chmod u+x $scripts_dir/run_fepprep_slurm.sh
# chmod u+x $scripts_dir/run_production_slurm.sh
# chmod u+x $scripts_dir/run_extract_output_slurm.sh
# chmod u+x $scripts_dir/run_analysis_slurm.sh

###########

echo "The folder for all these runs is $MAINDIRECTORY"
echo "The date is $(date)"
echo ${{lig_array[@]}}
echo ${{trans_array[@]}}
echo ${{eng_array[@]}}
echo ${{win_array[@]}}
echo "name is $name"

# make output dir for slurm out and err files
if [[ ! -d slurm_logs ]]; then
    mkdir slurm_logs
fi

# Run the runs
# ligand prep
jidlig=$(sbatch --parsable --array=0-$((${{#lig_array[@]}}-1)) $scripts_dir/run_ligprep_slurm.sh)
echo "ligand prep jobid is $jidlig"

# FEP prep
jidfep=$(sbatch --dependency=afterany:${{jidlig}} --parsable --array=0-$((${{#trans_array[@]}}-1)) $scripts_dir/run_fepprep_slurm.sh)
echo "FEP prep jobid is $jidfep"

# incase rewritten during fepprep, clean protocol file again
cp $prot_file $prot_file\_0
sed 's/\r$//' $prot_file\_0 > $prot_file
rm $prot_file\_0

# Production runs and analysis for the transformation
for i in "${{!trans_array[@]}}"; do
jidprod=$(sbatch --dependency=afterany:${{jidfep}} --parsable --array=0-$((${{win_array[i]}}-1)) $scripts_dir/run_production_slurm.sh ${{trans_array[i]}}$name ${{eng_array[i]}} ${{win_array[i]}})
echo "Production jobid for ${{trans_array[i]}}, ${{eng_array[i]}} is $jidprod"
jidextract=$(sbatch --dependency=afterany:${{jidprod}} --parsable $scripts_dir/run_extract_output_slurm.sh ${{trans_array[i]}} ${{eng_array[i]}})
echo "Extraction jobid for ${{trans_array[i]}}, ${{eng_array[i]}} is $jidextract"
jidana=$(sbatch --dependency=afterany:${{jidextract}} --parsable $scripts_dir/run_analysis_slurm.sh ${{trans_array[i]}} ${{eng_array[i]}})
echo "Analysis jobid for ${{trans_array[i]}}, ${{eng_array[i]}} is $jidana"
done        

"""
            )
