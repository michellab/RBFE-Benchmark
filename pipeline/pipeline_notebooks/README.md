# running the pipeline

To run the pipeline, it is possible to use the CLI:

`python initialise_pipeline.py -lf path_to_ligands_folder -pf path_to_protein_file -mf path_to_main_folder_to_setup`

alternatively:

`python initialise_pipeline.py`
and it will also ask for the above folder/file paths.

Additional input that will be asked for during setup:
   - Sampling time (per window)
   - MD engines to use
   - Whether HMR should be applied
   - How many repeats to run
   - Whether to save the trajectories

Then, the network can be edited as follows:
   - Remove any ligands to generate the network for
   - Add perturbations
   - Remove perturbations
   - Whether to also run in the reverse direction

Finally, it will ask for a path to a sources file that includes all the module exports/pythonpaths.
eg:
```
export BSS="/home/anna/anaconda3/bin/activate /home/anna/mambaforge/envs/openbiosim-dev"
export amber="/home/anna/amber22"
export gromacs="/usr/local/gromacs-23.1/bin/GMXRC"

# sourcing
source $BSS
source $amber/amber.sh
source $gromacs

export PYTHONPATH="$HOME/BioSimSpace/python:$PYTHONPATH"
export PYTHONPATH="$HOME/Documents/pipeline/python:$PYTHONPATH"

```
can also for example include lines such as:
```
module load amber/22
```
for the slurm scheduler.

The setup can also be carried out using the setup_notebook.ipynb in this folder. This provides more extensive protocol editing and visualisation options.

In the main folder, the following folders will be generated:
execution_model (contains: network.dat, ligands.dat, protocol.dat, analysis_protocol.dat)
scripts (contains: all scripts for runnning pipeline)
run_all_slurm.sh

The run_* files in the scripts folder need to be adjusted for the slurm scheduler (nodes, runtime, etc).

The first job, ligprep, parameterises the ligands, adds them to the protein system, solvates and equilibrates. It makes the prep folder of all the prepped ligands, which also contains folders where the equilibration files are kept. The ligand files are named sys for protein+ligand and lig for just the solvated ligand.

Fepprep creates all the input files for the runs. This includes merging the ligands and generating the files in the outputs folder.

Prod(uction) executes a bash script for running all the AFE runs with their respective MD engines. The allocation is one lambda window per GPU. 

Extract creates the outputs_extracted folder. This is incase the trajectory files as a result of the production runs need to be moved to alternative storage locations. The folder structure is the same as for the outputs folder. A config file for lambda_0.0 for each part of the run is also copied over to keep track of any passed settings.

Ana(lysis) analyses the outputs. This creates a results folder in the outputs_extracted folder, which contains the final_summary_*, free_repeat_*, bound_repeat_*, freenrg_repeat_* files, extension depending on engine and analysis method. These are needed later for the analysis notebook. By default, pickles are also saved of the analysed objects in a pickle folder.

running the run_all_slurm.sh should start all the runs.

The final directory structure should be:

main folder
 |_ execution_model
    |_ ligands file, network file, protocol files
 |_ prep
    |_ prepped ligand files
 |_ outputs
    |_ ENGINE
        |_ lig_0~lig_1
            |_ bound_0
            |_ free_0
 |_ outputs_extracted
    |_ ENGINE ....
    |_ results
    |_ analysis
 |_ scripts

The results can then be computed using the analyse_pipeline.py, but it is better to use the notebook (analysis_notebook.ipynb). This includes some examples of analyses that can be carried out, but are not exhaustive.
