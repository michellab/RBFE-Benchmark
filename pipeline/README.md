# pipeline

Copied from https://github.com/annamherz/pipeline and edited for clarity.

### To install the code:

First, create a conda/mamba environment:

`conda create --name pipeline python=3.10 pip`

Install requirements:

`conda install -c conda-forge openmm openff-toolkit openff-interchange openff-units openff-utilities openff-forcefields lomap2`

`conda install -c openbiosim sire==2024.3.0`

and also install the requirements listed:

`pip install -r requirements.txt`

Please note these requirements were generated from the conda environment used to run the project. Some dependencies may need to be adjusted.

Clone the BioSimSpace version used for the project at (https://github.com/michellab/BioSimSpace/releases/tag/RBFE-benchmark) and install this in the environment, or for more a more recent version please visit https://github.com/openbiosim/biosimspace .

Follow the instructions there to install the chosen branch.

To use flex align, kcombu also needs to be installed:

`conda install openbiosim::kcombu_bss`

Finally, go to the python directory and install the pipeline in the environment using:

`python setup.py install`


To run additional network analysis using either FreeEnergyNetworkAnalysis (https://github.com/michellab/FreeEnergyNetworkAnalysis) or MBARNet (https://gitlab.com/RutgersLBSR/fe-toolkit), please follow their instructions for installation and use. Default network analysis proceeds using cinnabar.


### Outline of folders:

pipeline_notebooks - notebooks and scripts for starting and analysing the pipeline

python - contains all the code 

other_scripts - contains additional notebooks. These were used for the analysis presented. Absolute file paths may need to be adjusted.


### Outline of the pipeline:

After the initial setup of the pipeline (pipeline_notebooks folder), a main folder with the settings in `execution_model` should have been created. This contains the `ligands.dat`, `network.dat`, `protocol.dat`, and `analysis_protocol.dat` that can be edited as required. In the main folder, running: `bash run_all_slurm.sh` will start the entire pipeline for a slurm workload manager in a series of dependencies. The run_*_slurm scripts generated in the `scripts` folder may need to be adjusted depending on the slurm cluster and resources available. After all the runs are finished, they can be analysed in a jupyter notebook following the example in the pipeline_notebooks folder.

The workflow and scripts are outlined in the figure below:

![](pipeline_outline.png)
