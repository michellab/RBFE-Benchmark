# pipeline

Copied from https://github.com/annamherz/pipeline and edited for clarity.

### To install the code:

First, create a mamba environment:

```mamba create --name pipeline python=3.10 pip```

Next, activate the environment:

```mamba activate pipeline```

Following this, install the following requirements:

```mamba install -c conda-forge openmm==8.1.2 openff-toolkit==0.16.6 openff-interchange==0.4.0 openff-units==0.2.2 openff-utilities==0.1.12 openff-forcefields==2024.09.0 lomap2==3.1.0```

```mamba install -c openbiosim sire==2024.3.0```

To use flex align, kcombu also needs to be installed:

```mamba install openbiosim::kcombu_bss```

The remaining requirements can be installed using:

```pip install -r requirements.txt```

Clone the BioSimSpace version used for the project at https://github.com/michellab/BioSimSpace/releases/tag/RBFE-benchmark and install this in the environment following the instructions on the repository.
More recent versions of BioSimSpace are available at https://github.com/openbiosim/biosimspace .

Finally, go to the python directory and install the pipeline in the environment using:

```python setup.py install```

To run additional network analysis using either FreeEnergyNetworkAnalysis (https://github.com/michellab/FreeEnergyNetworkAnalysis) or MBARNet (https://gitlab.com/RutgersLBSR/fe-toolkit), please follow their instructions for installation and use. Default network analysis proceeds using cinnabar.

To be able to run the pipeline, GROMACS (tested with version 23.1) and AMBER (tested with version 22) installations are also required. Please follow their instructions for installation and use, available at https://manual.gromacs.org/documentation/2023.1/install-guide/index.html and https://ambermd.org/Installation.php .

### Outline of folders:

**pipeline_notebooks** - notebooks and scripts for starting and analysing the pipeline

**python** - contains all the code for running the pipeline


### Outline of the pipeline:

To run the pipeline, follow instructions in the pipeline_notebooks folder. After the initial setup of the pipeline (pipeline_notebooks folder), a main folder with the settings in `execution_model` should have been created. This contains the `ligands.dat`, `network.dat`, `protocol.dat`, and `analysis_protocol.dat` that can be edited as required. In the main folder, running: `bash run_all_slurm.sh` will start the entire pipeline for a slurm workload manager in a series of dependencies. The `run_*_slurm` scripts generated in the `scripts` folder may need to be adjusted depending on the slurm cluster and resources available. After all the runs are finished, they can be analysed in a jupyter notebook following the example in the `pipeline_notebooks` folder.

The workflow and scripts are outlined in the figure below:

![](pipeline_outline.png)
