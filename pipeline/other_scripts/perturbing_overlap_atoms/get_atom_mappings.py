import BioSimSpace as BSS
import os
import sys

try:
    import pipeline
except:
    print("adding code to the pythonpath...")
    code = "/home/anna/Documents/code/python"
    if code not in sys.path:
        sys.path.insert(1, code)
    import pipeline

from pipeline.prep import *
from pipeline.utils import *

# Values from the corresponding bash arrays.
# The ligands for which the transformation is to be carried out.
trans = sys.argv[1]
lig_1 = trans.split("~")[0]
lig_2 = trans.split("~")[1]
engine_query = str(sys.argv[2]).upper()

# files that were set in the run_all script
main_dir = os.environ["MAINDIRECTORY"]
net_file = os.environ["net_file"]  # network file
prep_dir = f"{main_dir}/prep"  # define lig prep location
workdir = validate.folder_path(f"{main_dir}/outputs/atom_mappings", create=True)

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

    ligand_0, ligand_1, mapping = merge.atom_mappings(system_1, system_2, engine_query)

    write_atom_mappings(
        lig_1,
        lig_2,
        ligand_0,
        ligand_1,
        mapping,
        output_file=f"{workdir}/{engine}_atom_mappings.dat",
    )
