#!/usr/bin/python3

import rdkit
from rdkit import Chem
import yaml
import glob
import sys
import numpy as np

protein = sys.argv[1]

ligand_files = glob.glob(f"../protein_systems/{protein}/ligands_*/lig_*")

value_dict = {}

for f in ligand_files:
    lig_name = f.split("/")[-1].split(".")[0].strip()

    suppl = Chem.SDMolSupplier(f, removeHs=False)
    mol = suppl[0]
    prop_binding = "IC50[nM]" # IC50[uM](SPA)
    binding_val = mol.GetProp(prop_binding)
    unit = prop_binding.split("[")[-1].split("]")[0]
    err = float(0.3 * np.log(10) * float(binding_val))
    value_dict[lig_name] = {"measurement":{"type":"ic50", "error":err, "unit":unit,
        "value":float(binding_val)}, "name":lig_name}

with open(f"{protein}.yml", "w") as file:
    yaml.dump(value_dict, file)

