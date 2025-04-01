import BioSimSpace as BSS
from BioSimSpace.Units.Length import angstrom as _angstrom
from BioSimSpace.MD._md import _find_md_engines
from BioSimSpace import _Exceptions
import logging
from ..utils._validate import *

from typing import Union, Optional

from ..utils._validate import *


class ligprep:
    """class to store lig prep functions."""

    def __init__(self):
        pass

    @staticmethod
    def lig_paramaterise(
        molecule: BSS._SireWrappers._molecule.Molecule, ligff_query: str, **kwargs
    ):
        """_summary_

        Args:
            molecule (_BioSimSpace._SireWrappers.Molecule): unparameterised molecule
            ligff_query (str): ligand forcefield

        Returns:
            _BioSimSpace._SireWrappers.Molecule: parameterised molecule
        """
        # dicitonary of functions available
        validate.lig_ff(ligff_query)

        param_molecule = BSS.Parameters.parameterise(
            molecule, ligff_query, **kwargs
        ).getMolecule()

        return param_molecule

    @staticmethod
    def minimum_solvation(
        system: BSS._SireWrappers.System,
        solvent: str,
        box_type: str,
        box_edges: Union[int, float],
        box_edges_unit: str = "angstrom",
        ion_conc: float = 0.15,
        work_dir: Optional[str] = None,
    ):
        """minimum solvation

        Args:
            system (_BioSimSpace._SireWrappers.System): system to be solvated
            solvent (str): solvent forcefield
            box_type (str): type of box, eg cubic
            box_edges (int/float): box edges size added to box size calculated
            box_edges_unit (str, optional): unit of box edges. Defaults to "angstrom".
            ion_conc (float, optional): Ion conc added (NaCl). Defaults to 0.15.
            work_dir (str, optional): work dir for solvation

        Returns:
            _BioSimSpace._SireWrappers.System: the solvated system
        """

        # validate inputs
        try:
            solvent = validate.solvent_ff(solvent)
            box_edges = validate.integer(box_edges)
            box_edges_unit = validate.box_edges_unit(box_edges_unit)
            box_type = validate.box_type(box_type)
            ion_conc = validate.is_float(ion_conc)
            work_dir = validate.folder_path(work_dir, create=True)
        except Exception as e:
            logging.error(
                f"The provided arguments could not be validated.\n Exception is:\n {e}"
            )

        # type of solvation models available in BSS
        boxtype_dict = {
            "cubic": BSS.Box.cubic,
            "truncatedOctahedron": BSS.Box.truncatedOctahedron,
            "octahedral": BSS.Box.truncatedOctahedron,
        }

        # define the box sizes based on the sizes of what is being solvated
        box_min, box_max = system.getAxisAlignedBoundingBox()
        # calcualte the minimum box size needed
        box_size = [y - x for x, y in zip(box_min, box_max)]
        # add the user defined box size around the min system size
        box_sizes = [x + int(box_edges) * box_edges_unit for x in box_size]

        # for amber22 currently, eq fails if the overall box is less than 41 A
        # also best to have slightly larger, so when there is a small system issues don't happed with a too small box for amber during the production stage if truncated octahedron is used.
        # check the box size and adjust if needed
        min_size = 45 * _angstrom
        if max(box_sizes) < min_size:
            logging.error(
                f"max box size {max(box_sizes)} is below the min size {min_size}. This will be replaced."
            )
            new_max_size = max(box_sizes) + (min_size - max(box_sizes))
            # replace the max box size in the list with the new max box size
            # so that in the next part, this max box size is used for the box type
            for index, size in enumerate(box_sizes):
                if size == max(box_sizes):
                    box_sizes[index] = new_max_size

        # Solvate based on the box_type query
        # this also adds ions to balance the charge
        boxtype_func = boxtype_dict[box_type]
        box, angles = boxtype_func(max(box_sizes))

        mol_solvated = BSS.Solvent.solvate(
            solvent,
            molecule=system,
            box=box,
            angles=angles,
            ion_conc=ion_conc,
            work_dir=work_dir,
        )

        nmols = mol_solvated.nMolecules()

        logging.info(
            f"box dimensions for {box_edges} {box_edges_unit} {box_type} are:")
        logging.info(f"box_min : {box_min}")
        logging.info(f"box_max : {box_max}")
        logging.info(f"box_size : {box_size}")
        logging.info(f"box_sizes : {box_sizes}")
        logging.info(f"with the final box : {box} with angles as : {angles}")
        logging.info(f"The total no of molecules is : {nmols}")

        return mol_solvated


def run_process(
    system: BSS._SireWrappers.System,
    protocol: str,
    engine: str = "AMBER",
    exe: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> BSS._SireWrappers.System:
    """Given a solvated system (BSS object) and BSS protocol, run a process workflow with either
    AMBER or GROMACS. Returns the processed system.

    Args:
        system (BioSimSpace._SireWrappers.System): a BSS system
        protocol (pipeline.prep.pipeline_protocol): a pipeline protocol
        engine (str, optional): engine used for the process. Defaults to "AMBER".
        exe (str, optional): path to executable to run. Defaults to None.
        work_dir (str, optional): work dir for the BSS process. Defaults to None.

    Raises:
        _Exceptions.ThirdPartyError: If the process is not able to run.

    Returns:
        BioSimSpace._SireWrappers.System: the system after the process is completed.
    """

    eng_proc_dict = {
        "AMBER": BSS.Process.Amber,
        "GROMACS": BSS.Process.Gromacs,
        "SOMD": BSS.Process.Somd,
    }

    func = eng_proc_dict[engine]  # validate.engine(engine)

    if exe:
        pass
    else:
        found_engines, found_exes = _find_md_engines(
            system, protocol, engine, gpu_support=True
        )
        exe = found_exes[0]

    logging.info(f"using {exe} as an engine for the process...")

    # Create the process passing a working directory.
    process = func(system, protocol, work_dir=work_dir, exe=exe)

    # Start the process, wait for it to exit
    process.start()
    process.wait()

    # Check for errors.
    if process.isError():
        logging.error(process.stdout())
        logging.error(process.stderr())
        raise _Exceptions.ThirdPartyError("The process exited with an error!")

    # If it worked, try to get the system. No need to block, since it's already finished.
    system = process.getSystem()

    return system


def min_prots(lig_fep: str) -> (BSS.Protocol, BSS.Protocol):
    """define the minimisation protocols for the equilibration. For fepprep is at lambda 0.5

    Args:
        lig_fep (str): 'ligprep' or 'fepprep' ; which the protocols are for

    Returns:
        BSS.Protocol: the minimisation restrained and minimisation no restraints protocols.
    """

    prot_dict = {
        "ligprep": (BSS.Protocol.Minimisation, {}),
        "fepprep": (
            BSS.Protocol.FreeEnergyMinimisation,
            {"lam_vals": [0.5], "lam": 0.5},
        ),
    }

    func = prot_dict[lig_fep][0]
    args = prot_dict[lig_fep][1]

    # minimisation of solvent
    protocol_min_rest = func(steps=10000, restraint="heavy", **args)
    # minimisation full system
    protocol_min = func(steps=10000, **args)

    return protocol_min_rest, protocol_min


def nvt_prots(lig_fep: str) -> (BSS.Protocol):
    """define the nvt protocols for the equilibration. For fepprep is at lambda 0.5

    Args:
        lig_fep (str): 'ligprep' or 'fepprep' ; which the protocols are for

    Returns:
        BSS.Protocol: the nvt restrained all, restrained heavy, no restraints protocols.
    """

    prot_dict = {
        "ligprep": (BSS.Protocol.Equilibration, {}),
        "fepprep": (
            BSS.Protocol.FreeEnergyEquilibration,
            {"lam_vals": [0.5], "lam": 0.5},
        ),
    }

    temperature = 300

    func = prot_dict[lig_fep][0]
    args = prot_dict[lig_fep][1]

    # NVTs
    # NVT restraining all non solvent atoms
    protocol_nvt_sol = func(
        runtime=400 * BSS.Units.Time.picosecond,
        temperature_start=10 * BSS.Units.Temperature.kelvin,
        temperature_end=temperature * BSS.Units.Temperature.kelvin,
        restraint="all",
        timestep=1 * BSS.Units.Time.femtosecond,
        force_constant=100,
        restart=False,
        thermostat_time_constant=0.5 * BSS.Units.Time.picosecond, # so collision freq 2ps^-1 when running with AMBER
        **args,
    )

    return protocol_nvt_sol


def npt_prots(lig_fep: str) -> (BSS.Protocol, BSS.Protocol, BSS.Protocol):
    """define the npt protocols for the equilibration. For fepprep is at lambda 0.5

    Args:
        lig_fep (str): 'ligprep' or 'fepprep' ; which the protocols are for

    Returns:
        BSS.Protocol: the npt restrained heavy 10, restrained heavy 5, no restraints protocols.
    """

    prot_dict = {
        "ligprep": (BSS.Protocol.Equilibration, {}),
        "fepprep": (
            BSS.Protocol.FreeEnergyEquilibration,
            {"lam_vals": [0.5], "lam": 0.5},
        ),
    }

    temperature = 300

    func = prot_dict[lig_fep][0]
    args = prot_dict[lig_fep][1]

    # NPTs
    # NPT restraining all non solvent atoms
    protocol_npt_heavy = func(
        runtime=400 * BSS.Units.Time.picosecond,
        pressure=1 * BSS.Units.Pressure.atm,
        temperature=temperature * BSS.Units.Temperature.kelvin,
        restraint="all",
        force_constant=50,
        restart=False,
        thermostat_time_constant=0.5 * BSS.Units.Time.picosecond,  # so collision freq 2ps^-1
        **args,
    )
    # NPT with gradual release of restraints
    protocol_npt_heavy_lighter = func(
        runtime=400 * BSS.Units.Time.picosecond,
        pressure=1 * BSS.Units.Pressure.atm,
        temperature=temperature * BSS.Units.Temperature.kelvin,
        restraint="heavy",
        force_constant=10,
        restart=False,
        thermostat_time_constant=0.5 * BSS.Units.Time.picosecond,  # so collision freq 2ps^-1
        **args,
    )
    # NPT no restraints
    protocol_npt = func(
        runtime=800 * BSS.Units.Time.picosecond,
        pressure=1 * BSS.Units.Pressure.atm,
        temperature=temperature * BSS.Units.Temperature.kelvin,
        restart=False,
        thermostat_time_constant=0.5 * BSS.Units.Time.picosecond,  # so collision freq 2ps^-1
        **args,
    )

    return protocol_npt_heavy, protocol_npt_heavy_lighter, protocol_npt


def md_prots(runtime: int, **kwargs) -> BSS.Protocol:
    """define the npt protocols for the MD. Equilibrate before.

    Args:
        runtime (int): runtime of the MD simulation in nanoseconds.

    Returns:
        BSS.Protocol: MD protocol.
    """

    runtime = validate.integer(runtime)

    prot_dict = {
        "ligprep": (BSS.Protocol.Equilibration, {}),
        "fepprep": (
            BSS.Protocol.FreeEnergyEquilibration,
            {"lam_vals": [0.5], "lam": 0.5},
        ),
    }

    func = BSS.Protocol.Production
    args = kwargs

    temperature = 300

    # NPT no restraints
    protocol_md = func(
        runtime=runtime * BSS.Units.Time.nanosecond,
        pressure=1 * BSS.Units.Pressure.atm,
        temperature=temperature * BSS.Units.Temperature.kelvin,
        restart=False,
        **args,
    )

    return protocol_md


def minimise_equilibrate_leg(
    system_solvated: BSS._SireWrappers.System,
    engine: str = "AMBER",
    pmemd: Optional[str] = None,
    lig_fep: str = "ligprep",
    work_dir: Optional[str] = None,
    timestep: int = 2,
) -> BSS._SireWrappers.System: 
    """minimse and equilibrate for the given leg of the pipeline

    Args:
        system_solvated (BioSimSpace._SireWrappers.System): the solvated BSS system
        engine (str, optional): engine used for the process. Defaults to "AMBER".
        pmemd (str, optional): pmemd path for use with AMBER. Defaults to None.
        lig_fep (str, optional):'ligprep' or 'fepprep'. Defaults to "ligprep".
        work_dir (str, optional): location for the runs. Defaults to None.

    Returns:
        BioSimSpace._SireWrappers.System: the final minimised and equilibrated system
    """

    if work_dir:
        work_dir = validate.folder_path(work_dir, create=True)

    # define all the protocols
    logging.info("defining min and eq protocols...")
    # Minimisations
    protocol_min_rest, protocol_min = min_prots(lig_fep)

    # NVTs
    # protocol_nvt_sol, protocol_nvt_heavy, protocol_nvt = nvt_prots(lig_fep)
    protocol_nvt_sol = nvt_prots(lig_fep)

    # NPTs
    protocol_npt_heavy, protocol_npt_heavy_lighter, protocol_npt = npt_prots(
        lig_fep)

    protocol_npt_heavy.setTimeStep(timestep * BSS.Units.Time.femtosecond)
    protocol_npt_heavy_lighter.setTimeStep(
        timestep * BSS.Units.Time.femtosecond)
    protocol_npt.setTimeStep(timestep * BSS.Units.Time.femtosecond)

    # run all the protocols
    if lig_fep == "ligprep":
        logging.info("minimising...")
        minimised1 = run_process(
            system_solvated,
            protocol_min_rest,
            engine,
            pmemd,
            work_dir=f"{work_dir}/min1",
        )
        minimised2 = run_process(
            minimised1, protocol_min, engine, pmemd, work_dir=f"{work_dir}/min2"
        )
        logging.info("equilibrating NVT...")
        equil1 = run_process(
            minimised2, protocol_nvt_sol, engine, pmemd, work_dir=f"{work_dir}/nvt1"
        )

        logging.info("equilibrating NPT...")
        equil2 = run_process(
            equil1, protocol_npt_heavy, engine, pmemd, work_dir=f"{work_dir}/npt1"
        )
        equil3 = run_process(
            equil2,
            protocol_npt_heavy_lighter,
            engine,
            pmemd,
            work_dir=f"{work_dir}/npt2",
        )
        sys_equil_fin = run_process(
            equil3, protocol_npt, engine, pmemd, work_dir=f"{work_dir}/npt3"
        )

    if lig_fep == "fepprep":
        logging.info("minimising...")
        minimised1 = run_process(
            system_solvated, protocol_min_rest, engine, pmemd)
        minimised2 = run_process(minimised1, protocol_min, engine, pmemd)
        logging.info("equilibrating NVT...")
        equil_nvt = run_process(minimised2, protocol_nvt_sol, engine, pmemd)
        logging.info("equilibrating NPT...")
        sys_equil_fin = run_process(equil_nvt, protocol_npt, engine, pmemd)

    return sys_equil_fin


def md_leg(
    system_equil: BSS._SireWrappers.System,
    engine: str = "AMBER",
    pmemd: Optional[str] = None,
    runtime: int = 10,
    work_dir: Optional[str] = None,
    timestep: int = 2,
) -> BSS._SireWrappers.System:  # times at the top
    """minimse and equilibrate for the given leg of the pipeline

    Args:
        system_equil (BioSimSpace._SireWrappers.System): the equilibrated BSS system .
        engine (str, optional): engine used for the process. Defaults to "AMBER".
        pmemd (str, optional): pmemd path for use with AMBER. Defaults to None.
        runtime (int, optional): Runtime of the MD simulation. Defaults to 10.
        work_dir (str, optional): location for the runs. Defaults to None.

    Returns:
        BioSimSpace._SireWrappers.System: the final system
    """

    if work_dir:
        work_dir = validate.folder_path(work_dir, create=True)

    # define all the protocols

    # MD
    protocol_md = md_prots(runtime)
    protocol_md.setTimeStep(timestep * BSS.Units.Time.femtosecond)

    sys_md_fin = run_process(
        system_equil, protocol_md, engine, pmemd, work_dir=work_dir
    )

    return sys_md_fin
