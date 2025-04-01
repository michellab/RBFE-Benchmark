import BioSimSpace as BSS
from distutils.dir_util import copy_tree, remove_tree
import warnings as _warnings
import logging

try:
    amber_version = float(BSS._amber_version)
    # assume the amber version is 22
    if not amber_version:
        amber_version = 22
except:
    amber_version = 22

from ..utils import *
from ._merge import *
from ._ligprep import *
from ._protocol import *

from typing import Optional
from pytest import approx
from scipy.constants import proton_mass
from scipy.constants import physical_constants

hydrogen_amu = proton_mass / (physical_constants["atomic mass constant"][0])


def check_hmr(
    system: BSS._SireWrappers.System, protocol: pipeline_protocol
) -> BSS._SireWrappers.System:
    # from bss code

    system = validate.system(system)
    protocol = validate.pipeline_protocol(protocol)
    engine = protocol.engine()
    property_map = {}

    # HMR check
    # by default, if the timestep is greater than 4 fs this should be true.
    if protocol.hmr():
        # Set the expected HMR factor.
        hmr_factor = protocol.hmr_factor()

        if engine == "AMBER" or engine == "GROMACS":
            if hmr_factor == "auto":
                hmr_factor = 3
            else:
                pass # existing hmr factor
        # SOMD factor set during relative from the protocol.

        # Extract the molecules to which HMR applies.
        # set auto based on engine
        repartition_water = False
        if repartition_water == False:
            water = "no"
            # water_mols = system.getWaterMolecules()
            # molecules = system - water_mols
            molecules = system.search("not water", property_map)
        elif repartition_water == True:
            water = "yes"
            molecules = self._system.getMolecules()

        for mol in molecules:
            try:
                h = mol.search("element H")
                # mass = h._sire_object.property(mass_prop).value()
                mass = h[0]._sire_object.evaluate().mass(property_map).value()
                found_h = True
            except:
                found_h = False
            if found_h:
                break

        if not found_h:
            # in some cases, may not be able to find the mass based on element
            # if the only molecule is perturbable. Search for atom name with H.
            # need mass0 as this is one of the perturbable molecule properties.
            mass_prop = property_map.get("mass0", "mass0")
            molecule = system.getPerturbableMolecules()[0]
            for atom in molecule.getAtoms():
                # check if the atom is a H atom
                if atom.name().startswith("H"):
                    mass = atom._sire_object.property(mass_prop).value()
                    # check in case the mass at 0 is 0 as perturbable molecule.
                    if mass > 1:
                        found_h = True
                if found_h:
                    break

        # error if cant find a H mass
        if not found_h:
            raise TypeError("Can't find the mass of a H in the system.")

        # Check that the mass matches what is expected.
        if engine == "SOMD":
            # values should be in amu
            if mass == approx(hydrogen_amu, rel=1e-2):
                repartition = False
            else:
                raise TypeError(
                    "Please do not pass an already repartitioned system in for use with SOMD."
                )

        # check for amber or gromacs repartitioning
        elif engine == "AMBER" or engine == "GROMACS":
            # check if the system has been repartitioned at all. If not, repartition.
            if mass == approx(hydrogen_amu, rel=1e-2):
                repartition = True
            # check if system as been repartitioned with the decided factor.
            elif mass == approx(hmr_factor * hydrogen_amu, rel=1e-2):
                repartition = False
            # finally, check if the system is repartitioned with the wrong factor.
            elif mass != approx(hmr_factor * hydrogen_amu, rel=1e-2):
                raise TypeError(
                    """
                The system is repartitioned at a factor different from that specified in 'hmr_factor'
                or at the auto default for this engine (3 for AMBER and GROMACS, None for SOMD (as this is specified in the cfg file)).
                Please pass a correctly partitioned or entirely unpartitioned system."""
                )

        # Repartition if necessary.
        if repartition:
            _warnings.warn(
                f"The passed system is being repartitioned according to a factor of '{hmr_factor}'."
            )
            system.repartitionHydrogenMass(
                factor=hmr_factor, water=water, property_map=property_map
            )
        else:
            if engine != "SOMD":
                _warnings.warn(
                    "The passed system is already repartitioned. Proceeding without additional repartitioning."
                )

    else:
        pass

    return system


class fepprep:
    """class for fepprep
    makes all the protocols and writes the folders
    """

    def __init__(
        self,
        free_system: Optional[BSS._SireWrappers.System] = None,
        bound_system: Optional[BSS._SireWrappers.System] = None,
        protocol: Optional[pipeline_protocol] = None,
    ):
        # instantiate the class with the system and pipeline_protocol
        if free_system:
            self._merge_free_system = validate.system(free_system).copy()
        else:
            self._merge_free_system = None
            self._free_system_0 = None
            self._free_system_1 = None
            logging.error(
                "please add a system for free with lig0 and free with lig1 and merge these"
            )

        if bound_system:
            self._merge_bound_system = validate.system(bound_system).copy()
        else:
            self._merge_bound_system = None
            self._bound_system_0 = None
            self._bound_system_1 = None
            logging.error(
                "please add a system for bound with lig0 and bound with lig1 and merge these"
            )

        self._pipeline_protocol = validate.pipeline_protocol(
            protocol, fepprep=True)
        # generate the BSS protocols from the pipeline protocol
        fepprep._generate_bss_protocols(self)

    def add_system(
        self,
        system: BSS._SireWrappers.System,
        free_bound: Optional[str] = None,
        start_end: Optional[str] = None,
    ):
        """add a merged system to the fepprep and which state it is meant to represent;
        either the free/bound leg or the state at start(lambda 0.0)/end(lambda 1.0)

        Args:
            system (BioSimSpace._SireWrappers.System): _description_
            free_bound (str, optional): free or bound system. Defaults to None.
            start_end (_type_, optional): system represents start(lambda 0.0) or end(lambda 1.0). Defaults to None.

        Raises:
            ValueError: free_bound must be free or bound.
            ValueError: start_end must be start or end.
        """

        if free_bound not in ["free", "bound"]:
            raise ValueError("free_bound must be free or bound.")
        if start_end not in ["start", "end"]:
            raise ValueError("start_end must be start or end.")

        if free_bound == "free" and start_end == "start":
            self._free_system_0 = validate.system(system).copy()
        if free_bound == "free" and start_end == "end":
            self._free_system_1 = validate.system(system).copy()
        if free_bound == "bound" and start_end == "start":
            self._bound_system_0 = validate.system(system).copy()
        if free_bound == "bound" and start_end == "end":
            self._bound_system_1 = validate.system(system).copy()

    def merge_systems(
        self, align_to: str = "lig0", **kwargs
    ) -> (BSS._SireWrappers.System, BSS._SireWrappers.System):
        """merge the systems based on whether aligning to the lambda 0.0 coordinates or the lmabda 1.0 coordinates.

        Args:
            align_to (str): 'lig0' or 'lig1'.

        Returns:
            BioSimSpace._SireWrappers.System: the free system and the bound system
        """

        kwarg_dict = {"ALIGNTO": align_to}

        for key, value in kwargs.items():
            kwarg_dict[key.upper().replace(
                " ", "").replace("_", "").strip()] = value

        logging.info(f"merging using {kwarg_dict} ...")

        self._free_system_0 = check_hmr(
            self._free_system_0,
            self._pipeline_protocol
        )
        self._free_system_1 = check_hmr(
            self._free_system_1,
            self._pipeline_protocol,
        )
        self._bound_system_0 = check_hmr(
            self._bound_system_0,
            self._pipeline_protocol,
        )
        self._bound_system_1 = check_hmr(
            self._bound_system_1,
            self._pipeline_protocol,
        )

        try:
            free_system = merge.merge_system(
                self._free_system_0, self._free_system_1, **kwarg_dict
            )
            bound_system = merge.merge_system(
                self._bound_system_0, self._bound_system_1, **kwarg_dict
            )
        except:
            logging.error(
                "could not merge with the existing protocol. Will try merging with the allow ring breaking and allow ring size change arguments set to True..."
            )
            update_kwarg_dict = {
                "ALLOWRINGBREAKING": True, "ALLOWRINGSIZECHANGE": True}
            kwarg_dict.update(update_kwarg_dict)
            free_system = merge.merge_system(
                self._free_system_0, self._free_system_1, **kwarg_dict
            )
            bound_system = merge.merge_system(
                self._bound_system_0, self._bound_system_1, **kwarg_dict
            )

        self._merge_free_system = free_system
        self._merge_bound_system = bound_system

        return free_system, bound_system

    def _generate_bss_protocols(self):
        """internal function to generate bss protocols for setup based on passed pipeline protocol."""

        protocol = self._pipeline_protocol

        restart_interval = 10000

        if protocol.engine() == "AMBER":
            # for amber, this will be 1/value to give 2 for the collision frequency in ps-1
            thermostat_time_constant = BSS.Types.Time(0.5, "picosecond")
        elif protocol.engine() == "GROMACS":
            # in gromacs this is the tau-t in ps
            thermostat_time_constant = BSS.Types.Time(2, "picosecond")

        if protocol.engine() == "AMBER" or protocol.engine() == "GROMACS":
            min_protocol = BSS.Protocol.FreeEnergyMinimisation(
                num_lam=protocol.num_lambda(),
                steps=protocol.min_steps(),
            )

            heat_protocol = BSS.Protocol.FreeEnergyEquilibration(
                timestep=protocol.eq_timestep() * protocol.timestep_unit(),
                num_lam=protocol.num_lambda(),
                runtime=protocol.eq_runtime() * protocol.eq_runtime_unit(),
                pressure=None,
                temperature_start=protocol.start_temperature()
                * protocol.temperature_unit(),
                temperature_end=protocol.end_temperature()
                * protocol.temperature_unit(),
                restart_interval=restart_interval,
                # restraint="heavy",
                thermostat_time_constant=thermostat_time_constant,
            )
            eq_protocol = BSS.Protocol.FreeEnergyEquilibration(
                timestep=protocol.eq_timestep() * protocol.timestep_unit(),
                num_lam=protocol.num_lambda(),
                runtime=protocol.eq_runtime() * protocol.eq_runtime_unit(),
                temperature=protocol.temperature() * protocol.temperature_unit(),
                pressure=protocol.pressure() * protocol.pressure_unit(),
                restart=True,
                restart_interval=restart_interval,
                thermostat_time_constant=thermostat_time_constant,
            )
            freenrg_protocol = BSS.Protocol.FreeEnergyProduction(
                timestep=protocol.timestep() * protocol.timestep_unit(),
                num_lam=protocol.num_lambda(),
                runtime=protocol.sampling() * protocol.sampling_unit(),
                temperature=protocol.temperature() * protocol.temperature_unit(),
                pressure=protocol.pressure() * protocol.pressure_unit(),
                restart=True,
                restart_interval=restart_interval,
                thermostat_time_constant=thermostat_time_constant,
            )

        elif protocol.engine() == "SOMD":
            min_protocol = None
            heat_protocol = None

            eq_protocol = BSS.Protocol.FreeEnergy(
                timestep=protocol.eq_timestep() * protocol.timestep_unit(),
                num_lam=protocol.num_lambda(),
                temperature=protocol.temperature() * protocol.temperature_unit(),
                runtime=(protocol.eq_runtime() * 2) *
                protocol.eq_runtime_unit(),
                pressure=protocol.pressure() * protocol.pressure_unit(),
                restart_interval=restart_interval,
            )
            freenrg_protocol = BSS.Protocol.FreeEnergy(
                timestep=protocol.timestep() * protocol.timestep_unit(),
                num_lam=protocol.num_lambda(),
                runtime=protocol.sampling() * protocol.sampling_unit(),
                temperature=protocol.temperature() * protocol.temperature_unit(),
                pressure=protocol.pressure() * protocol.pressure_unit(),
                restart_interval=restart_interval,
            )

        # if protocol.engine() == "GROMACS":
        #     logging.info("using no heating for gromacs")
        #     heat_protocol = BSS.Protocol.FreeEnergyEquilibration(
        #         timestep=protocol.eq_timestep() * protocol.timestep_unit(),
        #         num_lam=protocol.num_lambda(),
        #         runtime=protocol.eq_runtime() * protocol.eq_runtime_unit(),
        #         pressure=None,
        #         temperature=protocol.temperature() * protocol.temperature_unit(),
        #         restart_interval=restart_interval,
        #         # restraint="heavy",
        #         thermostat_time_constant=thermostat_time_constant,
        #     )

        # set the new protocols to self as well
        self._min_protocol = min_protocol
        self._heat_protocol = heat_protocol
        self._eq_protocol = eq_protocol
        self._freenrg_protocol = freenrg_protocol

    ## currently not used
    def prep_system_middle(self, pmemd_path: str, work_dir: Optional[str] = None):
        """trying to prep the system at lambda 0.5 (not very robust currently)

        Args:
            pmemd_path (str): path to pmemd executable for use with amber
            work_dir (str, optional): the work directory for the runs. Defaults to None.

        Returns:
            BioSimSpace._SireWrappers.System: the free and bound merged systems
        """

        if self._pipeline_protocol.fepprep() == "middle":
            # Solvate and run each the bound and the free system.
            legs_mols, legs = [self._merge_free_system, self._merge_bound_system], [
                "lig",
                "sys",
            ]

            # zip together the molecules in that leg with the name for that leg
            for leg, leg_mol in zip(legs, legs_mols):
                logging.info(f"carrying out prep system middle for {leg}")
                leg_equil_final = minimise_equilibrate_leg(
                    leg_mol, "AMBER", pmemd_path, lig_fep="fepprep", work_dir=work_dir
                )
                if leg == "lig":
                    self._merge_free_system = leg_equil_final
                if leg == "sys":
                    self._merge_bound_system = leg_equil_final

        return self._merge_free_system, self._merge_bound_system

    def _generate_folders(
        self,
        system_free: BSS._SireWrappers.System,
        system_bound: BSS._SireWrappers.System,
        work_dir: str,
        rep: int = 0,
    ):
        """generating the folders for the free and the bound system

        Args:
            system_free (BioSimSpace._SireWrappers.System): the free merged system
            system_bound (BioSimSpace._SireWrappers.System): the bound merged system
            work_dir (str): the work dir for where the folders will be generated
        """

        protocol = self._pipeline_protocol
        min_protocol = self._min_protocol
        heat_protocol = self._heat_protocol
        eq_protocol = self._eq_protocol
        freenrg_protocol = self._freenrg_protocol

        logging.info(f"setting up FEP run in {work_dir}...")

        if protocol.engine() == "AMBER" or protocol.engine() == "GROMACS":
            # set up for each the bound and the free leg
            for leg, system in zip(["bound", "free"], [system_bound, system_free]):
                if protocol.engine() == "GROMACS":
                    gromacs_dict = {
                        "rlist": "0.8",
                        "rvdw": "0.8",
                        "rcoulomb": "0.8",
                    }

                    min_extra_options = gromacs_dict.copy()
                    heat_extra_options = gromacs_dict.copy()
                    eq_extra_options = gromacs_dict.copy()
                    prod_extra_options = gromacs_dict.copy()

                elif protocol.engine() == "AMBER":
                    if amber_version > 20:
                        # Options used in AMBER DDBoost ; http://ambermd.org/tutorials/advanced/tutorial39/index.php
                        amber_dict = {
                            "gti_cut": 1,  # add smoothing to SC-vDW, beginning at gti_cut_sc_on and ending at gti_cut_sc_off; using the second order smooth-step function
                            "gti_cut_sc_on": 8,
                            "gti_cut_sc_off": 10,
                            "gti_output": 1,  # output term by term detailed TI results
                            "gti_add_sc": 5,  # all interactions except vdw sc internal are scaled with lambda and not present in the dummy state
                            "gti_scale_beta": 1,  # new form of sc potential enabled
                            "scalpha": 0.5,  # default for gti_scale_beta
                            "scbeta": 1.0,  # default for gti_scale_beta
                            "gti_lam_sch": 1,  #
                            "gti_ele_sc": 1,  # smoothstep function used
                            "gti_vdw_sc": 1,  # smoothstep function used
                            # smooth vdw (1) and then also elec (2)
                            "gti_cut_sc": 2,
                            "gti_ele_exp": 2,
                            "gti_vdw_exp": 2,  # default value is 6
                            "cut": "10.0",
                            "iwrap": 0,
                            "gti_bat_sc": 0,
                            "tishake": 0,
                            # "gti_syn_mass": 3,
                        }
                    else:
                        amber_dict = {}
                    min_extra_options = amber_dict.copy()
                    heat_extra_options = amber_dict.copy()
                    eq_extra_options = amber_dict.copy()
                    prod_extra_options = amber_dict.copy()
                    prod_extra_options["iwrap"] = 1
                    eq_extra_options["barostat"] = 2

                for key, value in protocol.config_options()["all"].items():
                    min_extra_options[key] = value
                    heat_extra_options[key] = value
                    eq_extra_options[key] = value
                    prod_extra_options[key] = value
                for key, value in protocol.config_options()["min"].items():
                    min_extra_options[key] = value
                for key, value in protocol.config_options()["heat"].items():
                    heat_extra_options[key] = value
                for key, value in protocol.config_options()["eq"].items():
                    eq_extra_options[key] = value
                for key, value in protocol.config_options()["prod"].items():
                    prod_extra_options[key] = value

                BSS.FreeEnergy.Relative(
                    system,
                    min_protocol,
                    engine=f"{protocol.engine()}",
                    work_dir=f"{work_dir}/{leg}_{rep}/min",
                    extra_options=min_extra_options,
                    ignore_warnings=True,
                    explicit_dummies=False, # only applied to AMBER runs
                )

                BSS.FreeEnergy.Relative(
                    system,
                    heat_protocol,
                    engine=f"{protocol.engine()}",
                    work_dir=f"{work_dir}/{leg}_{rep}/heat",
                    extra_options=heat_extra_options,
                    ignore_warnings=True,
                    explicit_dummies=False,
                )

                BSS.FreeEnergy.Relative(
                    system,
                    eq_protocol,
                    engine=f"{protocol.engine()}",
                    work_dir=f"{work_dir}/{leg}_{rep}/eq",
                    extra_options=eq_extra_options,
                    ignore_warnings=True,
                    explicit_dummies=False,
                )

                BSS.FreeEnergy.Relative(
                    system,
                    freenrg_protocol,
                    engine=f"{protocol.engine()}",
                    work_dir=f"{work_dir}/{leg}_{rep}/prod",
                    extra_options=prod_extra_options,
                    ignore_warnings=True,
                    explicit_dummies=False,
                )

        if protocol.engine() == "SOMD":
            for leg, system in zip(["bound", "free"], [system_bound, system_free]):
                eq_extra_options = {
                    "minimise": True,
                    "minimise maximum iterations": protocol.min_steps(),
                    "equilibrate": False,
                    "coulomb power": 1, "shift delta": 1.0,
                    "cutoff distance": "12 angstrom",
                    "integrator_type": "langevinmiddle", # as eq is 2 fs, want to use the same as for prod
                }
                prod_extra_options = {"minimise": False, "equilibrate": False,
                                      "coulomb power": 1, "shift delta": 1.0,
                                      "cutoff distance": "12 angstrom",
                                      }

                # hmr factor
                if protocol.hmr():
                    if protocol.hmr_factor() == "auto":
                        eq_extra_options["hydrogen mass repartitioning factor"] = "1.5"
                        prod_extra_options["hydrogen mass repartitioning factor"] = "1.5"
                    else:
                        eq_extra_options["hydrogen mass repartitioning factor"] = f"{protocol.hmr_factor()}"
                        prod_extra_options["hydrogen mass repartitioning factor"] = f"{protocol.hmr_factor()}"

                for key, value in protocol.config_options()["all"].items():
                    eq_extra_options[key] = value
                    prod_extra_options[key] = value
                for key, value in protocol.config_options()["eq"].items():
                    eq_extra_options[key] = value
                for key, value in protocol.config_options()["prod"].items():
                    prod_extra_options[key] = value

                BSS.FreeEnergy.Relative(
                    system,
                    eq_protocol,
                    engine=f"{protocol.engine()}",
                    work_dir=f"{work_dir}/{leg}_{rep}/eq",
                    extra_options=eq_extra_options,
                    ignore_warnings=True,
                )

                BSS.FreeEnergy.Relative(
                    system,
                    freenrg_protocol,
                    engine=f"{protocol.engine()}",
                    work_dir=f"{work_dir}/{leg}_{rep}/prod",
                    extra_options=prod_extra_options,
                    ignore_warnings=True,
                )

    def generate_folders(self, work_dir: str, **kwargs):
        """generate the folders for the RBFE run.

        Args:
            work_dir (str): the folder to generate the folders in.
        """

        if self._pipeline_protocol.rerun():
            work_dir = validate.folder_path(work_dir, create=False)
            b_folders, f_folders = get_repeat_folders(work_dir)
            if len(b_folders) != len(f_folders):
                raise ValueError(
                    "work dir must have same number of free and bound folders for reruns."
                )
            # get which repeat on
            rep = len(b_folders)
            start_rep = rep
            self._pipeline_protocol.rerepeat(start_rep)
            self._pipeline_protocol.rewrite_protocol()
            if (self._pipeline_protocol.repeats() - rep) <= 0:
                raise ValueError("all repeats folders already exist.")
        else:
            work_dir = validate.folder_path(work_dir, create=True)
            rep = 0
            start_rep = 1

        # default options based on engine
        if (
            self._pipeline_protocol.engine() == "AMBER"
            or self._pipeline_protocol.engine() == "GROMACS"
        ):
            # use a hybrid topology for AMBER and GROMACS
            kwarg_dict = {"PRUNEPERTURBEDCONSTRAINTS": True}
        else:
            kwarg_dict = {}

        if amber_version < 22:
            if self._pipeline_protocol.hmr:
                if self._pipeline_protocol.engine() == "AMBER":
                    kwarg_dict["PRUNECROSSINGCONSTRAINTS"] = True

        if self._pipeline_protocol.engine() == "AMBER":
            kwarg_dict["PRUNEATOMTYPES"] = True

        # any pipeline kwargs overwrite this
        for key, value in self._pipeline_protocol.kwargs().items():
            kwarg_dict[key.upper().replace(
                " ", "").replace("_", "").strip()] = value

        # any final passed arguments in the script overwrite this
        for key, value in kwargs.items():
            kwarg_dict[key.upper().replace(
                " ", "").replace("_", "").strip()] = value

        if self._pipeline_protocol.fepprep() == "both":
            ligs = ["lig0", "lig1"]
            for lig in ligs:
                free_system, bound_system = self.merge_systems(
                    align_to=lig, **kwarg_dict
                )
                self._generate_folders(
                    free_system, bound_system, f"{work_dir}/{lig}", rep=rep
                )

            # get half of the lambdas
            lambdas_list = self._freenrg_protocol.getLambdaValues()
            middle_index = len(lambdas_list) // 2
            first_half = lambdas_list[:middle_index]
            sec_half = lambdas_list[middle_index:]

            # copy files to main folder
            logging.info(
                "copying generated folders for the endstates into a combined folder, so first half is lig0 and second half is lig1"
            )
            for lig, lam_list in zip(ligs, [first_half, sec_half]):
                for leg in ["bound", "free"]:
                    for part in ["min", "heat", "eq", "prod"]:
                        try:  # so will not copy if folders do not exist
                            for lam in lam_list:
                                copy_tree(
                                    f"{work_dir}/{lig}/{leg}_{rep}/{part}/lambda_{lam:.4f}",
                                    f"{work_dir}/{leg}_{rep}/{part}/lambda_{lam:.4f}",
                                )
                        except:
                            pass

                # remove the dir
                logging.info(f"removing directory for {lig} as copied...")
                remove_tree(f"{work_dir}/{lig}")

        else:
            if not self._merge_free_system or not self._merge_bound_system:
                logging.info("no merged systems, merging....")
                if self._pipeline_protocol.fepprep() == "start":
                    self.merge_systems(**kwarg_dict)
                elif self._pipeline_protocol.fepprep() == "end":
                    self.merge_systems(align_to="lig1", **kwarg_dict)
            self._generate_folders(
                self._merge_free_system, self._merge_bound_system, work_dir, rep=rep
            )

        # default folder is with no integer.
        # for the sake of analysis , doesnt matter as finds folders w names of leg
        more_repeats = list(
            range(start_rep, self._pipeline_protocol.repeats()))

        logging.info(
            f"there are {self._pipeline_protocol.repeats()} folder(s) being made for each leg..."
        )
        for r in more_repeats:
            for leg in ["bound", "free"]:
                copy_tree(f"{work_dir}/{leg}_{rep}", f"{work_dir}/{leg}_{r}")

        logging.info("done.")
