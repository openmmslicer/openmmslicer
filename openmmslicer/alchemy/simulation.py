import copy as _copy
import logging as _logging

import mdtraj as _mdtraj
import openmm as _openmm
import openmm.unit as _unit
import openmm.app as _app

from . import align as _align
from . import relative as _relative

_logger = _logging.getLogger(__name__)


class AlchemicalContext:
    """
    A wrapper for openmm.Context, which handles loading of states from trajectory files, as well as applying
    openmmslicer.move.Moves.

    Parameters
    ----------
    context : openmm.Context
        The OpenMM context to be wrapped.
    simulation : openmmslicer.alchemy.AlchemicalSimulation
        The parent alchemical simulation.
    """
    def __init__(self, context, simulation):
        self.__class__ = type(self.__class__.__name__, (self.__class__, context.__class__), {})
        self._context = context
        self._simulation = simulation

    def __getattr__(self, item):
        if item != "_context":
            return getattr(self._context, item)
        raise AttributeError

    def __setattr__(self, key, value):
        if key not in ["_context", "_simulation"] and hasattr(self, "_context"):
            setattr(self._context, key, value)
        else:
            super().__setattr__(key, value)

    def setState(self, state, new_lambda_=None, apply_hybrid_transforms=True):
        """
        Sets a given state to the current context.

        Parameters
        ----------
        state : openmm.State, openmmslicer.samplers.Walker
            Sets the state from either an openmm.State object or a openmmslicer.utils.Walker object.
        new_lambda_ : float, optional
            The lambda value at which the state will be run. Default is the current simulation lambda value.
        apply_hybrid_transforms : bool, optional
            Whether to apply the associated openmmslicer.moves.HybridMoves after loading the state.

        Returns
        -------
        log_jacobian : float
            The logarithm of the Jacobian determinant corresponding to the applied moves.
        """
        from openmmslicer.samplers.misc import Walker
        log_jacobian = 0.
        if isinstance(state, _openmm.State):
            self._context.setState(state)
        elif isinstance(state, Walker):
            if state.state is not None:
                self._context.setState(state.state)
            else:
                # load from trajectory file
                frame = _mdtraj.load_frame(state.reporter_filename, state.frame,
                                           self._simulation.alchemical_chain.aligned_structures.coordinates)
                positions = frame.xyz[0]
                periodic_box_vectors = frame.unitcell_vectors[0]
                self._context.setPositions(positions)
                self._context.setPeriodicBoxVectors(*periodic_box_vectors)

            # augment the lambdas with the intermediate checkpoints
            if new_lambda_ is None:
                new_lambda_ = self._simulation.lambda_
            bonded_func = self._simulation.alchemical_functions["lambda_bonds"]
            n_states = len(self._simulation.alchemical_chain.states) - 1
            checkpoint_lambdas = [i / n_states for i in range(n_states + 1)]
            min_lambda = min(state.original_lambda_, new_lambda_)
            max_lambda = max(state.original_lambda_, new_lambda_)
            lambdas = sorted({min_lambda} |
                             {x for x in checkpoint_lambdas if min_lambda < x < max_lambda} |
                             {max_lambda})
            if state.original_lambda_ > new_lambda_:
                lambdas = lambdas[::-1]

            # apply the endpoint moves
            moves = self._simulation.alchemical_chain.getMoves(lambdas[0])
            if moves is not None:
                moves.apply(self._context, transformations=state.transform)

            # apply the hybrid moves
            if apply_hybrid_transforms:
                for lambda0, lambda1 in zip(lambdas[:-1], lambdas[1:]):
                    moves = self._simulation.alchemical_chain.getMoves(0.5 * (lambda0 + lambda1))
                    if moves is not None:
                        simulation_index0, lambda0_local = self._simulation.globalToLocalLambda(lambda0)
                        simulation_index1, lambda1_local = self._simulation.globalToLocalLambda(lambda1)
                        if simulation_index1 > simulation_index0:
                            lambda1_local = int(not self._simulation._swaps[simulation_index0])
                        elif simulation_index1 < simulation_index0:
                            lambda0_local = int(not self._simulation._swaps[simulation_index1])
                        bonded_lambda0 = bonded_func(lambda0_local)
                        bonded_lambda1 = bonded_func(lambda1_local)
                        moves.apply(self._context, lambda0=bonded_lambda0, lambda1=bonded_lambda1)
                        log_jacobian += moves.getLogJacobian(lambda0=bonded_lambda0, lambda1=bonded_lambda1)
        else:
            raise TypeError(f"Unrecognised parameter type {type(state)}")
        return log_jacobian


class AlchemicalSimulation:
    """
    A wrapper for openmm.Simulation, which dynamically switches between multiple hybrid topologies for intermediate
    lambda windows and pure topologies for endpoint lambda windows.

    Parameters
    ----------
    alchemical_chain : openmmslicer.alchemy.AlchemicalChain
        The alchemical chain to be explored by the AlchemicalSimulation.
    integrator : openmm.Integrator
        The integrator used at each sampling step.
    alchemical_functions : dict, optional
        A dictionary whose keys correspond to the alchemical variables and whose values are functions with a domain
        and range between 0 and 1.
    platform : str, optional
        The platform which should be used for simulation. Default is the fastest available platform.
    platform_properties : dict, optional
        Additional platform properties.
    npt : bool, optional
        Whether to add a barostat. Default is True.
    md_config : dict, optional
        Additional simulation parameters passed to parmed.Structure.generateSimulation().
    alch_config : dict, optional
        Additional alchemical parameters passed to generateAlchSystem().
    barostat_config : dict, optional
        Additional barostat parameters passed to addBarostat().

    Attributes
    ----------
    alchemical_chain : openmmslicer.alchemy.AlchemicalChain
        The associated alchemical chain.
    alchemical_functions : dict
        The associated alchemical functions.
    kT : openmm.unit.Quantity
        The current temperature multiplied by the gas constant.
    lambda_ : float
        The current lambda value.
    temperature : openmm.unit.Quantity
        The temperature of the current integrator.
    """
    def __init__(self, alchemical_chain, integrator, alchemical_functions=None, lambda_=0., platform=None,
                 platform_properties=None, npt=True, md_config=None, alch_config=None, barostat_config=None):
        def generate_simulation(struct, system):
            if npt:
                self.addBarostat(system, temperature=integrator.getTemperature(), **barostat_config)
            if removeCMMotion:
                system.addForce(_openmm.CMMotionRemover())
            simulation = self.generateSimulation(struct, system, _copy.deepcopy(integrator), platform,
                                                 platform_properties)
            simulation.context = AlchemicalContext(simulation.context, self)
            return simulation

        if md_config is None:
            md_config = {}
        if alch_config is None:
            alch_config = {}
        if barostat_config is None:
            barostat_config = {}

        self.alchemical_chain = alchemical_chain
        self.alchemical_functions = alchemical_functions
        removeCMMotion = md_config.pop("removeCMMotion", True)
        md_config["removeCMMotion"] = False
        self._simulations = []
        self._checkpoint_simulations = {}
        self._swaps = []

        for i in range(len(self.alchemical_chain.states) - 1):
            # get the two endpoints
            name0, struct0, moves0 = self.alchemical_chain.states[i]
            name1, struct1, moves1 = self.alchemical_chain.states[i + 1]
            dummy_atoms0 = _align.AlignedStructures.getDummyIndices(struct0)
            dummy_atoms1 = _align.AlignedStructures.getDummyIndices(struct1)

            # generate the two endpoint systems
            system0 = self.generateSystem(struct0, **md_config)
            system1 = self.generateSystem(struct1, **md_config)

            # generate checkpoint simulations which are not alchemical
            if i not in self._checkpoint_simulations:
                self._checkpoint_simulations[i] = generate_simulation(_copy.copy(struct0), _copy.copy(system0))

            # always perturb small to big
            if not dummy_atoms0.issuperset(dummy_atoms1):
                swap = True
                name0, name1 = name1, name0
                moves0, moves1 = moves1, moves0
                struct0, struct1 = struct1, struct0
                system0, system1 = system1, system0
            else:
                swap = False

            # generate the alchemical interpolation
            alch_system = self.generateAlchSystem(system0, system1, **alch_config)

            # add extra forces, if applicable
            simulation = generate_simulation(struct0, alch_system)
            self._simulations += [simulation]
            self._swaps += [swap]
        self.__class__ = type(self.__class__.__name__, (self.__class__, simulation.__class__), {})
        self.lambda_ = lambda_

    def __getattr__(self, item):
        if item != "_simulation":
            return getattr(self._simulation, item)
        raise AttributeError

    def __setattr__(self, key, value):
        if hasattr(self, "_simulation") and key in self._simulation.__dict__:
            setattr(self._simulation, key, value)
        else:
            super().__setattr__(key, value)

    @property
    def alchemical_functions(self):
        return self._alchemical_functions

    @alchemical_functions.setter
    def alchemical_functions(self, val):
        if val is None:
            val = {}
        self._alchemical_functions = val
        for func in self.alchemical_functions.values():
            assert func(0) == 0 and func(1) == 1, "All alchemical functions must go from 0 to 1"

    @property
    def kT(self):
        return _unit.MOLAR_GAS_CONSTANT_R * self.temperature

    @property
    def lambda_(self):
        return self._lambda_

    @lambda_.setter
    def lambda_(self, val):
        unnormalised_lambda_ = (len(self.alchemical_chain.states) - 1) * val
        if abs(unnormalised_lambda_ - round(unnormalised_lambda_)) < 1e-8:
            unnormalised_lambda_ = round(unnormalised_lambda_)

        if unnormalised_lambda_ in self._checkpoint_simulations:
            self._set_simulation(self._checkpoint_simulations[unnormalised_lambda_], new_lambda_=val)
        else:
            # get one of the hybrid simulations and a corresponding local lambda
            simulation_idx, local_lambda_ = self.globalToLocalLambda(val)
            self._set_simulation(self._simulations[simulation_idx], new_lambda_=val)

            # update the lambda parameters
            valid_parameters = [x for x in self._simulation.context.getParameters()]
            for param, func in self.alchemical_functions.items():
                if param in valid_parameters:
                    param_val = float(func(local_lambda_))
                    assert 0 <= param_val <= 1, "All lambda functions must evaluate between 0 and 1"
                    self._simulation.context.setParameter(param, param_val)

        self._lambda_ = val

    def globalToLocalLambda(self, lambda_):
        """
        Converts a local lambda value (from 0 to 1) to a global lambda value.

        Parameters
        ----------
        lambda_ : float
            The input local lambda value.

        Returns
        -------
        simulation_idx : int
            The corresponding index of the relevant simulation.
        lambda_ : float
            The output global lambda value.
        """
        unnormalised_lambda_ = (len(self.alchemical_chain.states) - 1) * lambda_
        if abs(unnormalised_lambda_ - round(unnormalised_lambda_)) < 1e-8:
            unnormalised_lambda_ = round(unnormalised_lambda_)
        simulation_idx, local_lambda_ = int(unnormalised_lambda_), unnormalised_lambda_ % 1
        if simulation_idx == len(self._simulations):
            simulation_idx = len(self._simulations) - 1
            local_lambda_ = 1
        if self._swaps[simulation_idx]:
            local_lambda_ = 1 - local_lambda_
        return simulation_idx, local_lambda_

    def _set_simulation(self, val, **kwargs):
        if hasattr(self, "_simulation") and val is not self._simulation:
            val.context.setState(self._simulation.context.getState(getPositions=True, getVelocities=True), **kwargs)
        self._simulation = val

    @property
    def temperature(self):
        return self._simulation.integrator.getTemperature()

    @staticmethod
    def addBarostat(system, temperature=298 * _unit.kelvin, pressure=1 * _unit.atmospheres, frequency=25):
        """
        Adds a MonteCarloBarostat to the MD system.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        temperature : openmm.unit.Quantity
            Temperature (Kelvin) to be simulated at.
        pressure : openmm.unit.Quantity
            Pressure (atm) for Barostat for NPT simulations.
        frequency : int, default=25
            Frequency at which Monte Carlo pressure changes should be attempted (in time steps)

        Returns
        -------
        system : openmm.System
            The OpenMM System with the MonteCarloBarostat attached.
        """
        _logger.info(f"Adding MonteCarloBarostat with {pressure}. MD simulation will be {temperature} NPT.")
        # Add Force Barostat to the system
        system.addForce(_openmm.MonteCarloBarostat(pressure, temperature, frequency))
        return system

    @staticmethod
    def generateAlchSystem(system0, system1, **kwargs):
        """
        Generates a hybrid alchemical system based on two pre-aligned openmm.Systems.

        Parameters
        ----------
        args
            Positional arguments passed to openmmslicer.alchemy.HybridTopologyFactory().
        kwargs
            Keyword arguments passed to openmmslicer.alchemy.HybridTopologyFactory().

        Returns
        -------
        hybrid_system : openmm.System
            The hybrid system.
        """
        hybrid_topology_factory = _relative.HybridTopologyFactory(system0, system1, **kwargs)
        return hybrid_topology_factory.hybrid_system

    @staticmethod
    def generateSimulation(structure, system, integrator, platform=None, properties=None):
        """
        Generate the OpenMM Simulation objects from a given parmed.Structure()

        Parameters
        ----------
        structure : parmed.Structure
            ParmEd Structure object of the entire system to be simulated.
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        integrator : openmm.Integrator
            The OpenMM Integrator object for the simulation.
        platform : str, default = None
            Valid choices: 'Auto', 'OpenCL', 'CUDA'
            If None is specified, the fastest available platform will be used.
        properties : dict
            Additional platform properties.

        Returns
        -------
        simulation : openmm.Simulation
            The generated OpenMM Simulation from the parmed.Structure, openmm.System,
            amd the integrator.
        """
        if not properties:
            properties = {}
        # Specifying platform properties here used for local development.
        if platform is None:
            # Use the fastest available platform
            simulation = _app.Simulation(structure.topology, system, integrator)
        else:
            platform = _openmm.Platform.getPlatformByName(platform)
            # Make sure key/values are strings
            properties = {str(k): str(v) for k, v in properties.items()}
            simulation = _app.Simulation(structure.topology, system, integrator, platform, properties)

        # Set initial positions/velocities
        if structure.box_vectors:
            simulation.context.setPeriodicBoxVectors(*structure.box_vectors)
        simulation.context.setPositions(structure.positions)
        simulation.context.setVelocitiesToTemperature(integrator.getTemperature())

        return simulation

    @staticmethod
    def generateSystem(structure, **kwargs):
        """
        Construct an OpenMM System representing the topology described by the
        prmtop file. This function is just a wrapper for parmed Structure.createSystem().

        Parameters
        ----------
        structure : parmed.Structure()
            The parmed.Structure of the molecular system to be simulated
        nonbondedMethod : cutoff method
            This is the cutoff method. It can be either the NoCutoff,
            CutoffNonPeriodic, CutoffPeriodic, PME, or Ewald objects from the
            openmm.app namespace
        nonbondedCutoff : float or distance Quantity
            The nonbonded cutoff must be either a floating point number
            (interpreted as nanometers) or a Quantity with attached units. This
            is ignored if nonbondedMethod is NoCutoff.
        switchDistance : float or distance Quantity
            The distance at which the switching function is turned on for van
            der Waals interactions. This is ignored when no cutoff is used, and
            no switch is used if switchDistance is 0, negative, or greater than
            the cutoff
        constraints : None, app.HBonds, app.HAngles, or app.AllBonds
            Which type of constraints to add to the system (e.g., SHAKE). None
            means no bonds are constrained. HBonds means bonds with hydrogen are
            constrained
        rigidWater : bool=True
            If True, water is kept rigid regardless of the value of constraints.
            A value of False is ignored if constraints is not None.
        implicitSolvent : None, app.HCT, app.OBC1, app.OBC2, app.GBn, app.GBn2
            The Generalized Born implicit solvent model to use.
        implicitSolventKappa : float or 1/distance Quantity = None
            This is the Debye kappa property related to modeling saltwater
            conditions in GB. It should have units of 1/distance (1/nanometers
            is assumed if no units present). A value of None means that kappa
            will be calculated from implicitSolventSaltConc (below)
        implicitSolventSaltConc : float or amount/volume Quantity=0 moles/liter
            If implicitSolventKappa is None, the kappa will be computed from the
            salt concentration. It should have units compatible with mol/L
        temperature : float or temperature Quantity = 298.15 kelvin
            This is only used to compute kappa from implicitSolventSaltConc
        soluteDielectric : float=1.0
            The dielectric constant of the protein interior used in GB
        solventDielectric : float=78.5
            The dielectric constant of the water used in GB
        useSASA : bool=False
            If True, use the ACE non-polar solvation model. Otherwise, use no
            SASA-based nonpolar solvation model.
        removeCMMotion : bool=True
            If True, the center-of-mass motion will be removed periodically
            during the simulation. If False, it will not.
        hydrogenMass : float or mass quantity = None
            If not None, hydrogen masses will be changed to this mass and the
            difference subtracted from the attached heavy atom (hydrogen mass
            repartitioning)
        ewaldErrorTolerance : float=0.0005
            When using PME or Ewald, the Ewald parameters will be calculated
            from this value
        flexibleConstraints : bool=True
            If False, the energies and forces from the constrained degrees of
            freedom will NOT be computed. If True, they will (but those degrees
            of freedom will *still* be constrained).
        verbose : bool=False
            If True, the progress of this subroutine will be printed to stdout
        splitDihedrals : bool=False
            If True, the dihedrals will be split into two forces -- proper and
            impropers. This is primarily useful for debugging torsion parameter
            assignments.

        Returns
        -------
        system : openmm.System
            System formatted according to the prmtop file.

        Notes
        -----
        This function calls prune_empty_terms if any Topology lists have
        changed.
        """
        return structure.createSystem(**kwargs)
