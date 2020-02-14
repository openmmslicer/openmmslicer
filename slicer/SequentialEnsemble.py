import copy as _copy
import logging as _logging
import random as _random

import mdtraj.reporters as _reporters
import numpy as _np
import openmmtools as _openmmtools
from scipy.interpolate import interp1d as _interp1d
from scipy.misc import comb as _comb
from scipy.optimize import minimize as _minimize
import simtk.openmm as _openmm
import simtk.unit as _unit
import simtk.openmm.app as _app

from slicer.conformations import ConformationGenerator as _ConformationGenerator
from slicer import integrators as _integrators

# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = _unit.kilojoules_per_mole

class SequentialEnsemble:
    _read_only_properties = ["alchemical_atoms", "current_states", "lambda_", "ligand", "ligname", "structure",
                             "rotatable_bonds"]

    def __init__(self, structure, integrator, platform, ligname="LIG", rotatable_bonds=None, md_config=None,
                 alch_config=None, n_walkers=100):
        if not md_config:
            md_config = {}
        if not alch_config:
            alch_config = {}
        self._setStructure(structure, ligname=ligname)
        self.system = SequentialEnsemble.generateSystem(self._structure, **md_config)
        self.generateAlchemicalRegion(rotatable_bonds)
        self.alch_system = SequentialEnsemble.generateAlchSystem(self.system, self._alchemical_atoms, **alch_config)

        # TODO: implement parallelism?
        self.integrator = _copy.copy(integrator)
        self.simulation = self.generateSimFromStruct(structure, self.alch_system, self.integrator, platform=platform)
        # this is only used for energy evaluation
        self._dummy_simulation = self.generateSimFromStruct(structure, self.alch_system,
                                                            _integrators.DummyAlchemicalIntegrator(), platform=platform)

        self._lambda_ = 0
        self.n_walkers = n_walkers

        self._deltaE_history = []
        self._weight_history = []
        self._current_states = []

    def __getattr__(self, item):
        if item not in self._read_only_properties:
            return self.__getattribute__(item)
        else:
            try:
                return self.__getattribute__("_" + item)
            except AttributeError:
                return None

    @classmethod
    def generateSimFromStruct(cls, structure, system, integrator, platform=None, properties=None, **kwargs):
        """Generate the OpenMM Simulation objects from a given parmed.Structure()

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

    def run(self, *args, **kwargs):
        while self._lambda_ <= 1:
            self.runSingleIteration(*args, **kwargs)
            if self._lambda_ == 1:
                break

    def runSingleIteration(self, distribution="dihedrals", sampling="semi-deterministic", decorrelation_steps=500,
                           equilibration_steps=100000, maximum_weight=0.05, default_increment=0.1,
                           minimum_increment=0.01, reporter_filename=None, n_conformers=None,
                           output_equilibration=True):
        if not n_conformers:
            n_conformers = self.n_walkers
        decorrelation_steps = max(1, decorrelation_steps)

        # run either initial conformer generation or MD decorrelation
        if not self._lambda_:
            equilibration_steps = max(1, equilibration_steps)
            if reporter_filename and output_equilibration:
                self.simulation.reporters.append(_reporters.DCDReporter(reporter_filename.format("equil"), decorrelation_steps))
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
            self.simulation.step(equilibration_steps)
            confgen = _ConformationGenerator(self.system, self._structure, self.simulation.context,
                                             self._rotatable_bonds)
            self._current_states = confgen.generateConformers(n_conformers, distribution=distribution, sampling=sampling)
        else:
            if reporter_filename:
                self.simulation.reporters.append(_reporters.DCDReporter(
                    reporter_filename.format(round(self._lambda_, 3)), decorrelation_steps))
            for n, state in enumerate(self._current_states):
                self.simulation.context.setState(state)
                self.simulation.context.setVelocitiesToTemperature(self.temperature)
                self.simulation.step(decorrelation_steps)
                self._current_states[n] = self.simulation.context.getState(getPositions=True, getEnergy=True)
        # reset the reporters
        self.simulation.reporters = []

        # return if this is a final decorrelation step
        if self._lambda_ == 1:
            return

        # otherwise come up with a new lambda value
        dlambda = abs(default_increment)

        # here we calculate the weights and update the lambda value iteratively if we need to
        # maintain a maximum allowed weight
        previous_dlambdas = [0]
        previous_weights = [1 / len(self._current_states)]
        calculate_initial_energy = True
        current_reduced_potentials = []
        while True:
            # calculate energies based on dlambda
            new_reduced_potentials = []
            new_lambda = min(1, self._lambda_ + dlambda)
            dlambda = new_lambda - self._lambda_
            for state in self._current_states:
                self._dummy_simulation.context.setState(state)
                if calculate_initial_energy:
                    current_reduced_potentials += [self._dummy_simulation.integrator.getPotentialEnergyFromLambda(self.lambda_) / self.kT]
                new_reduced_potentials += [self._dummy_simulation.integrator.getPotentialEnergyFromLambda(new_lambda) / self.kT]
            calculate_initial_energy = False
            deltaEs = _np.array(new_reduced_potentials) - _np.array(current_reduced_potentials)
            weights = _np.exp(_np.nanmin(deltaEs)-deltaEs)
            weights[weights != weights] = 0
            weights /= sum(weights)
            
            # terminate if the maximum weight is within 10% of the user-specified limit or if we are not updating
            # the lambda value adaptively
            max_weight = _np.max(weights)
            if (0.9 * maximum_weight < max_weight < 1.1 * maximum_weight) or \
                    (max_weight < maximum_weight and new_lambda == 1) or \
                    (maximum_weight <= 1 / len(self._current_states)):
                break

            # otherwise predict a new dlambda based on interpolation / extrapolation of previous iterations
            # TODO: we assume always strictly increasing function - is that necessarily the case?
            interp = _interp1d(_np.log(previous_weights + [max_weight]), previous_dlambdas + [dlambda],
                               fill_value='extrapolate')
            new_dlambda = interp(_np.log(maximum_weight))
            dlambda = new_dlambda
            if minimum_increment:
                dlambda = max(minimum_increment, dlambda)

        self._deltaE_history += [deltaEs]
        self._weight_history += [weights]

        # sample new states based on weights
        integer_weights = self.minimumVarianceResampling(weights, n_walkers=self.n_walkers)
        self._current_states = sum([int(i) * [x] for i, x in zip(integer_weights, self._current_states)], [])

        # update the lambda
        self._lambda_ = new_lambda
        self.simulation.integrator.setGlobalVariableByName("lambda", self._lambda_)

    @property
    def kT(self):
        try:
            kB = _unit.BOLTZMANN_CONSTANT_kB * _unit.AVOGADRO_CONSTANT_NA
            kT = kB * self.integrator.getTemperature()
            return kT
        except:
            return None

    @property
    def temperature(self):
        try:
            T = self.integrator.getTemperature()
            return T
        except:
            return None

    def _setStructure(self, struct, ligname="LIG"):
        self._structure = struct
        self._ligname = ligname

        # only get the relevant residue
        idx_abs = sum(([x.idx for x in res.atoms] for res in self._structure.residues if res.name == self._ligname), [])
        lig = self.structure[":{}".format(ligname)]
        idx_rel = sum(([x.idx for x in res.atoms] for res in lig.residues if res.name == self._ligname), [])
        self._abs_to_rel = {x: y for x, y in zip(idx_abs, idx_rel)}
        self._rel_to_abs = {x: y for x, y in zip(idx_rel, idx_abs)}

    def generateAlchemicalRegion(self, rotatable_bonds):
        self._rotatable_bonds = [(self._rel_to_abs[i], self._rel_to_abs[j]) for (i, j) in rotatable_bonds]
        confgen = _ConformationGenerator(self.system, self._structure, None, self._rotatable_bonds)
        self._alchemical_atoms = set().union(*list(confgen._rotatable_atoms.values()))

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
            simtk.openmm.app namespace
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
        openmm.System
            System formatted according to the prmtop file.

        Notes
        -----
        This function calls prune_empty_terms if any Topology lists have
        changed.
        """
        return structure.createSystem(**kwargs)

    @staticmethod
    def generateAlchSystem(system,
                           atom_indices,
                           softcore_alpha=0.5,
                           softcore_a=1,
                           softcore_b=1,
                           softcore_c=6,
                           softcore_beta=0.0,
                           softcore_d=1,
                           softcore_e=1,
                           softcore_f=2,
                           alchemical_torsions=True,
                           annihilate_electrostatics=True,
                           annihilate_sterics=False,
                           disable_alchemical_dispersion_correction=True,
                           alchemical_pme_treatment='direct-space',
                           suppress_warnings=True,
                           **kwargs):
        """Returns the OpenMM System for alchemical perturbations.
        This function calls `openmmtools.alchemy.AbsoluteAlchemicalFactory` and
        `openmmtools.alchemy.AlchemicalRegion` to generate the System for the
        NCMC simulation.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System object corresponding to the reference system.
        atom_indices : list of int
            Atom indicies of the move or designated for which the nonbonded forces
            (both sterics and electrostatics components) have to be alchemically
            modified.
        annihilate_electrostatics : bool, optional
            If True, electrostatics should be annihilated, rather than decoupled
            (default is True).
        annihilate_sterics : bool, optional
            If True, sterics (Lennard-Jones or Halgren potential) will be annihilated,
            rather than decoupled (default is False).
        softcore_alpha : float, optional
            Alchemical softcore parameter for Lennard-Jones (default is 0.5).
        softcore_a, softcore_b, softcore_c : float, optional
            Parameters modifying softcore Lennard-Jones form. Introduced in
            Eq. 13 of Ref. [TTPham-JChemPhys135-2011]_ (default is 1).
        softcore_beta : float, optional
            Alchemical softcore parameter for electrostatics. Set this to zero
            to recover standard electrostatic scaling (default is 0.0).
        softcore_d, softcore_e, softcore_f : float, optional
            Parameters modifying softcore electrostatics form (default is 1).
        disable_alchemical_dispersion_correction : bool, optional, default=True
            If True, the long-range dispersion correction will not be included for the alchemical
            region to avoid the need to recompute the correction (a CPU operation that takes ~ 0.5 s)
            every time 'lambda_sterics' is changed. If using nonequilibrium protocols, it is recommended
            that this be set to True since this can lead to enormous (100x) slowdowns if the correction
            must be recomputed every time step.
        alchemical_pme_treatment : str, optional, default = 'direct-space'
            Controls how alchemical region electrostatics are treated when PME is used.
            Options are 'direct-space', 'coulomb', 'exact'.
            - 'direct-space' only models the direct space contribution
            - 'coulomb' includes switched Coulomb interaction
            - 'exact' includes also the reciprocal space contribution, but it's
            only possible to annihilate the charges and the softcore parameters
            controlling the electrostatics are deactivated. Also, with this
            method, modifying the global variable `lambda_electrostatics` is
            not sufficient to control the charges. The recommended way to change
            them is through the `AlchemicalState` class.

        Returns
        -------
        alch_system : alchemical_system
            System to be used for the NCMC simulation.

        References
        ----------
        .. [TTPham-JChemPhys135-2011] T. T. Pham and M. R. Shirts, J. Chem. Phys 135, 034114 (2011). http://dx.doi.org/10.1063/1.3607597
        """
        if suppress_warnings:
            # Lower logger level to suppress excess warnings
            _logging.getLogger("openmmtools.alchemy").setLevel(_logging.ERROR)

        # Disabled correction term due to increased computational cost
        factory = _openmmtools.alchemy.AbsoluteAlchemicalFactory(
            disable_alchemical_dispersion_correction=disable_alchemical_dispersion_correction,
            alchemical_pme_treatment=alchemical_pme_treatment)
        alch_region = _openmmtools.alchemy.AlchemicalRegion(
            alchemical_atoms=atom_indices,
            softcore_alpha=softcore_alpha,
            softcore_a=softcore_a,
            softcore_b=softcore_b,
            softcore_c=softcore_c,
            softcore_beta=softcore_beta,
            softcore_d=softcore_d,
            softcore_e=softcore_e,
            softcore_f=softcore_f,
            alchemical_torsions=alchemical_torsions,
            annihilate_electrostatics=annihilate_electrostatics,
            annihilate_sterics=annihilate_sterics,
            **kwargs
        )

        alch_system = factory.create_alchemical_system(system, alch_region)
        return alch_system

    @staticmethod
    def minimumVarianceResampling(weights, n_walkers=None, discrete_superensemble_size=50, minimum_weight=0.001):
        weights = _np.array(weights)
        weights /= sum(weights)
        if not n_walkers:
            n_walkers = weights.shape[0]

        exact_walkers = n_walkers * weights
        integer_walkers = _np.floor(exact_walkers)
        n_residual_walkers = int(n_walkers - sum(integer_walkers))

        if n_residual_walkers:
            walkers_ensemble = []
            weights_ensemble = _np.array([])

            def minfunc(input):
                input = _np.abs(input)
                input /= _np.sum(_np.abs(input))
                average_weights = _np.sum([i * walker for i, walker in zip(input, walkers_ensemble)], axis=0)
                return _np.average(_np.abs(average_weights - residual_probabilities))

            # we partition the rest of the walkers equally between the walkers with highest residual probabilities
            residual_probabilities = exact_walkers - integer_walkers
            current_residuals = residual_probabilities[:]

            discrete_superensemble_size = int(min(_comb(weights.shape[0], n_residual_walkers), discrete_superensemble_size))
            for i in range(discrete_superensemble_size):
                residual_walkers = _np.zeros(len(weights))
                sorted_residuals_indices = _np.argsort(current_residuals)[::-1] # descending
                residual_walkers[sorted_residuals_indices[:n_residual_walkers]] = 1
                sorted_residuals = current_residuals[sorted_residuals_indices]
                creation_annihilation_pairs = min(len(sorted_residuals[sorted_residuals < 0]), len(sorted_residuals[sorted_residuals > 0]))
                for j in range(creation_annihilation_pairs):
                    if sorted_residuals[n_residual_walkers + j] > -sorted_residuals[-(j + 1)]:
                        residual_walkers[sorted_residuals_indices[n_residual_walkers + j]] = 1
                        residual_walkers[sorted_residuals_indices[-(j + 1)]] = -1
                    else:
                        break

                walkers_ensemble += [residual_walkers]
                minim = _minimize(minfunc, _np.append(weights_ensemble, [1 / (1 + len(weights_ensemble))]), method="L-BFGS-B")
                weights_ensemble = _np.abs(minim.x)
                weights_ensemble /= _np.sum(weights_ensemble)
                average_weights = _np.sum([w * walker for w, walker in zip(weights_ensemble, walkers_ensemble)], axis=0)
                current_residuals = residual_probabilities - average_weights
                if min(weights_ensemble) < minimum_weight:
                    break

            chosen_walkers, = _random.choices(walkers_ensemble, weights_ensemble)
            integer_walkers += chosen_walkers

        return integer_walkers
