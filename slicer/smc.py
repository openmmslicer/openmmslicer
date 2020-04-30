import copy as _copy
import inspect as _inspect
import logging as _logging
import os as _os
import pickle as _pickle
import random as _random

import mdtraj as _mdtraj
import mdtraj.reporters as _reporters
import numpy as _np
import openmmtools as _openmmtools
from scipy.special import logsumexp as _logsumexp
import simtk.openmm as _openmm
import simtk.unit as _unit
import simtk.openmm.app as _app

import slicer.integrators as _integrators
from slicer.minimise import GreedyBisectingMinimiser as _GreedyBisectingMinimiser
import slicer.moves as _moves
import slicer.resampling_metrics as _resmetrics
import slicer.resampling_methods as _resmethods
import slicer.sampling_metrics as _sammetrics

_logger = _logging.getLogger(__name__)


class SequentialSampler:
    """
    A complete sequential Monte Carlo sampler which can enhance the sampling of certain degrees of freedom.

    Parameters
    ----------
    coordinates : str
        Path to a file which contains all coordinates.
    structure : parmed.Structure
        An object containing all structural information.
    integrator : openmm.Integrator
        The integrator which should be used during the sampling step.
    moves : slicer.moves.MoveList or [slicer.moves.Move]
        All moves which must be applied at lambda = 0.
    platform : str
        The platform which should be used for simulation.
    platform_properties : dict
        Additional platform properties.
    npt : bool
        Whether to add a barostat at 1 atm.
    checkpoint : str
        A path to a pickled checkpoint file to load SequentialSampler from. If this is None, SequentialSampler is
        initialised normally.
    md_config : dict
        Additional parameters passed to generateSystem().
    alch_config : dict
        Additional parameters passed to generateAlchSystem().

    Attributes
    ----------
    coordinates : str
        Path to a file which contains all coordinates.
    moves : [Move]
        All moves which must be applied at lambda = 0.
    structure : parmed.Structure
        An object containing all structural information.
    system : openmm.System
        The initial unmodified system.
    alch_system : openmm.System
        The modified alchemical system.
    platform : str
        The currently used platform.
    platform_properties : dict
        The currently used platform properties.
    integrator : openmm.integrator
        The integrator used during the sampling step.
    simulation : openmm.Simulation
        A simulation object of the current alchemical system.
    lambda_ : float
        The current lambda value.
    total_sampling_steps : int
        A counter keeping track of the number of times the integrator was called.
    lambda_history : list
        A list containing all past lambda values.
    deltaE_history : list
        A list containing all past deltaE values.
    weight_history : list
        A list containing all past weights used for resampling.
    reporter_history : list
        A list containing paths to all past trajectory files.
    current_states : [int] or [openmm.State]
        A list containing all current states.
    logZ : float
        The current estimate of the dimensionless free energy difference.
    """
    _picklable_attrs = ["lambda_", "total_sampling_steps", "lambda_history", "deltaE_history", "weight_history",
                        "reporter_history", "logZ"]

    def __init__(self, coordinates, structure, integrator, moves, platform=None, platform_properties=None,
                 npt=True, checkpoint=None, md_config=None, alch_config=None):
        if md_config is None:
            md_config = {}
        if alch_config is None:
            alch_config = {}

        self.coordinates = coordinates
        self.moves = moves
        self.structure = structure
        self.system = SequentialSampler.generateSystem(self.structure, **md_config)
        self.generateAlchemicalRegion()
        if "alchemical_torsions" not in alch_config.keys():
            alch_config["alchemical_torsions"] = self._alchemical_dihedral_indices
        self.alch_system = SequentialSampler.generateAlchSystem(self.system, self.alchemical_atoms, **alch_config)
        if npt:
            self.alch_system = self.addBarostat(self.alch_system, temperature=integrator.getTemperature())

        # TODO: implement parallelism?
        self.platform = platform
        self.platform_properties = platform_properties
        self.integrator = _copy.copy(integrator)
        self.simulation = self.generateSimFromStruct(
            structure, self.alch_system, self.integrator, platform, platform_properties)
        # this is only used for energy evaluation
        dummy_integrator = _integrators.AlchemicalEnergyEvaluator(alchemical_functions=integrator._alchemical_functions)
        self._dummy_simulation = self.generateSimFromStruct(
            structure, self.alch_system, dummy_integrator, platform, platform_properties)

        if checkpoint is not None:
            _logger.info("Loading checkpoint...")
            obj = _pickle.load(open(checkpoint, "rb"))
            for attr in self._picklable_attrs + ["current_states"]:
                self.__setattr__(attr, getattr(obj, attr))
        else:
            self.lambda_ = 0
            self.total_sampling_steps = 0
            self.lambda_history = [0]
            self.deltaE_history = []
            self.weight_history = []
            self.reporter_history = []
            self.current_states = []
            self.logZ = 0

    @classmethod
    def generateSimFromStruct(cls, structure, system, integrator, platform=None, properties=None, **kwargs):
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

    def run(self, save_checkpoint=True, checkpoint_filename="checkpoint.pickle", final_decorrelation_step=True,
            *args, **kwargs):
        """
        Performs a complete sequential Monte Carlo run until lambda = 1.

        Parameters
        ----------
        save_checkpoint : bool
            Whether to save a checkpoint file.
        checkpoint_filename : str
            Path to the pickle filename to save the checkpoint to. Only valid if save_checkpoint is True.
        final_decorrelation_step : bool
            Whether to decorrelate the final resampled walkers for another number of default_decorrelation_steps.
        args
            Positional arguments to be passed to runSingleIteration().
        kwargs
            Keyword arguments to be passed to runSingleIteration().
        """
        def runAndSave():
            self.runSingleIteration(*args, **kwargs)
            if save_checkpoint:
                self.saveCheckpoint(checkpoint_filename)

        while self.lambda_ < 1:
            runAndSave()
        if final_decorrelation_step:
            runAndSave()
        total_sampling_time = self.total_sampling_steps * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
        _logger.info("Total simulation time was {}".format(total_sampling_time))

    def runSingleIteration(self,
                           sampling_metric=_sammetrics.EnergyCorrelation,
                           resampling_method=_resmethods.SystematicResampler,
                           resampling_metric=_resmetrics.WorstCaseSampleSize,
                           target_metric_value=None,
                           target_metric_value_initial=None,
                           target_metric_tol=None,
                           maximum_metric_evaluations=20,
                           default_dlambda=0.1,
                           minimum_dlambda=0.01,
                           default_decorrelation_steps=500,
                           maximum_decorrelation_steps=5000,
                           reporter_filename=None,
                           n_walkers=1000,
                           n_conformers_per_walker=100,
                           equilibration_options=None,
                           dynamically_generate_conformers=True,
                           keep_walkers_in_memory=False):
        """
        Performs a single iteration of sequential Monte Carlo.

        Parameters
        ----------
        sampling_metric : class
            A sampling metric with callable methods as described in slicer.sampling_metrics. This metric is used
            to adaptively determine the optimal sampling time. None removes adaptive sampling.
        resampling_method : class
            A resampling method with callable methods as described in slicer.resampling_methods.
        resampling_metric : class
            A resampling metric with callable methods as described in slicer.resampling_metrics. This metric is used
            to adaptively determine the optimal next lambda. None removes adaptive resampling.
        target_metric_value : float
            The threshold for the resampling metric. None uses the default value given by the class.
        target_metric_value_initial : float
            Same as target_metric_value, but only valid for lambda = 0.
        target_metric_tol : float
            The relative tolerance for the resampling metric. None uses the default value given by the class.
        maximum_metric_evaluations : int
            The maximum number of energy evaluations to determine the resampling metric.
        default_dlambda : float
            Determines the next lambda value. Only used if resampling_metric is None.
        minimum_dlambda : float
            The minimum allowed change in lambda.
        default_decorrelation_steps : int
            The default number of decorrelation steps. If sampling metric is None, these denote the true number of
            decorrelation steps.
        maximum_decorrelation_steps : int
            The maximum number of decorrelation steps. Only used with adaptive sampling.
        reporter_filename : str
            A template containing the reporter filename. Must contain curly brackets, so that python's format() can
            be called.
        n_walkers : int
            The number of walkers to be resampled.
        n_conformers_per_walker : int
            How many conformers to generate for each walker using self.moves. Only used at lambda = 0.
        equilibration_options : dict
            Parameters to be passed to equilibrate(). Only used at lambda = 0.
        dynamically_generate_conformers : bool
            Whether to store the extra conformers as states or as transformations. The former is much faster, but also
            extremely memory-intensive. Only set to False if you are certain that you have enough memory.
        keep_walkers_in_memory : bool
            Whether to keep the walkers as states or load them dynamically from the hard drive. The former is much
            faster during the energy evaluation step but is more memory-intensive. Only set to True if you are certain
            that you have enough memory.
        """
        if not keep_walkers_in_memory and reporter_filename is None:
            raise ValueError("Need to set a reporter if trajectory is not kept in memory.")
        if not keep_walkers_in_memory and not dynamically_generate_conformers:
            raise ValueError("Cannot store extra conformers in memory if the walkers are not kept in memory.")
        if _inspect.isclass(sampling_metric):
            sampling_metric = sampling_metric(self)
        default_decorrelation_steps = max(1, default_decorrelation_steps)

        # equilibrate if lambda = 0
        if not self.lambda_:
            if equilibration_options is None:
                equilibration_options = {}
            if reporter_filename is not None and "reporter_filename" not in equilibration_options:
                equilibration_options["reporter_filename"] = reporter_filename.format("equil")
            self.equilibrate(**equilibration_options)
            self.current_states = [self.simulation.context.getState(getPositions=True, getEnergy=True)] * n_walkers

        # adaptively set decorrelation time, if needed
        if self.lambda_ and sampling_metric:
            sampling_metric.evaluateBefore()
        elapsed_steps = 0
        while True:
            extra_conformers = []

            # set up reporter, if applicable
            if reporter_filename:
                curr_reporter_filename = reporter_filename.format(round(self.lambda_, 8))
                if _os.path.exists(curr_reporter_filename):
                    if elapsed_steps:
                        prev_reporter_filebase, ext = _os.path.splitext(curr_reporter_filename)
                        prev_reporter_filename = prev_reporter_filebase + "_prev" + ext
                        _os.rename(curr_reporter_filename, prev_reporter_filename)
                        self.reporter_history = [x for x in self.reporter_history if x != prev_reporter_filename]
                        self.reporter_history[-1] = prev_reporter_filename
                    else:
                        backup_filename = _os.path.splitext(curr_reporter_filename)[0] + "_backup.dcd"
                        _os.rename(curr_reporter_filename, backup_filename)
                        self.reporter_history = [backup_filename if x == curr_reporter_filename else x
                                                 for x in self.reporter_history]
                    self.simulation.reporters = []
                self.simulation.reporters.append(_reporters.DCDReporter(curr_reporter_filename,
                                                                        default_decorrelation_steps))

            # sample
            for n, state in enumerate(self.current_states):
                prev_reporter_filename = None if not len(self.reporter_history) else self.reporter_history[-1]
                if type(state) is tuple:
                    state, transform = state
                else:
                    transform = None
                self.setState(state, self.simulation.context, reporter_filename=prev_reporter_filename,
                              transform=transform)
                self.simulation.context.setVelocitiesToTemperature(self.temperature)
                self.simulation.step(default_decorrelation_steps)

                # generate conformers if needed
                if not self.lambda_:
                    if not n:
                        _logger.info("Generating {} total conformers...".format(n_walkers * n_conformers_per_walker))
                    target_metric_value = target_metric_value_initial
                    transforms = self.moves.generateMoves(n_conformers_per_walker)
                    if dynamically_generate_conformers:
                        extra_conformers += [transforms]
                    else:
                        for t in transforms:
                            self.moves.applyMove(self.simulation.context, t)
                            extra_conformers += [self.simulation.context.getState(getPositions=True)]
                            self.simulation.context.setState(state)

                # update states
                if keep_walkers_in_memory:
                    self.current_states[n] = self.simulation.context.getState(getPositions=True, getEnergy=True)
                else:
                    self.current_states[n] = n
            elapsed_steps += default_decorrelation_steps
            self.total_sampling_steps += default_decorrelation_steps * n_walkers

            if not self.lambda_:
                if dynamically_generate_conformers:
                    extra_conformers = _np.concatenate(extra_conformers)

            # reset the reporters
            if reporter_filename:
                self.simulation.reporters = []
                self.reporter_history += [curr_reporter_filename]

            # return if this is a final decorrelation step
            if self.lambda_ == 1:
                _logger.info("Sampling at lambda = 1 terminated after {} steps".format(elapsed_steps))
                return

            # continue decorrelation if metric says so and break otherwise
            # only for metrics that don't need the next lambda value
            if self.lambda_ and sampling_metric and not sampling_metric.requireNextLambda:
                sampling_metric.evaluateAfter()
                _logger.debug("Sampling metric {:.8g}".format(sampling_metric.metric))
                if not sampling_metric.terminateSampling and elapsed_steps < maximum_decorrelation_steps:
                    continue

            # here we evaluate the baseline reduced energies
            self._current_reduced_potentials = self.calculateStateEnergies(self.lambda_, extra_conformers=extra_conformers)

            self._current_deltaEs = {}
            self._current_weights = {}

            # this is the function we are going to minimise
            def evaluateWeights(lambda_):
                new_lambda = float(min(1., lambda_))
                self._new_reduced_potentials = self.calculateStateEnergies(new_lambda, extra_conformers=extra_conformers)
                self._current_deltaEs[new_lambda] = self._new_reduced_potentials - self._current_reduced_potentials
                self._current_weights[new_lambda] = _np.exp(_np.nanmin(self._current_deltaEs[new_lambda]) - self._current_deltaEs[new_lambda])
                self._current_weights[new_lambda][self._current_weights[new_lambda] != self._current_weights[new_lambda]] = 0
                self._current_weights[new_lambda] /= sum(self._current_weights[new_lambda])
                if resampling_metric is not None:
                    val = resampling_metric.evaluate(self._current_weights[new_lambda])
                    _logger.debug("Resampling metric {:.8g} at next lambda {:.8g}".format(val, new_lambda))
                    return val

            # evaluate next lambda value
            if resampling_metric:
                if target_metric_value is None:
                    target_metric_value = resampling_metric.defaultValue(n_walkers)
                if target_metric_tol is None:
                    target_metric_tol = resampling_metric.defaultTol(n_walkers)

                # minimise and set optimal lambda value adaptively if possible
                length = max(len(self.current_states), len(extra_conformers))
                pivot_y = resampling_metric.evaluate([1 / length] * length)
                self.next_lambda_ = _GreedyBisectingMinimiser.minimise(evaluateWeights, target_metric_value,
                                                                       self.lambda_, 1., pivot_y=pivot_y,
                                                                       minimum_x=minimum_dlambda, tol=target_metric_tol,
                                                                       maxfun=maximum_metric_evaluations)
                _logger.debug("Tentative next lambda: {:.8g}".format(self.next_lambda_))
            else:
                # else use default_dlambda
                self.next_lambda_ = min(1., self.lambda_ + default_dlambda)
                evaluateWeights(self.next_lambda_)

            # continue decorrelation if metric says so and break otherwise
            # only for metrics that need the next lambda value
            if self.lambda_ and sampling_metric:
                if sampling_metric.requireNextLambda:
                    sampling_metric.evaluateAfter()
                    _logger.debug("Sampling metric {:.8g} at next lambda {:.8g}".format(sampling_metric.metric,
                                                                                       self.next_lambda_))
                    if not sampling_metric.terminateSampling and elapsed_steps < maximum_decorrelation_steps:
                        continue
                sampling_metric.reset()

            if reporter_filename and elapsed_steps != default_decorrelation_steps:
                _os.remove(self.reporter_history[-2])
            break

        # update histories, lambdas, and partition functions
        self.lambda_ = self.next_lambda_
        self.lambda_history += [self.lambda_]
        self.simulation.integrator.setGlobalVariableByName("lambda", self.lambda_)
        self._current_deltaEs = self._current_deltaEs[self.lambda_]
        self._current_weights = self._current_weights[self.lambda_]
        self.deltaE_history += [self._current_deltaEs]
        self.weight_history += [self._current_weights]
        current_deltaEs = self._current_deltaEs[self._current_deltaEs == self._current_deltaEs]
        self.logZ += _logsumexp(-current_deltaEs) - _np.log(self._current_deltaEs.shape[0])

        # sample new states based on weights
        if extra_conformers is not None and len(extra_conformers):
            if dynamically_generate_conformers:
                new_states = resampling_method.resample([i for i in range(len(extra_conformers))],
                                                        self._current_weights, n_walkers=n_walkers)[0]
                self.current_states = [(self.current_states[i // n_conformers_per_walker], extra_conformers[i])
                                       for i in new_states]
            else:
                self.current_states = resampling_method.resample(extra_conformers, self._current_weights,
                                                                 n_walkers=n_walkers)[0]
        else:
            self.current_states = resampling_method.resample(self.current_states, self._current_weights,
                                                             n_walkers=n_walkers)[0]
        _random.shuffle(self.current_states)

        _logger.info("Sampling at lambda = {:.8g} terminated after {} steps".format(self.lambda_history[-2],
                                                                                    elapsed_steps))
        _logger.info("Reporter path: \"{}\"".format(self.reporter_history[-1]))
        _logger.info("Current accumulated logZ: {:.8g}".format(self.logZ))
        _logger.info("Next lambda: {:.8g}".format(self.lambda_))

    def saveCheckpoint(self, filename="checkpoint.pickle", *args, **kwargs):
        """
        Saves a pickled checkpoint file.

        Parameters
        ----------
        filename : str
            Path to the pickle filename to save the checkpoint to.
        args
            Positional arguments to be passed to pickle.dump().
        kwargs
            Keyword arguments to be passed to pickle.dump().
        """
        new_self = _copy.copy(self)
        new_self.__dict__ = {x: y for x, y in new_self.__dict__.items() if x in self._picklable_attrs}
        new_self.current_states = [i for i in range(len(self.current_states))]
        if _os.path.exists(filename):
            _os.rename(filename, filename + ".old")
        _logger.info("Saving checkpoint...")
        _pickle.dump(new_self, open(filename, "wb"), *args, **kwargs)
        if _os.path.exists(filename + ".old"):
            _os.remove(filename + ".old")

    @property
    def alchemical_atoms(self):
        """[int]: The absolute indices of all alchemical atoms."""
        return self.moves.alchemical_atoms

    @property
    def kT(self):
        """openmm.unit.Quantity: The current temperature multiplied by the gas constant."""
        try:
            kB = _unit.BOLTZMANN_CONSTANT_kB * _unit.AVOGADRO_CONSTANT_NA
            kT = kB * self.integrator.getTemperature()
            return kT
        except:
            return None

    @property
    def moves(self):
        """The list of moves which must be performed at lambda = 0."""
        return self._moves

    @moves.setter
    def moves(self, val):
        if not isinstance(val, (list, _moves.MoveList)):
            val = _moves.MoveList([val])
        elif isinstance(val, list):
            val = _moves.MoveList(val)
        self._moves = val

    @property
    def n_walkers(self):
        """int: The number of current walkers."""
        return len(self.current_states)

    @property
    def temperature(self):
        """openmm.unit.Quantity: The temperature of the current integrator."""
        try:
            T = self.integrator.getTemperature()
            return T
        except:
            return None

    def calculateStateEnergies(self, lambda_, states=None, extra_conformers=None, *args, **kwargs):
        """
        Calculates the reduced potential energies of all states for a given lambda value.

        Parameters
        ----------
        lambda_ : float
            The desired lambda value.
        states : [int] or [openmm.State] or None
            Which states need to be used. If None, self.current_states are used. Otherwise, these could be in any
            format supported by setState().
        extra_conformers : list
            Extra conformers relevant for lambda = 0. These could be either a list of openmm.State or a list of
            transformations which can be generated dynamically.
        args
            Positional arguments to be passed to setState().
        kwargs
            Keyword arguments to be passed to setState().
        """
        if states is None:
            states = self.current_states

        energies = []
        conf_iterations = 0 if extra_conformers is None else len(extra_conformers) // len(states)
        if "reporter_filename" not in kwargs.keys():
            kwargs["reporter_filename"] = None if not len(self.reporter_history) else self.reporter_history[-1]

        for i, state in enumerate(states):
            # this is the for the case of lambda = 0
            if conf_iterations:
                # here we optimise by only loading the state once from the hard drive, if applicable
                if type(state) is int:
                    self.setState(state, self._dummy_simulation.context, *args, **kwargs)
                    state = self._dummy_simulation.context.getState(getPositions=True)
                # here we generate the conformers dynamically, if applicable
                for j in range(conf_iterations):
                    # we generate dynamically
                    if type(extra_conformers[conf_iterations * i + j]) is _np.ndarray:
                        self.setState(state, self._dummy_simulation.context,
                                      transform=extra_conformers[conf_iterations * i + j], *args, **kwargs)
                    # or we don't
                    else:
                        self.setState(extra_conformers[conf_iterations * i + j], self._dummy_simulation.context,
                                      *args, **kwargs)
                    energies += [self._dummy_simulation.integrator.getPotentialEnergyFromLambda(lambda_) / self.kT]
            # this is for all other cases
            else:
                # we load from the hard drive
                if type(state) is tuple:
                    state, transform = state
                # or we don't
                else:
                    transform = None
                kwargs["transform"] = transform
                self.setState(state, self._dummy_simulation.context, *args, **kwargs)
                energies += [self._dummy_simulation.integrator.getPotentialEnergyFromLambda(lambda_) / self.kT]

        return _np.asarray(energies, dtype=_np.float32)

    def setState(self, state, context, reporter_filename=None, transform=None):
        """
        Sets a given state to the current context.

        Parameters
        ----------
        state : int, openmm.State
            Either sets the state from a State object or from a frame of a trajectory file given by reporter_filename.
        context : openmm.Context
            The context to which the state needs to be applied.
        reporter_filename : str
            The path to the trajectory file containing the relevant frame, if applicable.
        transform :
            Optionally generate a transform dynamically from a format, specific to the underlying moves.
        """
        if type(state) is int:
            frame = _mdtraj.load_frame(reporter_filename, state, self.coordinates)
            positions = frame.xyz[0]
            periodic_box_vectors = frame.unitcell_vectors[0]
            context.setPositions(positions)
            context.setPeriodicBoxVectors(*periodic_box_vectors)
        else:
            context.setState(state)
        if transform is not None:
            self.moves.applyMove(context, transform)

    def generateAlchemicalRegion(self):
        """Makes sure that all rotated dihedrals are also made alchemical."""
        self._rotatable_bonds = [x.rotatable_bond for x in self.moves.moves if isinstance(x, _moves.DihedralMove)]
        all_rotatable_bonds = {frozenset(x) for x in self._rotatable_bonds}
        self._alchemical_dihedral_indices = [i for i, d in enumerate(self.structure.dihedrals) if not d.improper and
                                             {d.atom2.idx, d.atom3.idx} in all_rotatable_bonds]

    def equilibrate(self,
                    equilibration_steps=100000,
                    restrain_backbone=True,
                    force_constant=5.0 * _unit.kilocalories_per_mole / _unit.angstroms ** 2,
                    output_interval=100000,
                    reporter_filename=None):
        """
        Equilibrates the system at lambda = 0.

        Parameters
        ----------
        equilibration_steps : int
            The number of equilibration steps.
        restrain_backbone : bool
            Whether to restrain all atoms with the following names: 'CA', 'C', 'N'.
        force_constant : openmm.unit.Quantity
            The magnitude of the restraint force constant.
        output interval : int
            How often to output to a trajectory file.
        reporter_filename : str
            The path to the trajectory file.
        """
        if reporter_filename is not None:
            output_interval = min(output_interval, equilibration_steps)
            self.simulation.reporters.append(_reporters.DCDReporter(reporter_filename, output_interval))

        # add restraints, if applicable
        if restrain_backbone:
            force = _openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            force.addGlobalParameter("k", force_constant)
            force.addPerParticleParameter("x0")
            force.addPerParticleParameter("y0")
            force.addPerParticleParameter("z0")
            counter = 0
            for i, atom_crd in enumerate(self.structure.positions):
                if self.structure.atoms[i].name in ('CA', 'C', 'N'):
                    force.addParticle(i, atom_crd.value_in_unit(_unit.nanometers))
                    counter += 1
            if counter:
                _logger.info("Adding {} equilibration restraints...".format(counter))
                force_idx = self.alch_system.addForce(force)

        # run the equilibration
        _logger.info("Running initial equilibration...")
        self.simulation.integrator.setGlobalVariableByName("lambda", 0.)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        self.simulation.step(equilibration_steps)

        # remove the restraints, if applicable
        if restrain_backbone and counter:
            _logger.info("Removing {} equilibration restraints...".format(counter))
            self.alch_system.removeForce(force_idx)

        if reporter_filename is not None:
            self.simulation.reporters = []

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
        system : openmm.System
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
                           softcore_beta=0.5,
                           softcore_d=1,
                           softcore_e=1,
                           softcore_f=2,
                           alchemical_torsions=True,
                           annihilate_electrostatics=True,
                           annihilate_sterics=True,
                           disable_alchemical_dispersion_correction=True,
                           alchemical_pme_treatment='direct-space',
                           suppress_warnings=True,
                           **kwargs):
        """
        Returns the OpenMM System for alchemical perturbations.
        This function calls `openmmtools.alchemy.AbsoluteAlchemicalFactory` and
        `openmmtools.alchemy.AlchemicalRegion` to generate the System for the
        SMC simulation.

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
            System to be used for the SMC simulation.

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

    @classmethod
    def addBarostat(cls, system, temperature=298 * _unit.kelvin, pressure=1 * _unit.atmospheres, frequency=25, **kwargs):
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
        _logger.info('Adding MonteCarloBarostat with {}. MD simulation will be {} NPT.'.format(pressure, temperature))
        # Add Force Barostat to the system
        system.addForce(_openmm.MonteCarloBarostat(pressure, temperature, frequency))
        return system
