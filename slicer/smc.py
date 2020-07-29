from collections import Counter as _Counter
import copy as _copy
import inspect as _inspect
import logging as _logging
import os as _os
import pickle as _pickle
import random as _random
import warnings as _warnings

import anytree as _anytree
import mdtraj as _mdtraj
import numpy as _np
import openmmtools as _openmmtools
from scipy.special import logsumexp as _logsumexp
import simtk.openmm as _openmm
import simtk.openmm.app as _app
import simtk.unit as _unit

import slicer.integrators as _integrators
from slicer.minimise import BisectingMinimiser as _BisectingMinimiser
import slicer.moves as _moves
import slicer.reporters as _reporters
import slicer.resampling_metrics as _resmetrics
import slicer.resampling_methods as _resmethods
import slicer.sampling_metrics as _sammetrics

_logger = _logging.getLogger(__name__)


class GenericSMCSampler:
    """
    A generic sequential Monte Carlo sampler which can enhance the sampling of certain degrees of freedom.

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
        A path to a pickled checkpoint file to load GenericSMCSampler from. If this is None, GenericSMCSampler is
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
    log_weight_history : list
        A list containing all past weights used for resampling.
    reporter : slicer.reporters.MultistateDCDReporter, None
        The reporter containing all trajectory files.
    states : [int] or [openmm.State]
        A list containing all current states.
    transforms : list
        A list containing all relevant transforms to be applied for each state.
    logZ : float
        The current estimate of the dimensionless free energy difference.
    """
    _picklable_attrs = ["total_sampling_steps", "_lambda_", "lambda_history", "log_weights", "log_weight_history",
                        "deltaE_history", "logZ", "logZ_history", "transforms", "initialised", "state_tree",
                        "_all_tree_nodes"]

    def __init__(self, coordinates, structure, integrator, moves, platform=None, platform_properties=None,
                 npt=True, checkpoint=None, md_config=None, alch_config=None):
        if md_config is None:
            md_config = {}
        if alch_config is None:
            alch_config = {}

        self.coordinates = coordinates
        self.moves = moves
        self.structure = structure
        self.system = GenericSMCSampler.generateSystem(self.structure, **md_config)
        self.generateAlchemicalRegion()
        if "alchemical_torsions" not in alch_config.keys():
            alch_config["alchemical_torsions"] = self._alchemical_dihedral_indices
        self.alch_system = GenericSMCSampler.generateAlchSystem(self.system, self.alchemical_atoms, **alch_config)
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

        self.initialised = False
        self.total_sampling_steps = 0
        self.states = []
        self.transforms = []
        self._lambda_ = 0
        self.lambda_history = [0]
        self.log_weights = []
        self.log_weight_history = []
        self.deltaE_history = []
        self.reporters = []
        self.logZ = 0
        self.logZ_history = [0]
        self.state_tree = _anytree.Node(0, lambda_=None, iteration=None, transform=None)
        self._all_tree_nodes = {self.state_tree}

        if checkpoint is not None:
            _logger.info("Loading checkpoint...")
            obj = _pickle.load(open(checkpoint, "rb"))["self"]
            for attr in self._picklable_attrs + ["states"]:
                try:
                    self.__setattr__(attr, getattr(obj, attr))
                    if attr == "_lambda_":
                        self.simulation.integrator.setGlobalVariableByName("lambda", attr)
                except AttributeError:
                    continue

    @property
    def trajectory_reporter(self):
        reporter = [x for x in self.reporters if isinstance(x, _reporters.MultistateDCDReporter)]
        if not len(reporter):
            return None
        if len(reporter) > 1:
            _warnings.warn("Only the first slicer.reporters.MultistateDCDReporter will be used for state loading")
        return reporter[0]

    @property
    def state_data_reporters(self):
        reporters = [x for x in self.reporters if isinstance(x, _reporters.MultistateStateDataReporter)]
        if not len(reporters):
            return None
        return reporters

    @property
    def current_trajectory_filename(self):
        reporter = self.trajectory_reporter
        return None if reporter is None else reporter.current_filename

    @property
    def previous_trajectory_filename(self):
        reporter = self.trajectory_reporter
        return None if reporter is None or len(reporter.filename_history) < 2 else reporter.filename_history[-2]

    @property
    def alchemical_atoms(self):
        """[int]: The absolute indices of all alchemical atoms."""
        return self.moves.alchemical_atoms

    @property
    def iteration(self):
        return self.lambda_history.count(0) - 1

    @property
    def kT(self):
        """openmm.unit.Quantity: The current temperature multiplied by the gas constant."""
        try:
            kB = _unit.BOLTZMANN_CONSTANT_kB * _unit.AVOGADRO_CONSTANT_NA
            kT = kB * self.integrator.getTemperature()
            return kT
        except AttributeError:
            return None

    @property
    def lambda_(self):
        return self._lambda_

    @lambda_.setter
    def lambda_(self, val):
        if val is not None and val != self._lambda_:
            # update lambdas
            self._lambda_ = val
            self.lambda_history += [self._lambda_]
            self.simulation.integrator.setGlobalVariableByName("lambda", self.lambda_)

            # update deltaEs
            if self._lambda_ in self._current_deltaEs.keys():
                current_deltaEs = self._current_deltaEs[self._lambda_]
            else:
                current_deltaEs = self.calculateDeltaEs(val)
                self._current_deltaEs = current_deltaEs
            self.deltaE_history += [current_deltaEs]

            # update log_weights
            lengths = [len(x) if x is not None else 1 for x in self.transforms]
            log_weights_old = self.log_weights[sum([[i] * x for i, x in enumerate(lengths)], [])]
            weights_old = _np.exp(log_weights_old - _logsumexp(log_weights_old))
            self.log_weights = log_weights_old - current_deltaEs
            self.log_weights -= _logsumexp(self.log_weights)
            self.log_weight_history += [self.log_weights]

            # update logZs
            self.logZ += _logsumexp(-current_deltaEs, b=weights_old)
            self.logZ_history += [self.logZ]

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
        return len(self.states)

    @property
    def temperature(self):
        """openmm.unit.Quantity: The temperature of the current integrator."""
        try:
            T = self.integrator.getTemperature()
            return T
        except AttributeError:
            return None

    def tree_layer(self, lambda_=None, iteration=None, transform=None):
        if lambda_ is None:
            lambda_ = self.lambda_
        if iteration is None:
            iteration = self.iteration
        nodes = [x for x in self._all_tree_nodes if x.lambda_ == lambda_ and x.iteration == iteration]
        nodes.sort(key=lambda x: x.name)
        if transform is True:
            nodes = [x for x in nodes if x.transform is True]
        elif transform is False:
            nodes = [x for x in nodes if x.transform is False]
        elif transform is None:
            if any(x.transform == True for x in nodes):
                nodes = [x for x in nodes if x.transform is True]
            else:
                nodes = [x for x in nodes if x.transform is False]
        return nodes

    def serialise(self):
        new_self = _copy.copy(self)
        new_self.__dict__ = {x: y for x, y in new_self.__dict__.items() if x in self._picklable_attrs}
        new_self.states = [i for i in range(len(self.states))]
        return new_self

    def writeCheckpoint(self, data, filename="checkpoint.pickle", update=True, *args, **kwargs):
        backups = {}
        if update:
            try:
                backups = _pickle.load(open(filename, "rb"))
            except (FileNotFoundError, _pickle.UnpicklingError):
                pass
        backups.update(data)
        backups["self"] = self.serialise()
        _logger.debug("Writing checkpoint...")
        if _os.path.exists(filename):
            _os.rename(filename, filename + ".old")
        _pickle.dump(backups, open(filename, "wb"), *args, **kwargs)
        if _os.path.exists(filename + ".old"):
            _os.remove(filename + ".old")

    def calculateDeltaEs(self, lambda1, lambda0=None, states=None, transforms=None, *args, **kwargs):
        if lambda0 is None:
            lambda0 = self.lambda_
        if states is None:
            states = self.states
        if transforms is None:
            transforms = self.transforms
        old_potentials = self.calculateStateEnergies(lambda0, states=states, transforms=transforms, *args, **kwargs)
        new_potentials = self.calculateStateEnergies(lambda1, states=states, transforms=transforms, *args, **kwargs)
        return new_potentials - old_potentials

    def calculateStateEnergies(self, lambda_, states=None, transforms=None, *args, **kwargs):
        """
        Calculates the reduced potential energies of all states for a given lambda value.

        Parameters
        ----------
        lambda_ : float
            The desired lambda value.
        states : [int] or [openmm.State] or None
            Which states need to be used. If None, self.states are used. Otherwise, these could be in any
            format supported by setState().
        transforms : list
            Extra transforms to be passed to setState().
        args
            Positional arguments to be passed to setState().
        kwargs
            Keyword arguments to be passed to setState().
        """
        if states is None:
            states = self.states

        energies = []
        for i, state in enumerate(states):
            if transforms is None or transforms[i] is None:
                self.setState(state, self._dummy_simulation.context, transform=None, *args, **kwargs)
                energies += [self._dummy_simulation.integrator.getPotentialEnergyFromLambda(lambda_) / self.kT]
            else:
                # here we optimise by only loading the state once from the hard drive, if applicable
                if type(state) is int:
                    self.setState(state, self._dummy_simulation.context, *args, **kwargs)
                    state = self._dummy_simulation.context.getState(getPositions=True)
                for t in transforms[i]:
                    self.setState(state, self._dummy_simulation.context, transform=t, *args, **kwargs)
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
            The path to the trajectory file containing the relevant frame, if applicable. Default is
            current_trajectory_filename.
        transform :
            Optionally generate a transform dynamically from a format, specific to the underlying moves.
        """
        if reporter_filename is None:
            reporter_filename = self.current_trajectory_filename
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

    def equilibrate(self,
                    equilibration_steps=100000,
                    restrain_backbone=True,
                    restrain_resnames=None,
                    restrain_alchemical_atoms=False,
                    force_constant=5.0 * _unit.kilocalories_per_mole / _unit.angstroms ** 2,
                    output_interval=100000):
        """
        Equilibrates the system at lambda = 0.

        Parameters
        ----------
        equilibration_steps : int
            The number of equilibration steps.
        restrain_backbone : bool
            Whether to restrain all atoms with the following names: 'CA', 'C', 'N'.
        restrain_resnames : list
            A list of residue names to restrain. Default is: ["UNL", "LIG"]
        restrain_alchemical_atoms : bool, None
            True restrains all alchemical atoms, False removes restraints from all alchemical atoms and None has no
            effect on the restraints.
        force_constant : openmm.unit.Quantity
            The magnitude of the restraint force constant.
        output_interval : int
            How often to output to a trajectory file.
        """
        if restrain_resnames is None:
            restrain_resnames = ["LIG", "UNL"]

        # set up a reporter, if applicable
        if self.trajectory_reporter is not None:
            output_interval = min(output_interval, equilibration_steps)
            self.simulation.reporters.append(self.trajectory_reporter.generateReporter("equil", output_interval,
                                                                                       append=True))

        # add restraints, if applicable
        force = _openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", force_constant)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        atoms = set()
        if restrain_backbone:
            atoms |= {i for i, a in enumerate(self.structure.atoms) if a.name in ('CA', 'C', 'N')}
        if restrain_resnames:
            atoms |= {a.idx for r in self.structure.residues for a in r if r.name in restrain_resnames}
        if restrain_alchemical_atoms is False:
            atoms -= set(self.alchemical_atoms)
        if restrain_alchemical_atoms is True:
            atoms |= set(self.alchemical_atoms)
        counter = len(atoms)

        for i in atoms:
            force.addParticle(i, self.structure.positions[i].value_in_unit(_unit.nanometers))
        if counter:
            _logger.info("Adding {} equilibration restraints...".format(counter))
            force_idx = self.alch_system.addForce(force)

        # run the equilibration
        _logger.info("Running initial equilibration...")
        self.simulation.integrator.setGlobalVariableByName("lambda", 0.)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        self.simulation.step(equilibration_steps)

        # remove the restraints, if applicable
        if counter:
            _logger.info("Removing {} equilibration restraints...".format(counter))
            self.alch_system.removeForce(force_idx)

        # reset the reporter, if applicable
        if self.trajectory_reporter:
            del self.simulation.reporters[-1]
            self.trajectory_reporter.prune()

    def initialise(self, n_walkers):
        if not self.initialised:
            self.state_tree = _anytree.Node(0, lambda_=None, iteration=None, transform=None)
            if not len(self.states):
                self.states = [self.simulation.context.getState(getPositions=True, getEnergy=True)]
            equilibration_layer = [_anytree.Node(i, parent=self.state_tree, lambda_=None, iteration=self.iteration,
                                                 transform=None) for i in range(len(self.states))]
            if len(self.states) < n_walkers:
                indices = ([i for i in range(len(self.states))] * (n_walkers // len(self.states) + 1))[:n_walkers]
            else:
                indices = [i for i in range(len(self.states))]
            self.states = [self.states[i] for i in indices]
            self._all_tree_nodes = set(equilibration_layer)
            self._all_tree_nodes |= {_anytree.Node(i_new, parent=equilibration_layer[i_old], lambda_=self.lambda_,
                                                   iteration=self.iteration, transform=False)
                                     for i_new, i_old in enumerate(indices)}
            self.transforms = [None] * n_walkers
            self.log_weights = _np.log([1 / n_walkers] * n_walkers)
        self.initialised = True

    def sample(self,
               default_decorrelation_steps=500,
               keep_walkers_in_memory=False,
               write_checkpoint=None,
               load_checkpoint=None):
        if not keep_walkers_in_memory and self.trajectory_reporter is None:
            raise ValueError("Need to set a reporter if trajectory is not kept in memory.")
        if write_checkpoint and self.trajectory_reporter is None:
            raise ValueError("Need to set a reporter when storing a checkpoint.")
        default_decorrelation_steps = max(1, default_decorrelation_steps)

        # set up a reporter, if applicable
        if self.trajectory_reporter is not None:
            append = True if load_checkpoint else False
            self.simulation.reporters.append(self.trajectory_reporter.generateReporter(
                round(self.lambda_, 8), default_decorrelation_steps, append=append))
        if self.state_data_reporters is not None:
            self.simulation.reporters += self.state_data_reporters

        # load checkpoint, if applicable
        if load_checkpoint is not None:
            _logger.info("Loading instant checkpoint...")
            data = _pickle.load(open(load_checkpoint, "rb"))
            n = 0
            attrs = ["n"]
            if all(attr in data.keys() for attr in attrs):
                for attr in attrs:
                    exec("{0} = data.{0}".format(attr))
            generator = ((i, (i, self.transforms[i])) for i in range(n + 1, len(self.states)))
        else:
            generator = enumerate(zip(self.states, self.transforms))

        for n, (state, transform) in generator:
            # update the state data reporters, if applicable
            if self.state_data_reporters is not None:
                for r in self.state_data_reporters:
                    r.update(self, n)
            # sample
            self.setState(state, self.simulation.context, reporter_filename=self.previous_trajectory_filename,
                          transform=transform)
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
            self.simulation.step(default_decorrelation_steps)

            # update states
            if keep_walkers_in_memory:
                self.states[n] = self.simulation.context.getState(getPositions=True, getEnergy=True)
            else:
                self.states[n] = n
            if self.transforms is not None:
                self.transforms[n] = None

            # update statistics
            self.total_sampling_steps += default_decorrelation_steps

            # write checkpoint, if applicable
            if write_checkpoint is not None:
                data = {attr: locals()[attr] for attr in ["n"]}
                self.writeCheckpoint(data, filename=write_checkpoint, update=True)

        # reset the reporter
        self.simulation.reporters = [x for x in self.simulation.reporters if x not in self.reporters]
        if self.trajectory_reporter:
            self.trajectory_reporter.prune()

    def generateTransforms(self,
                           n_transforms_per_walker=100,
                           generate_transforms=None,
                           dynamically_generate_transforms=True):
        if generate_transforms or (generate_transforms is None and self.lambda_ == 0):
            _logger.info("Generating {} total transforms...".format(len(self.states) * n_transforms_per_walker))
            current_layer = self.tree_layer()
            new_layer = set()
            new_states = []
            new_transforms = []

            for n, state in enumerate(self.states):
                self.setState(state, self.simulation.context)
                transforms = self.moves.generateMoves(n_transforms_per_walker)
                new_layer |= {_anytree.Node(i, parent=current_layer[n], lambda_=self.lambda_,
                                            iteration=self.iteration, transform=True)
                              for i in range(n * n_transforms_per_walker, (n + 1) * n_transforms_per_walker)}

                if not dynamically_generate_transforms:
                    state = self.simulation.context.getState(getPositions=True)
                    for t in transforms:
                        self.moves.applyMove(self.simulation.context, t)
                        new_states += [self.simulation.context.getState(getPositions=True)]
                        self.simulation.context.setState(state)
                        new_transforms += [None]
                else:
                    new_states += [state]
                    new_transforms += [transforms]

            self.log_weights = self.log_weights[
                sum([[i] * n_transforms_per_walker for i in range(len(self.states))], [])]
            self.log_weights -= _logsumexp(self.log_weights)
            self._all_tree_nodes |= new_layer
            self.states = new_states
            self.transforms = new_transforms
        else:
            self.transforms = [None] * len(self.states)

    def reweight(self,
                 resampling_metric=_resmetrics.WorstCaseSampleSize,
                 target_metric_value=None,
                 target_metric_tol=None,
                 maximum_metric_evaluations=20,
                 default_dlambda=0.1,
                 minimum_dlambda=None,
                 maximum_dlambda=None,
                 target_lambda=1,
                 change_lambda=True):
        if self.lambda_ == target_lambda:
            return

        self._current_reduced_potentials = self.calculateStateEnergies(self.lambda_, transforms=self.transforms)
        self._current_deltaEs = {}
        self._current_weights = {}

        # this is the function we are going to minimise
        def evaluateWeights(lambda_):
            new_lambda = float(max(min(1., lambda_), 0.))
            self._new_reduced_potentials = self.calculateStateEnergies(new_lambda, transforms=self.transforms)
            self._current_deltaEs[new_lambda] = self._new_reduced_potentials - self._current_reduced_potentials
            self._current_weights[new_lambda] = _np.exp(
                _np.nanmin(self._current_deltaEs[new_lambda]) - self._current_deltaEs[new_lambda])
            self._current_weights[new_lambda][
                self._current_weights[new_lambda] != self._current_weights[new_lambda]] = 0
            self._current_weights[new_lambda] /= sum(self._current_weights[new_lambda])
            if resampling_metric is not None:
                val = resampling_metric.evaluate(self._current_weights[new_lambda])
                _logger.debug("Resampling metric {:.8g} at next lambda {:.8g}".format(val, new_lambda))
                return val

        # change direction, if needed
        sgn = _np.sign(target_lambda - self.lambda_)
        if default_dlambda is not None:
            default_dlambda = sgn * abs(default_dlambda)
        if minimum_dlambda is not None:
            minimum_dlambda = sgn * abs(minimum_dlambda)
            minimum_lambda = self.lambda_ + minimum_dlambda
        else:
            minimum_lambda = self.lambda_
        if maximum_dlambda is not None:
            target_lambda = self.lambda_ + sgn * abs(maximum_dlambda)

        # evaluate next lambda value
        if resampling_metric is not None:
            if target_metric_value is None:
                target_metric_value = resampling_metric.defaultValue()
            if target_metric_tol is None:
                target_metric_tol = resampling_metric.defaultTol()

            # minimise and set optimal lambda value adaptively if possible
            length = sum(1 if x is None else len(x) for x in self.transforms)
            current_y = resampling_metric.evaluate([1 / length] * length)
            initial_guess_x = None if default_dlambda is None else self.lambda_ + default_dlambda
            next_lambda_ = _BisectingMinimiser.minimise(evaluateWeights, target_metric_value, self.lambda_,
                                                        target_lambda, minimum_x=minimum_lambda,
                                                        initial_guess_x=initial_guess_x, current_y=current_y,
                                                        tol=target_metric_tol, maxfun=maximum_metric_evaluations)
            _logger.debug("Tentative next lambda: {:.8g}".format(next_lambda_))
        else:
            # else use default_dlambda
            next_lambda_ = max(min(1., self.lambda_ + default_dlambda), 0.)
            evaluateWeights(next_lambda_)

        del self._current_reduced_potentials, self._current_weights
        if change_lambda:
            # update histories, lambdas, and partition functions
            self.lambda_ = next_lambda_

        return next_lambda_

    def resample(self,
                 n_walkers=1000,
                 resampling_method=_resmethods.SystematicResampler,
                 exact_weights=False,
                 weights=None,
                 states=None,
                 transforms=None,
                 change_states=True):
        # prepare for resampling
        if weights is None:
            weights = _np.exp(self.log_weights)
        if states is None:
            states = self.states
        if transforms is None:
            transforms = self.transforms

        indices = []
        for i in range(len(self.states)):
            if self.transforms[i] is None:
                indices += [(i, None)]
            else:
                indices += [(i, j) for j in range(len(self.transforms[i]))]

        if resampling_method is not None:
            # resample
            resampled_states = resampling_method.resample(indices, weights, n_walkers=n_walkers)[0]
            _random.shuffle(resampled_states)
            states = [states[x[0]] for x in resampled_states]
            transforms = [None if x[1] is None else transforms[x[0]][x[1]] for x in resampled_states]
            weights = weights[[indices.index(x) for x in resampled_states]]

            # update the weights
            log_weights = _np.log(weights)
            if exact_weights:
                counts = _Counter(resampled_states)
                n_resampled = [counts[x] for x in resampled_states]
                log_weights -= _np.log(n_resampled)
            else:
                log_weights -= log_weights
            log_weights -= _logsumexp(log_weights)
            weights = _np.exp(log_weights)
        else:
            resampled_states = indices
            log_weights = _np.log(weights)

        # update the object, if applicable
        if change_states:
            old_state_indices = [indices.index(x) for x in resampled_states]
            current_iteraton = self.iteration if self.lambda_ else self.iteration - 1
            current_nodes = self.tree_layer(lambda_=self.lambda_history[-2], iteration=current_iteraton)
            self._all_tree_nodes |= {_anytree.Node(i_new, parent=current_nodes[i_old], lambda_=self.lambda_,
                                                   transform=False, iteration=current_iteraton)
                                     for i_new, i_old in enumerate(old_state_indices)}
            self.log_weights = log_weights
            self.states = states
            self.transforms = transforms

        return weights, states, transforms, resampled_states

    def runSingleIteration(self,
                           sampling_metric=_sammetrics.EnergyCorrelation,
                           resampling_method=_resmethods.SystematicResampler,
                           resampling_metric=_resmetrics.WorstCaseSampleSize,
                           exact_weights=False,
                           target_metric_value=None,
                           target_metric_tol=None,
                           maximum_metric_evaluations=20,
                           default_dlambda=0.1,
                           minimum_dlambda=None,
                           maximum_dlambda=None,
                           target_lambda=1,
                           default_decorrelation_steps=500,
                           maximum_decorrelation_steps=5000,
                           n_walkers=1000,
                           generate_transforms=None,
                           n_transforms_per_walker=100,
                           dynamically_generate_transforms=True,
                           keep_walkers_in_memory=False,
                           write_checkpoint=None,
                           load_checkpoint=None):
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
        exact_weights : bool
            Whether to change the weights based on the actual resampling, instead of the intended one.
        target_metric_value : float
            The threshold for the resampling metric. None uses the default value given by the class.
        target_metric_tol : float
            The relative tolerance for the resampling metric. None uses the default value given by the class.
        maximum_metric_evaluations : int
            The maximum number of energy evaluations to determine the resampling metric.
        default_dlambda : float
            Determines the next lambda value. Only used if resampling_metric is None.
        minimum_dlambda : float
            The minimum allowed change in lambda.
        maximum_dlambda : float
            The maximum allowed change in lambda.
        target_lambda : float
            What the final intended lambda value is.
        default_decorrelation_steps : int
            The default number of decorrelation steps. If sampling metric is None, these denote the true number of
            decorrelation steps.
        maximum_decorrelation_steps : int
            The maximum number of decorrelation steps. Only used with adaptive sampling.
        n_walkers : int
            The number of walkers to be resampled.
        generate_transforms : bool, None
            None generates transforms from self.moves only at lambda = 0, True generates them regardless of lambda and
            False doesn't generate any transforms.
        n_transforms_per_walker : int
            How many transforms to generate for each walker. Only used if generate_transforms is not False or None.
        dynamically_generate_transforms : bool
            Whether to store the extra transforms as states or as transformations. The former is much faster, but also
            extremely memory-intensive. Only set to False if you are certain that you have enough memory.
        keep_walkers_in_memory : bool
            Whether to keep the walkers as states or load them dynamically from the hard drive. The former is much
            faster during the energy evaluation step but is more memory-intensive. Only set to True if you are certain
            that you have enough memory.
        write_checkpoint : str
            Describes a path to a checkpoint file written after completing the whole SMC iteration. None means no such
            file will be written.
        load_checkpoint : str
            Describes a path to a checkpoint file written with write_checkpoint. In order to use this, the same
            arguments must be passed to this function as when the checkpoint file was written. This option also
            typically requires that the GenericSMCSampler was instantiated with the same checkpoint file. None
            means that no checkpoint will be read.
        """
        if _inspect.isclass(sampling_metric):
            sampling_metric = sampling_metric(self)

        reweight_kwargs = dict(
            resampling_metric=resampling_metric,
            target_metric_value=target_metric_value,
            target_metric_tol=target_metric_tol,
            maximum_metric_evaluations=maximum_metric_evaluations,
            default_dlambda=default_dlambda,
            minimum_dlambda=minimum_dlambda,
            maximum_dlambda=maximum_dlambda,
            target_lambda=target_lambda
        )
        self.initialise(n_walkers)

        if self.lambda_ and sampling_metric:
            sampling_metric.evaluateBefore()
        elapsed_steps = 0
        while True:
            # load checkpoint, if applicable
            skip_sampling = False
            if load_checkpoint is not None:
                _logger.info("Loading instant checkpoint...")
                data = _pickle.load(open(load_checkpoint, "rb"))
                attrs = ["elapsed_steps"]
                if all(attr in data.keys() for attr in attrs):
                    skip_sampling = True
                    for attr in attrs:
                        exec("{0} = data.{0}".format(attr))

            # sample
            if not skip_sampling:
                self.sample(
                    default_decorrelation_steps=default_decorrelation_steps,
                    keep_walkers_in_memory=keep_walkers_in_memory,
                    write_checkpoint=write_checkpoint,
                    load_checkpoint=load_checkpoint
                )
                elapsed_steps += default_decorrelation_steps

            # write checkpoint, if applicable
            if write_checkpoint is not None:
                data = {attr: locals()[attr] for attr in ["elapsed_steps"]}
                self.writeCheckpoint(data, filename=write_checkpoint, update=False)
                load_checkpoint = None

            # generate transforms
            self.generateTransforms(
                n_transforms_per_walker=n_transforms_per_walker,
                generate_transforms=generate_transforms,
                dynamically_generate_transforms=dynamically_generate_transforms
            )

            # return if final decorrelation
            if self.lambda_ == target_lambda:
                return

            # evaluate sampling metric
            next_lambda = None
            if self.lambda_ and sampling_metric:
                if sampling_metric.requireNextLambda:
                    next_lambda = self.reweight(**reweight_kwargs, change_lambda=False)
                    sampling_metric.evaluateAfter(next_lambda)
                    _logger.debug("Sampling metric {:.8g} at next lambda {:.8g}".format(sampling_metric.metric,
                                                                                        next_lambda))
                else:
                    sampling_metric.evaluateAfter()
                    _logger.debug("Sampling metric {:.8g} ".format(sampling_metric.metric))
                if not sampling_metric.terminateSampling and elapsed_steps < maximum_decorrelation_steps:
                    continue
                sampling_metric.reset()

            # reweight if needed and change lambda
            if next_lambda is None:
                self.reweight(**reweight_kwargs)
            else:
                self.lambda_ = next_lambda
            break

        # resample
        self.resample(
            n_walkers=n_walkers,
            resampling_method=resampling_method,
            exact_weights=exact_weights
        )

        # dump info to logger
        _logger.info("Sampling at lambda = {:.8g} terminated after {} steps per walker".format(self.lambda_history[-2],
                                                                                               elapsed_steps))
        _logger.info("Trajectory path: \"{}\"".format(self.current_trajectory_filename))
        _logger.info("Current accumulated logZ: {:.8g}".format(self.logZ))
        _logger.info("Next lambda: {:.8g}".format(self.lambda_))

    def run(self,
            *args,
            n_equilibrations=1,
            equilibration_steps=100000,
            restrain_backbone=True,
            restrain_resnames=None,
            restrain_alchemical_atoms=False,
            force_constant=5.0 * _unit.kilocalories_per_mole / _unit.angstroms ** 2,
            output_interval=100000,
            final_decorrelation_step=True,
            **kwargs):
        """
        Performs a complete sequential Monte Carlo run until lambda = 1.

        Parameters
        ----------
        n_equilibrations : int
            The number of equilibrations.
        equilibration_steps : int
            The number of equilibration steps per equilibration.
        restrain_backbone : bool
            Whether to restrain all atoms with the following names: 'CA', 'C', 'N'.
        restrain_resnames : list
            A list of residue names to restrain. Default is: ["UNL", "LIG"]
        restrain_alchemical_atoms : bool, None
            True restrains all alchemical atoms, False removes restraints from all alchemical atoms and None has no
            effect on the restraints.
        force_constant : openmm.unit.Quantity
            The magnitude of the restraint force constant.
        output_interval : int
            How often to output to a trajectory file.
        final_decorrelation_step : bool
            Whether to decorrelate the final resampled walkers for another number of default_decorrelation_steps.
        args
            Positional arguments to be passed to runSingleIteration().
        kwargs
            Keyword arguments to be passed to runSingleIteration().
        """
        def runOnce():
            self.runSingleIteration(*args, **kwargs)
            if "load_checkpoint" in kwargs.keys():
                kwargs.pop("load_checkpoint")

        if self.lambda_ == 0 and not self.initialised:
            for _ in range(n_equilibrations):
                old_state = self.simulation.context.getState(getPositions=True)
                self.equilibrate(equilibration_steps=equilibration_steps,
                                 restrain_backbone=restrain_backbone,
                                 restrain_resnames=restrain_resnames,
                                 restrain_alchemical_atoms=restrain_alchemical_atoms,
                                 force_constant=force_constant,
                                 output_interval=output_interval)
                self.states += [self.simulation.context.getState(getPositions=True, getEnergy=True)]
                self.setState(old_state, self.simulation.context)
            self.transforms = [None] * n_equilibrations
            self.log_weights = _np.log([1 / n_equilibrations] * n_equilibrations)

        while self.lambda_ < 1:
            runOnce()
        if final_decorrelation_step:
            runOnce()
        total_sampling_time = self.total_sampling_steps * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
        _logger.info("Total simulation time was {}".format(total_sampling_time))

    def generateAlchemicalRegion(self):
        """Makes sure that all rotated dihedrals are also made alchemical."""
        self._rotatable_bonds = [x.rotatable_bond for x in self.moves.moves if isinstance(x, _moves.DihedralMove)]
        all_rotatable_bonds = {frozenset(x) for x in self._rotatable_bonds}
        self._alchemical_dihedral_indices = [i for i, d in enumerate(self.structure.dihedrals) if not d.improper and
                                             {d.atom2.idx, d.atom3.idx} in all_rotatable_bonds]
        self._sample_torsions = []
        remaining_bonds = all_rotatable_bonds.copy()
        for d in self.structure.dihedrals:
            a1, a2, a3, a4 = d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx
            if frozenset([a2, a3]) in remaining_bonds:
                self._sample_torsions += [(a1, a2, a3, a4)]
                remaining_bonds -= {frozenset([a2, a3])}
            if remaining_bonds == set():
                break

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
    def generateSimFromStruct(cls, structure, system, integrator, platform=None, properties=None):
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

    @classmethod
    def addBarostat(cls, system, temperature=298 * _unit.kelvin, pressure=1 * _unit.atmospheres, frequency=25):
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
