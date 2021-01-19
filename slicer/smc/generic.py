from collections import Counter as _Counter
import copy as _copy
import inspect as _inspect
import logging as _logging
import math as _math
import os as _os
import pickle as _pickle
import random as _random
import warnings as _warnings

import mdtraj as _mdtraj
import numpy as _np
import openmmtools as _openmmtools
from scipy.special import logsumexp as _logsumexp
import simtk.openmm as _openmm
import simtk.openmm.app as _app
import simtk.unit as _unit

from .misc import Walker as _Walker
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
    lambda_history : dict
        A dictionary containing all past lambda values.
    reporters : [slicer.reporters.MultistateDCDReporter, slicer.reporters.MulsistateStateDataReporter]
        The reporter list containing all multistate reporters.
    walkers : [slicer.smc.misc.Walker]
        A list containing all current walkers.
    logZ : float
        The current estimate of the dimensionless free energy difference.
    """
    # TODO: make pickle work
    _picklable_attrs = ["total_sampling_steps", "_lambda_", "lambda_history", "walkers", "initialised", "walker_tree",
                        "_all_walkers", "reporters"]
    default_alchemical_functions = {
        'lambda_sterics': lambda x: min(1.25 * x, 1.),
        'lambda_electrostatics': lambda x: max(0., 5. * x - 4.),
        'lambda_torsions': lambda x: min(1.25 * x, 1.),
    }

    def __init__(self, coordinates, structure, integrator, moves, alchemical_functions=None, platform=None,
                 platform_properties=None, npt=True, checkpoint=None, md_config=None, alch_config=None):
        if md_config is None:
            md_config = {}
        if alch_config is None:
            alch_config = {}
        if alchemical_functions is None:
            alchemical_functions = {}
        self.alchemical_functions = {**self.default_alchemical_functions, **alchemical_functions}
        for func in self.alchemical_functions.values():
            assert func(0) == 0 and func(1) == 1, "All alchemical functions must go from 0 to 1"

        self.coordinates = coordinates
        self.moves = moves
        self.structure = structure
        self.system = GenericSMCSampler.generateSystem(self.structure, **md_config)
        self.generateAlchemicalRegion()
        if "alchemical_torsions" not in alch_config.keys():
            alch_config["alchemical_torsions"] = self._alchemical_dihedral_indices
        self.alch_system = self.generateAlchSystem(self.system, self.alchemical_atoms, **alch_config)
        if npt:
            self.alch_system = self.addBarostat(self.alch_system, temperature=integrator.getTemperature())

        # TODO: implement parallelism?
        self.platform = platform
        self.platform_properties = platform_properties
        self.integrator = _copy.copy(integrator)
        self.simulation = self.generateSimFromStruct(
            structure, self.alch_system, self.integrator, platform, platform_properties)

        self.initialised = False
        self.total_sampling_steps = 0
        self._lambda_ = 0.
        self.lambda_history = [0.]
        self.reporters = []
        self.walker_tree = _Walker(0, lambda_=None, iteration=None, transform=None)
        self._all_walkers = [self.walker_tree]
        self.walkers = []

        if checkpoint is not None:
            _logger.info("Loading checkpoint...")
            obj = _pickle.load(open(checkpoint, "rb"))["self"]
            for attr in self._picklable_attrs:
                try:
                    self.__setattr__(attr, getattr(obj, attr))
                    if attr == "_lambda_":
                        self._update_alchemical_lambdas(self.lambda_)
                except AttributeError:
                    _warnings.warn("There was missing or incompatible data from the checkpoint: {}".format(attr))

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
    def kT(self):
        """openmm.unit.Quantity: The current temperature multiplied by the gas constant."""
        try:
            kB = _unit.BOLTZMANN_CONSTANT_kB * _unit.AVOGADRO_CONSTANT_NA
            kT = kB * self.integrator.getTemperature()
            return kT
        except AttributeError:
            return None

    def iteration(self, lambda_=None):
        if lambda_ is None:
            lambda_ = self.lambda_
        # TODO: fix rounding
        return len([x for x in self.lambda_history if round(x, 8) == round(lambda_, 8)]) - 1

    @property
    def lambda_(self):
        return self._lambda_

    @lambda_.setter
    def lambda_(self, val):
        if val is not None:
            assert 0 <= val <= 1, "The lambda value must be between 0 and 1"
            current_weights = _np.asarray([0 if walker.logW is None else walker.logW for walker in self.walkers])

            # update deltaEs
            if val != self._lambda_:
                deltaEs = self.calculateDeltaEs(val)
                new_weights = current_weights - deltaEs
            else:
                new_weights = current_weights

            # update lambdas
            self._lambda_ = float(val)
            self.lambda_history += [self._lambda_]
            self._update_alchemical_lambdas(self._lambda_)

            # update walkers
            self.walkers = [_Walker(i,
                                    state=walker.state,
                                    transform=walker.transform,
                                    reporter_filename=walker.reporter_filename,
                                    frame=walker.frame,
                                    lambda_=self.lambda_,
                                    iteration=self.iteration(),
                                    logW=new_weights[i])
                            for i, (logW, walker) in enumerate(zip(new_weights, self.walkers))]

    @property
    def log_weights(self):
        relative_logW = _np.asarray([0 if walker.logW is None else walker.logW for walker in self.walkers])
        return relative_logW - _logsumexp(relative_logW)

    @property
    def logZ(self):
        return _logsumexp([0 if walker.logW is None else walker.logW for walker in self.walkers]) - _np.log(len(self.walkers))

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
        return len(self.walkers)

    @property
    def walkers(self):
        return self._walkers

    @walkers.setter
    def walkers(self, val):
        self._all_walkers = [x for x in self._all_walkers if x not in val] + val
        self._walkers = val
        for walker in self._all_walkers:
            if walker not in val and walker.state is not None:
                walker.state = None

    @property
    def temperature(self):
        """openmm.unit.Quantity: The temperature of the current integrator."""
        try:
            T = self.integrator.getTemperature()
            return T
        except AttributeError:
            return None

    @property
    def weights(self):
        return _np.exp(self.log_weights)

    def serialise(self):
        new_self = _copy.copy(self)
        new_self.__dict__ = {x: y for x, y in new_self.__dict__.items() if x in self._picklable_attrs}
        new_self.walkers = [i for i in range(len(self.walkers))]
        new_self.reporters = [new_self.trajectory_reporter]
        return new_self

    def writeCheckpoint(self, data, filename="checkpoint.pickle", update=True, *args, **kwargs):
        backups = {}
        if update:
            try:
                backups = _pickle.load(open(filename, "rb"))
            except (FileNotFoundError, _pickle.UnpicklingError, EOFError):
                pass
        backups.update(data)
        backups["self"] = self.serialise()
        _logger.debug("Writing checkpoint...")
        if _os.path.exists(filename):
            _os.rename(filename, filename + ".old")
        _pickle.dump(backups, open(filename, "wb"), *args, **kwargs)
        if _os.path.exists(filename + ".old"):
            _os.remove(filename + ".old")

    def calculateDeltaEs(self, lambda1=None, lambda0=None, walkers=None, **kwargs):
        if walkers is None:
            walkers = self.walkers
        old_potentials = self.calculateStateEnergies(lambda0, walkers=walkers, **kwargs)
        new_potentials = self.calculateStateEnergies(lambda1, walkers=walkers, **kwargs)
        return new_potentials - old_potentials

    def calculateStateEnergies(self, lambda_=None, walkers=None, **kwargs):
        """
        Calculates the reduced potential energies of all states for a given lambda value.

        Parameters
        ----------
        lambda_ : float or list
            The desired lambda value.
        walkers : [int] or [openmm.State] or None
            Which walkers need to be used. If None, self.walkers are used. Otherwise, these could be in any
            format supported by setState().
        kwargs
            Keyword arguments to be passed to setState().
        """
        if walkers is None:
            walkers = self.walkers
        if lambda_ is None:
            lambdas = _np.asarray([[walker.lambda_] for walker in walkers])
        else:
            lambda_ = _np.asarray(lambda_)
            lambdas = _np.full((len(walkers), lambda_.size), lambda_)

        energies = _np.zeros(lambdas.shape)

        # determine unique walkers for optimal loading from hard drive
        unique_walkers = {}
        for i, walker in enumerate(walkers):
            key = None
            if isinstance(walker, _Walker) and walker.state is None:
                key = (walker.reporter_filename, walker.frame)
                if any(x is None for x in key):
                    raise ValueError("Walkers need to contain either an OpenMM State or a valid trajectory path and "
                                     "frame number")
            if key not in unique_walkers.keys():
                unique_walkers[key] = []
            unique_walkers[key] += [i]

        for key, group in unique_walkers.items():
            # modify all walkers with a single read from the hard drive, if applicable
            if key is not None:
                previous_states = [walkers[i].state for i in group]
                dummy_state = _copy.copy(walkers[group[0]])
                dummy_state.transform = None
                self.setState(dummy_state)
                state = self.simulation.context.getState(getPositions=True)
                for i in group:
                    walkers[i].setStateKeepCache(state)

            # calculate the energies
            for i in group:
                walker = walkers[i]
                lambda_ = lambdas[i]

                state_already_set = False
                for j, value in enumerate(lambda_):
                    # get cached energy and skip energy evaluation, if applicable
                    if isinstance(walkers[i], _Walker):
                        energy = walkers[i].getCachedEnergy(value)
                        if energy is not None:
                            energies[i, j] = energy
                            continue

                    # calculate energy and set cache
                    if not state_already_set:
                        self.setState(walker, **kwargs)
                        state_already_set = True
                    self._update_alchemical_lambdas(value)
                    energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy() / self.kT
                    if isinstance(walker, _Walker):
                        walker.setCachedEnergy(value, energy)
                    energies[i, j] = energy

            # restore original walkers
            if key is not None:
                for i, previous_state in zip(group, previous_states):
                    walkers[i].setStateKeepCache(previous_state)

        self._update_alchemical_lambdas(self.lambda_)

        energies = energies.T
        if energies.shape[0] == 1:
            energies = energies[0]
        return energies

    def setState(self, state, context=None):
        """
        Sets a given state to the current context.

        Parameters
        ----------
        state : openmm.State, slicer.utils.Walker
            Sets the state from either an openmm.State object or a slicer.utils.Walker object.
        context : openmm.Context
            The context to which the state needs to be applied. Default is GenericSMCSampler.simulation.context.
        """
        if context is None:
            context = self.simulation.context
        if isinstance(state, _openmm.State):
            if self.initialised:
                _warnings.warn("Manually changing the state in an initialised SMC run can break functionality")
            context.setState(state)
        elif isinstance(state, _Walker):
            if state.state is not None:
                context.setState(state.state)
            else:
                frame = _mdtraj.load_frame(state.reporter_filename, state.frame, self.coordinates)
                positions = frame.xyz[0]
                periodic_box_vectors = frame.unitcell_vectors[0]
                context.setPositions(positions)
                context.setPeriodicBoxVectors(*periodic_box_vectors)
            if state.transform is not None:
                self.moves.applyMove(context, state.transform)
        else:
            raise TypeError("Unrecognised parameter type {}".format(type(state)))

    def _update_alchemical_lambdas(self, lambda_):
        valid_parameters = [x for x in self.simulation.context.getParameters()]
        for param, func in self.alchemical_functions.items():
            if param in valid_parameters:
                val = float(func(lambda_))
                assert 0 <= val <= 1, "All lambda functions must evaluate between 0 and 1"
                self.simulation.context.setParameter(param, val)

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
        self._update_alchemical_lambdas(0.)
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
            # root layer
            self.walker_tree = _Walker(0)
            if not len(self.walkers):
                state = self.simulation.context.getState(getPositions=True, getEnergy=True)
                self.walkers = [_Walker(0, state=state)]
            # lambda = 0 layer
            self.walkers = [_Walker(i,
                                    state=self.walkers[i % len(self.walkers)].state,
                                    reporter_filename=self.walkers[i % len(self.walkers)].reporter_filename,
                                    frame=self.walkers[i % len(self.walkers)].frame,
                                    lambda_=self.lambda_,
                                    iteration=self.iteration(),
                                    logW=0)
                            for i in range(max(n_walkers, len(self.walkers)))]
        self.initialised = True

    def sample(self,
               default_decorrelation_steps=500,
               keep_walkers_in_memory=False,
               write_checkpoint=None,
               load_checkpoint=None,
               reporter_filename=None,
               append=False):
        if not keep_walkers_in_memory and self.trajectory_reporter is None:
            raise ValueError("Need to set a reporter if trajectory is not kept in memory.")
        if write_checkpoint and self.trajectory_reporter is None:
            raise ValueError("Need to set a reporter when storing a checkpoint.")

        initial_frame = 0

        # set up a reporter, if applicable
        if self.trajectory_reporter is not None:
            append = True if load_checkpoint else append
            label = reporter_filename if reporter_filename is not None else str(round(self.lambda_, 8))
            self.simulation.reporters.append(self.trajectory_reporter.generateReporter(
                label, default_decorrelation_steps, append=append))
            if load_checkpoint:
                self.trajectory_reporter.prune()
            if append:
                duplicates = [w for w in self._all_walkers if w.reporter_filename == self.current_trajectory_filename]
                initial_frame = max([w.frame for w in duplicates]) + 1 if len(duplicates) else 0
        if self.state_data_reporters is not None:
            self.simulation.reporters += self.state_data_reporters

        # load checkpoint, if applicable
        if load_checkpoint is not None:
            # TODO take care of reporters and state updates
            _logger.info("Loading instant checkpoint...")
            data = _pickle.load(open(load_checkpoint, "rb"))
            n = 0 if "n" not in data.keys() else data["n"]
            generator = ((i, self.walkers[i]) for i in range(n + 1, len(self.walkers)))
        else:
            generator = enumerate(self.walkers)

        for walker in self.walkers:
            if walker.reporter_filename == self.current_trajectory_filename:
                walker.reporter_filename = self.previous_trajectory_filename

        for n, walker in generator:
            # update the state data reporters, if applicable
            if self.state_data_reporters is not None:
                for r in self.state_data_reporters:
                    r.update(self, n)
            # sample
            self.setState(walker)
            self.simulation.context.setVelocitiesToTemperature(self.temperature)
            self.simulation.step(default_decorrelation_steps)

            # update walkers
            if keep_walkers_in_memory:
                walker.state = self.simulation.context.getState(getPositions=True, getEnergy=True)
            else:
                walker.state = None
            walker.transform = None
            walker.reporter_filename = self.current_trajectory_filename
            walker.frame = n + initial_frame if walker.reporter_filename is not None else None

            # update statistics
            self.total_sampling_steps += default_decorrelation_steps

            # write checkpoint, if applicable
            if write_checkpoint is not None:
                self.writeCheckpoint({"n": n}, filename=write_checkpoint, update=True)

        # reset the reporter
        self.simulation.reporters = [x for x in self.simulation.reporters if x not in self.reporters]
        if self.trajectory_reporter is not None:
            self.simulation.reporters.remove(self.trajectory_reporter.current_reporter)
            self.trajectory_reporter.prune()

    def generateTransforms(self,
                           n_transforms_per_walker=100,
                           generate_transforms=None):
        if generate_transforms or (generate_transforms is None and self.lambda_ == 0):
            _logger.info("Generating {} total transforms...".format(len(self.walkers) * n_transforms_per_walker))
            new_walkers = []

            i = 0
            for walker in self.walkers:
                self.setState(walker)
                transforms = self.moves.generateMoves(n_transforms_per_walker)
                for t in transforms:
                    walker_new = _Walker(i, state=walker.state, lambda_=self.lambda_, iteration=self.iteration(),
                                         transform=t, reporter_filename=walker.reporter_filename,
                                         frame=walker.frame, logW=walker.logW)
                    new_walkers += [walker_new]
                    i += 1

            self.walkers = new_walkers

    def reweight(self,
                 resampling_metric=_resmetrics.WorstCaseSampleSize,
                 target_metric_value=None,
                 target_metric_tol=None,
                 maximum_metric_evaluations=20,
                 default_dlambda=0.1,
                 minimum_dlambda=None,
                 maximum_dlambda=None,
                 target_lambda=1,
                 fixed_lambdas=None,
                 change_lambda=True):
        if self.lambda_ == target_lambda:
            return

        if fixed_lambdas is None:
            fixed_lambdas = []

        if _inspect.isclass(resampling_metric):
            resampling_metric = resampling_metric(self)

        # this is the function we are going to minimise
        def evaluateWeights(lambda_):
            new_lambda = float(max(min(1., lambda_), 0.))
            deltaEs = self.calculateDeltaEs(new_lambda)
            weights = _np.exp(_np.nanmin(deltaEs) - deltaEs)
            weights[weights != weights] = 0
            weights /= _np.sum(weights)
            if resampling_metric is not None:
                val = resampling_metric(weights)
                _logger.debug("Resampling metric {:.8g} at next lambda {:.8g}".format(val, new_lambda))
                return val

        # change direction, if needed
        sgn = int(_np.sign(target_lambda - self.lambda_))
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
                target_metric_value = resampling_metric.defaultValue
            if target_metric_tol is None:
                target_metric_tol = resampling_metric.defaultTol

            # minimise and set optimal lambda value adaptively if possible
            current_y = resampling_metric([1 / len(self.walkers)] * len(self.walkers))
            initial_guess_x = None if default_dlambda is None else self.lambda_ + default_dlambda
            next_lambda_ = _BisectingMinimiser.minimise(evaluateWeights, target_metric_value, self.lambda_,
                                                        target_lambda, minimum_x=minimum_lambda,
                                                        initial_guess_x=initial_guess_x, current_y=current_y,
                                                        tol=target_metric_tol, maxfun=maximum_metric_evaluations)
            fixed_lambdas_filtered = [x for x in sorted(fixed_lambdas)
                                      if sgn * self.lambda_ < sgn * x < sgn * next_lambda_]
            if len(fixed_lambdas_filtered):
                next_lambda_ = fixed_lambdas_filtered[-1 * sgn]
            _logger.debug("Tentative next lambda: {:.8g}".format(next_lambda_))
        else:
            # else use default_dlambda
            next_lambda_ = max(min(1., self.lambda_ + default_dlambda), 0.)

        if change_lambda:
            # update histories, lambdas, and partition functions
            self.lambda_ = next_lambda_

        return next_lambda_

    def resample(self,
                 n_walkers=1000,
                 resampling_method=_resmethods.SystematicResampler,
                 exact_weights=False,
                 walkers=None,
                 change_walkers=True):
        # prepare for resampling
        if walkers is None:
            walkers = self.walkers
        new_walkers = walkers

        if resampling_method is not None:
            # get weights
            log_relative_weights = _np.asarray([0 if walker.logW is None else walker.logW for walker in walkers])
            weights = _np.exp(log_relative_weights - _logsumexp(log_relative_weights))

            # resample
            indices = [i for i in range(len(walkers))]
            resampled_indices = resampling_method.resample(indices, weights, n_walkers=n_walkers)[0]
            _random.shuffle(resampled_indices)

            logWs = _np.repeat(_logsumexp(log_relative_weights) - _np.log(len(walkers)), n_walkers)
            if exact_weights:
                counts = _Counter(resampled_indices)
                n_resampled = _np.asarray([counts[x] for x in resampled_indices], dtype=_np.float32)
                log_weights = _np.log(weights[resampled_indices]) - _np.log(n_resampled / _np.sum(n_resampled))
                logWs += log_weights - _logsumexp(log_weights) + _np.log(log_weights.shape)

            new_walkers = [_Walker(i_new,
                                   state=walkers[i_old].state,
                                   transform=walkers[i_old].transform,
                                   reporter_filename=walkers[i_old].reporter_filename,
                                   frame=walkers[i_old].frame,
                                   lambda_=self.lambda_,
                                   iteration=self.iteration(),
                                   logW=logWs[i_new])
                           for i_new, i_old in enumerate(resampled_indices)]

            # update the object, if applicable
            if change_walkers:
                self._all_walkers = [x for x in self._all_walkers if x not in walkers]
                self.walkers = new_walkers

        return new_walkers

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
                           fixed_lambdas=None,
                           default_decorrelation_steps=500,
                           maximum_decorrelation_steps=5000,
                           n_walkers=1000,
                           generate_transforms=None,
                           n_transforms_per_walker=100,
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
            target_lambda=target_lambda,
            fixed_lambdas=fixed_lambdas
        )
        self.initialise(n_walkers)

        if sampling_metric is not None and not load_checkpoint:
            sampling_metric.evaluateBefore()
        elapsed_steps = 0
        while True:
            # load checkpoint, if applicable
            skip_sampling = False
            if load_checkpoint is not None:
                _logger.info("Loading instant checkpoint...")
                data = _pickle.load(open(load_checkpoint, "rb"))
                # continue the simulation from the relevant iteration
                if "elapsed_steps" in data.keys():
                    elapsed_steps = data["elapsed_steps"]
                    skip_sampling = True
                if "sampling_metric" in data.keys() and data["sampling_metric"] is not None:
                    sampling_metric.__dict__.update(data["sampling_metric"])
                # continue the iteration from a particular walker
                if "n" in data.keys():
                    skip_sampling = False
                    # write a checkpoint so that it can get updated by the sampler
                    if write_checkpoint is not None:
                        self.writeCheckpoint(data, filename=write_checkpoint, update=False)

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
                sampling_backup = sampling_metric.serialise() if sampling_metric is not None else None
                data = {"self": self.serialise(), "elapsed_steps": elapsed_steps, "sampling_metric": sampling_backup}
                self.writeCheckpoint(data, filename=write_checkpoint, update=False)
                load_checkpoint = None

            # evaluate sampling metric
            next_lambda = None
            if sampling_metric is not None:
                if sampling_metric.requireNextLambda:
                    next_lambda = self.reweight(**reweight_kwargs, change_lambda=False)
                    sampling_metric.evaluateAfter(next_lambda)
                    _logger.debug("Sampling metric {:.8g} at next lambda {:.8g}".format(sampling_metric.metric,
                                                                                        next_lambda))
                else:
                    sampling_metric.evaluateAfter()
                    _logger.debug("Sampling metric {:.8g}".format(sampling_metric.metric))
                if not sampling_metric.terminateSampling and elapsed_steps < maximum_decorrelation_steps:
                    continue
                sampling_metric.reset()

            # generate transforms
            self.generateTransforms(
                n_transforms_per_walker=n_transforms_per_walker,
                generate_transforms=generate_transforms,
            )

            # reweight if needed and change lambda
            if self.lambda_ != target_lambda:
                if next_lambda is None:
                    self.reweight(**reweight_kwargs)
                else:
                    self.lambda_ = next_lambda
            break

        if self.lambda_ == target_lambda:
            return

        # resample
        self.resample(
            n_walkers=n_walkers,
            resampling_method=resampling_method,
            exact_weights=exact_weights
        )

        # write checkpoint, if applicable
        if write_checkpoint:
            self.writeCheckpoint({"self": self.serialise()}, filename=write_checkpoint, update=False)

        # dump info to logger
        _logger.info("Sampling at lambda = {:.8g} terminated after {} steps per walker".format(self.lambda_history[-2],
                                                                                               elapsed_steps))
        if self.current_trajectory_filename is not None:
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
            target_lambda=1,
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
        restrain_resnames : list, False
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
        target_lambda : float
            What the final intended lambda value is.
        args
            Positional arguments to be passed to runSingleIteration().
        kwargs
            Keyword arguments to be passed to runSingleIteration().
        """
        kwargs["target_lambda"] = target_lambda
        direction = _np.sign(target_lambda - self.lambda_)

        def runOnce():
            self.runSingleIteration(*args, **kwargs)
            if "load_checkpoint" in kwargs.keys():
                kwargs.pop("load_checkpoint")

        # TODO: fix equilibration
        if self.lambda_ == 0 and not self.initialised:
            if output_interval:
                assert equilibration_steps % output_interval == 0, "The equilibration steps must be a multiple of " \
                                                                   "the output interval"
            frame_step = equilibration_steps // output_interval if output_interval else None
            self.walkers = []
            for i in range(n_equilibrations):
                old_state = self.simulation.context.getState(getPositions=True)
                self.equilibrate(equilibration_steps=equilibration_steps,
                                 restrain_backbone=restrain_backbone,
                                 restrain_resnames=restrain_resnames,
                                 restrain_alchemical_atoms=restrain_alchemical_atoms,
                                 force_constant=force_constant,
                                 output_interval=output_interval)
                self.walkers += [_Walker(i,
                                         state=self.simulation.context.getState(getPositions=True, getEnergy=True),
                                         reporter_filename=self.current_trajectory_filename,
                                         frame=(i + 1) * frame_step - 1) if frame_step is not None else None]
                self.setState(old_state, self.simulation.context)

        initial_sampling_steps = self.total_sampling_steps
        while _np.sign(target_lambda - self.lambda_) == direction != 0:
            runOnce()
        if final_decorrelation_step:
            runOnce()
        sampling_step_difference = self.total_sampling_steps - initial_sampling_steps
        total_sampling_time = sampling_step_difference * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
        _logger.info("The SMC run cycle finished in {}".format(total_sampling_time))

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
                           alchemical_factory=_openmmtools.alchemy.AbsoluteAlchemicalFactory,
                           softcore_alpha=0.5,
                           softcore_a=1,
                           softcore_b=1,
                           softcore_c=6,
                           softcore_beta=0,
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
        factory = alchemical_factory(
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
    def generateSimFromStruct(structure, system, integrator, platform=None, properties=None):
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
