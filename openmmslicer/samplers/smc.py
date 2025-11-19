from collections import Counter as _Counter
import copy as _copy
import inspect as _inspect
import logging as _logging
import os as _os
import random as _random
import threading as _threading
import warnings as _warnings

import dill as _dill
import numpy as _np
from scipy.special import logsumexp as _logsumexp
import openmm as _openmm
import openmm.unit as _unit

from .misc import Walker as _Walker, WalkerMemo as _WalkerMemo
from openmmslicer.minimise import BisectingMinimiser as _BisectingMinimiser
import openmmslicer.alchemy as _alchemy
from openmmslicer.misc import quantity_round as _quantity_round
import openmmslicer.moves as _moves
import openmmslicer.reporters as _reporters
import openmmslicer.resampling_metrics as _resmetrics
import openmmslicer.resampling_methods as _resmethods
import openmmslicer.sampling_metrics as _sammetrics

_logger = _logging.getLogger(__name__)


class SMCSampler:
    """
    A generic sequential Monte Carlo sampler which can enhance the sampling of certain degrees of freedom.

    Parameters
    ----------
    alchemical_chain : openmmslicer.alchemy.AlchemicalChain
        The alchemical chain to be explored by the SMCSampler.
    integrator : openmm.Integrator
        The integrator used at each sampling step.
    alchemical_functions : dict, optional
        A dictionary whose keys correspond to the alchemical variables and whose values are functions with a domain
        and range between 0 and 1. Default is given by self.default_alchemical_functions.
    platform : str, optional
        The platform which should be used for simulation. Default is the fastest available platform.
    platform_properties : dict, optional
        Additional platform properties.
    npt : bool, optional
        Whether to add a barostat. Default is True.
    checkpoint : str
        A path to a pickled checkpoint file to load SMCSampler from. If None, SMCSampler is initialised normally.
    md_config : dict, optional
        Additional simulation parameters passed to openmmslicer.alchemy.AlchemicalSimulation().
    alch_config : dict, optional
        Additional alchemical parameters passed to openmmslicer.alchemy.AlchemicalSimulation().
    barostat_config : dict, optional
        Additional barostat parameters passed to openmmslicer.alchemy.AlchemicalSimulation().

    Attributes
    ----------
    trajectory_reporter : [openmmslicer.reporters.MultistateDCDReporter]
        All associated MultistateDCDReporters.
    state_data_reporters : [openmmslicer.reporters.MulsistateStateDataReporter]
        All associated MulsistateStateDataReporters.
    current_trajectory_filename : str
        The current filename used in the main openmmslicer.reporters.MultistateDCDReporter.
    previous_trajectory_filename : str
        The previous filename used in the main openmmslicer.reporters.MultistateDCDReporter.
    initialised : bool
        Whether the system has been initialised after equilibration.
    lambda_ : float
        The current lambda value.
    log_weights : numpy.ndarray
        The logarithm of the normalised weights of each of the current walkers.
    logZ : float
        The logarithm of the average unnormalised walker weights.
    moves : [openmmslicer.moves.MoveList]
        A list of all the moves relevant to the current lambda value.
    n_walkers : int
        The number of current walkers.
    walkers : [openmm.samplers.Walker]
        All current walkers.
    weights : numpy.ndarray
        The normalised weights of each of the current walkers.
    alchemical_chain : openmmslicer.alchemy.AlchemicalChain
        The alchemical chain to be explored by the SMCSampler.
    simulation : openmmslicer.alchemy.AlchemicalSimulation
        The openmmslicer.alchemy.AlchemicalSimulation() object sampled by the SMCSampler.
    total_sampling_steps : int
        The total number of integration steps performed.
    reporters : [openmmslicer.reporters.MultistateDCDReporter, openmmslicer.reporters.MulsistateStateDataReporter]
        The reporter list containing all multistate reporters.
    walker_memo : openmmslicer.samplers.WalkerMemo
        A container storing the total sampling history of the simulation.
    """
    _picklable_attrs = ["total_sampling_steps", "_lambda_", "walkers", "walker_memo", "_initialised", "reporters"]
    default_alchemical_functions = {
        'lambda_bonds': lambda x: min(1.25 * x, 1.),
        'lambda_angles': lambda x: min(1.25 * x, 1.),
        'lambda_electrostatics': lambda x: max(0., 5. * x - 4.),
        'lambda_sterics': lambda x: min(1.25 * x, 1.),
        'lambda_torsions': lambda x: min(1.25 * x, 1.),
    }

    def __init__(self, alchemical_chain, integrator, alchemical_functions=None, platform=None,
                 platform_properties=None, npt=True, checkpoint=None, md_config=None, alch_config=None,
                 barostat_config=None):
        if alchemical_functions is None:
            alchemical_functions = {}
        alchemical_functions = {**self.default_alchemical_functions, **alchemical_functions}
        self.alchemical_chain = alchemical_chain
        self.simulation = _alchemy.AlchemicalSimulation(alchemical_chain, integrator,
                                                        alchemical_functions=alchemical_functions, platform=platform,
                                                        platform_properties=platform_properties, npt=npt,
                                                        md_config=md_config, alch_config=alch_config,
                                                        barostat_config=barostat_config)

        self._initialised = False
        self.total_sampling_steps = 0
        self._lambda_ = 0.
        self.reporters = []
        self.walker_memo = _WalkerMemo()
        self._walkers = []

        if checkpoint is not None:
            self.loadCheckpoint(checkpoint)

    @property
    def state_data_reporters(self):
        reporters = [x for x in self.reporters if isinstance(x, _reporters.MultistateStateDataReporter)]
        if not len(reporters):
            return None
        return reporters

    @property
    def trajectory_reporter(self):
        reporter = [x for x in self.reporters if isinstance(x, _reporters.MultistateDCDReporter)]
        if not len(reporter):
            return None
        if len(reporter) > 1:
            _warnings.warn("Only the first openmmslicer.reporters.MultistateDCDReporter will be used for state loading")
        return reporter[0]

    @property
    def current_trajectory_filename(self):
        reporter = self.trajectory_reporter
        return None if reporter is None else reporter.current_filename

    @property
    def previous_trajectory_filename(self):
        reporter = self.trajectory_reporter
        return None if reporter is None or len(reporter.filename_history) < 2 else reporter.filename_history[-2]

    @property
    def initialised(self):
        return self._initialised

    def iteration(self, lambda_=None):
        """
        Returns the number of times a lambda value has been visited.

        Parameters
        ----------
        lambda_ : float, optional
            The lambda value. Default is the current lambda value.

        Returns
        -------
        n : int
            The number of times lambda_ has been visited.
        """
        if lambda_ is None:
            lambda_ = self.lambda_
        # TODO: fix rounding
        return _np.sum(_np.isclose(self.walker_memo.timestep_lambdas, lambda_, rtol=1e-8))

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
            self.simulation.lambda_ = self._lambda_

            # update walkers
            self.walkers = [_Walker(i,
                                    state=walker.state,
                                    transform=walker.transform,
                                    reporter_filename=walker.reporter_filename,
                                    frame=walker.frame,
                                    original_lambda_=walker.original_lambda_,
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
        return self.alchemical_chain.getMoves(self.lambda_)

    @property
    def n_walkers(self):
        return len(self.walkers)

    @property
    def walkers(self):
        return self._walkers

    @walkers.setter
    def walkers(self, val):
        self.walker_memo.updateWalkers(val)
        self._walkers = val

    @property
    def weights(self):
        return _np.exp(self.log_weights)

    def serialise(self):
        """
        Return all serialisable attributes of the object.

        Returns
        -------
        pickle_dict : dict
            A dictionary containing all serialised attributes of the object.
        """
        pickle_dict = {}
        for attr in self._picklable_attrs:
            try:
                pickle_dict[attr] = getattr(getattr(self, attr), "serialise")()
            except AttributeError:
                pickle_dict[attr] = getattr(self, attr)
        return pickle_dict

    def loadCheckpoint(self, filename="checkpoint.pickle"):
        """
        Loads a previously written checkpoint.

        Parameters
        ----------
        filename : str, optional
            A full path to the pickled checkpoint. Default name is "checkpoint.pickle".
        """
        _logger.info("Loading checkpoint...")
        obj = _dill.load(open(filename, "rb"))["self"]
        for attr in self._picklable_attrs:
            try:
                if attr == "_lambda_":
                    self._lambda_ = obj[attr]
                    self.simulation.lambda_ = self.lambda_
                else:
                    setattr(self, attr, obj[attr])
                if attr == "walker_memo":
                    self.walker_memo._lock = _threading.RLock()
            except AttributeError:
                _warnings.warn(f"There was missing or incompatible data from the checkpoint: {attr}")

    def writeCheckpoint(self, filename="checkpoint.pickle", update=False, overwrite=False, *args, **kwargs):
        """
        Writes a checkpoint file based on all serialisable attributes in the object.

        Parameters
        ----------
        filename : str, optional
            A full path to the pickled checkpoint. Default name is "checkpoint.pickle".
        update : bool, optional
            If filename exists, this parameter determines whether its contents will be loaded and updated before
            pickling. Default is False.
        overwrite : bool, optional
            If filename exists, this parameter determines whether it will be overwritten or renamed. Default is False.
        args
            Positional arguments to be passed to dill.dump().
        kwargs
            Keyword arguments to be passed to dill.dump().
        """
        backups = {}
        if update:
            try:
                backups = _dill.load(open(filename, "rb"))
            except (FileNotFoundError, _dill.UnpicklingError, EOFError):
                pass
        backups["self"] = self.serialise()
        _logger.info("Writing checkpoint...")
        if _os.path.exists(filename) and not overwrite:
            _os.rename(filename, filename + ".old")
        _dill.dump(backups, open(filename, "wb"), *args, **kwargs)
        if _os.path.exists(filename + ".old"):
            _os.remove(filename + ".old")

    def calculateDeltaEs(self, lambda1=None, lambda0=None, walkers=None):
        """
        Calculates the dimensionless energy difference of a list of walkers between two lambda values.

        Parameters
        ----------
        lambda1 : float or list or numpy.ndarray, optional
            The final lambda value to be passed to calculateStateEnergies().
        lambda0 : float or list or numpy.ndarray, optional
            The initial lambda value to be passed to calculateStateEnergies().
        walkers: [openmm.samplers.Walker], optional
            A list of all walkers to be passed to calculateStateEnergies().

        Returns
        -------
        energies : np.ndarray
            An array containing all per-walker energy differences.
        """
        if walkers is None:
            walkers = self.walkers
        old_potentials = self.calculateStateEnergies(lambda0, walkers=walkers)
        new_potentials = self.calculateStateEnergies(lambda1, walkers=walkers)
        return new_potentials - old_potentials

    def calculateStateEnergies(self, lambda_=None, walkers=None):
        """
        Calculates the dimensionless potential energies of all states for a given lambda value.

        Parameters
        ----------
        lambda_ : float or list or numpy.ndarray, optional
            The desired lambda value or per-walker lambda values. Default is None, which means that the lambda value
            used for generating each of the walkers will be used.
        walkers : [openmm.samplers.Walker], optional
            A list of all walkers to be evaluated. Default is all current walkers.

        Returns
        -------
        energies : np.ndarray
            An array containing all per-walker energies.
        """
        def hasHybridMoves(lambda0, lambda1):
            n_states = len(self.simulation.alchemical_chain.states) - 1
            checkpoint_lambdas = [i / n_states for i in range(n_states + 1)]
            min_lambda = min(lambda0, lambda1)
            max_lambda = max(lambda0, lambda1)
            lambdas = [min_lambda] + [x for x in checkpoint_lambdas if min_lambda < x < max_lambda] + [max_lambda]
            for lambda_ in lambdas:
                moves = self.simulation.alchemical_chain.getMoves(lambda_)
                if moves is not None and any(isinstance(move, _moves.HybridMove) for move in moves._moves):
                    return True
            return False

        if walkers is None:
            walkers = self.walkers
        if lambda_ is None:
            lambdas = _np.asarray([[walker.lambda_] for walker in walkers])
        else:
            lambda_ = _np.asarray(lambda_)
            lambdas = _np.full((len(walkers), lambda_.size), lambda_)

        energies = _np.zeros(lambdas.shape)
        kT = self.simulation.kT

        # determine unique walkers for optimal loading from hard drive
        unique_walkers = {}
        for i, walker in enumerate(walkers):
            key = None
            if isinstance(walker, _Walker) and walker.state is None:
                key = (walker.reporter_filename, walker.frame)
                if any(x is None for x in key):
                    raise ValueError("Walkers need to contain either an OpenMM State or a valid trajectory path and "
                                     "frame number")
            if key not in unique_walkers:
                unique_walkers[key] = []
            unique_walkers[key] += [i]

        for key, group in unique_walkers.items():
            # modify all walkers with a single read from the hard drive, if applicable
            if key is not None:
                previous_states = [walkers[i].state for i in group]
                dummy_state = _copy.copy(walkers[group[0]])
                dummy_state.transform = None
                self.simulation.context.setState(dummy_state, apply_hybrid_transforms=False)
                state = self.simulation.context.getState(getPositions=True)
                for i in group:
                    walkers[i].setStateKeepCache(state)

            # calculate the energies
            for i in group:
                walker = walkers[i]
                lambda_ = lambdas[i]

                state_already_set = False
                has_hybrid_moves = False
                for j, value in enumerate(lambda_):
                    # get cached energy and skip energy evaluation, if applicable
                    if isinstance(walker, _Walker):
                        energy = walker.getCachedEnergy(value)
                        if energy is not None:
                            energies[i, j] = energy
                            continue

                    # calculate energy and set cache
                    self.simulation.lambda_ = value
                    if hasHybridMoves(walker.original_lambda_, value):
                        has_hybrid_moves = True
                    if not state_already_set or has_hybrid_moves:
                        log_jacobian = self.simulation.context.setState(walker)
                        state_already_set = True
                    else:
                        log_jacobian = 0.
                    energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy() / kT - log_jacobian
                    energies[i, j] = energy

                    # update cache
                    if isinstance(walker, _Walker):
                        walker.setCachedEnergy(value, energy)

            # restore original walkers
            if key is not None:
                for i, previous_state in zip(group, previous_states):
                    walkers[i].setStateKeepCache(previous_state)

        self.simulation.lambda_ = self.lambda_

        energies = energies.T
        if energies.shape[0] == 1:
            energies = energies[0]

        # TODO: handle infinities in a more robust way?
        if _np.any(_np.isnan(energies)):
            raise ValueError("NaN values encountered during energy calculation")

        return energies

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
        equilibration_steps : int, optional
            The number of equilibration steps. Default is 100000.
        restrain_backbone : bool, optional
            Whether to restrain all atoms with the following names: 'CA', 'C', 'N'. Default it True.
        restrain_resnames : list, optional
            A list of residue names to restrain. Default is ["UNL", "LIG", "MOL"].
        restrain_alchemical_atoms : bool, None, optional
            True restrains all alchemical atoms, False removes restraints from all alchemical atoms and None has no
            effect on the restraints. Default is False.
        force_constant : openmm.unit.Quantity, optional
            The magnitude of the restraint force constant. Default is 5 kcal/mol/angstroms^2
        output_interval : int, optional
            How often to output to the trajectory file. Default is every 100000 steps.
        """
        if restrain_resnames is None:
            restrain_resnames = ["LIG", "UNL", "MOL"]

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
        structure = self.alchemical_chain.aligned_structures[self.alchemical_chain.states[0][0]]
        if restrain_backbone:
            atoms |= {i for i, a in enumerate(structure.atoms) if a.name in ('CA', 'C', 'N')}
        if restrain_resnames:
            atoms |= {a.idx for r in structure.residues for a in r if r.name in restrain_resnames}
        if self.moves is not None:
            alch_atoms = {atom.idx for atom in structure.atoms if "du" in atom.type} | set(self.moves.alchemical_atoms)
            if restrain_alchemical_atoms is False:
                atoms -= alch_atoms
            if restrain_alchemical_atoms is True:
                atoms |= alch_atoms
        counter = len(atoms)

        for i in atoms:
            force.addParticle(i, structure.positions[i].value_in_unit(_unit.nanometers))
        if counter:
            _logger.info(f"Adding {counter} equilibration restraints...")
            force_idx = self.simulation.system.addForce(force)

        # run the equilibration
        _logger.info("Running initial equilibration...")
        self.simulation.lambda_ = 0.
        self.simulation.context.setVelocitiesToTemperature(self.simulation.temperature)
        self.simulation.step(equilibration_steps)

        # remove the restraints, if applicable
        if counter:
            _logger.info(f"Removing {counter} equilibration restraints...")
            self.simulation.system.removeForce(force_idx)

        # reset the reporter, if applicable
        if self.trajectory_reporter:
            del self.simulation.reporters[-1]
            self.trajectory_reporter.prune()

    def initialise(self, n_walkers):
        if not self.initialised:
            # root layer
            if not len(self.walkers):
                state = self.simulation.context.getState(getPositions=True, getEnergy=True)
                self._walkers = [_Walker(0, state=state)]
            # lambda = 0 layer
            self.walkers = [_Walker(i,
                                    state=self.walkers[i % len(self.walkers)].state,
                                    reporter_filename=self.walkers[i % len(self.walkers)].reporter_filename,
                                    frame=self.walkers[i % len(self.walkers)].frame,
                                    original_lambda_=self.lambda_,
                                    lambda_=self.lambda_,
                                    iteration=self.iteration(),
                                    logW=0)
                            for i in range(max(n_walkers, len(self.walkers)))]
            self._initialised = True

    def sample(self,
               default_decorrelation_steps=500,
               presamplers=None,
               keep_walkers_in_memory=False,
               reporter_filename=None,
               append=False):
        """
        Performs a single sequential importance sampling iteration.

        Parameters
        ----------
        default_decorrelation_steps : int, optional
            The number of integration steps per walker. Default is 500 steps.
        presamplers : list, optional
            A list of callables to be called before sampling with their only argument being the current object. This
            can be used for consecutively calling other custom samplers. Default is None.
        keep_walkers_in_memory : bool, optional
            Whether to keep the walkers as states or load them dynamically from the hard drive. The former is much
            faster during the energy evaluation step but is more memory-intensive. Only set to True if you are certain
            that you have enough memory.
        reporter_filename : str, optional
            The label to be used when initialising the openmmslicer.reporters.MultistateDCDReporter. Default is the
            current lambda value rounded to eight decimal places.
        append : bool, optional
            Whether to append to the old trajectory or overwrite it. Default is False.
        """
        if not keep_walkers_in_memory and self.trajectory_reporter is None:
            raise ValueError("Need to set a reporter if trajectory is not kept in memory.")
        if presamplers is None:
            presamplers = []

        initial_frame = 0

        # set up a reporter, if applicable
        if self.trajectory_reporter is not None:
            label = reporter_filename if reporter_filename is not None else str(round(self.lambda_, 8))
            self.simulation.reporters.append(self.trajectory_reporter.generateReporter(
                label, default_decorrelation_steps, append=append))
            if append:
                duplicates = [w for w in self.walker_memo.walkers
                              if w.reporter_filename == self.current_trajectory_filename]
                initial_frame = max([w.frame for w in duplicates]) + 1 if len(duplicates) else 0
        if self.state_data_reporters is not None:
            self.simulation.reporters += self.state_data_reporters

        for walker in self.walkers:
            if walker.reporter_filename == self.current_trajectory_filename:
                walker.reporter_filename = self.previous_trajectory_filename

        for n, walker in enumerate(self.walkers):
            # update the state data reporters, if applicable
            if self.state_data_reporters is not None:
                for r in self.state_data_reporters:
                    r.update(self, n)
            # sample
            self.simulation.context.setState(walker)
            for presampler in presamplers:
                presampler(self.simulation)
            self.simulation.context.setVelocitiesToTemperature(self.simulation.temperature)
            self.simulation.step(default_decorrelation_steps)

            # update walkers
            if keep_walkers_in_memory:
                walker.state = self.simulation.context.getState(getPositions=True, getEnergy=True)
            else:
                walker.state = None
            walker.original_lambda_ = self.lambda_
            walker.transform = None
            walker.reporter_filename = self.current_trajectory_filename
            walker.frame = n + initial_frame if walker.reporter_filename is not None else None

            # update statistics
            self.total_sampling_steps += default_decorrelation_steps

        # reset the reporter
        self.simulation.reporters = [x for x in self.simulation.reporters if x not in self.reporters]
        if self.trajectory_reporter is not None:
            self.simulation.reporters.remove(self.trajectory_reporter.current_reporter)
            self.trajectory_reporter.prune()

    def generateTransforms(self, n_transforms_per_walker=100):
        """
        Generates transformed walkers according to self.moves.

        Parameters
        ----------
        n_transforms_per_walker : int, optional
            How many transforms to generate for each walker. Default is 100.
        """
        if self.moves is not None and any(isinstance(move, _moves.EndpointMove) for move in self.moves._moves):
            _logger.info(f"Generating {len(self.walkers) * n_transforms_per_walker} total transforms...")
            new_walkers = []

            i = 0
            for walker in self.walkers:
                self.simulation.context.setState(walker)
                transforms = self.moves.generate(n_transforms_per_walker)
                for t in transforms:
                    walker_new = _Walker(i, state=walker.state, original_lambda_=walker.lambda_, lambda_=walker.lambda_,
                                         iteration=walker.iteration, transform=t,
                                         reporter_filename=walker.reporter_filename, frame=walker.frame,
                                         logW=walker.logW)
                    new_walkers += [walker_new]
                    i += 1

            self.walkers = new_walkers

    def reweight(self,
                 resampling_metric=_resmetrics.ExpectedSampleSize,
                 target_metric_value=None,
                 target_metric_tol=None,
                 maximum_metric_evaluations=20,
                 default_dlambda=0.1,
                 minimum_dlambda=None,
                 maximum_dlambda=None,
                 target_lambda=1,
                 fixed_lambdas=None,
                 change_lambda=True):
        """
        Reweights the current walkers based on a new lambda value which can be determined adaptively.

        Parameters
        ----------
        resampling_metric : class or None, optional
            A resampling metric with callable methods as described in openmmslicer.resampling_metrics. This metric is
            used to adaptively determine the next lambda value None removes adaptive resampling. None means no adaptive
            resampling will be performed. Default is openmmslicer.resampling_metrics.ExpectedSampleSize.
        target_metric_value : float, optional
            The target metric value which will be used to determine the next lambda value. Default is given by
            resampling_metric.defaultValue.
        target_metric_tol : float, optional
            The target metric error tolerance when determining the next lambda value. Default is given by
            resampling_metric.defaultTol.
        maximum_metric_evaluations : int, optional
            The maximum number of metric evaluations before selecting a new lambda value. Default is 20.
        default_dlambda : float, optional
            The initial dlambda to be passed to the bisection algorithm. If the resampling metric is None, the next
             lambda value will be given by self.lambda_ + default_dlambda. Default is 0.1.
        minimum_dlambda : float or None, optional
            The minimum dlambda for the next lambda value. Default is None.
        maximum_dlambda : float or None, optional
            The maximum dlambda for the next lambda value. Default is None.
        target_lambda : float, optional
            The target lambda value which marks the end of the sequential importance sampling algorithm. Default is 1.
        fixed_lambdas : list, optional
            If self.lambda_ + dlambda is higher than the lowest value in fixed_lambdas higher than self.lambda_, it
            will be used as the next lambda value instead of self.lambda_ + dlambda. Default is None.
        change_lambda : bool, optional
            Whether to change self.lambda_. This will trigger reweighting of all walkers.
        """
        if self.lambda_ == target_lambda:
            return

        if fixed_lambdas is None:
            fixed_lambdas = self.alchemical_chain.fixed_lambdas

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
                _logger.debug(f"Resampling metric {val} at next lambda {new_lambda}")
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
            _logger.debug(f"Tentative next lambda: {next_lambda_}")
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
        """
        Resamples the current walkers based on their weights.

        Parameters
        ----------
        n_walkers : int, optional
            The number of walkers to resample. Default is 1000.
        resampling_method : class, optional
            A resampling method with callable methods as described in openmmslicer.resampling_methods. Default is
            openmmslicer.resampling_methods.SystematicResampler.
        exact_weights : bool, optional
            Whether to modify the weights based on the actual resampled populations or on the expected populations.
            Default is False.
        walkers : [openmmslicer.samplers.Walker], optional
            The walkers to resample. Default is self.walkers.
        change_walkers : bool, optional
            Whether to set self.walkers to the resampled walkers. Default is True.
        """
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

            if change_walkers:
                self.walker_memo.lock.acquire()
                self.walker_memo.removeWalkers(walkers)

            new_walkers = [_Walker(i_new,
                                   state=walkers[i_old].state,
                                   transform=walkers[i_old].transform,
                                   reporter_filename=walkers[i_old].reporter_filename,
                                   frame=walkers[i_old].frame,
                                   original_lambda_=walkers[i_old].original_lambda_,
                                   lambda_=self.lambda_,
                                   iteration=self.iteration(),
                                   logW=logWs[i_new])
                           for i_new, i_old in enumerate(resampled_indices)]

            # update the object, if applicable
            if change_walkers:
                self.walkers = new_walkers
                self.walker_memo.lock.release()

        return new_walkers

    def runSingleIteration(self,
                           sampling_metric=_sammetrics.EnergyCorrelation,
                           resampling_method=_resmethods.SystematicResampler,
                           resampling_metric=_resmetrics.ExpectedSampleSize,
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
                           presamplers=None,
                           n_walkers=1000,
                           n_transforms_per_walker=100,
                           keep_walkers_in_memory=False):
        """
        Performs a single iteration of the sampler.

        Parameters
        ----------
        sampling_metric : class, optional
            A sampling metric with callable methods as described in openmmslicer.sampling_metrics. This metric is used
            to adaptively determine the optimal sampling time. None removes adaptive sampling. Default is
            openmmslicer.sampling_metrics.EnergyCorrelation.
        resampling_method : class, optional
            A resampling method with callable methods as described in openmmslicer.resampling_methods. Default is
            openmmslicer.resampling_methods.SystematicResampler.
        resampling_metric : class or None, optional
            A resampling metric with callable methods as described in openmmslicer.resampling_metrics. This metric is
            used to adaptively determine the next lambda value None removes adaptive resampling. None means no adaptive
            resampling will be performed. Default is openmmslicer.resampling_metrics.ExpectedSampleSize.
        exact_weights : bool, optional
            Whether to modify the weights based on the actual resampled populations or on the expected populations.
            Default is False.
        target_metric_value : float, optional
            The target metric value which will be used to determine the next lambda value. Default is given by
            resampling_metric.defaultValue.
        target_metric_tol : float, optional
            The target metric error tolerance when determining the next lambda value. Default is given by
            resampling_metric.defaultTol.
        maximum_metric_evaluations : int, optional
            The maximum number of metric evaluations before selecting a new lambda value. Default is 20.
        default_dlambda : float, optional
            The initial dlambda to be passed to the bisection algorithm. If the resampling metric is None, the next
             lambda value will be given by self.lambda_ + default_dlambda. Default is 0.1.
        minimum_dlambda : float or None, optional
            The minimum dlambda for the next lambda value. Default is None.
        maximum_dlambda : float or None, optional
            The maximum dlambda for the next lambda value. Default is None.
        target_lambda : float, optional
            The target lambda value which marks the end of the sequential importance sampling algorithm. Default is 1.
        default_decorrelation_steps : int, optional
            The number of integration steps per walker. Default is 500 steps.
        maximum_decorrelation_steps : int, optional
            The maximum number of decorrelation steps. Only used with adaptive sampling. Default is 5000 steps.
        presamplers : list, optional
            A list of callables to be called before sampling with their only argument being the current object. This
            can be used for consecutively calling other custom samplers. Default is None.
        n_walkers : int, optional
            The number of walkers to resample. Default is 1000.
        n_transforms_per_walker : int, optional
            How many transforms to generate for each walker, if applicable. Default is 100.
        keep_walkers_in_memory : bool, optional
            Whether to keep the walkers as states or load them dynamically from the hard drive. The former is much
            faster during the energy evaluation step but is more memory-intensive. Only set to True if you are certain
            that you have enough memory.
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

        if sampling_metric is not None:
            sampling_metric.evaluateBefore()
        elapsed_steps = 0
        while True:
            # sample
            self.sample(
                default_decorrelation_steps=default_decorrelation_steps,
                presamplers=presamplers,
                keep_walkers_in_memory=keep_walkers_in_memory
            )
            elapsed_steps += default_decorrelation_steps

            # evaluate sampling metric
            next_lambda = None
            if sampling_metric is not None:
                if sampling_metric.requireNextLambda:
                    next_lambda = self.reweight(**reweight_kwargs, change_lambda=False)
                    sampling_metric.evaluateAfter(next_lambda)
                    _logger.debug(f"Sampling metric {sampling_metric.metric} at next lambda {next_lambda}")
                else:
                    sampling_metric.evaluateAfter()
                    _logger.debug(f"Sampling metric {sampling_metric.metric}")
                if not sampling_metric.terminateSampling and elapsed_steps < maximum_decorrelation_steps:
                    continue
                sampling_metric.reset()

            # generate transforms
            self.generateTransforms(n_transforms_per_walker=n_transforms_per_walker)

            # reweight if needed and change lambda
            if self.lambda_ != target_lambda:
                if next_lambda is None:
                    self.reweight(**reweight_kwargs)
                else:
                    self.lambda_ = next_lambda
            break

        if self.lambda_ != target_lambda:
            # resample
            self.resample(
                n_walkers=n_walkers,
                resampling_method=resampling_method,
                exact_weights=exact_weights
            )

        # dump info to logger
        _logger.info(f"Sampling at lambda = {self.walker_memo.timestep_lambdas[-1]} terminated after {elapsed_steps} "
                     f"steps per walker")
        if self.current_trajectory_filename is not None:
            _logger.info(f"Trajectory path: \"{self.current_trajectory_filename}\"")
        _logger.info(f"Current accumulated logZ: {self.logZ}")
        _logger.info(f"Next lambda: {self.lambda_}")

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
        n_equilibrations : int, optional
            The number of equilibrations.
        equilibration_steps : int, optional
            The number of equilibration steps. Default is 100000.
        restrain_backbone : bool, optional
            Whether to restrain all atoms with the following names: 'CA', 'C', 'N'. Default it True.
        restrain_resnames : list, optional
            A list of residue names to restrain. Default is ["UNL", "LIG", "MOL"].
        restrain_alchemical_atoms : bool, None, optional
            True restrains all alchemical atoms, False removes restraints from all alchemical atoms and None has no
            effect on the restraints. Default is False.
        force_constant : openmm.unit.Quantity, optional
            The magnitude of the restraint force constant. Default is 5 kcal/mol/angstroms^2
        output_interval : int, optional
            How often to output to the trajectory file. Default is every 100000 steps.
        final_decorrelation_step : bool, optional
            Whether to decorrelate the final resampled walkers for another number of default_decorrelation_steps.
            Default is True.
        target_lambda : float, optional
            The target lambda value which marks the end of the sequential importance sampling algorithm. Default is 1.
        args
            Positional arguments to be passed to runSingleIteration().
        kwargs
            Keyword arguments to be passed to runSingleIteration().
        """
        kwargs["target_lambda"] = target_lambda
        direction = _np.sign(target_lambda - self.lambda_)

        # TODO: fix equilibration
        if self.lambda_ == 0 and not self.initialised:
            if output_interval:
                assert equilibration_steps % output_interval == 0, "The equilibration steps must be a multiple of " \
                                                                   "the output interval"
            frame_step = equilibration_steps // output_interval if output_interval else None
            self._walkers = []
            for i in range(n_equilibrations):
                old_state = self.simulation.context.getState(getPositions=True)
                self.equilibrate(equilibration_steps=equilibration_steps,
                                 restrain_backbone=restrain_backbone,
                                 restrain_resnames=restrain_resnames,
                                 restrain_alchemical_atoms=restrain_alchemical_atoms,
                                 force_constant=force_constant,
                                 output_interval=output_interval)
                self._walkers += [_Walker(i,
                                          state=self.simulation.context.getState(getPositions=True, getEnergy=True),
                                          reporter_filename=self.current_trajectory_filename,
                                          frame=(i + 1) * frame_step - 1) if frame_step is not None else None]
                self.simulation.context.setState(old_state)

        initial_sampling_steps = self.total_sampling_steps
        while _np.sign(target_lambda - self.lambda_) == direction != 0:
            self.runSingleIteration(*args, **kwargs)
        if final_decorrelation_step:
            self.runSingleIteration(*args, **kwargs)
        sampling_step_difference = self.total_sampling_steps - initial_sampling_steps
        dt = self.simulation.integrator.getStepSize().in_units_of(_unit.nanosecond)
        total_sampling_time = _quantity_round(sampling_step_difference * dt, 6)
        _logger.info(f"The SMC run cycle finished in {total_sampling_time}")
