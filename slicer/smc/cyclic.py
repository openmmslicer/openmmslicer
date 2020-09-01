import inspect as _inspect
import logging as _logging
import math as _math
import random as _random

import numpy as _np
import simtk.unit as _unit

from .generic import GenericSMCSampler as _GenericSMCSampler
import slicer.fe_estimators as _fe_estimators
import slicer.resampling_methods as _resmethods

_logger = _logging.getLogger(__name__)


class CyclicSMCSampler(_GenericSMCSampler):
    # TODO: correctly reweight adaptive walkers
    # TODO: make pickle work
    _picklable_attrs = _GenericSMCSampler._picklable_attrs + ["target_lambda", "N_opt", "adaptive_mode",
                                                              "sampling_history", "fe_estimator"]

    def __init__(self, *args, fe_estimator=_fe_estimators.EqualMetropolis, **kwargs):
        self.target_lambda = 1
        self.N_opt = None
        self.adaptive_mode = True
        self.sampling_history = []
        if not issubclass(fe_estimator, _fe_estimators.AbstractFEEstimator):
            raise TypeError("The free energy estimator must be inherited from the abstract base class")
        self.fe_estimator = fe_estimator(self)
        super().__init__(*args, **kwargs)

    def walker_filter(self, lambda_, **kwargs):
        initial_filter = [w for w in self._all_walkers if w.lambda_ is not None and _math.isclose(w.lambda_, lambda_)]
        return [w for w in initial_filter if all(getattr(w, k) == v for k, v in kwargs.items())]

    @property
    def target_lambda(self):
        return self._target_lambda

    @target_lambda.setter
    def target_lambda(self, val):
        assert val in [0, 1], "The target lambda must be one of the terminal lambda states"
        self._target_lambda = val

    def sample(self, *args, reporter_filename=None, **kwargs):
        if reporter_filename is None:
            if self.adaptive_mode:
                reporter_filename = "{}_adapt".format(round(self.lambda_, 8))
                kwargs["append"] = False
            else:
                reporter_filename = str(round(self.lambda_, 8))
                kwargs["append"] = True if self.iteration() > 1 else False
        super().sample(*args, reporter_filename=reporter_filename, **kwargs)

    def reweight(self, *args, **kwargs):
        if self.adaptive_mode:
            return super().reweight(*args, **kwargs)

        # get the protocol schedule from the adaptive step
        idx = next(i for i, x in enumerate(self.lambda_history) if _math.isclose(x, self.lambda_))
        sign = -1 if self.lambda_ != 0 and (self.target_lambda > self.lambda_ or self.lambda_ == 1) else 1
        prev_lambda = self.lambda_history[(idx + sign)]
        sign = 1 if self.lambda_ != 1 and (self.target_lambda > self.lambda_ or self.lambda_ == 0) else -1
        next_lambda = self.lambda_history[(idx + sign)]

        # get acceptance criterion
        fe_fwd = self.fe_estimator(next_lambda, self.lambda_)
        deltaEs_fwd = self.calculateDeltaEs(next_lambda, self.lambda_)
        acc_fwd = min(1., _np.average(_np.exp(-deltaEs_fwd + fe_fwd)))
        _logger.debug("Forward probability: {}".format(acc_fwd))
        
        # accept or reject move and/or swap direction
        randnum = _random.random()
        if acc_fwd != 1. and randnum >= acc_fwd:
            if not _math.isclose(next_lambda, prev_lambda):
                fe_bwd = self.fe_estimator(prev_lambda, self.lambda_)
                deltaEs_bwd = self.calculateDeltaEs(prev_lambda, self.lambda_)
                acc_bwd = max(0., min(1., _np.average(_np.exp(-deltaEs_bwd + fe_bwd))) - acc_fwd)
                _logger.debug("Backward probability: {}".format(acc_bwd))

                if acc_bwd != 0. and randnum - acc_fwd < acc_bwd:
                    self.target_lambda = int(not self.target_lambda)
            # trigger walker update
            self.lambda_ = self.lambda_
        else:
            self.lambda_ = next_lambda

        return self.lambda_

    def run(self,
            *args,
            n_cycles=0,
            n_equilibrations=1,
            equilibration_steps=100000,
            restrain_backbone=True,
            restrain_resnames=None,
            restrain_alchemical_atoms=False,
            force_constant=5.0 * _unit.kilocalories_per_mole / _unit.angstroms ** 2,
            output_interval=100000,
            target_lambda=1,
            maximum_duration=None,
            **kwargs):
        """
        Performs a complete sequential Monte Carlo run until lambda = 1.

        Parameters
        ----------
        n_cycles : int, None
            The number of SMC cycles to run. None means no limit.
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
        target_lambda : float
            What the final intended lambda value is.
        maximum_duration : simtk.unit.Quantity, None
            The simulation will be stopped at the cycle immediately after this value. None means no limit.
        args
            Positional arguments to be passed to super().run().
        kwargs
            Keyword arguments to be passed to super().run().
        """
        kwargs["target_lambda"] = target_lambda
        if "final_decorrelation_step" in kwargs.keys():
            kwargs.pop("final_decorrelation_step")

        initial_sampling_steps = self.total_sampling_steps
        if self.adaptive_mode:
            super().run(
                *args,
                n_equilibrations=n_equilibrations,
                equilibration_steps=equilibration_steps,
                restrain_backbone=restrain_backbone,
                restrain_resnames=restrain_resnames,
                restrain_alchemical_atoms=restrain_alchemical_atoms,
                force_constant=force_constant,
                output_interval=output_interval,
                final_decorrelation_step=True,
                **kwargs
            )
            self.adaptive_mode = False

        current_cycle = 0
        while True:
            dsteps = self.total_sampling_steps - initial_sampling_steps
            dt = dsteps * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
            if maximum_duration is not None and dt >= maximum_duration.in_units_of(_unit.nanosecond):
                break
            if n_cycles is not None and current_cycle > n_cycles:
                break
            self._runCycle(*args, **kwargs)
            current_cycle += 1

        lambdas = self.lambda_history[:self.lambda_history.index(1) + 1]
        fe = sum(self.fe_estimator(lambdas[i + 1], lambdas[i]) for i in range(len(lambdas) - 1))
        _logger.info("Grand total simulation time was {}".format(dt))
        _logger.info("Final dimensionless free energy difference (-logZ) between lambda = 0 and lambda = 1 is: {}".
                     format(fe))

    def runSingleIteration(self, *args, n_walkers_adapt=50, n_walkers=1, **kwargs):
        if not self.adaptive_mode:
            all_kwargs = _inspect.signature(self._runPostAdaptiveIteration).parameters
            valid_kwargs = {k: v for k, v in kwargs.items() if k in all_kwargs.keys()}
            self._runPostAdaptiveIteration(*args, n_walkers=n_walkers, **valid_kwargs)
        else:
            all_kwargs = _inspect.signature(super().runSingleIteration).parameters
            valid_kwargs = {k: v for k, v in kwargs.items() if k in all_kwargs.keys()}
            steps_before = self.total_sampling_steps
            n_walkers_adapt = len(self.walkers) if self.initialised else max(len(self.walkers), n_walkers_adapt)
            super().runSingleIteration(*args, n_walkers=n_walkers_adapt, **valid_kwargs)
            self.sampling_history += [(self.total_sampling_steps - steps_before) // n_walkers_adapt]

    def _runPostAdaptiveIteration(self,
                                  resampling_method=_resmethods.SystematicResampler,
                                  exact_weights=False,
                                  n_walkers=1,
                                  walker_metric=None,
                                  generate_transforms=None,
                                  n_transforms_per_walker=100,
                                  keep_walkers_in_memory=False,
                                  write_checkpoint=None,
                                  load_checkpoint=None):
        if resampling_method is None:
            raise ValueError("Resampling must be performed after the adaptative cycle")

        # get next lambda and decorrelation time
        prev_lambda = self.lambda_
        self.reweight()
        idx = next(i for i, x in enumerate(self.lambda_history) if _math.isclose(x, self.lambda_))
        next_decorr = self.sampling_history[idx]

        # get next number of walkers, if applicable
        if walker_metric is not None:
            if _inspect.isclass(walker_metric):
                walker_metric = walker_metric(self)
            n_walkers = walker_metric(self.lambda_)

        # resample only if we need to
        if self.lambda_ != prev_lambda or n_walkers != len(self.walkers):
            self.resample(
                n_walkers=n_walkers,
                resampling_method=resampling_method,
                exact_weights=exact_weights,
            )

        _logger.info("Sampling {} walkers at lambda = {:.8g} for {} steps per walker...".format(
            n_walkers, self.lambda_, next_decorr))

        # generate uncorrelated samples
        self.sample(
            default_decorrelation_steps=next_decorr,
            keep_walkers_in_memory=keep_walkers_in_memory,
            write_checkpoint=write_checkpoint,
            load_checkpoint=load_checkpoint
        )

        # generate transforms, if applicable
        self.generateTransforms(
            n_transforms_per_walker=n_transforms_per_walker,
            generate_transforms=generate_transforms,
        )

        if self.current_trajectory_filename is not None:
            _logger.info("Trajectory path: \"{}\"".format(self.current_trajectory_filename))

    def _runCycle(self, *args, **kwargs):
        def runOnce():
            kwargs["target_lambda"] = self.target_lambda
            self.runSingleIteration(*args, **kwargs)
            if self.lambda_ == self.target_lambda:
                self.target_lambda = int(not self.target_lambda)
            if "load_checkpoint" in kwargs.keys():
                kwargs.pop("load_checkpoint")

        if self.lambda_ == self.target_lambda:
            self.target_lambda = int(not self.target_lambda)
        initial_target_lambda = self.target_lambda
        finished_half_cycle = False
        initial_lambda_ = self.lambda_
        initial_sampling_steps = self.total_sampling_steps

        while True:
            runOnce()
            if self.lambda_ == initial_target_lambda:
                finished_half_cycle = True
            if finished_half_cycle and self.lambda_ == int(not initial_target_lambda):
                d_steps = self.total_sampling_steps - initial_sampling_steps
                time = d_steps * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
                _logger.info("Cycle from lambda = {} to lambda = {} finished after {} of cumulative sampling".format(
                    initial_lambda_, self.lambda_, time))
                break
