import inspect as _inspect
import logging as _logging
import math as _math
import random as _random

import numpy as _np
from scipy.interpolate import interp1d as _interp1d
import openmm.unit as _unit

from .smc import SMCSampler as _SMCSampler
from openmmslicer.correlation_metrics import EffectiveDecorrelationTime as _EffectiveDecorrelationTime
import openmmslicer.fe_estimators as _fe_estimators
from openmmslicer.misc import quantity_round as _quantity_round
import openmmslicer.resampling_methods as _resmethods
from openmmslicer.protocol import Protocol as _Protocol

_logger = _logging.getLogger(__name__)


class STSampler(_SMCSampler):
    """
    An irreversible simulated tempering sampler.

    Parameters
    ----------
    fe_estimator : class, optional
        The free energy estimator used to estimate the free energies between the lambda windows as defined in
        openmmslicer.fe_estimators. Default is openmmslicer.fe_estimators.EnsembleBAR.
    n_bootstraps : int or None, optional
        The number of bootstraps used for free energy estimation. None means all samples will be used. Default is None.
    n_decorr : int or None, optional
        The maximum number of decorrelated sets of walkers used for free energy estimation. The decorrelation time is
        estimated from openmmslicer.correlation_metrics.EffectiveDecorrelationTime and is used to decimate the samples.
        None means no sample decimation is done. Default is None.
    fe_update_func : callable, optional
        A callable which when called as fe_update_func(fe_estimator) returns the number of times sample() will be called
        before the next free energy update is performed. Default is lambda self: 1 + 0.01 * self.effective_sample_size.
    fe_parallel : bool, optional
        Whether to run the free energy estimator in parallel. Default is True.
    fe_background : bool, optional
        Whether to run the free energy estimator in the background. Default is True.
    significant_figures : int or None, optional
        The maximum number of significant figures per lambda value. None means no such discretisation will be performed.
        Default is None.
    args
        Positional arguments to be passed to super().__init__().
    kwargs
        Keyword arguments to be passed to super().__init__().

    Attributes
    ----------
    adaptive_mode : bool
        Whether the simulation is in its adaptive sequential Monte Carlo stage.
    fe_estimator : openmmslicer.fe_estimators.AbstractFEEstimator
        The current free energy estimator.
    next_lambda : float
        The next lambda value in the sequence.
    previous_lambda : float
        The previous lambda value in the sequence.
    protocol : openmmslicer.protocol.Protocol
        The associated protocol.
    target_lambda : int
        The target lambda value. One of 0 and 1.
    sampling_history : list
        The number of integration steps to be performed per lambda value.
    """
    # TODO: fix protocol
    _picklable_attrs = _SMCSampler._picklable_attrs + [
        "target_lambda", "adaptive_mode", "sampling_history", "_fe_estimator", "_protocol"
    ]

    def __init__(self, *args, fe_estimator=_fe_estimators.EnsembleBAR, n_bootstraps=None, n_decorr=None,
                 fe_update_func=lambda self: 1 + 0.01 * self.effective_sample_size, fe_parallel=True,
                 fe_background=True, significant_figures=None, checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_lambda = 1
        self.adaptive_mode = True
        # TODO: make sampling_history part of walker_memo
        self.sampling_history = []
        self.fe_estimator = None
        self.protocol = None
        self._post_adaptive_kwargs = dict(fe_estimator=fe_estimator, n_bootstraps=n_bootstraps, n_decorr=n_decorr,
                                          fe_update_func=fe_update_func, fe_parallel=fe_parallel,
                                          fe_background=fe_background, significant_figures=significant_figures)
        if checkpoint is not None:
            self.adaptive_mode = False
            self.loadCheckpoint(checkpoint)

    @property
    def adaptive_mode(self):
        return self._adaptive_mode

    @adaptive_mode.setter
    def adaptive_mode(self, val):
        if not val:
            self._initialise_fe_estimator()
            self._initialise_protocol()
        self._adaptive_mode = val

    @property
    def fe_estimator(self):
        return self._fe_estimator

    @fe_estimator.setter
    def fe_estimator(self, val):
        if val is None:
            self._fe_estimator = val
        elif _inspect.isclass(val):
            if not issubclass(val, _fe_estimators.AbstractFEEstimator):
                raise TypeError("The free energy estimator must be inherited from the abstract base class")
            self._fe_estimator = val(self.walker_memo)
        else:
            if not isinstance(val, _fe_estimators.AbstractFEEstimator):
                raise TypeError("The free energy estimator must be inherited from the abstract base class")
            self._fe_estimator = val
            val.walker_memo = self.walker_memo

    @property
    def next_lambda(self):
        if self.protocol is None:
            return None
        idx = _np.where(_np.isclose(self.protocol.value, self.lambda_, rtol=1e-8))[0][0]
        sign = 1 if self.lambda_ != 1 and (self.target_lambda > self.lambda_ or self.lambda_ == 0) else -1
        return self.protocol.value[idx + sign]

    @property
    def previous_lambda(self):
        if self.protocol is None:
            return None
        idx = _np.where(_np.isclose(self.protocol.value, self.lambda_, rtol=1e-8))[0][0]
        sign = -1 if self.lambda_ != 0 and (self.target_lambda > self.lambda_ or self.lambda_ == 1) else 1
        return self.protocol.value[idx + sign]

    @property
    def protocol(self):
        return self._protocol

    @protocol.setter
    def protocol(self, val):
        self._protocol = val
        if isinstance(self.fe_estimator, _fe_estimators.EnsembleMBAR) and self._post_adaptive_kwargs["n_decorr"]:
            self._fe_estimator._interval.protocol = val

    @property
    def target_lambda(self):
        return self._target_lambda

    @target_lambda.setter
    def target_lambda(self, val):
        assert val in [0, 1], "The target lambda must be one of the terminal lambda states"
        self._target_lambda = val

    @_SMCSampler.walkers.setter
    def walkers(self, val):
        if self.protocol is None:
            self.walker_memo.updateWalkers(val)
        else:
            self.walker_memo.updateWalkersAndEnergies(val, self, self.protocol.value)
        self._walkers = val

    def _initialise_protocol(self):
        """Initialises the protocol."""
        idx = _np.where(self.walker_memo.timestep_lambdas == 1)[0][0]
        self.protocol = _Protocol(self.walker_memo.timestep_lambdas[:idx + 1],
                                  significant_figures=self._post_adaptive_kwargs["significant_figures"])
        self.walker_memo.updateEnergies(self, self.protocol.value)

    def _initialise_fe_estimator(self):
        """Initialises the free energy estimator."""
        fe_estimator = self._post_adaptive_kwargs["fe_estimator"]
        if _inspect.isclass(fe_estimator):
            if self._post_adaptive_kwargs["n_bootstraps"] or self._post_adaptive_kwargs["n_decorr"]:
                if fe_estimator is _fe_estimators.BAR:
                    fe_estimator = _fe_estimators.EnsembleBAR
                elif fe_estimator is _fe_estimators.MBAR:
                    fe_estimator = _fe_estimators.EnsembleMBAR
                kwargs = dict(n_bootstraps=self._post_adaptive_kwargs["n_bootstraps"],
                              n_decorr=self._post_adaptive_kwargs["n_decorr"],
                              update_func=self._post_adaptive_kwargs["fe_update_func"],
                              parallel=self._post_adaptive_kwargs["fe_parallel"],
                              background=self._post_adaptive_kwargs["fe_background"])
            else:
                kwargs = dict(update_func=self._post_adaptive_kwargs["fe_update_func"],
                              parallel=self._post_adaptive_kwargs["fe_parallel"],
                              background=self._post_adaptive_kwargs["fe_background"])
            fe_estimator = fe_estimator(self.walker_memo, **kwargs)
            if self._post_adaptive_kwargs["n_decorr"]:
                _logger.debug(f"_post_adaptive_kwargs")
                fe_estimator.interval = _EffectiveDecorrelationTime(fe_estimator=fe_estimator,
                                                                    protocol=self.protocol)
        self.fe_estimator = fe_estimator

    def decorrelationSteps(self, lambda_=None):
        """The number of decorrelation steps associated with a particular lambda value."""
        if self.protocol is None:
            return None
        protocol = self.walker_memo.timestep_lambdas[:_np.where(self.walker_memo.timestep_lambdas == 1)[0][0] + 1]
        if lambda_ is None:
            lambda_ = self.lambda_
        return _np.round(_interp1d(protocol, self.sampling_history)(lambda_)).astype(int)

    def sample(self, *args, reporter_filename=None, **kwargs):
        if reporter_filename is None:
            if self.adaptive_mode:
                reporter_filename = f"{round(self.lambda_, 8)}_adapt"
                kwargs["append"] = False
            else:
                reporter_filename = str(round(self.lambda_, 8))
                kwargs["append"] = True if self.iteration() > 0 else False
        super().sample(*args, reporter_filename=reporter_filename, **kwargs)

    def reweight(self, *args, **kwargs):
        # TODO: correctly reweight in case of > 1 walkers
        if self.adaptive_mode:
            return super().reweight(*args, **kwargs)

        with self.fe_estimator.walker_memo.lock:
            # get acceptance criterion
            fe_fwd = self.fe_estimator(self.lambda_, self.next_lambda)
            deltaEs_fwd = self.calculateDeltaEs(self.next_lambda, self.lambda_)
            samples_fwd = _np.sum(_np.meshgrid(fe_fwd, -deltaEs_fwd), axis=0)
            acc_fwd = _np.average(_np.exp(_np.minimum(samples_fwd, 0.)))
            _logger.debug(f"DeltaEs are {deltaEs_fwd}, samples fwd are {samples_fwd}")
            _logger.debug(f"Forward probability to lambda = {self.next_lambda} is {acc_fwd} with dimensionless "
                          f"free energy = {fe_fwd}")

            # accept or reject move and/or swap direction
            randnum = _random.random()
            _logger.debug(f"The random number is: {randnum}")
            if acc_fwd != 1. and randnum >= acc_fwd:
                _logger.debug(f"Checking for bwd...")
                if not _math.isclose(self.next_lambda, self.previous_lambda):
                    fe_bwd = self.fe_estimator(self.lambda_, self.previous_lambda)
                    deltaEs_bwd = self.calculateDeltaEs(self.previous_lambda, self.lambda_)
                    samples_bwd = _np.sum(_np.meshgrid(fe_bwd, -deltaEs_bwd), axis=0)
                    acc_bwd = max(0., _np.average(_np.exp(_np.minimum(samples_bwd, 0.))) - acc_fwd)
                    _logger.debug(f"DeltaEs bwd are {deltaEs_bwd}, samples bwd are {samples_bwd}")
                    _logger.debug(f"Backward probability to lambda = {self.previous_lambda} is {acc_bwd} with "
                                  f"dimensionless free energy = {fe_bwd}")

                    if acc_bwd != 0. and randnum - acc_fwd < acc_bwd:
                        _logger.debug(f"acc_bwd != 0. and randnum - acc_fwd < acc_bwd, switching directions")
                        self.target_lambda = int(not self.target_lambda)
                        _logger.debug(f"Direction is towards {self.target_lambda}")

                # trigger walker update
                self.lambda_ = self.lambda_
                _logger.debug(f"self.lambda_ = self.lambda_, move rejected")
            else:
                self.lambda_ = self.next_lambda
                _logger.debug(f"self.lambda_ = self.next_lambda, move accepted")
                if self.lambda_ == 1.0 or self.lambda_ == 0.0:
                    self.target_lambda = int(not self.lambda_)
                    _logger.debug(f"Direction is towards {self.target_lambda}")
        return self.lambda_

    def run(self,
            *args,
            n_equilibrations=1,
            equilibration_steps=100000,
            restrain_backbone=True,
            restrain_resnames=None,
            restrain_alchemical_atoms=False,
            force_constant=5.0 * _unit.kilocalories_per_mole / _unit.angstroms ** 2,
            output_interval=100000,
            target_lambda=1,
            duration=None,
            adapt_kwargs=None,
            **kwargs):
        """
        Performs a complete sequential Monte Carlo run until lambda = 1.

        Parameters
        ----------
        duration : openmm.unit.Quantity, None, optional
            The total length of the run. None means only the initial adaptive SMC run is performed. Default is None.
        adapt_kwargs : dict, optional
            Keyword arguments to be passed to super().run() during the initial adaptive SMC run.
        args
            Positional arguments to be passed to super().run().
        kwargs
            Keyword arguments to be passed to super().run().
        """
        default_adapt_kwargs = {"n_walkers": 50, "target_lambda": target_lambda}
        if adapt_kwargs is not None:
            default_adapt_kwargs.update(adapt_kwargs)
        default_adapt_kwargs["final_decorrelation_step"] = True

        initial_sampling_steps = self.total_sampling_steps
        if self.adaptive_mode:
            super().run(
                n_equilibrations=n_equilibrations,
                equilibration_steps=equilibration_steps,
                restrain_backbone=restrain_backbone,
                restrain_resnames=restrain_resnames,
                restrain_alchemical_atoms=restrain_alchemical_atoms,
                force_constant=force_constant,
                output_interval=output_interval,
                **default_adapt_kwargs
            )
            self.adaptive_mode = False

        round_trips = self.walker_memo.round_trips
        round_trip_steps = self.total_sampling_steps
        dt = self.simulation.integrator.getStepSize().in_units_of(_unit.nanosecond)
        while duration:
            # report after every round trip
            if self.walker_memo.round_trips > round_trips:
                round_trips = self.walker_memo.round_trips
                round_trip_time = _quantity_round((self.total_sampling_steps - round_trip_steps) * dt, 6)
                _logger.info(f"Round trip from lambda = 0 to lambda = 1 finished after {round_trip_time} of cumulative "
                             f"sampling")
                round_trip_steps = self.total_sampling_steps

                # report free energy difference
                # TODO: possibly move the multi-window free energy calculation interface to BAR
                lambdas = self.protocol.value
                with self.fe_estimator.walker_memo.lock:
                    fe = sum(self.fe_estimator(lambdas[i], lambdas[i + 1]) for i in range(len(lambdas) - 1))
                _logger.info(f"Dimensionless free energy difference between lambda = 0 and lambda = 1 is: {fe}")

            # break if run for longer than the maximum duration
            time = _quantity_round((self.total_sampling_steps - initial_sampling_steps) * dt, 6)
            if time >= duration:
                _logger.info(f"Grand total simulation time was {time}")
                break

            # run for a single iteration
            self.runSingleIteration(*args, **kwargs)

    def runSingleIteration(self, *args, **kwargs):
        # TODO: restructure
        if not self.adaptive_mode:
            self._runPostAdaptiveIteration(*args, **kwargs)
        else:
            steps_before = self.total_sampling_steps
            if self.initialised:
                if "n_walkers" in kwargs:
                    kwargs["n_walkers"] = max(len(self.walkers), kwargs["n_walkers"])
                else:
                    # TODO: fix
                    kwargs["n_walkers"] = len(self.walkers)
            significant_figures = self._post_adaptive_kwargs["significant_figures"]
            if significant_figures is not None:
                minimum_value = 1. / 10 ** significant_figures
                for key in ["default_dlambda", "minimum_dlambda", "maximum_dlambda", "target_lambda"]:
                    if key in kwargs:
                        kwargs[key] = _np.sign(kwargs[key]) * max(minimum_value, round(abs(kwargs[key]),
                                                                                       significant_figures))
            additional_fixed_lambdas = {i / (len(self.alchemical_chain.states) - 1)
                                        for i in range(1, len(self.alchemical_chain.states) - 1)}
            if "fixed_lambdas" not in kwargs or kwargs["fixed_lambdas"] is None:
                kwargs["fixed_lambdas"] = sorted(additional_fixed_lambdas)
            else:
                kwargs["fixed_lambdas"] = sorted(set(kwargs["fixed_lambdas"]) | additional_fixed_lambdas)
            super().runSingleIteration(*args, **kwargs)
            self.sampling_history += [(self.total_sampling_steps - steps_before) // kwargs["n_walkers"]]

    def _runPostAdaptiveIteration(self,
                                  resampling_method=_resmethods.SystematicResampler,
                                  exact_weights=False,
                                  n_walkers=1,
                                  walker_metric=None,
                                  presamplers=None,
                                  n_transforms_per_walker=1,
                                  keep_walkers_in_memory=False):
        """Runs a single ST iteration."""
        if resampling_method is None:
            raise ValueError("Resampling must be performed after the adaptive cycle")

        # get next lambda and decorrelation time
        prev_lambda = self.lambda_
        self.reweight()
        next_decorr = int(self.decorrelationSteps())

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

        _logger.info(f"Sampling {n_walkers} walkers at lambda = {self.lambda_} for {next_decorr} steps per walker...")

        # generate uncorrelated samples
        self.sample(
            default_decorrelation_steps=next_decorr,
            presamplers=presamplers,
            keep_walkers_in_memory=keep_walkers_in_memory
        )

        # generate transforms, if applicable
        self.generateTransforms(n_transforms_per_walker=n_transforms_per_walker)

        if self.current_trajectory_filename is not None:
            _logger.info(f"Trajectory path: \"{self.current_trajectory_filename}\"")
