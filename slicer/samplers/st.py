import inspect as _inspect
import logging as _logging
import math as _math
import random as _random

import numpy as _np
from scipy.interpolate import interp1d as _interp1d
import simtk.unit as _unit

from .smc import SMCSampler as _SMCSampler
from slicer.correlation_metrics import EffectiveDecorrelationTime as _EffectiveDecorrelationTime
import slicer.fe_estimators as _fe_estimators
import slicer.resampling_methods as _resmethods
from slicer.protocol import Protocol as _Protocol

_logger = _logging.getLogger(__name__)


class STSampler(_SMCSampler):
    # TODO: make pickle work
    # TODO: fix protocol
    _picklable_attrs = _SMCSampler._picklable_attrs + ["target_lambda", "adaptive_mode",
                                                       "sampling_history", "fe_estimator"]

    def __init__(self, *args, fe_estimator=_fe_estimators.EnsembleBAR, n_bootstraps=None, n_decorr=None,
                 fe_update_func=lambda self: 1 + 0.01 * self.effective_sample_size, fe_parallel=False,
                 significant_figures=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_lambda = 1
        self.adaptive_mode = True
        # TODO: make sampling_history part of walker_memo
        self.sampling_history = []
        self.fe_estimator = None
        self.protocol = None
        self._post_adaptive_kwargs = dict(fe_estimator=fe_estimator, n_bootstraps=n_bootstraps, n_decorr=n_decorr,
                                          fe_update_func=fe_update_func, fe_parallel=fe_parallel,
                                          significant_figures=significant_figures)

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

    @_SMCSampler.lambda_.setter
    def lambda_(self, val):
        if val is not None:
            if self._post_adaptive_kwargs["significant_figures"] is not None:
                val = round(val, self._post_adaptive_kwargs["significant_figures"])
            _SMCSampler.lambda_.fset(self, val)

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
        idx = _np.where(self.walker_memo.timestep_lambdas == 1)[0][0]
        self.protocol = _Protocol(self.walker_memo.timestep_lambdas[:idx + 1],
                                  significant_figures=self._post_adaptive_kwargs["significant_figures"])
        self.walker_memo.updateEnergies(self, self.protocol.value)

    def _initialise_fe_estimator(self):
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
                              parallel=self._post_adaptive_kwargs["fe_parallel"])
            else:
                kwargs = dict(update_func=self._post_adaptive_kwargs["fe_update_func"],
                              parallel=self._post_adaptive_kwargs["fe_parallel"])
            fe_estimator = fe_estimator(self.walker_memo, **kwargs)
            if self._post_adaptive_kwargs["n_decorr"]:
                fe_estimator.interval = _EffectiveDecorrelationTime(fe_estimator=fe_estimator,
                                                                    protocol=self.protocol)
        self.fe_estimator = fe_estimator

    def decorrelationSteps(self, lambda_=None):
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

        # get acceptance criterion
        fe_fwd = self.fe_estimator(self.lambda_, self.next_lambda)
        deltaEs_fwd = self.calculateDeltaEs(self.next_lambda, self.lambda_)
        samples_fwd = _np.sum(_np.meshgrid(fe_fwd, -deltaEs_fwd), axis=0)
        acc_fwd = _np.average(_np.exp(_np.minimum(samples_fwd, 0.)))
        _logger.debug(f"Forward probability to lambda = {self.next_lambda} is {acc_fwd} with dimensionless "
                      f"free energy = {fe_fwd}")
        
        # accept or reject move and/or swap direction
        randnum = _random.random()
        if acc_fwd != 1. and randnum >= acc_fwd:
            if not _math.isclose(self.next_lambda, self.previous_lambda):
                fe_bwd = self.fe_estimator(self.lambda_, self.previous_lambda)
                deltaEs_bwd = self.calculateDeltaEs(self.previous_lambda, self.lambda_)
                samples_bwd = _np.sum(_np.meshgrid(fe_bwd, -deltaEs_bwd), axis=0)
                acc_bwd = max(0., _np.average(_np.exp(_np.minimum(samples_bwd, 0.))) - acc_fwd)
                _logger.debug(f"Backward probability to lambda = {self.previous_lambda} is {acc_bwd} with "
                              f"dimensionless free energy = {fe_bwd}")

                if acc_bwd != 0. and randnum - acc_fwd < acc_bwd:
                    self.target_lambda = int(not self.target_lambda)
            # trigger walker update
            self.lambda_ = self.lambda_
        else:
            self.lambda_ = self.next_lambda

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
            adapt_kwargs=None,
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

        current_cycle = 0
        while True:
            dsteps = self.total_sampling_steps - initial_sampling_steps
            dt = dsteps * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
            if maximum_duration is not None and dt >= maximum_duration.in_units_of(_unit.nanosecond):
                break
            if n_cycles is not None and current_cycle >= n_cycles:
                break
            self._runRoundTrip(*args, maximum_duration=maximum_duration, **kwargs)
            current_cycle += 1

            lambdas = self.protocol.value
            fe = sum(self.fe_estimator(lambdas[i], lambdas[i + 1]) for i in range(len(lambdas) - 1))
            _logger.info("Dimensionless free energy difference (-logZ) between lambda = 0 and lambda = 1 is: {}".
                         format(fe))

        _logger.info(f"Grand total simulation time was {dt}")

    def runSingleIteration(self, *args, **kwargs):
        # TODO: restructure
        if not self.adaptive_mode:
            self._runPostAdaptiveIteration(*args, **kwargs)
        else:
            steps_before = self.total_sampling_steps
            if self.initialised:
                if "n_walkers" in kwargs.keys():
                    kwargs["n_walkers"] = max(len(self.walkers), kwargs["n_walkers"])
                else:
                    # TODO: fix
                    kwargs["n_walkers"] = len(self.walkers)
            significant_figures = self._post_adaptive_kwargs["significant_figures"]
            if significant_figures is not None:
                minimum_value = 1. / 10 ** significant_figures
                for key in ["default_dlambda", "minimum_dlambda", "maximum_dlambda", "target_lambda"]:
                    if key in kwargs.keys():
                        kwargs[key] = _np.sign(kwargs[key]) * max(minimum_value, round(abs(kwargs[key]),
                                                                                       significant_figures))
            if "fixed_lambdas" not in kwargs.keys() or kwargs["fixed_lambdas"] is None:
                kwargs["fixed_lambdas"] = []
            else:
                kwargs["fixed_lambdas"] = sorted(set(kwargs["fixed_lambdas"]))
            super().runSingleIteration(*args, **kwargs)
            self.sampling_history += [(self.total_sampling_steps - steps_before) // kwargs["n_walkers"]]

    def _runPostAdaptiveIteration(self,
                                  resampling_method=_resmethods.SystematicResampler,
                                  exact_weights=False,
                                  n_walkers=1,
                                  walker_metric=None,
                                  presamplers=None,
                                  generate_transforms=None,
                                  n_transforms_per_walker=100,
                                  keep_walkers_in_memory=False,
                                  write_checkpoint=None,
                                  load_checkpoint=None):
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
            _logger.info(f"Trajectory path: \"{self.current_trajectory_filename}\"")

    def _runRoundTrip(self, *args, maximum_duration=None, **kwargs):
        def runOnce():
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
            dt = self.total_sampling_steps * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
            if maximum_duration is not None and dt >= maximum_duration.in_units_of(_unit.nanosecond):
                break
            if self.lambda_ == initial_target_lambda:
                finished_half_cycle = True
            if finished_half_cycle and self.lambda_ == int(not initial_target_lambda):
                d_steps = self.total_sampling_steps - initial_sampling_steps
                time = d_steps * self.integrator.getStepSize().in_units_of(_unit.nanosecond)
                _logger.info(f"Round trip from lambda = {initial_lambda_} to lambda = {self.lambda_} finished after "
                             f"{time} of cumulative sampling")
                break
