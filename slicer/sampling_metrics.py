import copy as _copy

import scipy.stats as _ss


class EnergyCorrelation:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.E_before = []
        self.E_after = []

    def evaluateBefore(self):
        self.E_before = self.ensemble.calculateStateEnergies(self.ensemble.lambda_)

    def evaluateAfter(self):
        self.E_after = self.ensemble.calculateStateEnergies(self.ensemble.lambda_)

    @property
    def min(self):
        return -0.1

    @property
    def max(self):
        return 0.1

    @property
    def metric(self):
        if len(set(self.E_before)) <= 1 or len(set(self.E_after)) <= 1:
            return 0
        else:
            return _ss.pearsonr(self.E_before, self.E_after)[0]

    @property
    def terminateSampling(self):
        if len(self.E_before) and len(self.E_after):
            return self.min <= self.metric <= self.max
        else:
            return False

    def reset(self):
        self.E_before = []
        self.E_after = []


class DeltaEnergyCorrelation(EnergyCorrelation):
    def evaluateBefore(self):
        self.initial_states = _copy.copy(self.ensemble.current_states)

    def evaluateAfter(self):
        self.E_before = self.ensemble.calculateStateEnergies(self.ensemble.next_lambda_, states=self.initial_states) - \
                        self.ensemble.calculateStateEnergies(self.ensemble.lambda_, states=self.initial_states)
        self.E_after = self.ensemble.calculateStateEnergies(self.ensemble.next_lambda_) - \
                       self.ensemble.calculateStateEnergies(self.ensemble.lambda_)