import scipy.stats as _ss


class EnergyCorrelation:
    """
    A sampling metric which determines the energy autocorrelation time and stops sampling after enough decorrelation.

    Parameters
    ----------
    ensemble : slicer.smc.SequentialSampler
        A sampler, containing all of the information of the system.

    Attributes
    ----------
    ensemble : slicer.smc.SequentialSampler
        The associated SMC sampler.
    E_before : [float]
        A list of all reduced initial potential energies for each walker.
    E_after : [float]
        A list of all reduced current potential energies for each walker.
    """
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.E_before = []
        self.E_after = []

    def evaluateBefore(self):
        """Evaluates the energies immediately after resampling."""
        self.E_before = self.ensemble.calculateStateEnergies(self.ensemble.lambda_)

    def evaluateAfter(self):
        """Evaluates the energies at the current timestep."""
        self.E_after = self.ensemble.calculateStateEnergies(self.ensemble.lambda_)

    @property
    def requireNextLambda(self):
        """bool: Whether a potentially expensive calculation of the next lambda value is needed."""
        return False

    @property
    def min(self):
        """float: The minimum value of the metric. Default is -0.1."""
        return -0.1

    @property
    def max(self):
        """float: The maximum value of the metric. Default is 0.1."""
        return 0.1

    @property
    def metric(self):
        """float: The energy autocorrelation coefficient."""
        if len(set(self.E_before)) <= 1 or len(set(self.E_after)) <= 1:
            return 0
        else:
            return _ss.pearsonr(self.E_before, self.E_after)[0]

    @property
    def terminateSampling(self):
        """bool: Whether the metric is within the desired bounds."""
        if len(self.E_before) and len(self.E_after):
            return self.min <= self.metric <= self.max
        else:
            return False

    def reset(self):
        """Resets the sampling metric."""
        self.E_before = []
        self.E_after = []
