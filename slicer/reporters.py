import itertools as _it
import os as _os

import numpy as _np
from simtk.openmm.app.dcdreporter import DCDReporter as _DCDReporter
from simtk.openmm.app.statedatareporter import StateDataReporter as _StateDataReporter


class MultistateDCDReporter:
    def __init__(self, filebase):
        self.filebase = _os.path.abspath(filebase)
        self.reset()

    def generateReporter(self, lambda_, *args, append=False, **kwargs):
        self.current_filename = self.filebase.format(lambda_)

        if not append and _os.path.exists(self.current_filename):
            prev_filebase, ext = _os.path.splitext(self.current_filename)
            if len(self.filename_history) and self.filename_history[-1] == self.current_filename:
                suffix = "_prev"
                prev_filename = prev_filebase + suffix + ext
                self.prunable_filenames += [prev_filename]
                self.filename_history = [prev_filename if x == self.current_filename else x
                                         for x in self.filename_history]
            else:
                suffix = "_backup"
                prev_filename = prev_filebase + suffix + ext
            _os.rename(self.current_filename, prev_filename)
        elif not _os.path.exists(self.current_filename):
            append = False

        self.current_reporter = _DCDReporter(self.current_filename, *args, append=append, **kwargs)
        self.filename_history += [self.current_filename]

        return self.current_reporter

    def prune(self):
        for filename in self.prunable_filenames:
            try:
                _os.remove(filename)
            except FileNotFoundError:
                pass
        self.prunable_filenames = []
        self.filename_history = [x for x, _ in _it.groupby(self.filename_history)]

    def reset(self):
        self.current_filename = None
        self.current_reporter = None
        self.filename_history = []
        self.prunable_filenames = []


class MultistateStateDataReporter(_StateDataReporter):
    def __init__(self, *args, current_lambda=False, logZ=False, walker_ids=False, weights=False, log_weights=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._smc_extras = {
            "Lambda": [current_lambda, None],
            "LogZ": [logZ, None],
            "Walker ID": [walker_ids, None],
            "Weight": [weights, None],
            "Log weight": [log_weights, None]
        }

    def update(self, smc_sampler, current_walker_id):
        self._smc_extras["Lambda"][1] = smc_sampler.lambda_
        self._smc_extras["LogZ"][1] = smc_sampler.logZ
        self._smc_extras["Walker ID"][1] = current_walker_id
        self._smc_extras["Weight"][1] = _np.exp(smc_sampler.log_weights[current_walker_id])
        self._smc_extras["Log weight"][1] = smc_sampler.log_weights[current_walker_id]

    def _constructHeaders(self):
        headers = []
        for attr in ["Lambda", "LogZ", "Walker ID", "Weight", "Log weight"]:
            if self._smc_extras[attr][0]:
                headers.append(attr)
        headers += super()._constructHeaders()
        return headers

    def _constructReportValues(self, simulation, state):
        values = []
        for attr in ["Lambda", "LogZ", "Walker ID", "Weight", "Log weight"]:
            if self._smc_extras[attr][0]:
                values.append(self._smc_extras[attr][1])
        values += super()._constructReportValues(simulation, state)
        return values
