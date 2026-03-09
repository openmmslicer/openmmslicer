import copy as _copy
import itertools as _it
import os as _os

import numpy as _np
from openmm.app.dcdreporter import DCDReporter as _DCDReporter
from openmm.app.statedatareporter import StateDataReporter as _StateDataReporter

__all__ = ["MultistateDCDReporter"]


class MultistateDCDReporter:
    """
    A DCDReporter which handles multiple trajectories.

    Parameters
    ----------
    filebase : str
        The name of the filebase, to be supplied with additional curly brackets, so that str.format() can be called.
    workdir : str, optional
        The directory where all trajectory files will be written. Default is the current working directory

    Attributes
    ----------
    current_filename : str
        The filename of the current trajectory file.
    current_reporter : openmm.app.dcdreporter.DCDReporter
        The current DCDReporter.
    filename_history : list
        A list with all previously written filenames.
    prunable_filenames : list
        Temporary files which will be removed by the reporter.
    filebase : str
        The trajectory filebase.
    workdir : str
        The directory where all trajectory files are written.
    """
    def __init__(self, filebase, workdir=None):
        self.workdir = workdir
        self.filebase = filebase
        self.reset()

    @property
    def workdir(self):
        return self._workdir

    @workdir.setter
    def workdir(self, val):
        if val is None:
            val = _os.getcwd()
        if not _os.path.exists(val):
            _os.makedirs(val, exist_ok=True)
        if val[-1] != "/":
            val += "/"
        if hasattr(self, "_workdir"):
            self.filename_history = [x.replace(self._workdir, val) for x in self.filename_history]
            self.prunable_filenames = [x.replace(self._workdir, val) for x in self.prunable_filenames]
            self.current_filename = self.current_filename.replace(self._workdir, val)
            self.current_reporter._out.name.replace(self._workdir, val)
            self.current_reporter._dcd = None
        self._workdir = val

    def generateReporter(self, label, *args, append=False, **kwargs):
        """
        Creates an openmm.app.dcdreporter.DCDReporter based on a label.

        Parameters
        ----------
        label : str or int or float
            A unique label which will be used to format the filebase.
        append : bool, optional
            Whether to append to a previous trajectory file or start a new one. Default is False.
        args
            Positional arguments to be passed to openmm.app.dcdreporter.DCDReporter().
        kwargs
            Keyword arguments to be passed to openmm.app.dcdreporter.DCDReporter().
        """
        self.current_filename = self.workdir + self.filebase.format(label)

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
        """Removes all files in prunable_filenames."""
        for filename in self.prunable_filenames:
            try:
                _os.remove(filename)
            except FileNotFoundError:
                pass
        self.prunable_filenames = []
        self.filename_history = [x for x, _ in _it.groupby(self.filename_history)]

    def reset(self):
        """Resets the reporter."""
        self.current_filename = None
        self.current_reporter = None
        self.filename_history = []
        self.prunable_filenames = []

    def serialise(self):
        """Serialises the reporter."""
        new_self = _copy.copy(self)
        new_self.current_reporter = None
        return new_self


class MultistateStateDataReporter(_StateDataReporter):
    # TODO: This reporter is no longer usable after all the code refactoring and needs to be fixed.
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
