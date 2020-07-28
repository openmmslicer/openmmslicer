import itertools as _it
import os as _os

from simtk.openmm.app.dcdreporter import DCDReporter as _DCDReporter


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
