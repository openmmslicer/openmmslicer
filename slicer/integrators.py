import openmmtools.integrators as _integrators
import simtk.unit as _unit


def make_alchemical(cls):
    def __init__(self, alchemical_functions=None, initial_lambda=0, *args, **kwargs):
        if not alchemical_functions:
            alchemical_functions = {
                'lambda_sterics': 'min(lambda, 1)',
                'lambda_electrostatics': 'min(lambda, 1)',
                'lambda_torsions': 'min(lambda, 1)',
            }

        self._alchemical_functions = alchemical_functions
        super(cls, self).__init__(*args, **kwargs)
        self.addGlobalVariable("lambda", initial_lambda)

    def _add_integrator_steps(self):
        self._add_update_alchemical_parameters_step()
        super(cls, self)._add_integrator_steps()

    def _add_update_alchemical_parameters_step(self):
        for key, value in self._alchemical_functions.items():
            self.addComputeGlobal(key, value)

    cls.__init__ = __init__
    cls._add_integrator_steps = _add_integrator_steps
    cls._add_update_alchemical_parameters_step = _add_update_alchemical_parameters_step

    return cls


@make_alchemical
class AlchemicalLangevinIntegrator(_integrators.LangevinIntegrator):
    pass


@make_alchemical
class AlchemicalGeodesicBAOABIntegrator(_integrators.GeodesicBAOABIntegrator):
    pass


class AlchemicalEnergyEvaluator(AlchemicalLangevinIntegrator):
    def __init__(self, *args, **kwargs):
        super(AlchemicalEnergyEvaluator, self).__init__(*args, **kwargs)
        self.addGlobalVariable("potential", 0)

    def _add_integrator_steps(self):
        self._add_update_alchemical_parameters_step()
        self.addComputeGlobal("potential", "energy")

    def getPotentialEnergyFromLambda(self, lambda_):
        self.setGlobalVariableByName("lambda", lambda_)
        self.step(1)
        return self.getGlobalVariableByName("potential") * _unit.kilojoules_per_mole
