from openmmtools.integrators import LangevinIntegrator as _LangevinIntegrator
import simtk.unit as _unit

# Energy unit used by OpenMM unit system
_OPENMM_ENERGY_UNIT = _unit.kilojoules_per_mole


class AlchemicalLangevinIntegrator(_LangevinIntegrator):
    def __init__(self, *args, alchemical_functions=None, initial_lambda=0, **kwargs):
        if not alchemical_functions:
            alchemical_functions = {
                'lambda_sterics': 'lambda',
                'lambda_electrostatics': 'lambda',
                'lambda_torsions': 'lambda',
            }

        self._alchemical_functions = alchemical_functions
        self._system_parameters = {system_parameter for system_parameter in alchemical_functions.keys()}
        self._lambda_ = initial_lambda

        super(AlchemicalLangevinIntegrator, self).__init__(*args, **kwargs)

        self.addGlobalVariable("lambda", initial_lambda)

    def _add_integrator_steps(self):
        self._add_update_alchemical_parameters_step()
        super(AlchemicalLangevinIntegrator, self)._add_integrator_steps()

    def _add_update_alchemical_parameters_step(self):
        """
        Add step to update Context parameters according to provided functions.
        """
        for context_parameter in self._alchemical_functions:
            if context_parameter in self._system_parameters:
                self.addComputeGlobal(context_parameter, self._alchemical_functions[context_parameter])



class DummyAlchemicalIntegrator(AlchemicalLangevinIntegrator):
    def __init__(self, *args, **kwargs):
        super(DummyAlchemicalIntegrator, self).__init__(*args, **kwargs)
        self.addGlobalVariable("potential", 0)

    def _add_integrator_steps(self):
        self._add_update_alchemical_parameters_step()
        self.addComputeGlobal("potential", "energy")

    def getPotentialEnergyFromLambda(self, lambda_):
        self.setGlobalVariableByName("lambda", lambda_)
        self.step(1)
        return self.getGlobalVariableByName("potential") * _OPENMM_ENERGY_UNIT