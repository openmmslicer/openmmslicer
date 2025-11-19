from collections import defaultdict as _defaultdict
import copy as _copy
import itertools as _it
import logging as _logging

from cached_property import cached_property as _cached_property
import numpy as _np
from openmmtools.constants import ONE_4PI_EPS0 as _ONE_4PI_EPS0
import openmm as _openmm
import openmm.unit as _unit

_logger = _logging.getLogger(__name__)


# This class has been copied from a pre-release version of perses: https://github.com/choderalab/perses
# TODO: remove and use stable API when perses is officially released


class HybridTopologyFactory:
    def __init__(self, system1, system2,
                 softcore_alpha=0.5, softcore_a=1, softcore_b=1, softcore_c=6,
                 softcore_beta=0., softcore_d=1, softcore_e=1, softcore_f=2,
                 disable_alchemical_dispersion_correction=False):
        """
        Creates a hybrid OpenMM system based on two aligned ParmEd structures.

        Parameters
        ----------
        system1 : parmed.Structure
            The first system.
        system2 : parmed.Structure
            The second system
        kwargs
            Keyword arguments used to instantiate various attributes.

        Attributes
        ----------
        system1 : parmed.Structure
            The first system.
        system2 : parmed.Structure
            The second system
        softcore_alpha : float, optional
            Alchemical softcore parameter for Lennard-Jones (default is 0.5).
        softcore_a, softcore_b, softcore_c : float
            Parameters modifying softcore Lennard-Jones form. Introduced in
            Eq. 13 of Ref. [TTPham-JChemPhys135-2011]_.
        softcore_beta : float, optional
            Alchemical softcore parameter for electrostatics. Set this to zero
            to recover standard electrostatic scaling (default is 0.0).
        softcore_d, softcore_e, softcore_f : float
            Parameters modifying softcore electrostatics form .
        disable_alchemical_dispersion_correction : bool
            Whether to disable the alchemical dispersion correction. Note that this will not in general be updated
            at different lambda values, so it needs to be calculated by other means.
        hybrid_system : openmm.System
            The hybrid OpenMM system.
        """
        # TODO: add more parameters
        self.softcore_alpha = softcore_alpha
        self.softcore_a = softcore_a
        self.softcore_b = softcore_b
        self.softcore_c = softcore_c
        self.softcore_beta = softcore_beta
        self.softcore_d = softcore_d
        self.softcore_e = softcore_e
        self.softcore_f = softcore_f
        self.disable_alchemical_dispersion_correction = disable_alchemical_dispersion_correction
        self.system1 = _copy.deepcopy(system1)
        self.system2 = _copy.deepcopy(system2)

    @property
    def use_sterics_softcore(self):
        return (self.softcore_alpha, self.softcore_a) != (0, 1)

    @property
    def use_electrostatics_softcore(self):
        return (self.softcore_beta, self.softcore_d) != (0, 1)

    @staticmethod
    def _check_constraints(system1, system2):
        system1_constraints = [system1.getConstraintParameters(i) for i in range(system1.getNumConstraints())]
        system1_dict = {frozenset({i, j}): distance for i, j, distance in system1_constraints}
        system2_constraints = [system2.getConstraintParameters(i) for i in range(system2.getNumConstraints())]
        system2_dict = {frozenset({i, j}): distance for i, j, distance in system2_constraints}
        assert system1_dict == system2_dict, "The constraints must be the same in both systems"

    @staticmethod
    def _check_nonbonded_forces(force1, force2):
        attr_list = ["getCutoffDistance", "getEwaldErrorTolerance", "getLJPMEParameters", "getNonbondedMethod",
                     "getPMEParameters", "getReactionFieldDielectric", "getSwitchingDistance",
                     "getUseDispersionCorrection", "getUseSwitchingFunction"]
        for attr in attr_list:
            assert getattr(force1, attr)() == getattr(force2, attr)(), "Incompatible nonbonded forces"

    @staticmethod
    def _index_force(force, mode):
        map_dict = {"Bond": 2, "Angle": 3, "Torsion": 4, "Exception": 2}
        assert mode in map_dict, "Unsupported mode"
        getparamfunc = f"get{mode}Parameters"
        n_terms = getattr(force, f"getNum{mode}s")()
        n_atoms = map_dict[mode]
        indices = _defaultdict(list)
        for i in range(n_terms):
            key = tuple(getattr(force, getparamfunc)(i)[:n_atoms])
            key = key if key[0] < key[-1] else key[::-1]
            indices[key] += [i]
        indices.default_factory = None
        return indices

    @staticmethod
    def _element_map(container1, container2):
        common_keys = set(container1) & set(container2)
        elements = [(container1[key], container2[key]) for key in common_keys] + \
                   [(container1[key], [None]) for key in set(container1.keys()) - common_keys] + \
                   [([None], container2[key]) for key in set(container2.keys()) - common_keys]
        return elements

    def _merge_bonded(self, shared_elements, force1, force2, common_force, custom_force, mode):
        def zero(parameters):
            return parameters[:-1] + [0 * parameters[-1]]

        map_dict = {"Bond": 2, "Angle": 3, "Torsion": 4}
        assert mode in map_dict, "Unsupported mode"
        getfunc = f"get{mode}Parameters"
        addfunc = f"add{mode}"
        n_atoms = map_dict[mode]

        # only add the alchemical bonded interactions to the custom force
        for indices1, indices2 in shared_elements:
            for index1, index2 in _it.zip_longest(indices1, indices2):
                if index1 is not None:
                    result1 = getattr(force1, getfunc)(index1)
                    atoms1, parameters1 = result1[:n_atoms], result1[n_atoms:]
                if index2 is not None:
                    result2 = getattr(force2, getfunc)(index2)
                    atoms2, parameters2 = result2[:n_atoms], result2[n_atoms:]
                assert index1 is None or index2 is None or atoms1 in [atoms2, atoms2[::-1]], "Mismatch in atom indices"
                if index1 is None:
                    getattr(custom_force, addfunc)(*atoms2, [*zero(parameters2), *parameters2])
                elif index2 is None:
                    getattr(custom_force, addfunc)(*atoms1, [*parameters1, *zero(parameters1)])
                elif parameters1 != parameters2:
                    getattr(custom_force, addfunc)(*atoms1, [*parameters1, *parameters2])
                else:
                    getattr(common_force, addfunc)(*atoms1, *parameters1)

        return common_force, custom_force

    def _merge_nonbonded(self, total_exceptions, force1, force2, custom_sterics_force, custom_electrostatics_force,
                         custom_exceptions_force):
        def generate_exception(i, j, force):
            charge_i, sigma_i, epsilon_i = force.getParticleParameters(i)
            charge_j, sigma_j, epsilon_j = force.getParticleParameters(j)
            return charge_i * charge_j, 0.5 * (sigma_i + sigma_j), (epsilon_i * epsilon_j) ** 0.5

        common_force = _copy.deepcopy(force1)
        if self.disable_alchemical_dispersion_correction:
            common_force.setUseDispersionCorrection(False)
        n_particles1 = force1.getNumParticles()
        n_particles2 = force2.getNumParticles()
        assert n_particles1 == n_particles2, "Mismatch in atom indices"
        all_indices = list(range(n_particles1))
        any_custom_sterics, any_custom_electrostatics = False, False

        common_indices, custom_sterics_indices, custom_electrostatics_indices = set(), set(), set()
        added_sterics, added_electrostatics = False, False
        for index in all_indices:
            # get the relevant parameters
            charge1, sigma1, epsilon1 = force1.getParticleParameters(index)
            charge2, sigma2, epsilon2 = force2.getParticleParameters(index)

            # now we handle all alchemical atoms
            custom_sterics, custom_electrostatics = False, False
            if (charge1, sigma1, epsilon1) != (charge2, sigma2, epsilon2):
                is_dummy = not (epsilon1 and epsilon2)
                # we either add parameter offset or move the parameter to custom_sterics_force which has softcore
                if (sigma1, epsilon1) != (sigma2, epsilon2):
                    if not self.use_sterics_softcore or not is_dummy:
                        if not added_sterics:
                            common_force.addGlobalParameter("lambda_sterics", 0.)
                            added_sterics = True
                        common_force.addParticleParameterOffset("lambda_sterics", index, 0., sigma2 - sigma1,
                                                                epsilon2 - epsilon1)
                        common_indices |= {index}
                    else:
                        custom_sterics, any_custom_sterics = True, True
                        custom_sterics_indices |= {index}

                # we either add parameter offset or move the parameter to custom_electrostatics_force which has softcore
                if charge1 != charge2:
                    if not self.use_electrostatics_softcore or not is_dummy:
                        if not added_electrostatics:
                            common_force.addGlobalParameter("lambda_electrostatics", 0.)
                            added_electrostatics = True
                        common_force.addParticleParameterOffset("lambda_electrostatics", index, charge2 - charge1,
                                                                0., 0.)
                        common_indices |= {index}
                    else:
                        custom_electrostatics, any_custom_electrostatics = True, True
                        custom_electrostatics_indices |= {index}

                common_force.setParticleParameters(index, charge1 * (not custom_electrostatics), sigma1,
                                                   epsilon1 * (not custom_sterics))

            # add all atoms to the custom forces
            custom_sterics_force.addParticle([sigma1, epsilon1, sigma2, epsilon2])
            custom_electrostatics_force.addParticle([sigma1, charge1, sigma2, charge2])

        alchemical_indices = common_indices | custom_sterics_indices | custom_electrostatics_indices
        # add interactions to the custom forces
        if any_custom_sterics:
            custom_sterics_force.addInteractionGroup(sorted(custom_sterics_indices), all_indices)
        if any_custom_electrostatics:
            custom_electrostatics_force.addInteractionGroup(sorted(custom_electrostatics_indices), all_indices)

        # add exceptions implemented as custom bonded force
        for [index1], [index2] in total_exceptions:
            # get the relevant exceptions
            if index1 is not None:
                i1, j1, charge_prod1, sigma1, epsilon1 = force1.getExceptionParameters(index1)

            if index2 is not None:
                i2, j2, charge_prod2, sigma2, epsilon2 = force2.getExceptionParameters(index2)

            if index1 is None:
                i1, j1 = i2, j2
                charge_prod1, sigma1, epsilon1 = generate_exception(i1, j1, force1)
                index1 = common_force.addException(i1, j1, charge_prod1, sigma1, epsilon1)

            if index2 is None:
                i2, j2 = i1, j1
                charge_prod2, sigma2, epsilon2 = generate_exception(i2, j2, force2)

            assert {i1, j1} == {i2, j2}, "Mismatch in atom indices"

            # add them as exclusions to the custom potentials
            if any_custom_sterics:
                custom_sterics_force.addExclusion(i1, j1)
            if any_custom_electrostatics:
                custom_electrostatics_force.addExclusion(i1, j1)

            # now we handle all exceptions involving at least one alchemical atom
            custom_sterics, custom_electrostatics = False, False
            if {i1, j1} & alchemical_indices:
                is_dummy = not (epsilon1 and epsilon2)
                # try to handle the steric part of the exception in the original nonbonded force
                if (sigma1, epsilon1) != (sigma2, epsilon2):
                    if not self.use_sterics_softcore or not is_dummy:
                        common_force.addExceptionParameterOffset(
                            "lambda_sterics", index1, 0., sigma2 - sigma1, epsilon2 - epsilon1)
                    else:
                        custom_sterics = True

                # try to handle the electrostatic part of the exception in the original nonbonded force
                if charge_prod1 != charge_prod2:
                    if not self.use_electrostatics_softcore or not is_dummy:
                        common_force.addExceptionParameterOffset(
                            "lambda_electrostatics", index1, charge_prod2 - charge_prod1, 0., 0.)
                    else:
                        custom_electrostatics = True

                # reset the terms which use softcore - we will handle these using a custom bond
                common_force.setExceptionParameters(index1, i1, j1,
                                                    charge_prod1 * (not custom_electrostatics), sigma1,
                                                    epsilon1 * (not custom_sterics))

                # we handle softcore terms using a custom bond
                if custom_sterics or custom_electrostatics:
                    epsilon1 *= custom_sterics
                    epsilon2 *= custom_sterics
                    charge_prod1 *= custom_electrostatics
                    charge_prod2 *= custom_electrostatics
                    custom_exceptions_force.addBond(
                        i1, j1, [charge_prod1, sigma1, epsilon1, charge_prod2, sigma2, epsilon2])
            elif (charge_prod1, sigma1, epsilon1) != (charge_prod2, sigma2, epsilon2):
                raise ValueError("Mismatching exceptions in the non-perturbed part of the molecule")

        masks = [True, any_custom_sterics, any_custom_electrostatics, any_custom_sterics or any_custom_electrostatics]
        forces = [common_force, custom_sterics_force, custom_electrostatics_force, custom_exceptions_force]
        return tuple(force for force, mask in zip(forces, masks) if mask)

    def _add_bonded(self, force1, force2, system, index_to_remove, mode):
        map_dict = {"Bond": self._alchemical_harmonic_bond_force,
                    "Angle": self._alchemical_harmonic_angle_force,
                    "Torsion": self._alchemical_periodic_torsion_force}
        assert mode in map_dict, "Unsupported mode"
        terms1 = self._index_force(force1, mode)
        terms2 = self._index_force(force2, mode)
        total_terms = self._element_map(terms1, terms2)
        common_force = getattr(_openmm, force1.__class__.__name__)()
        custom_force = map_dict[mode]()
        forces_to_add = self._merge_bonded(total_terms, force1, force2, common_force, custom_force, mode)
        system.removeForce(index_to_remove)
        for force in forces_to_add:
            system.addForce(force)

    def _add_nonbonded(self, force, force1, force2, system, index_to_remove):
        self._check_nonbonded_forces(force1, force2)
        exceptions1 = self._index_force(force1, "Exception")
        exceptions2 = self._index_force(force2, "Exception")
        total_exceptions = self._element_map(exceptions1, exceptions2)
        sterics_force, electrostatics_force, exceptions_force = self._alchemical_nonbonded_force(force)
        forces_to_add = self._merge_nonbonded(total_exceptions, force1, force2, sterics_force, electrostatics_force,
                                              exceptions_force)
        system.removeForce(index_to_remove)
        for force in forces_to_add:
            system.addForce(force)

    def _alchemical_harmonic_bond_force(self):
        energy_expression = """
        (K/2)*(r-length)^2;
        K = (1-lambda_bonds)*K1 + lambda_bonds*K2;
        length = (1-lambda_bonds)*length1 + lambda_bonds*length2;
        """
        custom_force = _openmm.CustomBondForce(energy_expression)
        custom_force.addGlobalParameter('lambda_bonds', 0.)
        for parameter in ["length1", "K1", "length2", "K2"]:
            custom_force.addPerBondParameter(parameter)
        return custom_force

    def _alchemical_harmonic_angle_force(self):
        energy_expression = """
        (K/2)*(theta-theta0)^2;
        K = (1.0-lambda_angles)*K1 + lambda_angles*K2;
        theta0 = (1.0-lambda_angles)*theta0_1 + lambda_angles*theta0_2;
        """
        custom_force = _openmm.CustomAngleForce(energy_expression)
        custom_force.addGlobalParameter('lambda_angles', 0.)
        for parameter in ["theta0_1", "K1", "theta0_2", "K2"]:
            custom_force.addPerAngleParameter(parameter)
        return custom_force

    def _alchemical_periodic_torsion_force(self):
        energy_expression = """
        (1-lambda_torsions)*U1 + lambda_torsions*U2;
        U1 = K1*(1+cos(periodicity1*theta-phase1));
        U2 = K2*(1+cos(periodicity2*theta-phase2));
        """
        custom_force = _openmm.CustomTorsionForce(energy_expression)
        custom_force.addGlobalParameter('lambda_torsions', 0.)
        for parameter in ["periodicity1", "phase1", "K1", "periodicity2", "phase2", "K2"]:
            custom_force.addPerTorsionParameter(parameter)
        return custom_force

    def _alchemical_sterics_common(self):
        energy_expression = f"""
        reff_stericsA = sigma*(softcore_alpha*lambda_sterics^softcore_b + (r/sigma)^softcore_c)^(1/softcore_c);
        reff_stericsB = sigma*(softcore_alpha*(1-lambda_sterics)^softcore_b + (r/sigma)^softcore_c)^(1/softcore_c);
        sigma = (1-lambda_sterics)*sigmaA + lambda_sterics*sigmaB;
        softcore_alpha = {self.softcore_alpha};
        softcore_a = {self.softcore_a};
        softcore_b = {self.softcore_b};
        softcore_c = {self.softcore_c};
        """
        return energy_expression

    def _alchemical_electrostatics_common(self):
        energy_expression = f"""
        reff_electrostaticsA = sigma*(softcore_beta*lambda_electrostatics^softcore_e + (r/sigma)^softcore_f)^(1/softcore_f);
        reff_electrostaticsB = sigma*(softcore_beta*(1-lambda_electrostatics)^softcore_e + (r/sigma)^softcore_f)^(1/softcore_f);
        sigma = (1-lambda_electrostatics)*sigmaA + lambda_electrostatics*sigmaB;
        chargeprod = (1-lambda_electrostatics)*chargeprodA + lambda_electrostatics*chargeprodB;
        softcore_beta = {self.softcore_beta};
        softcore_d = {self.softcore_d};
        softcore_e = {self.softcore_e};
        softcore_f = {self.softcore_f};
        ONE_4PI_EPS0 = {_ONE_4PI_EPS0};
        """
        return energy_expression

    def _alchemical_sterics_mixing(self):
        energy_expression = """
        epsilonA = sqrt(epsilonA1*epsilonA2);
        epsilonB = sqrt(epsilonB1*epsilonB2);
        sigmaA = 0.5*(sigmaA1 + sigmaA2);
        sigmaB = 0.5*(sigmaB1 + sigmaB2);
        """
        return energy_expression

    def _alchemical_electrostatics_mixing(self):
        energy_expression = """
        sigmaA = 0.5*(sigmaA1 + sigmaA2);
        sigmaB = 0.5*(sigmaB1 + sigmaB2);
        chargeprodA = chargeA1*chargeA2;
        chargeprodB = chargeB1*chargeB2;
        """
        return energy_expression

    def _alchemical_sterics(self):
        energy_expression = """
        U_sterics = (1-lambda_sterics)^softcore_a*U_stericsA + lambda_sterics^softcore_a*U_stericsB;
        U_stericsA = 4*epsilonA*xA*(xA-1.0); xA = (sigma/reff_stericsA)^6;
        U_stericsB = 4*epsilonB*xB*(xB-1.0); xB = (sigma/reff_stericsB)^6;
        """
        return energy_expression

    def _alchemical_electrostatics_nocutoff(self):
        energy_expression = """
        U_electrostatics = ONE_4PI_EPS0*((1-lambda_electrostatics)^softcore_d*U_electrostaticsA + lambda_electrostatics^softcore_d*U_electrostaticsB); 
        U_electrostaticsA = chargeprodA/reff_electrostaticsA;
        U_electrostaticsB = chargeprodB/reff_electrostaticsB;
        """
        return energy_expression

    def _alchemical_electrostatics_cutoff(self, force):
        epsilon_solvent = force.getReactionFieldDielectric()
        r_cutoff = force.getCutoffDistance()
        k_rf = r_cutoff ** (-3) * ((epsilon_solvent - 1) / (2 * epsilon_solvent + 1))
        c_rf = r_cutoff ** (-1) * ((3 * epsilon_solvent) / (2 * epsilon_solvent + 1))
        energy_expression = f"""
        U_electrostatics = ONE_4PI_EPS0*((1-lambda_electrostatics)^softcore_d*U_electrostaticsA + lambda_electrostatics^softcore_d*U_electrostaticsB); 
        U_electrostaticsA = chargeprodA*(reff_electrostaticsA^(-1) +  k_rf*reff_electrostaticsA^2 - c_rf);
        U_electrostaticsB = chargeprodB*(reff_electrostaticsB^(-1) +  k_rf*reff_electrostaticsB^2 - c_rf);
        k_rf = {k_rf.value_in_unit_system(_unit.md_unit_system)};
        c_rf = {c_rf.value_in_unit_system(_unit.md_unit_system)};
        """
        return energy_expression

    def _alchemical_electrostatics_ewald(self, force):
        alpha_ewald, nx, ny, nz = force.getPMEParameters()
        if alpha_ewald == 0.:
            # If alpha is 0., alpha_ewald is computed by OpenMM from from the error tolerance.
            delta = force.getEwaldErrorTolerance()
            r_cutoff = force.getCutoffDistance()
            alpha_ewald = _np.sqrt(-_np.log(2 * delta)) / r_cutoff
        energy_expression = f"""
        U_electrostatics = ONE_4PI_EPS0*((1-lambda_electrostatics)^softcore_d*U_electrostaticsA + lambda_electrostatics^softcore_d*U_electrostaticsB); 
        U_electrostaticsA = chargeprodA*erfc(alpha_ewald*reff_electrostaticsA)/reff_electrostaticsA;
        U_electrostaticsB = chargeprodB*erfc(alpha_ewald*reff_electrostaticsB)/reff_electrostaticsB;
        alpha_ewald = {alpha_ewald.value_in_unit_system(_unit.md_unit_system)};
        """
        return energy_expression

    def _alchemical_nonbonded_force(self, force):
        # select functional form based on nonbonded method.
        method = force.getNonbondedMethod()
        sterics_energy_expression = self._alchemical_sterics()
        if method in [_openmm.NonbondedForce.NoCutoff]:
            electrostatics_energy_expression = self._alchemical_electrostatics_nocutoff()
        elif method in [_openmm.NonbondedForce.CutoffPeriodic, _openmm.NonbondedForce.CutoffNonPeriodic]:
            electrostatics_energy_expression = self._alchemical_electrostatics_cutoff(force)
        elif method in [_openmm.NonbondedForce.PME, _openmm.NonbondedForce.Ewald]:
            electrostatics_energy_expression = self._alchemical_electrostatics_ewald(force)
        else:
            raise ValueError(f"Nonbonded method {str(method)} not supported yet")

        # add common alchemical terms
        sterics_energy_expression += self._alchemical_sterics_common()
        electrostatics_energy_expression += self._alchemical_electrostatics_common()

        # create sterics force
        alchemical_sterics_force = _openmm.CustomNonbondedForce("U_sterics;" + sterics_energy_expression +
                                                                self._alchemical_sterics_mixing())
        alchemical_sterics_force.addGlobalParameter("lambda_sterics", 0.)
        for parameter in ["sigmaA", "epsilonA", "sigmaB", "epsilonB"]:
            alchemical_sterics_force.addPerParticleParameter(parameter)

        # create electrostatics force
        alchemical_electrostatics_force = _openmm.CustomNonbondedForce("U_electrostatics;" +
                                                                       electrostatics_energy_expression +
                                                                       self._alchemical_electrostatics_mixing())
        alchemical_electrostatics_force.addGlobalParameter("lambda_electrostatics", 0.)
        for parameter in ["sigmaA", "chargeA", "sigmaB", "chargeB"]:
            alchemical_electrostatics_force.addPerParticleParameter(parameter)

        # create exception force
        exception_energy_expression = "U_sterics + U_electrostatics;" + self._alchemical_sterics() + \
                                      self._alchemical_sterics_common() + self._alchemical_electrostatics_nocutoff() + \
                                      self._alchemical_electrostatics_common()
        alchemical_exception_force = _openmm.CustomBondForce(exception_energy_expression)
        alchemical_exception_force.addGlobalParameter("lambda_electrostatics", 0.)
        alchemical_exception_force.addGlobalParameter("lambda_sterics", 0.)
        for parameter in ["chargeprodA", "sigmaA", "epsilonA", "chargeprodB", "sigmaB", "epsilonB"]:
            alchemical_exception_force.addPerBondParameter(parameter)

        # set some parameters
        attrs = {
            "getCutoffDistance": "setCutoffDistance",
            "getSwitchingDistance": "setSwitchingDistance",
            "getUseSwitchingFunction": "setUseSwitchingFunction"
        }
        if method in [_openmm.NonbondedForce.PME, _openmm.NonbondedForce.Ewald]:
            method = _openmm.NonbondedForce.CutoffPeriodic
        for nonbonded_force in [alchemical_sterics_force, alchemical_electrostatics_force]:
            for get, set in attrs.items():
                getattr(nonbonded_force, set)(getattr(force, get)())
            nonbonded_force.setNonbondedMethod(method)
            nonbonded_force.setUseLongRangeCorrection(False)

        return alchemical_sterics_force, alchemical_electrostatics_force, alchemical_exception_force

    @_cached_property
    def hybrid_system(self):
        system = _copy.deepcopy(self.system1)
        self._check_constraints(system, self.system2)

        forces = [system.getForce(index) for index in range(system.getNumForces())]
        forces1 = {self.system1.getForce(index).__class__.__name__: self.system1.getForce(index) for index in
                   range(self.system1.getNumForces())}
        forces2 = {self.system2.getForce(index).__class__.__name__: self.system2.getForce(index) for index in
                   range(self.system2.getNumForces())}

        for force in forces:
            force_name = force.__class__.__name__
            force1 = forces1[force_name]
            force2 = forces2[force_name]
            force_to_idx = {system.getForce(index).__class__.__name__: index for index in range(system.getNumForces())}
            if force_name == 'HarmonicBondForce':
                self._add_bonded(force1, force2, system, force_to_idx[force_name], "Bond")
            elif force_name == 'HarmonicAngleForce':
                self._add_bonded(force1, force2, system, force_to_idx[force_name], "Angle")
            elif force_name == 'PeriodicTorsionForce':
                self._add_bonded(force1, force2, system, force_to_idx[force_name], "Torsion")
            elif force_name == 'NonbondedForce':
                self._add_nonbonded(force, force1, force2, system, force_to_idx[force_name])

        return system
