import openmmtools.alchemy as _alchemy


# TODO: maybe have own parameters instead of using softcore a, b and c?
class AbsoluteAlchemicalGaussianSoftcoreFactory(_alchemy.AbsoluteAlchemicalFactory):
    """Uses the soft-core potential defined in: https://doi.org/10.1021/acs.jctc.0c00163"""
    def _get_sterics_energy_expressions(self, lambda_variable_suffixes):
        """Return the energy expressions for sterics.

        Parameters
        ----------
        lambda_variable_suffixes : List[str]
            A list with suffixes for the global variable "lambda_sterics" that
            will control the energy. If no suffix is necessary (i.e. there are
            no multiple alchemical regions) just set lambda_variable_suffixes[0] = ''.
            If the list has more than one element, the energy is controlled by
            the multiplication of lambda_sterics_suffix1 * lambda_sterics_suffix2.
        """

        # Sterics mixing rules.
        if lambda_variable_suffixes[0] == '':
            lambda_variable_name = 'lambda_sterics'
        else:
            if len(lambda_variable_suffixes) > 1:
                lambda_variable_name = 'lambda_sterics{0}*lambda_sterics{1}'.format(
                    lambda_variable_suffixes[0], lambda_variable_suffixes[1])
            else:
                lambda_variable_name = 'lambda_sterics{}'.format(lambda_variable_suffixes[0])

        sterics_mixing_rules = ('epsilon = sqrt(epsilon1*epsilon2);'  # Mixing rule for epsilon.
                                'sigma = 0.5*(sigma1 + sigma2);')  # Mixing rule for sigma.

        # Soft-core Lennard-Jones. 0.89089871814 = 2^-6 and is related to the minimal distance.
        exceptions_sterics_energy_expression = ('U_sterics;'
                                                'U_sterics = U_real+U_alch;'
                                                'U_real = max(2.0*{0}-1.0,0.0)*4*epsilon*x*(x-1.0);'
                                                'U_alch = (1.0-2.0*abs({0}-0.5))*softcore_a*exp(-softcore_b*y);'
                                                'y = (0.89089871814*r/sigma)^softcore_c;'
                                                'x = (sigma/r)^6;').format(lambda_variable_name)

        # Define energy expression for electrostatics.
        return sterics_mixing_rules, exceptions_sterics_energy_expression