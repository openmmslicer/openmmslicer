import numpy as np
from openmmtools import alchemy, integrators
import parmed as pmd
from simtk import unit, openmm

from slicer.alchemy import AbsoluteAlchemicalGaussianSoftcoreFactory as GSC


def test_GSC():
    structure = pmd.load_file("shared/butene_vac.top", xyz="shared/butene_vac.gro")
    system = structure.createSystem()
    indices = list(range(12))
    alch_region_gsc = alchemy.AlchemicalRegion(alchemical_atoms=indices, softcore_a=5 * unit.kilocalorie_per_mole,
                                               softcore_b=5, softcore_c=4, annihilate_sterics=True)
    alch_region_dum = alchemy.AlchemicalRegion(alchemical_atoms=indices, softcore_a=0 * unit.kilocalorie_per_mole,
                                               softcore_b=5, softcore_c=4, annihilate_sterics=True)
    alch_region_csc = alchemy.AlchemicalRegion(alchemical_atoms=indices, softcore_alpha=0.5, softcore_beta=0.5,
                                               annihilate_sterics=True)
    
    alch_system_gsc = GSC().create_alchemical_system(system, alch_region_gsc)
    alch_system_dum = GSC().create_alchemical_system(system, alch_region_dum)
    alch_system_csc = alchemy.AbsoluteAlchemicalFactory().create_alchemical_system(system, alch_region_csc)

    context_gsc = openmm.Context(alch_system_gsc, integrators.LangevinIntegrator())
    context_gsc.setPositions(structure.positions)
    context_dum = openmm.Context(alch_system_dum, integrators.LangevinIntegrator())
    context_dum.setPositions(structure.positions)
    context_csc = openmm.Context(alch_system_csc, integrators.LangevinIntegrator())
    context_csc.setPositions(structure.positions)

    contexts = [context_gsc, context_dum, context_csc]

    for lambda_, val in zip([0, 1], [90.29867553710938, 91.51290893554688]):
        for context in contexts:
            context.setParameter("lambda_sterics", lambda_)
        energies = [context.getState(getEnergy=True).getPotentialEnergy()._value for context in contexts]
        assert np.sum(np.isclose(energies, val)) == 3
