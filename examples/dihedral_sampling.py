from slicer.integrators import AlchemicalLangevinIntegrator
from slicer.resampling_metrics import LogWorstCaseSampleSize
from slicer.SequentialEnsemble import SequentialEnsemble

import numpy as np
import parmed as pmd
import simtk.unit as unit
import simtk.openmm.app as app

# load the file
gro = "PATH_TO_GRO/file.gro"
top = "PATH_TO_TOP/file.top"
output_dcd_template = "PATH_TO_DCD/trajectory_{}.dcd"
structure = pmd.load_file(top, xyz=gro)

# the rotatable bonds with all atoms attached to index 2 being moved
rotatable_bonds = [(4, 8)]

md_config = {
    "nonbondedMethod": app.forcefield.PME,
    "nonbondedCutoff": 1.2*unit.nanometer,
    "constraints": app.HBonds,
}

# set up integrator, platform, system and minimise
integrator = AlchemicalLangevinIntegrator(298, 1, 0.002)
platform = "CUDA"
ensemble = SequentialEnsemble(structure, integrator, platform, rotatable_bonds=rotatable_bonds, ligname='LIG', md_config=md_config)
ensemble.simulation.minimizeEnergy()

# run simulation
ensemble.run(equilibration_steps=100000,
             n_walkers=1000,
             n_conformers_per_walker=25,
             resampling_metric=LogWorstCaseSampleSize,
             target_metric_value=np.log(100),
             target_metric_value_initial=np.log(2500),
             distribution="uniform",
             decorrelation_steps=500,
             default_dlambda=0.1,
             reporter_filename=output_dcd_template)
