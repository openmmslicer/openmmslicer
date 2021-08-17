from openmmslicer.smc import SequentialSampler
from openmmslicer.integrators import AlchemicalLangevinIntegrator
from openmmslicer.sampling_metrics import EnergyCorrelation
from openmmslicer.resampling_metrics import WorstCaseSampleSize
from openmmslicer.moves import DihedralMove
import parmed as pmd
import simtk.unit as unit
import simtk.openmm.app as app

import argparse
import logging
import os

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--protocol", choices=["split", "unified"], help="Type of alchemical protocol", required=True)
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU ID to be used")
parser.add_argument("-w", "--workdir", type=str, default="Run", help="Working directory")
args = parser.parse_args()

# set up current working directory and logging
current_dirname = args.workdir + "/"
os.makedirs(current_dirname, exist_ok=True)
logging.basicConfig(filename=current_dirname + "log.txt", filemode="a", level=logging.DEBUG)

# load the gro and top file into ParmEd
gro = "structure.gro"
top = "structure.top"
structure = pmd.load_file(top, xyz=gro)

# set MD configuration
md_config = {
    "nonbondedMethod": app.forcefield.PME,
    "nonbondedCutoff": 1.2*unit.nanometer,
    "constraints": app.HBonds,
}

# set alchemical configuration
if args.protocol == "split":
    # we switch on sterics and dihedrals before electrostatics
    alchemical_functions = {
        'lambda_sterics': 'min(1.25*lambda, 1)',
        'lambda_electrostatics': 'max(0, 5*lambda-4)',
        'lambda_torsions': 'min(1.25*lambda, 1)',
    }
    # we don't use Coulomb softcore
    alch_config = {"softcore_alpha": 0.5, "softcore_beta": 0.}
else:
    # we switch on all interactions simultaneously
    alchemical_functions = {
        'lambda_sterics': 'lambda',
        'lambda_electrostatics': 'lambda',
        'lambda_torsions': 'lambda',
    }
    # we use Coulomb softcore
    alch_config = {"softcore_alpha": 0.5, "softcore_beta": 0.5}

# set integrator
integrator = AlchemicalLangevinIntegrator(temperature=298.,
                                          collision_rate=1. / unit.picoseconds,
                                          timestep=2. * unit.femtoseconds,
                                          alchemical_functions=alchemical_functions)

# set OpenMM platform
platform = "CUDA"
properties = {'CudaDeviceIndex': str(args.gpu), 'CudaPrecision': 'mixed'}

# this corresponds to Val111 dihedral rotation
moves = [DihedralMove(structure, (1737, 1739))]

# create a simulation, minimise and run
ensemble = SequentialSampler(gro, structure, integrator, moves, platform, platform_properties=properties,
                             md_config=md_config, alch_config=alch_config)
ensemble.simulation.minimizeEnergy()
ensemble.run(n_walkers=500,
             n_conformers_per_walker=100,
             sampling_metric=EnergyCorrelation,
             resampling_metric=WorstCaseSampleSize,
             target_metric_value=100,
             target_metric_value_initial=10000,
             default_decorrelation_steps=500,
             reporter_filename=current_dirname + "simulation_{}.dcd")
