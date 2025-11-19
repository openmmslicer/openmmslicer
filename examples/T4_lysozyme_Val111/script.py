import os
import argparse
import logging

import parmed as pmd
import openmm.unit as unit
import openmm.app as app

from openmmtools.integrators import LangevinIntegrator
from openmmslicer.alchemy import *
from openmmslicer.samplers import *
from openmmslicer.reporters import *
from openmmslicer.resampling_metrics import *
from openmmslicer.sampling_metrics import *
from openmmslicer.moves import *

# set up parser
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--protocol", choices=["split", "unified"], help="Type of alchemical protocol", required=True)
parser.add_argument('-pt', '--platform', default="CUDA", type=str, help="GPU platform to use: either CUDA or OpenCL")
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

aligned_structures = AlignedStructures("mol", structure, gro)

solute_atoms = [atom.idx for atom in aligned_structures.main_structure.residues[0].atoms]

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
        'lambda_sterics': lambda x: min(1.25 * x, 1.),
        'lambda_electrostatics': lambda x: max(0., 5. * x - 4.),
        'lambda_torsions': lambda x: min(1.25 * x, 1.),
    }
    # we don't use Coulomb softcore
    alch_config = {"softcore_alpha": 0.5, "softcore_beta": 0.}
else:
    # we switch on all interactions simultaneously
    alchemical_functions = {
        'lambda_sterics': lambda x: x,
        'lambda_electrostatics': lambda x: x,
        'lambda_torsions': lambda x: x,
    }
    # we use Coulomb softcore
    alch_config = {"softcore_alpha": 0.5, "softcore_beta": 0.5}

# set integrator
integrator = LangevinIntegrator(temperature=298.,
                                collision_rate=1. / unit.picoseconds,
                                timestep=2. * unit.femtoseconds)

# set OpenMM platform
if args.platform.lower() == "cuda":
    platform = "CUDA"
    properties = {'CudaDeviceIndex': str(args.gpu), 'CudaPrecision': 'mixed'}
elif args.platform.lower() == "opencl":
    platform = "OpenCL"
    properties = {'OpenCLDeviceIndex': str(args.gpu), 'OpenCLPrecision': 'mixed'}
else:
    raise ValueError(f"Unrecognised platform: {args.platform}")

# this corresponds to Val111 dihedral rotation
moves = [DihedralMove(structure, (1737, 1739))]

alch_chain =  AlchemicalChain(aligned_structures, [("mol", moves), ("mol", None)])

npt = True

pickle_filename = current_dirname + "checkpoint.pickle"
if not os.path.exists(pickle_filename):
    pickle_filename = None

# create a simulation, minimise and run
ensemble = SMCSampler(alch_chain, integrator, alchemical_functions=alchemical_functions, platform=platform,
                       platform_properties=properties, npt=npt, md_config=md_config, alch_config=alch_config,
                       checkpoint=pickle_filename)

# If we are not continuing the simulation, we define a DCD reporter and minimise the energy of the system
if pickle_filename is None:
    reporter = MultistateDCDReporter("simulation_{}.dcd", workdir=current_dirname)
    ensemble.reporters.append(reporter)
    ensemble.simulation.minimizeEnergy()

# These are the keyword arguments corresponding to the AASMC procedure
ensemble.run(n_equilibrations=1,
             n_walkers=500,
             n_transforms_per_walker=100,
             sampling_metric=EnergyCorrelation,
             resampling_metric=WorstCaseSampleSize,
             target_metric_value=100,
             default_decorrelation_steps=500)

# Finally write a checkpoint which dumps the whole ensemble object and can be used for free energy analysis
ensemble.writeCheckpoint(current_dirname + "checkpoint.pickle")

