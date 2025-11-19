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
parser.add_argument('-pt', '--platform', default="CUDA", type=str, help="GPU platform to use: either CUDA or OpenCL")
parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU ID to be used")
parser.add_argument("-l", "--leg", choices=["solv", "gas"], required=True, help="Which simulation leg to run")
parser.add_argument('-n', '--number', default=1, type=int, help="A unique ID parameter which can be used to run repeats")
args = parser.parse_args()

# set up current working directory and logging
path = os.getcwd() + f"/sampl4_001/{args.leg}/" 
current_dirname = path + f"Run_{args.number}/"
os.makedirs(current_dirname, exist_ok=True)
logging.basicConfig(filename=current_dirname + "log.txt", filemode="a", level=logging.DEBUG)

# load the gro and top file into ParmEd
gro = path + "structure.gro"
top = path + "structure.top"
solvated_ligand = pmd.load_file(top, xyz=gro)

aligned_structures = AlignedStructures("mol", solvated_ligand, gro)

solute_atoms = [atom.idx for atom in aligned_structures.main_structure.residues[0].atoms]

# set MD configuration
md_config = {
    "nonbondedMethod": app.forcefield.PME,
    "nonbondedCutoff": 1.2*unit.nanometer,
    "constraints": app.HBonds
}

# set alchemical configuration
alch_functions = {
        'lambda_sterics': LinearAlchemicalFunction(0, 0.5),
        'lambda_electrostatics': LinearAlchemicalFunction(0.5, 1)
}

alch_config = {"softcore_alpha": 0.5, "softcore_beta": 0.0}

# set integrator
integrator = LangevinIntegrator(298*unit.kelvin, 
                                1. / unit.picoseconds, 
                                2. * unit.femtoseconds)

# set OpenMM platform
if args.platform.lower() == "cuda":
    platform = "CUDA"
    properties = {'CudaDeviceIndex': str(args.gpu), 'CudaPrecision': 'mixed'}
elif args.platform.lower() == "opencl":
    platform = "OpenCL"
    properties = {'OpenCLDeviceIndex': str(args.gpu), 'OpenCLPrecision': 'mixed'}
else:
    raise ValueError(f"Unrecognised platform: {args.platform}")

# this is the move that defines the alchemical atoms (all ligand interactions are fully decoupled)
moves = [
    TemperingAtomMove(aligned_structures.main_structure, movable_atoms=[solute_atoms[i]], 
    scaling_factor=0, movable_residue="UNK") for i in range(len(solute_atoms))
]

alch_chain = AlchemicalChain(aligned_structures, [("mol", moves), ("mol", None)])

if args.leg == "solv":
    npt = True
else:
    npt = False

# These are some of the FAST configuration arguments
fast_config = {
        'n_decorr': 1,
        'n_bootstraps': 1,
        'significant_figures': 3,
        'fe_parallel': True,
        'fe_background': True,
        'fe_update_func': lambda self: 1 + 0.01 * self.effective_sample_size,
        'prot_update_func': lambda self: 1000 + 0.1 * self.ensemble.fe_estimator.effective_sample_size
}

# These are the keyword arguments corresponding to the AASMC procedure
adapt_kwargs = {
        'n_walkers': 50,
        'n_transforms_per_walker': 100,
        'sampling_metric': None,
        'resampling_metric': ExpectedSampleSize,
        'target_metric_value': .5,
        'default_decorrelation_steps': 500,
        'default_dlambda': .1, 
        'keep_walkers_in_memory': False
}

pickle_filename = current_dirname + 'checkpoint.pickle'
if not os.path.exists(pickle_filename):
    pickle_filename = None

# create a simulation, minimise and run
ensemble = FASTSampler(alch_chain, integrator, alchemical_functions=alch_functions, platform=platform,
                       platform_properties=properties, npt=npt, md_config=md_config, alch_config=alch_config,
                       checkpoint=pickle_filename, **fast_config)

if pickle_filename is None:
    reporter = MultistateDCDReporter('simulation_{}.dcd', workdir=current_dirname)
    ensemble.reporters.append(reporter)
    ensemble.simulation.minimizeEnergy()

ensemble.run(n_equilibrations=1, restrain_resnames=None, restrain_backbone=False, 
                 equilibration_steps=50000, output_interval=50000, n_walkers=1, 
                 n_transforms_per_walker=1, duration=40*unit.nanosecond,  
                 adapt_kwargs=adapt_kwargs)

# Finally write a checkpoint which dumps the whole ensemble object and can be used for free energy analysis
ensemble.writeCheckpoint(current_dirname + "checkpoint.pickle")
