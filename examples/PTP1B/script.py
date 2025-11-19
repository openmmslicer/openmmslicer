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
parser.add_argument(
    "-e",
    "--enhanced",
    action="store_true",
    help="Whether to perform a free energy calculation with enhanced sampling",
)
parser.add_argument(
    "-l",
    "--leg",
    choices=["bound", "free"],
    help="Which simulation leg to run",
    required=True
)
parser.add_argument("-g", "--gpu", type=int, default=0,
                    help="GPU ID to be used")
parser.add_argument('-p', '--platform', default="CUDA", type=str, help="GPU platform to use: either CUDA or OpenCL")
parser.add_argument('-n', '--number', default=1, type=int, help="A unique ID parameter which can be used to run repeats")
args = parser.parse_args()

# set up current working directory and logging
path = os.getcwd() + f"/{args.leg}/"
current_dirname = path + f"{args.enhanced}/Run_{args.number}/"
os.makedirs(current_dirname, exist_ok=True)
logging.basicConfig(
    filename=current_dirname + "log.txt", filemode="a", level=logging.DEBUG
)

# load the gro and top files into ParmEd
structure0 = pmd.load_file(path + "LigA.top", xyz=path + "LigA.gro")
structure1 = pmd.load_file(path + "LigB.top", xyz=path + "LigB.gro")

# align the structures with a supplied maximum common substructure
mcs = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
    (17, 17),
    (18, 18),
    (19, 19),
    (20, 20),
    (21, 21),
    (22, 22),
    (23, 23),
    (24, 24),
    (25, 25),
    (26, 26),
    (27, 27),
    (29, 28),
    (30, 29),
    (31, 30),
    (32, 31),
    (33, 32),
    (34, 33),
    (35, 34),
    (36, 35),
    (37, 36),
    (38, 37),
    (39, 38),
    (40, 39),
    (41, 40),
    (42, 41),
    (43, 42),
    (44, 43),
    (45, 44),
    (46, 45),
    (47, 46),
    (48, 47),
    (49, 48),
]
aligned_structures = AlignedStructures(
    "SO2Me", structure1, path + "LigB.gro")
aligned_structures.addStructure("COMe", structure0, mcs, residue="LIG")

# set MD configuration
md_config = {
    "nonbondedMethod": app.forcefield.PME,
    "nonbondedCutoff": 1.2 * unit.nanometer,
    "constraints": app.HBonds,
}

# set alchemical configuration
alchemical_functions = {
    "lambda_bonds": LinearAlchemicalFunction(0, 0.5, full_interpolation=True),
    "lambda_angles": LinearAlchemicalFunction(0, 0.5, full_interpolation=True),
    "lambda_sterics": LinearAlchemicalFunction(0, 0.5, full_interpolation=True),
    "lambda_electrostatics": LinearAlchemicalFunction(0.5, 1, full_interpolation=True),
    "lambda_torsions": LinearAlchemicalFunction(0, 0.5, full_interpolation=True),
}
alch_config = {"softcore_alpha": 0.5, "softcore_beta": 0.0}

# set integrator
integrator = LangevinIntegrator(
    temperature=298.0,
    collision_rate=1.0 / unit.picoseconds,
    timestep=2.0 * unit.femtoseconds,
)

# set OpenMM platform
if args.platform.lower() == "cuda":
    platform = "CUDA"
    properties = {'CudaDeviceIndex': str(args.gpu), 'CudaPrecision': 'mixed'}
elif args.platform.lower() == "opencl":
    platform = "OpenCL"
    properties = {'OpenCLDeviceIndex': str(args.gpu), 'OpenCLPrecision': 'mixed'}
else:
    raise ValueError(f"Unrecognised platform: {args.platform}")

# set up the dihedral moves
moves = [
    DihedralMove(aligned_structures.main_structure,
                 (4, 8), movable_residue="LIG"),
    DihedralMove(aligned_structures.main_structure,
                 (10, 20), movable_residue="LIG"),
    DihedralMove(aligned_structures.main_structure,
                 (20, 21), movable_residue="LIG"),
    DihedralMove(aligned_structures.main_structure,
                 (24, 27), movable_residue="LIG"),
    DihedralMove(aligned_structures.main_structure,
                 (27, 30), movable_residue="LIG")
]

# set up the bond rescaling moves
if args.leg == "free":
    different_bonds = [(24, 27), (27, 29), (27, 30)]
else:
    different_bonds = [(5186, 5189), (5189, 5191), (5189, 5192)]

print("Perturbing bonds:", different_bonds)
bonded_moves = [
    BondMove(aligned_structures, (i, j), "COMe", "SO2Me")
    for i, j in different_bonds
]

# define the alchemical chain
if args.enhanced:
    alch_chain = AlchemicalChain(
        aligned_structures,
        [("COMe", moves), ("COMe", bonded_moves), ("SO2Me", bonded_moves)],
    )
else:
    alch_chain = AlchemicalChain(
        aligned_structures, [("COMe", bonded_moves), ("SO2Me", bonded_moves)]
    )

# check if a previously pickled checkpoint has been written
pickle_filename = current_dirname + "checkpoint.pickle"
if not os.path.exists(pickle_filename):
    pickle_filename = None
    
# create the FASTSampler
fast_config = {
    "n_decorr": 10,
    "n_bootstraps": 1,
    "significant_figures": 3,
    "fe_parallel": True,
    "fe_background": True,
    "fe_update_func": lambda self: 1 + 0.01 * self.effective_sample_size,
    "prot_update_func": lambda self: 1000 + 0.1 * self.ensemble.fe_estimator.effective_sample_size
}

ensemble = FASTSampler(alch_chain, integrator, alchemical_functions=alchemical_functions, platform=platform,
                       platform_properties=properties, npt=True, md_config=md_config, alch_config=alch_config,
                       checkpoint=pickle_filename, **fast_config)

# if we are not continuing the simulation, we define a DCD reporter and minimise the energy of the system
if pickle_filename is None:
    reporter = MultistateDCDReporter("simulation_{}.dcd", workdir=current_dirname)
    ensemble.reporters.append(reporter)
    ensemble.simulation.minimizeEnergy()

# these are the keyword arguments corresponding to the initial AASMC procedure
adapt_kwargs = dict(
    n_walkers=50,
    n_transforms_per_walker=100,
    sampling_metric=None,
    resampling_metric=ExpectedSampleSize,
    target_metric_value=.5,
    default_decorrelation_steps=500,
    default_dlambda=0.1,
    keep_walkers_in_memory=False,
)
# run a 160 ns FAST/MBAR simulation with 50000 steps of initial equilibration
ensemble.run(n_equilibrations=1,
             restrain_resnames=None,
             restrain_backbone=True,
             equilibration_steps=50000,
             output_interval=50000,
             n_walkers=1,
             n_transforms_per_walker=1,
             keep_walkers_in_memory=True,
             duration=160 * unit.nanosecond,
             exact_weights=False,
             adapt_kwargs=adapt_kwargs)

# finally write a checkpoint which dumps the whole ensemble object and can be used for free energy analysis
ensemble.writeCheckpoint(current_dirname + "checkpoint.pickle")
