import numpy as _np
from scipy.spatial.transform import Rotation as _Rotation
from simtk import openmm as _openmm
from simtk import unit as _unit


class Move:
    @staticmethod
    def sampler(n_samples, sampling="systematic"):
        sampling = sampling.lower()

        if sampling == "systematic":
            samples = _np.random.uniform(0, 1 / n_samples) + _np.linspace(0, 1, num=n_samples, endpoint=False)
            _np.random.shuffle(samples)
        elif sampling == "multinomial":
            samples = _np.random.uniform(size=n_samples)
        else:
            raise ValueError("Only systematic and multinomial sampling are currently supported")

        return samples

    def generateMoves(self, *args, **kwargs):
        raise NotImplementedError("This is a virtual method that needs to be overloaded")

    def applyMove(self, *args, **kwargs):
        raise NotImplementedError("This is a virtual method that needs to be overloaded")


class MoveList:
    def __init__(self, moves):
        self.moves = moves

    @property
    def alchemical_atoms(self):
        return list(set().union(*[move.alchemical_atoms for move in self.moves]))

    def generateMoves(self, n_samples):
        moves = [move.generateMoves(n_samples) for move in self.moves]
        moves = list(zip(*moves))
        return moves

    def applyMove(self, input, move):
        context = None
        if isinstance(input, _openmm.Context):
            context = input
            input = input.getState(getPositions=True)
            state = input
        if isinstance(input, _openmm.State):
            input = input.getPositions(True)
        if not isinstance(input, _openmm.unit.Quantity):
            raise TypeError("Array with units, state or context expected")

        for move, single_move in zip(self.moves, move):
            input = move.applyMove(input, single_move)

        if context is not None:
            context.setPositions(input)

        return input


class TranslationMove(Move):
    def __init__(self, structure, translatable_molecule, sampling="systematic", origin=None, radius=1 * _unit.nanometer,
                 region="spherical"):
        if type(translatable_molecule) is str:
            for i, residue in enumerate(structure.residues):
                if residue.name == translatable_molecule:
                    translatable_molecule = i
                    break

        if type(translatable_molecule) is not int:
            raise TypeError("Invalid input: {}. Need either a valid residue name or residue number.".format(
                translatable_molecule))

        self.sampling = sampling
        self.movable_atoms = [a.idx for a in structure.residues[translatable_molecule]]
        self.alchemical_atoms = self.movable_atoms[:]
        self.masses = _np.asarray([a.mass for a in structure.residues[translatable_molecule]], dtype=_np.float32)

        if origin is None:
            self.origin = 0.1 * (self.masses @ structure.coordinates[self.movable_atoms]) / _np.sum(self.masses) * _unit.nanometer
        else:
            self.origin = origin

        self.radius = radius
        self.region = region

    def generateMoves(self, n_states):
        self.region = self.region.lower()

        if self.region == "cubic":
            samples = _np.transpose(_np.asarray([self.sampler(n_states, self.sampling),
                                                 self.sampler(n_states, self.sampling),
                                                 self.sampler(n_states, self.sampling)], dtype=_np.float32))
        elif self.region == "spherical":
            r = self.sampler(n_states, self.sampling) ** (1 / 3)
            cos_theta = 2 * (self.sampler(n_states, self.sampling) - 0.5)
            phi = 2 * _np.pi * self.sampler(n_states, self.sampling)

            x = r * _np.sqrt(1 - cos_theta ** 2) * _np.cos(phi)
            y = r * _np.sqrt(1 - cos_theta ** 2) * _np.sin(phi)
            z = r * cos_theta

            samples = _np.transpose(_np.asarray([x, y, z], dtype=_np.float32))
        else:
            raise ValueError("Only cubic and spherical regions are currently supported")

        translations = self.origin + self.radius * samples

        return translations

    def applyMove(self, input, translation):
        context = None
        if isinstance(input, _openmm.Context):
            context = input
            input = input.getState(getPositions=True)
            state = input
        if isinstance(input, _openmm.State):
            input = input.getPositions(True)
        if not isinstance(input, _openmm.unit.Quantity):
            raise TypeError("Array with units, state or context expected")

        coords = input[self.movable_atoms].value_in_unit(_unit.nanometer)
        centre_of_mass = self.masses @ coords / _np.sum(self.masses)
        input[self.movable_atoms] += translation - centre_of_mass * _unit.nanometer

        if context is not None:
            context.setPositions(input)

        return input


class RotationMove(Move):
    def __init__(self, structure, rotatable_molecule, sampling="systematic"):
        if type(rotatable_molecule) is str:
            for i, residue in enumerate(structure.residues):
                if residue.name == rotatable_molecule:
                    rotatable_molecule = i
                    break

        if type(rotatable_molecule) is not int:
            raise TypeError("Invalid input: {}. Need either a valid residue name or residue number.".format(
                rotatable_molecule))

        self.sampling = sampling
        self.movable_atoms = [a.idx for a in structure.residues[rotatable_molecule]]
        self.alchemical_atoms = self.movable_atoms[:]
        self.masses = _np.asarray([a.mass for a in structure.residues[rotatable_molecule]], dtype=_np.float32)

    def generateMoves(self, n_states):
        rotations = _np.zeros((n_states, 3))

        thetas = _np.arccos(2 * self.sampler(n_states, self.sampling) - 1)
        phis = 2 * _np.pi * self.sampler(n_states, self.sampling)
        psis = 2 * _np.pi * self.sampler(n_states, self.sampling)

        rotations[:, 0] = psis * _np.sin(thetas) * _np.cos(phis)
        rotations[:, 1] = psis * _np.sin(thetas) * _np.cos(thetas)
        rotations[:, 2] = psis * _np.cos(thetas)

        return rotations

    def applyMove(self, input, rotation):
        context = None
        if isinstance(input, _openmm.Context):
            context = input
            input = input.getState(getPositions=True)
            state = input
        if isinstance(input, _openmm.State):
            input = input.getPositions(True)
        if not isinstance(input, _openmm.unit.Quantity):
            raise TypeError("Array with units, state or context expected")

        coords = input[self.movable_atoms].value_in_unit(_unit.nanometer)
        rotation = _Rotation.from_rotvec(rotation)
        centre_of_mass = self.masses @ coords / _np.sum(self.masses)
        coords -= centre_of_mass
        input[self.movable_atoms] = (rotation.apply(coords)  + centre_of_mass) * input.unit

        if context is not None:
            context.setPositions(input)

        return input


class DihedralMove(Move):
    def __init__(self, structure, rotatable_bond, residue=None, sampling="systematic"):
        if residue is not None:
            if type(residue) is str:
                for i, res in enumerate(structure.residues):
                    if res.name == residue:
                        residue = i
                        break
            residue = structure.residues[residue]
            rotatable_bond = tuple([residue[i].idx for i in rotatable_bond])

        self.sampling = sampling
        self.rotatable_bond = rotatable_bond

        movable_atoms = {rotatable_bond[1]}
        new_neighbours = {rotatable_bond[1]}
        while True:
            new_neighbours = {y.idx for x in new_neighbours for y in structure.atoms[x].bond_partners} - {rotatable_bond[0]} - movable_atoms
            if len(new_neighbours):
                movable_atoms |= new_neighbours
            else:
                break
            pass
        self.movable_atoms = list(movable_atoms)
        self.alchemical_atoms = self.movable_atoms[:]

    def generateMoves(self, n_states):
        return 2 * _np.pi * self.sampler(n_states, self.sampling)

    def applyMove(self, input, rotation):
        context = None
        if isinstance(input, _openmm.Context):
            context = input
            input = input.getState(getPositions=True)
            state = input
        if isinstance(input, _openmm.State):
            input = input.getPositions(True)
        if not isinstance(input, _openmm.unit.Quantity):
            raise TypeError("Array with units, state or context expected")

        coords = input[self.movable_atoms].value_in_unit(_unit.nanometer)
        bond_vector = (input[self.rotatable_bond[0], :] - input[self.rotatable_bond[1], :]).value_in_unit(
            _unit.nanometers)
        bond_vector /= _np.linalg.norm(bond_vector)
        rotation = _Rotation.from_rotvec(bond_vector * rotation)
        origin = input[self.rotatable_bond[1], :].value_in_unit(_unit.nanometers)
        coords -= origin
        input[self.movable_atoms] = (rotation.apply(coords) + origin) * input.unit

        if context is not None:
            context.setPositions(input)

        return input
