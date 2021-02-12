import warnings as _warnings

import numpy as _np
from scipy.spatial.transform import Rotation as _Rotation
from simtk import openmm as _openmm
from simtk import unit as _unit

import slicer.resampling_methods as _resmethods


class Move:
    @staticmethod
    def sampler(n_samples, sampling="systematic"):
        """
        Samples from the uniform distribution between 0 and 1 using different sampling methods.

        Parameters
        ----------
        n_samples : int
            Number of uniform samples to be generated.
        sampling : str
            One of "multinomial" - independent sampling of each sample and "systematic" - sampling on an evenly
            spaced grid with a randomly generated offset.

        Returns
        -------
        samples : numpy.ndarray
            An array of all generated samples.
        """
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
        """
        Generates an array of moves which can be read by applyMove(). This is a virtual function which must be
        overloaded in the child classes.
        """
        raise NotImplementedError("This is a virtual method that needs to be overloaded")

    def applyMove(self, *args, **kwargs):
        """
        Applies a move generated from generateMoves(). This is a virtual function which must be overloaded in
        the child classes.
        """
        raise NotImplementedError("This is a virtual method that needs to be overloaded")

    @staticmethod
    def _contextToQuantity(input, return_pbc=False):
        """
        A helper function which converts a number of input formats to standardised objects to be passed to applyMove().

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input which contains coordinate information with units.

        Returns
        -------
        context : openmm.Context or None
            The context, if applicable.
        input : openmm.unit.Quantity
            An array of coordinates with units.
        """
        context = None
        if isinstance(input, _openmm.Context):
            context = input
            input = input.getState(getPositions=True)
        if return_pbc:
            if isinstance(input, _openmm.State):
                pbc = input.getPeriodicBoxVectors(True)
            else:
                raise TypeError("Need to pass a valid context or state in order to obtain periodic box vector "
                                "information")

        if isinstance(input, _openmm.State):
            input = input.getPositions(True)
        if not isinstance(input, _openmm.unit.Quantity):
            raise TypeError("Array with units, state or context expected")

        if return_pbc:
            return context, input, pbc
        else:
            return context, input


class MoveList(Move):
    """
    A class which combines multiple moves.

    Parameters
    ----------
    moves : [Move]
        The instantiated moves to be combined.

    Attributes
    ----------
    moves : [Move]
        A list of moves.
    """
    def __init__(self, moves):
        self.moves = moves

    @property
    def alchemical_atoms(self):
        """[int]: Returns the combined alchemical atoms of all moves."""
        return list(set().union(*[move.alchemical_atoms for move in self.moves]))

    def generateMoves(self, n_states):
        """
        Generates a list of transformations for each move.

        Parameters
        ----------
        n_states : int
            Number of samples to be generated.

        Returns
        -------
        transformations : list
            A list of all transformations.
        """
        transformations = [move.generateMoves(n_states) for move in self.moves]
        transformations = list(zip(*transformations))
        return transformations

    def applyMove(self, input, transformations):
        """
        Applies a single transformation to a state, context or quantity. If a context is passed, it it also modified
        in place.

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input containing coordinate information.
        transformations : list
            A list of all transformations to be passed to self.moves. Each transformation must be of an appropriate
            format for the underlying move.

        Returns
        -------
        input : openmm.unit.Quantity
            The transformed coordinates.
        """
        context, input = self._contextToQuantity(input)
        failed_moves = []

        for move, transformation in zip(self.moves, transformations):
            try:
                input = move.applyMove(input, transformation)
            except:
                if context is None:
                    raise TypeError("Need to pass a valid context or state in order to obtain periodic box vector "
                                    "information")
                failed_moves += [(move, transformation)]

        for move, transformation in failed_moves:
            context.setPositions(input)
            input = move.applyMove(context, transformation)

        if context is not None:
            context.setPositions(input)

        return input


class TranslationMove(Move):
    """
    A class which performs translation around one or more centres.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    translatable_molecule : str or int
        If string, this is the first residue which has this residue name. If number, this is the index of the desired
        residue (starting from 0).
    sampling : str
        One of "multinomial" or "systematic" - to be passed to Move.sampler().
    origins : [openmm.Quantity] or None
        Initialises origins. None stands for translation around current centre of mass.
    radii : [openmm.Quantity]
        Initialises radii. Must be of the same length as origins.
    region : str
        Initialises region. One of "spherical" or "cubic". If cubic, the radii stand for half of the translation box
        length.

    Attributes
    ----------
    sampling : str
        One of "multinomial" or "systematic" - to be passed to Move.sampler().
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    alchemical_atoms : [int]
        A list of the absolute indices of all alchemical atoms.
    masses : numpy.ndarray
        An array of masses of the residue to be translated.
    origins : [openmm.Quantity]
        A list of one or more origins around which the samples should be centred.
    radii : [openmm.Quantity]
        A list of radii which determine the radius of sampling around each of the origins.
    region : str
        Determines the shape of the sampling area. One of "spherical" or "cubic".
    """
    def __init__(self, structure, translatable_molecule, sampling="systematic", origins=None, radii=None,
                 region="spherical"):
        if type(translatable_molecule) is str:
            for i, residue in enumerate(structure.residues):
                if residue.name == translatable_molecule:
                    translatable_molecule = i
                    break

        if type(translatable_molecule) is not int:
            raise TypeError(f"Invalid input: {translatable_molecule}. Need either a valid residue name or residue "
                            f"number.")

        self.sampling = sampling
        self.movable_atoms = [a.idx for a in structure.residues[translatable_molecule]]
        self.alchemical_atoms = self.movable_atoms[:]
        self.masses = _np.asarray([a.mass for a in structure.residues[translatable_molecule]], dtype=_np.float32)

        Iterable = (list, tuple, _np.ndarray)

        if isinstance(origins, Iterable):
            self.origins = origins
        else:
            self.origins = [origins]

        for i, origin in enumerate(self.origins):
            if origin is None:
                self.origins[i] = 0.1 * (self.masses @ structure.coordinates[self.movable_atoms]) / _np.sum(
                    self.masses) * _unit.nanometer

        self.radii = radii
        self.region = region
        self.periodic_box_vectors = _np.asarray(structure.box_vectors.value_in_unit(_unit.nanometer))

    @property
    def radii(self):
        return self._radii

    @radii.setter
    def radii(self, val):
        Iterable = (list, tuple, _np.ndarray)
        if not isinstance(val, Iterable):
            if val is None:
                self._radii = None
            else:
                self._radii = [val] * len(self.origins)
        else:
            self._radii = val
            if len(self.radii) != len(self.origins):
                raise ValueError("The number of origins must match the number of radii")

    def generateMoves(self, n_states):
        """
        Generates a list of translations.

        Parameters
        ----------
        n_states : int
            Number of samples to be generated.

        Returns
        -------
        translations : numpy.ndarray
            A numpy array with units containing n_states rows and 3 columns, which specifies the new dimensionless
            centres of mass.
        """
        self.region = self.region.lower()
        if self.radii is None and self.region != "cubic":
            _warnings.warn("Only 'cubic' region shape is supported with radii = None. Changing to 'cubic'...")
            self.region = "cubic"

        if self.region == "cubic":
            samples = _np.transpose(_np.asarray([self.sampler(n_states, self.sampling),
                                                 self.sampler(n_states, self.sampling),
                                                 self.sampler(n_states, self.sampling)], dtype=_np.float32))
            # we return unscaled coordinates and we obtain the box size when we apply the transformation
            if self.radii is None:
                return samples
            samples = 2 * samples - 1
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

        x = [i for i in range(len(self.origins))]
        w = [1 / len(self.origins)] * len(self.origins)
        if self.sampling == "systematic":
            resamples = _resmethods.SystematicResampler.resample(x, w, n_walkers=n_states)[0]
        elif self.sampling == "multinomial":
            resamples = _resmethods.MultinomialResampler.resample(x, w, n_walkers=n_states)[0]
        else:
            raise ValueError("Only systematic and multinomial sampling are currently supported")

        _np.random.shuffle(resamples)
        origins = _np.asarray([self.origins[i].value_in_unit(_unit.nanometer) for i in resamples])
        radii = _np.asarray([[self.radii[i].value_in_unit(_unit.nanometer) for i in resamples]])

        translations = (origins + radii.transpose() * samples) @ _np.linalg.inv(self.periodic_box_vectors)

        return translations

    def applyMove(self, input, translation):
        """
        Applies a single translation to a state, context or quantity. If a context is passed, it it also modified
        in place.

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input containing coordinate information.
        translation : openmm.unit.Quantity
            Coordinates of the new centre of mass.

        Returns
        -------
        input : openmm.unit.Quantity
            The transformed coordinates.
        """
        context, input, pbc = self._contextToQuantity(input, return_pbc=True)
        translation = (pbc.value_in_unit(_unit.nanometer) @ translation) * _unit.nanometer

        coords = input[self.movable_atoms].value_in_unit(_unit.nanometer)
        centre_of_mass = self.masses @ coords / _np.sum(self.masses) * _unit.nanometer
        input[self.movable_atoms] += translation - centre_of_mass

        if context is not None:
            context.setPositions(input)

        return input


class RotationMove(Move):
    """
    A class which performs rotation around centre of mass.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    rotatable_molecule : str or int
        If string, this is the first residue which has this residue name. If number, this is the index of the desired
        residue (starting from 0).
    sampling : str
        One of "multinomial" or "systematic" - to be passed to Move.sampler().

    Attributes
    ----------
    sampling : str
        One of "multinomial" or "systematic" - to be passed to Move.sampler().
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    alchemical_atoms : [int]
        A list of the absolute indices of all alchemical atoms.
    masses : numpy.ndarray
        An array of masses of the residue to be translated.
    """
    def __init__(self, structure, rotatable_molecule, sampling="systematic"):
        if type(rotatable_molecule) is str:
            for i, residue in enumerate(structure.residues):
                if residue.name == rotatable_molecule:
                    rotatable_molecule = i
                    break

        if type(rotatable_molecule) is not int:
            raise TypeError(f"Invalid input: {rotatable_molecule}. Need either a valid residue name or residue number.")

        self.sampling = sampling
        self.movable_atoms = [a.idx for a in structure.residues[rotatable_molecule]]
        self.alchemical_atoms = self.movable_atoms[:]
        self.masses = _np.asarray([a.mass for a in structure.residues[rotatable_molecule]], dtype=_np.float32)

    def generateMoves(self, n_states):
        """
        Generates a list of translations.

        Parameters
        ----------
        n_states : int
            Number of samples to be generated.

        Returns
        -------
        rotations : numpy.ndarray
            A numpy array containing n_states rows and 3 columns, which specifies the rotations around the centre of
            mass.
        """
        rotations = _np.zeros((n_states, 3))

        thetas = _np.arccos(2 * self.sampler(n_states, self.sampling) - 1)
        phis = 2 * _np.pi * self.sampler(n_states, self.sampling)
        psis = 2 * _np.pi * self.sampler(n_states, self.sampling)

        rotations[:, 0] = psis * _np.sin(thetas) * _np.cos(phis)
        rotations[:, 1] = psis * _np.sin(thetas) * _np.cos(thetas)
        rotations[:, 2] = psis * _np.cos(thetas)

        return rotations

    def applyMove(self, input, rotation):
        """
        Applies a single rotation to a state, context or quantity. If a context is passed, it it also modified
        in place.

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input containing coordinate information.
        rotation : numpy.ndarray
            A numpy vector whose direction specifies the axis of rotation and whose norm specifies the angle of
            anticlockwise rotaion.

        Returns
        -------
        input : openmm.unit.Quantity
            The transformed coordinates.
        """
        context, input = self._contextToQuantity(input)

        coords = input[self.movable_atoms].value_in_unit(_unit.nanometer)
        rotation = _Rotation.from_rotvec(rotation)
        centre_of_mass = self.masses @ coords / _np.sum(self.masses)
        coords -= centre_of_mass
        input[self.movable_atoms] = (rotation.apply(coords) + centre_of_mass) * input.unit

        if context is not None:
            context.setPositions(input)

        return input


class DihedralMove(Move):
    """
    A class which performs rotation around centre of mass.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    rotatable_bond : (int, int)
        A tuple of the indices of the two atoms defining the rotatable bond. If residue is None, absolute indices
        are required. Otherwise, relative indices must be used. Only the substituents of the second atom are moved.
    residue : str or int or None
        If string, this is the first residue which has this residue name. If number, this is the index of the desired
        residue (starting from 0). If None, absolute atom indices must be passed to rotatable_bond.
    sampling : str
        One of "multinomial" or "systematic" - to be passed to Move.sampler().

    Attributes
    ----------
    sampling : str
        One of "multinomial" or "systematic" - to be passed to Move.sampler().
    rotatable_bond : (int, int)
        A tuple of the absolute indices of the two atoms defining the rotatable bond. Only the substituents of the
        second atom are moved.
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    alchemical_atoms : [int]
        A list of the absolute indices of all alchemical atoms.

    """
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
        self.movable_atoms = list(movable_atoms)
        self.alchemical_atoms = self.movable_atoms[:]

    def generateMoves(self, n_states):
        """
        Generates a list of translations.

        Parameters
        ----------
        n_states : int
            Number of samples to be generated.

        Returns
        -------
        rotations : numpy.ndarray
            A numpy array containing n_states rows and 3 columns, which specifies the rotations around the rotatable
            bond.
        """
        return 2 * _np.pi * self.sampler(n_states, self.sampling)

    def applyMove(self, input, rotation):
        """
        Applies a single dihedral rotation to a state, context or quantity. If a context is passed, it it also modified
        in place.

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input containing coordinate information.
        rotation : numpy.ndarray
            A numpy vector whose direction specifies the axis of rotation and whose norm specifies the angle of
            anticlockwise rotaion.

        Returns
        -------
        input : openmm.unit.Quantity
            The transformed coordinates.
        """
        context, input = self._contextToQuantity(input)
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
