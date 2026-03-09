import abc
import abc as _abc
import warnings as _warnings

from cached_property import cached_property as _cached_property
import numpy as _np
from scipy.spatial.transform import Rotation as _Rotation
import openmm as _openmm
import openmm.unit as _unit

import openmmslicer.resampling_methods as _resmethods


class Move(_abc.ABC):
    @property
    @_abc.abstractmethod
    def movable_atoms(self):
        pass

    @property
    def movable_residue(self):
        return self._movable_residue

    @movable_residue.setter
    def movable_residue(self, val):
        if val is None:
            self._movable_residue = None
        else:
            if type(val) is str:
                for i, residue in enumerate(self._structure.residues):
                    if residue.name == val:
                        val = i
                        break
                else:
                    raise ValueError(f"Could not find residue {val}")
            elif type(val) is not int:
                raise TypeError(f"Invalid input: {val}. Need either a valid residue name or residue number.")
            self._movable_residue = self.structure.residues[val]
        self.reset()

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, val):
        import openmmslicer.alchemy.align as _align
        if isinstance(val, _align.AlignedStructures):
            self._structure = val.main_structure
        else:
            self._structure = val
        self._movable_residue = None
        self.reset()

    def reset(self):
        for key, value in self.__class__.__dict__.items():
            if isinstance(value, _cached_property):
                self.__dict__.pop(key, None)

    @_abc.abstractmethod
    def generate(self, *args, **kwargs):
        """
        Generates an input for apply(). This is a virtual function which must be overloaded in the child classes.
        """
        pass

    @_abc.abstractmethod
    def apply(self, *args, **kwargs):
        """
        Applies a move generated from generate(). This is a virtual function which must be overloaded in
        the child classes.
        """
        pass

    @staticmethod
    def _contextToQuantity(input, return_pbc=False):
        """
        A helper function which converts a number of input formats to standardised objects to be passed to apply().

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


class MoveList:
    """
    A class which combines multiple moves.

    Parameters
    ----------
    moves : [openmmslicer.moves.Move]
        The instantiated moves to be combined.

    Attributes
    ----------
    alchemical_atoms : [int]
        A list of the absolute indices of all alchemical atoms.
    alchemical_dihedrals: [int]
        A list of the absolute indices of all alchemical dihedral terms.
    endpoint_moves : [openmmslicer.moves.EndpointMove]
        A list of all endpoint moves.
    hybrid_moves : [openmmslicer.moves.HybridMove]
        A list of all hybrid moves.
    tempering_moves : [openmmslicer.moves.TemperingAtomMove] or [openmmslicer.moves.TemperingDihedralMove]
        A list of all tempering moves.
    moves : [openmmslicer.moves.Move]
        A list of moves.
    """
    def __init__(self, moves):
        self._moves = moves
        if not all(self._moves[0].structure is move.structure for move in self._moves[1:]):
            raise ValueError("All moves in a MoveList must be bound to the same structure")

    @_cached_property
    def alchemical_atoms(self):
        return sorted(set().union(*[move.movable_atoms for move in self.endpoint_moves]))

    @_cached_property
    def alchemical_dihedrals(self):
        return sorted(set().union(*[move.movable_dihedrals for move in self.endpoint_moves
                                    if hasattr(move, "movable_dihedrals")]))

    @property
    def endpoint_moves(self):
        return [move for move in self._moves if isinstance(move, EndpointMove)]

    @property
    def hybrid_moves(self):
        return [move for move in self._moves if isinstance(move, HybridMove)]

    @property
    def tempering_moves(self):
        return [move for move in self._moves if isinstance(move, (TemperingAtomMove, TemperingDihedralMove))]

    def getLogJacobian(self, lambda0, lambda1):
        """
        The associated logarithm of the Jacobian determinant for the transformation between two bonded lambda values,
        where linear switching is assumed.

        Parameters
        ----------
        lambda0 : float
            The first lambda value.
        lambda1 : float
            The second lambda value.
        """
        return sum(move.getLogJacobian(lambda0, lambda1) for move in self._moves if hasattr(move, "getLogJacobian"))

    def generate(self, n_states):
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
        transformations = [move.generate(n_states) for move in self.endpoint_moves]
        transformations = list(zip(*transformations))
        return transformations

    def apply(self, input, transformations=None, lambda0=None, lambda1=None):
        """
        Applies a single transformation to a state, context or quantity. If a context is passed, it it also modified
        in place.

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input containing coordinate information.
        transformations : list
            A list of all transformations to be passed to self._moves. Each transformation must be of an appropriate
            format for the underlying move.

        Returns
        -------
        input : openmm.unit.Quantity
            The transformed coordinates.
        """
        context, input = Move._contextToQuantity(input)
        failed_moves = []

        if transformations is not None:
            for move, transformation in zip(self.endpoint_moves, transformations):
                try:
                    input = move.apply(input, transformation)
                except:
                    if context is None:
                        raise TypeError("Need to pass a valid context or state in order to obtain periodic box vector "
                                        "information")
                    failed_moves += [(move, transformation)]

        if lambda0 is not None and lambda1 is not None:
            for move in self.hybrid_moves:
                input = move.apply(input, lambda0=lambda0, lambda1=lambda1)

        for move, transformation in failed_moves:
            context.setPositions(input)
            if isinstance(move, EndpointMove) and transformations is not None:
                input = move.apply(context, transformation)
            elif isinstance(move, HybridMove) and lambda0 is not None and lambda1 is not None:
                input = move.apply(context, lambda0=lambda0, lambda1=lambda1)

        if context is not None:
            context.setPositions(input)

        return input


class TemperingAtomMove(Move):
    """
    A virtual move which performs solute scaling on a particular set of atoms.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    movable_atoms : list
        The absolute atom indices to be scaled. If movable_residue is None, absolute indices are required. Otherwise,
        relative indices must be used.
    scaling_factor : float, optional
        The scaling factor to be applied. Default is 0, i.e. complete decoupling.
    scale_charges_linearly : bool, optional
        Whether to scale the charges linearly or as the square root of the scaling factor. Default is True.
    movable_residue : str or int
        If string, this is the first residue which has this residue name. If number, this is the index of the desired
        residue (starting from 0).

    Attributes
    ----------
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    """
    def __init__(self, structure, movable_atoms, scaling_factor, scale_charges_linearly=True, movable_residue=None):
        self.structure = structure
        self.movable_residue = movable_residue
        self.movable_atoms = movable_atoms
        self.scaling_factor = scaling_factor
        self.scale_charges_linearly = scale_charges_linearly

    @property
    def movable_atoms(self):
        return self._movable_atoms

    @movable_atoms.setter
    def movable_atoms(self, val):
        if not len(val):
            raise ValueError("No movable atoms supplied")
        val = set(val)
        if self.movable_residue is not None:
            val = {self.movable_residue.atoms[x].idx for x in val}
        elif min(val) < 0 or max(val) >= len(self.structure.atoms) :
            raise ValueError("Invalid atom indices supplied")
        self._movable_atoms = sorted(val)
        self.reset()

    def generate(self, *args, **kwargs):
        raise NotImplementedError("This class can only apply moves")

    def apply(self, input, lambda0, lambda1):
        pass


class TemperingDihedralMove(Move):
    """
    A virtual move which performs solute scaling on a particular dihedral bond.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    rotatable_bond : (int, int)
        A tuple of the indices of the two atoms defining the dihedral bond. If movable_residue is None, absolute
        indices are required. Otherwise, relative indices must be used.
    scaling_factor : float, optional
        The scaling factor to be applied. Default is 0, i.e. complete decoupling.
    movable_residue : str or int
        If string, this is the first residue which has this residue name. If number, this is the index of the desired
        residue (starting from 0).

    Attributes
    ----------
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    movable_dihedrals: [int]
        A list of the absolute indices of all movable dihedral terms.
    rotatable_bond : (int, int)
        A tuple of the absolute indices of the two atoms defining the rotatable bond.
    """
    def __init__(self, structure, rotatable_bond, scaling_factor, movable_residue=None):
        self.structure = structure
        self.movable_residue = movable_residue
        self.rotatable_bond = rotatable_bond
        self.scaling_factor = scaling_factor

    @property
    def movable_atoms(self):
        return []

    @_cached_property
    def movable_dihedrals(self):
        dihedrals = [i for i, d in enumerate(self._structure.dihedrals)
                     if not d.improper and {d.atom2.idx, d.atom3.idx} == set(self.rotatable_bond)]
        return sorted(dihedrals)

    @property
    def rotatable_bond(self):
        return self._rotatable_bond

    @rotatable_bond.setter
    def rotatable_bond(self, val):
        if len(val) != 2:
            raise ValueError("The rotatable bond needs to be a tuple of two atom indices")
        if self.movable_residue is not None:
            val = [self.movable_residue.atoms[val[0]].idx, self.movable_residue.atoms[val[1]].idx]
        if self.structure.atoms[val[1]] not in self.structure.atoms[val[0]].bond_partners:
            raise ValueError(f"Rotatable bond {val} is not valid")
        self._rotatable_bond = tuple(val)
        self.reset()

    def generate(self, *args, **kwargs):
        raise NotImplementedError("This class can only apply moves")

    def apply(self, input, lambda0, lambda1):
        pass


class EndpointMove(Move):
    """
    An abstract class which defines move applied on an endstate.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    movable_residue : str or int
        If string, this is the first residue which has this residue name. If number, this is the index of the desired
        residue (starting from 0).
    sampling : str
        One of "multinomial" or "systematic" - to be passed to Move.sampler().

    Attributes
    ----------
    structure : parmed.Structure
        The associated ParmEd structure.
    movable_residue : str or None
        The name of the movable residue, if applicable, otherwise None.
    sampling : str
        The type of sampling.
    """
    def __init__(self, structure, movable_residue, sampling="systematic"):
        self.structure = structure
        self.movable_residue = movable_residue
        self.sampling = sampling

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, val):
        self._sampling = val.lower()

    def sample(self, n_samples):
        """
        Samples from the uniform distribution between 0 and 1 using different sampling methods.

        Parameters
        ----------
        n_samples : int
            Number of uniform samples to be generated.

        Returns
        -------
        samples : numpy.ndarray
            An array of all generated samples.
        """
        if self.sampling == "systematic":
            samples = _np.random.uniform(0, 1 / n_samples) + _np.linspace(0, 1, num=n_samples, endpoint=False)
            _np.random.shuffle(samples)
        elif self.sampling == "multinomial":
            samples = _np.random.uniform(size=n_samples)
        else:
            raise ValueError("Only systematic and multinomial sampling are currently supported")

        return samples

    @_abc.abstractmethod
    def generate(self, n_states):
        """
        Generates an array of moves which can be read by apply(). This is a virtual function which must be
        overloaded in the child classes.
        """
        pass

    @_abc.abstractmethod
    def apply(self, input, transformation):
        """
        Applies a move generated from generate(). This is a virtual function which must be overloaded in
        the child classes.
        """
        pass


class TranslationMove(EndpointMove):
    """
    A class which performs translation around one or more centres.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    movable_residue : str or int
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
        The type of sampling.
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    masses : numpy.ndarray
        An array of masses of the residue to be translated.
    origins : [openmm.Quantity]
        A list of one or more origins around which the samples should be centred.
    periodic_box_vectors : numpy.ndarray
        An array containing the periodic box vectors of the structure.
    radii : [openmm.Quantity]
        A list of radii which determine the radius of sampling around each of the origins.
    region : str
        Determines the shape of the sampling area. One of "spherical" or "cubic".
    """
    def __init__(self, structure, movable_residue, sampling="systematic", origins=None, radii=None, region="spherical"):
        super().__init__(structure, movable_residue, sampling=sampling)

        if isinstance(origins, (list, tuple, _np.ndarray)):
            self.origins = origins
        else:
            self.origins = [origins]

        for i, origin in enumerate(self.origins):
            if origin is None:
                self.origins[i] = 0.1 * (self.masses @ self.structure.coordinates[self.movable_atoms]) / _np.sum(
                    self.masses) * _unit.nanometer

        self.radii = radii
        self.region = region

    @_cached_property
    def masses(self):
        return _np.asarray([atom.mass for atom in self.movable_residue])

    @_cached_property
    def movable_atoms(self):
        return sorted([atom.idx for atom in self.movable_residue])

    @_cached_property
    def periodic_box_vectors(self):
        return _np.asarray(self.structure.box_vectors.value_in_unit(_unit.nanometer))

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

    def generate(self, n_states):
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
            samples = _np.transpose(_np.asarray([self.sample(n_states), self.sample(n_states),
                                                 self.sample(n_states)], dtype=_np.float32))
            # we return unscaled coordinates and we obtain the box size when we apply the transformation
            if self.radii is None:
                return samples
            samples = 2 * samples - 1
        elif self.region == "spherical":
            r = self.sample(n_states) ** (1 / 3)
            cos_theta = 2 * (self.sample(n_states) - 0.5)
            phi = 2 * _np.pi * self.sample(n_states)

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

    def apply(self, input, translation):
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


class RotationMove(EndpointMove):
    """
    A class which performs rotation around centre of mass.

    Attributes
    ----------
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    masses : numpy.ndarray
        An array of masses of the residue to be translated.
    """
    @_cached_property
    def masses(self):
        return _np.asarray([atom.mass for atom in self.movable_residue])

    @_cached_property
    def movable_atoms(self):
        return sorted([atom.idx for atom in self.movable_residue])

    def generate(self, n_states):
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

        thetas = _np.arccos(2 * self.sample(n_states) - 1)
        phis = 2 * _np.pi * self.sample(n_states)
        psis = 2 * _np.pi * self.sample(n_states)

        rotations[:, 0] = psis * _np.sin(thetas) * _np.cos(phis)
        rotations[:, 1] = psis * _np.sin(thetas) * _np.cos(thetas)
        rotations[:, 2] = psis * _np.cos(thetas)

        return rotations

    def apply(self, input, rotation):
        """
        Applies a single rotation to a state, context or quantity. If a context is passed, it it also modified
        in place.

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input containing coordinate information.
        rotation : numpy.ndarray
            A numpy vector whose direction specifies the axis of rotation and whose norm specifies the angle of
            anticlockwise rotation.

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


class DihedralMove(EndpointMove):
    """
    A class which performs rotation around centre of mass.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    rotatable_bond : (int, int)
        A tuple of the indices of the two atoms defining the rotatable bond. If movable_residue is None, absolute
        indices are required. Otherwise, relative indices must be used. Only the substituents of the second atom are
        moved.
    movable_residue : str or int or None
        If string, this is the first residue which has this residue name. If number, this is the index of the desired
        residue (starting from 0). If None, absolute atom indices must be passed to rotatable_bond.
    sampling : str
        One of "multinomial" or "systematic" - to be passed to Move.sampler().

    Attributes
    ----------
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    movable_dihedrals: [int]
        A list of the absolute indices of all movable dihedral terms.
    rotatable_bond : (int, int)
        A tuple of the absolute indices of the two atoms defining the rotatable bond. Only the substituents of the
        second atom are moved.
    """
    def __init__(self, structure, rotatable_bond, movable_residue=None, sampling="systematic"):
        super().__init__(structure, movable_residue, sampling=sampling)
        self.rotatable_bond = rotatable_bond

    @_cached_property
    def movable_atoms(self):
        atoms = {self.rotatable_bond[1]}
        new_neighbours = {self.rotatable_bond[1]}
        while True:
            new_neighbours = {y.idx for x in new_neighbours for y in self.structure.atoms[x].bond_partners} - \
                             {self.rotatable_bond[0]} - atoms
            if len(new_neighbours):
                atoms |= new_neighbours
            else:
                break
        return sorted(atoms)

    @_cached_property
    def movable_dihedrals(self):
        dihedrals = [i for i, d in enumerate(self._structure.dihedrals)
                     if not d.improper and {d.atom2.idx, d.atom3.idx} == set(self.rotatable_bond)]
        return sorted(dihedrals)

    @property
    def rotatable_bond(self):
        return self._rotatable_bond

    @rotatable_bond.setter
    def rotatable_bond(self, val):
        if len(val) != 2:
            raise ValueError("The rotatable bond needs to be a tuple of two atom indices")
        if self.movable_residue is not None:
            val = [self.movable_residue.atoms[val[0]].idx, self.movable_residue.atoms[val[1]].idx]
        if self.structure.atoms[val[1]] not in self.structure.atoms[val[0]].bond_partners:
            raise ValueError(f"Rotatable bond {val} is not valid")
        self._rotatable_bond = tuple(val)
        self.reset()

    def generate(self, n_states):
        """
        Generates a list of dihedral moves.

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
        return 2 * _np.pi * self.sample(n_states)

    def apply(self, input, rotation):
        """
        Applies a single dihedral rotation to a state, context or quantity. If a context is passed, it it also modified
        in place.

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input containing coordinate information.
        rotation : numpy.ndarray
            A numpy vector whose direction specifies the axis of rotation and whose norm specifies the angle of
            anticlockwise rotation.

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


class HybridMove(Move):
    """
    An abstract class which defines a move which is executed at intermediate states.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    movable_residue : str or int or None, optional
        If a string, this is the first residue which has this residue name. If an integer, this is the index of the
        desired residue (starting from 0). If None, absolute atom indices must be passed to rotatable_bond. Default is
        None.
    state0 : str
        The name of the first state as defined in the alchemical chain.
    state1 : str
        The name of the second state as defined in the alchemical chain.

    Attributes
    ----------
    structure : parmed.Structure
        The associated ParmEd structure.
    state0 : str
        The name of the first state as defined in the alchemical chain.
    state1 : str
        The name of the second state as defined in the alchemical chain.
    """
    def __init__(self, structure, movable_residue, state0, state1):
        self.structure = structure
        self.state0 = state0
        self.state1 = state1
        self.movable_residue = movable_residue

    @property
    def structure(self):
        return self._aligned_structures.main_structure

    @structure.setter
    def structure(self, val):
        import openmmslicer.alchemy.align as _align
        if not isinstance(val, _align.AlignedStructures):
            raise ValueError("The structure needs to be a subclass of AlignedStructures")
        self._aligned_structures = val
        self._movable_residue = None
        self.reset()

    @property
    def state0(self):
        return self._aligned_structures[self._state0]

    @state0.setter
    def state0(self, val):
        self._state0 = val

    @property
    def state1(self):
        return self._aligned_structures[self._state1]

    @state1.setter
    def state1(self, val):
        self._state1 = val

    @abc.abstractmethod
    def getLogJacobian(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        raise NotImplementedError("This class can only apply moves")

    @_abc.abstractmethod
    def apply(self, input, lambda0, lambda1):
        pass


class BondMove(HybridMove):
    """
    A class which performs bond rescaling.

    Parameters
    ----------
    structure : parmed.Structure
        An object containing all of the information about the system.
    scalable_bond : (int, int)
        A tuple of the indices of the two atoms defining the scalable bond. If movable_residue is None, absolute indices
        are required. Otherwise, relative indices must be used. Only the substituents connected to the second atom will
        be moved.
    state0 : str
        The name of the first state as defined in the alchemical chain.
    state1 : str
        The name of the second state as defined in the alchemical chain.
    movable_residue : str or int or None, optional
        If a string, this is the first residue which has this residue name. If an integer, this is the index of the
        desired residue (starting from 0). If None, absolute atom indices must be passed to rotatable_bond. Default is
        None.

    Attributes
    ----------
    scalable_bond : (int, int)
        A tuple of the absolute indices of the two atoms defining the scalable bond.
    movable_atoms : [int]
        A list of the absolute indices of all movable atoms.
    """
    def __init__(self, structure, scalable_bond, state0, state1, movable_residue=None):
        super().__init__(structure, movable_residue, state0, state1)
        self.scalable_bond = scalable_bond

    @property
    def movable_atoms(self):
        return self._movable_atoms

    @property
    def scalable_bond(self):
        return self._scalable_bond

    @scalable_bond.setter
    def scalable_bond(self, val):
        if len(val) != 2:
            raise ValueError("The scalable_bond bond needs to be a tuple of two atom indices")
        if self.movable_residue is not None:
            val = [self.movable_residue.atoms[val[0]].idx, self.movable_residue.atoms[val[1]].idx]
        if self.structure.atoms[val[1]] not in self.structure.atoms[val[0]].bond_partners:
            raise ValueError(f"Scalable bond {val} is not valid")

        # raise error if bond is in a ring
        neighbours0, neighbours1 = {val[0]}, {val[1]}
        new_neighbours0, new_neighbours1 = {val[0]}, {val[1]}
        while True:
            new_neighbours0 = {y.idx for x in new_neighbours0 for y in self.structure.atoms[x].bond_partners} - {val[1]}
            new_neighbours1 = {y.idx for x in new_neighbours1 for y in self.structure.atoms[x].bond_partners} - {val[0]}
            if new_neighbours0.issubset(neighbours0) and new_neighbours1.issubset(neighbours1):
                break
            neighbours0 |= new_neighbours0
            neighbours1 |= new_neighbours1
            if neighbours0 & neighbours1:
                raise ValueError("Cannot scale bonds in rings")

        self._movable_atoms = sorted(neighbours1)
        self._scalable_bond = tuple(val)
        self.reset()

    def getLogJacobian(self, lambda0, lambda1):
        """
        The associated logarithm of the Jacobian determinant for the bond rescaling transformation between two bonded
        lambda values, where linear switching is assumed.

        Parameters
        ----------
        lambda0 : float
            The first lambda value.
        lambda1 : float
            The second lambda value.
        """
        if lambda0 == lambda1:
            return 0.
        bond0 = next(bond for bond in self.state0.bonds if {bond.atom1.idx, bond.atom2.idx} == set(self.scalable_bond))
        bond1 = next(bond for bond in self.state1.bonds if {bond.atom1.idx, bond.atom2.idx} == set(self.scalable_bond))
        delta_req = bond1.type.req - bond0.type.req
        req_ratio = _np.log(bond0.type.req + delta_req * lambda1) - _np.log(bond0.type.req + delta_req * lambda0)
        return 3 * float(req_ratio)

    def apply(self, input, lambda0, lambda1):
        """
        Applies a rescaling bond move to a state, context or quantity. If a context is passed, it it also modified in
        place.

        Parameters
        ----------
        input : openmm.Context or openmm.State or openmm.unit.Quantity
            Input containing coordinate information.
        lambda0 : float
            The first lambda value.
        lambda1 : float
            The second lambda value.

        Returns
        -------
        input : openmm.unit.Quantity
            The transformed coordinates.
        """
        context, input = self._contextToQuantity(input)
        if lambda0 == lambda1:
            return input
        bond_vector = (input[self.scalable_bond[1], :] - input[self.scalable_bond[0], :])
        scaling_factor = _np.exp(self.getLogJacobian(lambda0, lambda1) / 3)
        delta_bond = (scaling_factor - 1) * bond_vector
        input[self.movable_atoms] += delta_bond

        if context is not None:
            context.setPositions(input)

        return input
