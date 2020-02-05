import itertools as _it

import numpy as _np
from scipy.interpolate import interp1d as _interp1d
from scipy.spatial.transform import Rotation as _Rotation
import simtk.unit as _unit

_OPENMM_DISTANCE_UNIT = _unit.nanometer


class ConformationGenerator():
    _read_only_properties = ["rotatable_atoms", "rotatable_bonds"]

    def __init__(self, system, topology, context, rotatable_bonds, temperature=298):
        self.system = system
        self.topology = topology
        self.context = context
        self.rotatable_bonds = rotatable_bonds
        self.temperature = temperature

    def __getattr__(self, item):
        if item not in self._read_only_properties:
            return self.__getattribute__(item)
        else:
            try:
                return self.__getattribute__("_" + item)
            except AttributeError:
                return None

    @property
    def rotatable_bonds(self):
        return self._rotatable_bonds

    @rotatable_bonds.setter
    def rotatable_bonds(self, val):
        self._rotatable_bonds = val
        rotatable_bonds_sets = {frozenset(x) for x in self._rotatable_bonds}
        self._total_relevant_dihedrals = {x for x in self.topology.dihedrals if
                                          frozenset([x.atom2.idx, x.atom3.idx]) in rotatable_bonds_sets}
        # this ensures that all dihedrals come from the same resname
        ligname, = {x.atom1.residue.name for x in self._total_relevant_dihedrals}

        # here we generate the rotatable atom indices by tracing the bonded terms iteratively until convergence
        all_bonds = {frozenset([bond.atom1.idx, bond.atom2.idx]) for bond in self.topology.bonds
                     if bond.atom1.residue.name == ligname}
        all_atoms = set().union(*all_bonds)
        self._rotatable_atoms = {}
        neighbour_dict = {k: set().union(*[x for x in all_bonds if k in x]) for k in all_atoms}

        # not an optimal tracing algorithm but one that's easy to understand
        for bond in self._rotatable_bonds:
            current_neighbours = {bond[1]}
            while True:
                new_neighbours = set().union(*[neighbour_dict[x] for x in current_neighbours]) - {bond[0]}
                if current_neighbours == new_neighbours:
                    break
                current_neighbours = new_neighbours
            self._rotatable_atoms[bond] = current_neighbours

    def _generateInverseCDFs(self):
        funcs = {}
        state = self.context.getState(getPositions=True)

        for bond in self._rotatable_bonds:
            relevant_dihedrals = [x for x in self._total_relevant_dihedrals if {x.atom2.idx, x.atom3.idx} == set(bond)]
            initial_angles = _np.array([self.measureDihedral(state, [x.atom1.idx, x.atom2.idx, x.atom3.idx, x.atom4.idx])
                                        for x in relevant_dihedrals])
            initial_angles -= initial_angles[0]

            def dihedral_func(x):
                result = 0
                for dihedral, offset in zip(relevant_dihedrals, initial_angles):
                    phase, phi_k, mult = dihedral.type.phase, dihedral.type.phi_k, dihedral.type.per
                    result += phi_k * (1 + _np.cos(mult * (x + offset) - phase * _np.pi / 180))
                return result * _unit.kilocalorie_per_mole

            x = _np.linspace(-_np.pi, _np.pi, 1000)
            kBT = _unit.BOLTZMANN_CONSTANT_kB * _unit.AVOGADRO_CONSTANT_NA * self.temperature * _unit.kelvin
            y = _np.array([dihedral_func(val) / kBT for val in x])
            y = _np.exp(min(y) - y)
            y /= sum(y)
            cdf = list(_it.accumulate(y))
            cdf[0] = 0
            cdf[-1] = 1

            # this is the inverse CDF for this dihedral
            d = relevant_dihedrals[0]
            funcs[bond] = (d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx), _interp1d(cdf, x)

        return funcs

    def generateConformers(self, n_conformers):
        # we regenerate the CDFs every time because they are sensitive to the current state which we can't track
        cdfs = self._generateInverseCDFs()
        initial_state = self.context.getState(getPositions=True)
        new_states = []

        for n in range(n_conformers):
            delta_phis = []
            for bond, (dihedral, cdf) in cdfs.items():
                new_angle = cdf(_np.random.rand()).item(0)
                current_angle = self.measureDihedral(self.context.getState(getPositions=True), dihedral)
                delta_phi = new_angle - current_angle
                delta_phis += [delta_phi]
            # we call the function only once so that the coordinates are not copied for every dihedral rotation
            self.rotateDihedrals(cdfs.keys(), delta_phis)
            new_states += [self.context.getState(getPositions=True)]

        self.context.setState(initial_state)
        return new_states

    @staticmethod
    def measureDihedral(state, dihedral_indices):
        positions = state.getPositions(True)
        positions = [positions[i].value_in_unit(_OPENMM_DISTANCE_UNIT) for i in dihedral_indices]
        b1 = positions[1] - positions[0]
        b1 /= _np.sqrt(_np.dot(b1, b1))
        b2 = positions[2] - positions[1]
        b2 /= _np.sqrt(_np.dot(b2, b2))
        b3 = positions[3] - positions[2]
        b3 /= _np.sqrt(_np.dot(b3, b3))
        n1 = _np.cross(b1, b2)
        n1 /= _np.sqrt(_np.dot(n1, n1))
        n2 = _np.cross(b2, b3)
        n2 /= _np.sqrt(_np.dot(n2, n2))
        x = _np.dot(n1, n2)
        y = _np.dot(_np.cross(n1, b2), n2)
        angle = _np.arctan2(y, x)
        return angle

    def rotateDihedrals(self, rotatable_bonds, delta_phis):
        positions = self.context.getState(getPositions=True).getPositions(asNumpy=True)
        new_positions = positions.copy()

        for rotatable_bond, delta_phi in zip(rotatable_bonds, delta_phis):
            # we always rotate the atoms attached to rotatable_bond[1]
            bond_vector = (positions[rotatable_bond[0], :] - positions[rotatable_bond[1], :]).value_in_unit(_unit.nanometers)
            bond_vector /= _np.linalg.norm(bond_vector)
            rotation = _Rotation.from_rotvec(bond_vector * delta_phi)
            origin = positions[rotatable_bond[1], :]
            rotatable_atoms = self._rotatable_atoms[rotatable_bond]

            for i in rotatable_atoms:
                shifted_position = positions[i, :] - origin
                new_positions[i, :] = rotation.apply(shifted_position) * positions.unit + origin

        self.context.setPositions(new_positions)
