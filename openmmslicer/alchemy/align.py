from collections import defaultdict as _defaultdict
import math as _math
import copy as _copy
import warnings as _warnings

import numpy as _np
import parmed as _pmd

import openmmslicer.moves as _moves


class AlignedStructures:
    """
    An object used to make two structures compatible so that a hybrid topology can be created later.

    Parameters
    ----------
    main_name : str
        The unique name of the main structure used for identification. This is the structure which will later contain
        no dummy atoms and whose coordinates are used for initialisation.
    main_structure : parmed.Structure
        The associated ParmEd structure.
    main_coordinates : str
        A path to the main coordinate file.

    Attributes
    ----------
    main_name : str
        The unique name of the main structure used for identification.
    coordinates : str
        A path to the main coordinate file.
    """
    def __init__(self, main_name, main_structure, main_coordinates):
        self.coordinates = main_coordinates
        self.main_name = main_name
        self._structures = {main_name: main_structure.copy(main_structure.__class__, split_dihedrals=True)}

    def __getitem__(self, item):
        return self._structures[item]

    @property
    def main_structure(self):
        return self[self.main_name]

    @staticmethod
    def _append_term(term, attr):
        atoms_attr = [f"atom{i + 1}" for i in range(4) if f"atom{i + 1}" in dir(term)]
        for atom_attr in atoms_attr:
            getattr(getattr(term, atom_attr), attr).append(term)
        return term

    @staticmethod
    def _get_term(term, attr, attr_type, structure, mapping=None):
        new_term = _copy.copy(term)
        atoms_attr = [f"atom{i + 1}" for i in range(4) if f"atom{i + 1}" in dir(term)]
        for atom_attr in atoms_attr:
            atom_index = getattr(term, atom_attr).idx
            if mapping is not None:
                atom_index = mapping[atom_index]
            setattr(new_term, atom_attr, structure.atoms[atom_index])
        new_term.type = AlignedStructures._get_type(new_term.type, attr_type, structure)
        try:
            return next(x for x in getattr(structure, attr) if x == term)
        except StopIteration:
            return new_term

    @staticmethod
    def _get_type(term_type, attr_type, structure):
        try:
            return next(x for x in getattr(structure, attr_type) if x == term_type)
        except StopIteration:
            new_type = _copy.deepcopy(term_type)
            new_type.list = getattr(structure, attr_type)
            getattr(structure, attr_type).append(new_type)
            return new_type

    @staticmethod
    def _terms_to_dict(structure, attr, dummy_indices, mapping=None):
        map_dict = {"bonds": 2, "angles": 3, "dihedrals": 4}
        assert attr in map_dict, "Unsupported mode"
        term_dict = _defaultdict(list)
        for term in getattr(structure, attr):
            key = tuple(getattr(term, f"atom{i + 1}").idx for i in range(map_dict[attr]))
            if mapping is not None:
                if not set(key).issubset(set(mapping)):
                    continue
                key = tuple(mapping[x] for x in key)
            if key[0] > key[-1]:
                key = key[::-1]
            if len(set(key) | set(dummy_indices)):
                term_dict[key] += [term]
        term_dict.default_factory = None
        return term_dict
    
    @staticmethod
    def _update_parameters(structure):
        for adjust in structure.adjusts:
            adjust.type = None
        for adjust_type in reversed(structure.adjust_types):
            structure.adjust_types.remove(adjust_type)
        structure.parameterset = _pmd.ParameterSet.from_structure(structure)
        structure.update_dihedral_exclusions()
        structure.parametrize()
        return structure
    
    @staticmethod
    def _scale_dihedral_type(dihedral_type, scaling_factor=0.):
        new_dihedral_type = _copy.deepcopy(dihedral_type)
        new_dihedral_type.phi_k *= scaling_factor
        return new_dihedral_type

    def addStructure(self, name, structure, mcs, residue=None):
        """
        Aligns a new structure based on a predefined maximum common substructure (MCS).

        Parameters
        ----------
        name : str
            The unique name of the new structure used for identification.
        structure : parmed.Structure
            The associated ParmEd structure.
        mcs : [tuple]
            The maximum common substructure between the two structures or residues. This is a list of tuples, where
            each tuple consists of an atom index of the main_structure and the corresponding atom index of the added
            structure. All atoms of the new structure or residue need to be part of the mcs, otherwise an error will be
            thrown.
        residue : str, optional
            The residue name of the relevant part of the system to be aligned. If specified, all indices in the MCS
            are interpreted as relative indices, starting from 0 for the first atom of the residue. Both this structure
            and the main_structure need to have the same residue name.

        Returns
        -------
        structure : parmed.Structure
            The aligned structure.
        """
        if len(self._structures) >= 2:
            # TODO: extend to more structures
            raise NotImplementedError("Only a maximum of two aligned structures is currently supported")

        if residue is None:
            res_pairs = {(self.main_structure[atom0].residue, structure[atom1].residue) for atom0, atom1 in mcs}
            if sum(len(res_pair[1]) for res_pair in res_pairs) != len(mcs):
                raise ValueError("The MCS must contain every atom from the aligned topology")
        else:
            if type(residue) is str:
                for i, res in enumerate(structure.residues):
                    if res.name == residue:
                        residue = i
                        break
                else:
                    raise ValueError(f"Residue {residue} not found")
            res_pair = self.main_structure.residues[residue], structure.residues[residue]
            res_pairs = {res_pair}
            mcs = [(res_pair[0][i].idx, res_pair[1][j].idx) for i, j in mcs]

        structure = structure.copy(structure.__class__, split_dihedrals=True)
        structure_new = self.main_structure.copy(self.main_structure.__class__, split_dihedrals=True)

        # some bookkeeping
        atom_indices0 = {x[0] for x in mcs}
        mcs_dict_bwd = {y: x for x, y in mcs}

        # create dummy atoms with corresponding nonbonded interactions
        dummy_indices = {atom.idx for res_pair in res_pairs for atom in res_pair[0]} - atom_indices0
        self.softenAtoms(structure_new, dummy_indices)

        # replace the rest of the nonbonded interactions
        for atom0, atom1 in mcs:
            atom_type0 = _copy.deepcopy(structure_new.atoms[atom0].atom_type)
            atom_type1 = _copy.deepcopy(structure.atoms[atom1].atom_type)

            # we never perturb constraints
            if atom0 not in dummy_indices:
                if atom_type1.atomic_number == 1 and atom_type0.atomic_number != 1:
                    atom_type1.atomic_number = atom_type0.atomic_number
                    atom_type1.mass = atom_type0.mass
                    atom_type1.name += f"_{atom_type0.atomic_number}"
                elif atom_type0.atomic_number == 1 and atom_type1.atomic_number != 1:
                    atom_type0.atomic_number = atom_type1.atomic_number
                    atom_type0.mass = atom_type1.mass
                    atom_type0.name += f"_{atom_type1.atomic_number}"
                    for struct in [self.main_structure, structure_new]:
                        struct.atoms[atom0].atom_type = atom_type0
                        struct.atoms[atom0].element = structure.atoms[atom1].element
                        struct.atoms[atom0].type = atom_type0.name

            structure_new.atoms[atom0].atom_type = atom_type1
            structure_new.atoms[atom0].type = atom_type1.name
            for attr in ["charge", "name"]:
                setattr(structure_new[atom0], attr, getattr(structure.atoms[atom1], attr))

        # replace the non-dummy bonded terms
        for term_name, n_terms in zip(["bond", "angle", "dihedral"], [2, 3, 4]):
            attr = term_name + "s"
            attr_type = term_name + "_types"
            terms_to_remove0, terms_to_remove1 = [], []
            terms_to_add0, terms_to_add1 = [], []

            term_dict = self._terms_to_dict(structure, attr, dummy_indices, mapping=mcs_dict_bwd)
            term_dict_new = self._terms_to_dict(structure_new, attr, dummy_indices)
            common_keys = set(term_dict.keys()) & set(term_dict_new.keys())
            unique_terms0 = {key: term_dict_new[key] for key in set(term_dict_new.keys()) - common_keys}
            unique_terms1 = {key: term_dict[key] for key in set(term_dict.keys()) - common_keys}

            for key in common_keys:
                terms0, terms1 = term_dict_new[key], term_dict[key]
                types0, types1 = {x.type for x in terms0}, {x.type for x in terms1}
                assert len(types0) == len(terms0) and len(types1) == len(terms1), "Type duplicates found"
                common_types = types0 & types1
                unique_types0 = types0 - common_types
                unique_types1 = types1 - common_types

                if term_name in ["bond", "angle"]:
                    assert len(terms0) == len(terms1) == 1, f"Invalid number of {term_name}s"
                    if not common_types:
                        terms0[0].type = self._get_type(terms1[0].type, attr_type, structure_new)
                else:
                    per0 = {x.per for x in unique_types0}
                    per1 = {x.per for x in unique_types1}
                    common_pers = per0 & per1

                    # iterate over the unique reference subterms and the common subterms
                    for term in terms0:
                        if term.type.per not in common_pers and term.type in unique_types0:
                            term = self._get_term(term, attr, attr_type, structure_new)
                            new_term = self._append_term(_copy.copy(term), attr)
                            new_term.type = self._get_type(self._scale_dihedral_type(new_term.type), attr_type,
                                                           structure_new)
                            terms_to_remove1 += [term]
                            terms_to_add1 += [new_term]
                        elif term.type.per in common_pers:
                            new_term = next(x for x in terms1 if x.type.per == term.type.per)
                            new_term = self._get_term(new_term, attr, attr_type, structure_new, mapping=mcs_dict_bwd)
                            new_term = self._append_term(new_term, attr)
                            terms_to_remove1 += [term]
                            terms_to_add1 += [new_term]

                    # iterate over the unique new subterms
                    for term in terms1:
                        if term.type.per not in common_pers and term.type in unique_types1:
                            term0 = self._get_term(term, attr, attr_type, self.main_structure, mapping=mcs_dict_bwd)
                            new_term0 = self._append_term(_copy.copy(term0), attr)
                            new_term0.type = self._get_type(self._scale_dihedral_type(new_term0.type), attr_type,
                                                            self.main_structure)
                            terms_to_remove0 += [term0]
                            terms_to_add0 += [new_term0]

                            term1 = self._get_term(term, attr, attr_type, structure_new, mapping=mcs_dict_bwd)
                            new_term1 = self._append_term(_copy.copy(term1), attr)
                            new_term1.type = self._get_type(new_term1.type, attr_type, structure_new)
                            terms_to_remove1 += [term1]
                            terms_to_add1 += [new_term1]

            # iterate over the unique reference terms
            for atoms, terms in unique_terms0.items():
                for term in terms:
                    term = self._get_term(term, attr, attr_type, structure_new)
                    new_term = self._append_term(_copy.copy(term), attr)
                    new_term.type = self._get_type(term.type, attr_type, structure_new)
                    terms_to_remove1 += [term]
                    terms_to_add1 += [new_term]

            # iterate over the unique new terms
            if term_name in ["bond", "angle"]:
                assert not len(unique_terms1), f"The new structure has more {term_name}s than expected"
            for atoms, terms in unique_terms1.items():
                for term in terms:
                    new_term0 = self._get_term(term, attr, attr_type, self.main_structure, mapping=mcs_dict_bwd)
                    new_term0 = self._append_term(_copy.copy(new_term0), attr)
                    new_term0.type = self._get_type(self._scale_dihedral_type(term.type), attr_type, self.main_structure)
                    terms_to_add0 += [new_term0]
                    new_term1 = self._get_term(term, attr, attr_type, structure_new, mapping=mcs_dict_bwd)
                    new_term1 = self._append_term(_copy.copy(new_term1), attr)
                    new_term1.type = self._get_type(term.type, attr_type, structure_new)
                    terms_to_add1 += [new_term1]

            # remove old terms
            for term in terms_to_remove0:
                try:
                    getattr(self.main_structure, attr).remove(term)
                except ValueError:
                    pass
            for term in terms_to_remove1:
                try:
                    getattr(structure_new, attr).remove(term)
                except ValueError:
                    pass

            # add new terms
            setattr(self.main_structure, attr, getattr(self.main_structure, attr) + terms_to_add0)
            setattr(structure_new, attr, getattr(structure_new, attr) + terms_to_add1)

            # sort by atom index
            if term_name in ["bond", "angle"]:
                sortfunc = lambda x: [getattr(x, f"atom{i + 1}").idx for i in range(n_terms)]
            else:
                sortfunc = lambda x: [getattr(x, f"atom{i + 1}").idx for i in range(n_terms)] + \
                                     [getattr(getattr(x, "type"), y) for y in ["per", "phase", "phi_k"]]
            getattr(self.main_structure, attr).sort(key=sortfunc)
            getattr(structure_new, attr).sort(key=sortfunc)

        # update the parameter set and the exclusions
        self._update_parameters(structure_new)

        charge1_new = sum(atom.charge for atom in structure_new)
        charge1 = sum(atom.charge for atom in structure)
        if not _math.isclose(charge1, charge1_new, abs_tol=1e-8):
            _warnings.warn(f"The charge is not conserved between the old structure ({charge1}) and the new structure "
                           f"({charge1_new}). Check your MCS")

        self._structures[name] = structure_new

        return structure_new

    @staticmethod
    def getDummyIndices(structure):
        """Returns the atom indices corresponding to all dummy atoms in a ParmEd Structure."""
        return {atom.idx for atom in structure if "du" in atom.type}

    @staticmethod
    def softenAtoms(structure, atom_indices, scaling_factor=0., scale_charges_linearly=True):
        """
        Softens the Lennard-Jones energy parameters and the charges of a set of atoms.

        Parameters
        ----------
        structure : parmed.Structure
            The relevant ParmEd structure object to be modified.
        atom_indices : list
            The absolute atom indices to be scaled.
        scaling_factor : float, optional
            The scaling factor to be applied. Default is 0, i.e. complete decoupling.
        scale_charges_linearly : bool, optional
            Whether to scale the charges linearly or as the square root of the scaling factor. Default is True.

        Returns
        -------
        structure : parmed.Structure
            The modified structure.
        """
        charge_scaling_factor = scaling_factor if scale_charges_linearly else scaling_factor ** 0.5
        for atom_idx in atom_indices:
            atom = structure.atoms[atom_idx]
            atom.atom_type = _copy.deepcopy(atom.atom_type)
            atom.atom_type.epsilon *= scaling_factor
            atom.atom_type.charge = 0.
            if scaling_factor:
                atom.atom_type.name = f"{atom.atom_type.name}_{str(scaling_factor)}"
            else:
                atom.atom_type.name = "du_" + atom.atom_type.name
            atom.charge *= charge_scaling_factor
            atom.type = atom.atom_type.name
        return AlignedStructures._update_parameters(structure)

    @staticmethod
    def softenDihedrals(structure, dihedral_indices, scaling_factor=0.):
        """
        Softens the dihedral energy parameters of a set of dihedrals.

        Parameters
        ----------
        structure : parmed.Structure
            The relevant ParmEd structure object to be modified.
        dihedral_indices : list
            The absolute dihedral indices to be scaled.
        scaling_factor : float, optional
            The scaling factor to be applied. Default is 0, i.e. complete decoupling.

        Returns
        -------
        structure : parmed.Structure
            The modified structure.
        """
        for index in dihedral_indices:
            dihedral_type = AlignedStructures._scale_dihedral_type(structure.dihedrals[index].type, scaling_factor)
            structure.dihedrals[index].type = AlignedStructures._get_type(dihedral_type, "dihedral_types", structure)
        return structure


class AlchemicalChain:
    """
    An object used to define a linear Markov chain of states which will be traversed with a particular sampler.

    Parameters
    ----------
    aligned_structures : openmmslicer.alchemy.AlignedStructures
        The AlignedStructures object which will be used to define the AlchemicalChain.
    states : [(str, openmmslicer.moves.MoveList or None)]
        A list of all states which are defined as a tuple of their corresponding structure name described in
        aligned_structure and the corresponding moves which define the alchemical changes of this state. All
        openmmslicer.moves.BondMoves need to be given on both corresponding end states, otherwise an error will be
        thrown.

    Attributes
    ----------
    aligned_structures : openmmslicer.alchemy.AlignedStructures
        The associated AlignedStructures object.
    fixed_lambdas : numpy.ndarray
        An array of the lambda values corresponding to each of the states.
    states : [(str, openmmslicer.moves.MoveList or None)]
        A list of all states which are defined as a tuple of their corresponding structure name described in
        aligned_structure and the corresponding moves which define the alchemical changes of this state.
    """
    def __init__(self, aligned_structures, states):
        if len(states) < 2:
            raise ValueError("The number of states needs to be >= 2")

        self.aligned_structures = aligned_structures
        all_moves = [self._cast_moves(moves) for name, moves in states]
        self.states = [(name, self._cast_structure(self.aligned_structures[name], moves), moves)
                       for (name, _), moves in zip(states, all_moves)]
        dummy_atoms = [AlignedStructures.getDummyIndices(structure) for name, structure, moves in self.states]
        for x, y in zip(dummy_atoms, dummy_atoms[1:]):
            if not x.issubset(y) and not x.issuperset(y):
                raise ValueError("Simultaneous creation and annihilation of dummy atoms is not supported")

    @property
    def fixed_lambdas(self):
        return _np.linspace(0., 1., num=len(self.states))

    def getMoves(self, lambda_):
        """Returns an openmmslicer.moves.MoveList corresponding to a particular lambda value."""
        idx_float = (len(self.states) - 1) * lambda_
        idx_int = int(round(idx_float))
        if _math.isclose(idx_int, idx_float, abs_tol=1e-8):
            if self.states[idx_int][2]._moves:
                return self.states[idx_int][2]
        else:
            idx0 = _math.floor(idx_float)
            moves0 = self.states[idx0][2]._moves if self.states[idx0][2] is not None else []
            moves1 = self.states[idx0 + 1][2]._moves if self.states[idx0 + 1][2] is not None else []
            total_moves = [move for move in set(moves0) & set(moves1) if isinstance(move, _moves.HybridMove)]
            if total_moves:
                return _moves.MoveList(list(total_moves))
            else:
                return None

    @staticmethod
    def _cast_moves(moves):
        if moves is None:
            moves = []
        if not isinstance(moves, (list, _moves.MoveList)):
            moves = _moves.MoveList([moves])
        elif isinstance(moves, list):
            moves = _moves.MoveList(moves)
        return moves

    @staticmethod
    def _cast_structure(structure, moves):
        new_structure = _copy.copy(structure)
        AlignedStructures.softenAtoms(new_structure, moves.alchemical_atoms)
        AlignedStructures.softenDihedrals(new_structure, moves.alchemical_dihedrals)
        for move in moves.tempering_moves:
            if isinstance(move, _moves.TemperingAtomMove):
                AlignedStructures.softenAtoms(new_structure, move.movable_atoms, scaling_factor=move.scaling_factor,
                                              scale_charges_linearly=move.scale_charges_linearly)
            elif isinstance(move, _moves.TemperingDihedralMove):
                AlignedStructures.softenDihedrals(new_structure, move.movable_dihedrals,
                                                  scaling_factor=move.scaling_factor)
        return new_structure
