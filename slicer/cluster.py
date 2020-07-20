import numpy as _np

import simtk.unit as _unit

from slicer.moves import Move as _Move


class DihedralClustering:
    @staticmethod
    def measureDihedrals(input, all_dihedral_indices):
        original_positions = _Move._contextToQuantity(input)[1]
        all_angles = []
        for dihedral_indices in all_dihedral_indices:
            positions = [original_positions[i].value_in_unit(_unit.nanometer) for i in dihedral_indices]
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
            all_angles += [_np.arctan2(y, x)]
        return _np.asarray(all_angles)

    @staticmethod
    def clusterDihedrals(old_dihedrals, new_dihedrals):
        assert old_dihedrals.shape == new_dihedrals.shape, "The two arrays must have the same shape"
        TWOPI = 2 * _np.pi

        # make sure all angles are between 0 and 2 pi
        old_dihedrals = old_dihedrals % TWOPI
        old_dihedrals[old_dihedrals < 0] += TWOPI
        new_dihedrals = new_dihedrals % TWOPI
        new_dihedrals[new_dihedrals < 0] += TWOPI

        all_labels = []
        for old_dihedral, new_dihedral in zip(old_dihedrals, new_dihedrals):
            # express all angle transitions as line segments
            line_segments = []
            for x, y in zip(old_dihedral, new_dihedral):
                minimum, maximum = min(x, y), max(x, y)
                if maximum - minimum > _np.pi:
                    line_segments += [(0, minimum), (maximum, TWOPI)]
                else:
                    line_segments += [(minimum, maximum)]

            # find the gaps between the line segments and label the clusters
            line_segments = list(DihedralClustering.group(line_segments))
            clusters = [i for i in range(len(line_segments))]
            if line_segments[-1][-1] == TWOPI and line_segments[0][0] == 0:
                clusters[-1] = 0

            # label the new dihedrals
            labels = []
            for y in new_dihedral:
                for segment, cluster in zip(line_segments, clusters):
                    if segment[0] <= y <= segment[1]:
                        labels += [cluster]
                        break
            all_labels += [labels]

        # map the n-dimensional cluster labels onto 1D
        all_labels = list(map(tuple, zip(*all_labels)))
        mapping_dict = {x: i for i, x in enumerate(list(set(all_labels)))}
        all_labels = _np.asarray([mapping_dict[x] for x  in all_labels])

        return all_labels

    @staticmethod
    def group(data):
        data = sorted(data)
        it = iter(data)
        a, b = next(it)
        for c, d in it:
            if b >= c:
                b = max(b, d)
            else:
                yield a, b
                a, b = c, d
        yield a, b
