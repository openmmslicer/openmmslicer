import random as _random

import numpy as _np


class MaximumEntropyResampler:
    @staticmethod
    def resample(weights, n_walkers=None, n_samples=1, tol=1e-8):
        weights = _np.asarray(weights)
        weights /= sum(weights)
        n_weights = len(weights)

        if n_walkers is None:
            n_walkers = n_weights

        # the floor of the exact float walkers provides a certain lower bound to the chosen number of integer walkers
        exact_walkers = n_walkers * weights
        integer_walkers = _np.floor(exact_walkers)
        residual_walkers = exact_walkers - integer_walkers
        n_residual_walkers = n_walkers - int(_np.sum(integer_walkers))

        returned_samples = _np.array([integer_walkers] * n_samples, dtype=int)

        # this is a straightforward special case
        if n_residual_walkers == 1:
            for i in range(n_samples):
                new_proposal = _np.zeros(n_weights, dtype=int)
                proposal_index = _random.choices([i for i in range(n_weights)], weights=residual_walkers, k=1)
                new_proposal[proposal_index] = 1
                returned_samples[i] += new_proposal
        # otherwise we generate maximum entropy decomposition on the residual float walkers
        elif n_residual_walkers:
            current_approximation = _np.zeros(n_weights)
            cumulative_weights = 0
            randnums = list(_np.sort(_np.random.uniform(size=n_samples))[::-1])

            # we only generate the integer decomposition up to the largest random number we need
            while len(randnums) and randnums[-1] >= cumulative_weights:
                # we sort the residual difference in a descending order
                difference = residual_walkers - current_approximation
                sorted_indices = _np.argsort(difference)[::-1]
                sorted_difference = difference[sorted_indices]

                # we only work with positive differences (there are no negative ones in our algorithm)
                sorted_indices = sorted_indices[sorted_difference > tol]
                sorted_difference = difference[sorted_indices]

                # n_large represents the large degenerate differences and n_small - all the other ones
                # we try to expand n_large by reducing the degenerate probabilities down to a pivot value
                # (effectively increasing the degeneracy every step) and redistributing them equally onto the other
                # values beyond the pivot
                n_large = sorted_difference[_np.abs(sorted_difference - sorted_difference[0]) < tol].shape[0]
                n_small = sorted_difference.shape[0] - n_large

                # the residual walkers are split between the large differences and the small ones apart from a pivot
                # value in the middle. We are always try to put as many as possible walkers put onto the degenerate
                # states.
                n_residual_large = min(n_residual_walkers - 1, n_large)
                n_residual_small = min(n_residual_walkers - n_residual_large, max(n_small - 1, 0))
                n_residual_large = n_residual_walkers - n_residual_small
                rel_weight_large = n_residual_large / n_large
                rel_weight_small = n_residual_small and n_residual_small / (n_small - 1)

                # here we obtain the weights and the average float particle values per residual walker splitting
                # combination
                if n_large >= sorted_difference.shape[0]:
                    # here the pivot is one of our degenerate walkers, i.e. termination condition enforced later
                    new_weight = sorted_difference[0] * n_large / n_residual_walkers
                    assert abs(1 - cumulative_weights - new_weight) < tol
                    new_weight = 1 - cumulative_weights
                    average_particles = _np.array([rel_weight_large] * n_large)
                else:
                    # otherwise we try to increase the number of degenerate particles as long as the lowest weight
                    # walker permits that. Otherwise we reduce the lowest weight walker instead.
                    min_weight_large = sorted_difference[n_large - 1] - sorted_difference[n_large]
                    min_weight_small = sorted_difference[-1]

                    if rel_weight_small and min_weight_large / rel_weight_large > min_weight_small / rel_weight_small:
                        new_weight = min_weight_small / rel_weight_small
                    else:
                        new_weight = min_weight_large / rel_weight_large
                    average_particles = _np.array([rel_weight_large] * n_large + [0] +
                                                  [rel_weight_small] * (n_small - 1))

                average_proposal = _np.zeros(n_weights)
                average_proposal[sorted_indices] = average_particles
                current_approximation += new_weight * average_proposal
                cumulative_weights += new_weight

                # here we sample from the decomposed values if needed
                while True:
                    if len(randnums) and randnums[-1] < cumulative_weights:
                        randnums.pop(-1)
                        # we need to generate a specific combination based on the average walker values
                        # here we can finally sample without replacement and with certainty!
                        new_proposal = _np.zeros(n_weights, dtype=int)
                        large_indices = _random.sample([i for i in range(n_large)], k=n_residual_large)
                        small_indices = _random.sample([n_large + 1 + i for i in range(n_small - 1)],
                                                       k=n_residual_small)
                        sampled_indices = large_indices + small_indices
                        new_proposal[sorted_indices[sampled_indices]] = 1
                        returned_samples[len(randnums)] += new_proposal
                    else:
                        break

            # shuffle the samples so that they are not sorted by random number value
            _np.random.shuffle(returned_samples)

        return returned_samples
