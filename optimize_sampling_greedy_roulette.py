import numpy as np
import pdb

from tensorflow import flags
from copy import deepcopy

FLAGS = flags.FLAGS

def euclidean_proj_simplex(v, s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def optimize_q(c, sq_norms):
    '''Return q propto sqrt(sq_norms / c)'''
    c_diffs = np.array([c[i] - (0. if i < 1 else c[i-1]) for i in range(len(c))])
    c_diffs[np.logical_not(np.isfinite(c_diffs))] = 0.0
    #sq_norms[not np.isfinite(sq_norms)]
    p_j_geq_i_unnormalized = np.sqrt(sq_norms / (c_diffs+1e-14))
    for i in range(len(p_j_geq_i_unnormalized)-1):
        if p_j_geq_i_unnormalized[-(i+2)] < p_j_geq_i_unnormalized[-i+1]:
            p_j_geq_i_unnormalized[-(i+2)] = p_j_geq_i_unnormalized[-i+1]
    q_unnormalized = np.array(
        [p - (0. if i+1 == len(p_j_geq_i_unnormalized)
              else p_j_geq_i_unnormalized[i+1])
         for i, p in enumerate(p_j_geq_i_unnormalized)
        ])
    q_unnormalized = np.maximum(0., q_unnormalized)
    q = np.array(q_unnormalized) / sum(q_unnormalized)
    return q

def compute_and_variance(q, c, sq_norms):
    weights = [1/(q[i:].sum()) for i in range(len(q))]
    wn = np.array(weights) * sq_norms
    wn_sum = np.cumsum(wn)
    expected_variance = (q*wn_sum).sum()
    expected_compute = (q*c).sum()
    return expected_compute, expected_variance

def get_sq_norm_seq(sq_norms_matrix, idxs):
    sq_norms = []
    for i in range(len(idxs)):
        idx1 = 0 if i < 1 else idxs[i-1] + 1
        idx2 = idxs[i]
        v = sq_norms_matrix[idx1, idx2]
        sq_norms.append(v)
    return np.array(sq_norms)

def get_c_seq(c, idxs):
    c_seq = []
    try:
        compute_penalty = FLAGS.partial_update or FLAGS.compute_penalty
    except Exception as e:
        compute_penalty = False
    if compute_penalty:
        for i in range(len(idxs)):
            c_seq.append(sum([c[idxs[j]] for j in range(0, i+1)]))
    else:
        c_seq = [c[i] for i in idxs]
    return np.array(c_seq)

def cost(sq_norm_matrix, c, idxs, return_cv=False):
    sq_norms = get_sq_norm_seq(sq_norm_matrix, idxs)
    c = get_c_seq(c, idxs)
    q = optimize_q(c, sq_norms)
    cval, vval = compute_and_variance(q, c, sq_norms)
    costval = cval * (vval ** FLAGS.variance_weight)
    if not np.isfinite(costval):
        pdb.set_trace()
    if return_cv:
        return costval, cval, vval
    else:
        return costval

def get_q(sq_norm_matrix, c, idxs):
    sq_norms = get_sq_norm_seq(sq_norm_matrix, idxs)
    c = get_c_seq(c, idxs)
    return optimize_q(c, sq_norms)

def optimize_remove(sq_norm_matrix, c, idxs, verbose=False):
    idxs = deepcopy(idxs)
    baseline = cost(sq_norm_matrix, c, idxs)
    converged = False
    while not converged and len(idxs) > 1:
        if verbose:
            print("Not yet converged")
        converged = True
        i = len(idxs)-2
        while i >= 0:
            # Try eliminating every intermediate value
            idxs_minus_i = idxs[:i] + idxs[i+1:]
            cost_minus_i = cost(sq_norm_matrix, c, idxs_minus_i)
            if cost_minus_i < baseline:
                if verbose:
                    print("{}, trial cost {} under baseline {}".format(
                        i, cost_minus_i, baseline))
                    print("removing idx {}, remaining {}".format(idxs[i], idxs_minus_i))
                baseline = cost_minus_i
                idxs = idxs_minus_i
                converged = False
                break
            else:
                if verbose:
                    print("{}, trial cost {} not under baseline {}".format(
                        i, cost_minus_i, baseline))
            i -= 1
    q = get_q(sq_norm_matrix, c, idxs)
    if verbose:
        print("Converged. Final idxs: {}. Final ps: {}".format(idxs, q))
    return idxs, q

def idxs_from_negative(negative_idxs, idxs):
    return [i for i in idxs if i not in negative_idxs]

def optimize_add(sq_norm_matrix, c, idxs, verbose=False, logger=None):
    idxs = deepcopy(idxs)
    negative_idxs = idxs[:-1]
    baseline = cost(sq_norm_matrix, c, idxs_from_negative(negative_idxs, idxs))
    converged = False
    while not converged and len(negative_idxs) > 0:
        if verbose:
            print("Not yet converged")
        converged = True
        i = 0
        while i <= len(negative_idxs)-1:
            idxs_minus_i = negative_idxs[:i] + negative_idxs[i+1:]
            cost_minus_i = cost(sq_norm_matrix, c,
                                idxs_from_negative(idxs_minus_i, idxs))
            if cost_minus_i < baseline:
                if verbose:
                    print("{}, trial cost {} under baseline {}".format(
                        i, cost_minus_i, baseline))
                    print("adding idx {}, giving {}".format(
                        negative_idxs[i],
                        idxs_from_negative(idxs_minus_i, idxs)))
                baseline = cost_minus_i
                negative_idxs = idxs_minus_i
                converged = False
                break
            else:
                if verbose:
                    print("{}, trial cost {} not under baseline {}".format(
                        i, cost_minus_i, baseline))
            i += 1
    idxs = idxs_from_negative(negative_idxs, idxs)
    q = get_q(sq_norm_matrix, c, idxs)
    if verbose:
        print("Converged. Final idxs: {}. Final ps: {}".format(idxs, q))
    return idxs, q

def optimize_greedy_roulette(sq_norm_matrix, c,
                             idxs, verbose=False, logger=None):
    '''Greedily optimize a RT sampler.
    Args:
        sq_norm_matrix: N+1 x N array
                        entries [0, j]: sq norm of g_j
                        entries [i+1, j]: sq norm of g_j - g_i

        idxs:           all remaining nodes under consideration
    '''
    # Try greedily optimizing idxs by starting with all and removing
    base_cost, base_c, base_v = cost(sq_norm_matrix, c, [idxs[-1]],
                                     return_cv=True)
    try:
        force_all_idxs = FLAGS.force_all_idxs
    except Exception as e:
        force_all_idxs = False
    if force_all_idxs:
        if verbose:
            print("Forcing using all idxs")
        q = get_q(sq_norm_matrix, c, idxs)
    else:
        idxs_remove, q_remove = optimize_remove(sq_norm_matrix, c, idxs, verbose)
        idxs_add, q_add = optimize_add(sq_norm_matrix, c, idxs, verbose)
        cost_remove = cost(sq_norm_matrix, c, idxs_remove)
        cost_add = cost(sq_norm_matrix, c, idxs_add)

        if cost_remove < cost_add:
            if verbose:
                print("Greedy remove cost {} < greedy add cost {}.".format(
                    cost_remove, cost_add))
                print("Returning greedy remove idxs {} instead of greedy add "
                      "idxs {}".format(idxs_remove, idxs_add))
            idxs = idxs_remove
            q = q_remove

        else:
            if verbose:
                print("Greedy add cost {} <= greedy remove cost {}.".format(
                    cost_add, cost_remove))
                print("Returning greedy add idxs {} instead of greedy remove "
                      "idxs {}".format(idxs_add, idxs_remove))
            idxs = idxs_add
            q = q_add

    costval, cval, vval = cost(sq_norm_matrix, c, idxs, return_cv=True)

    if logger:
        logger.info(
            "Optimized RT. idxs: {}. q: {}".format(idxs, q))
        logger.info("RT estimator has cost " +
            "{:.2f}, compute: {:.2f}, variance: {:.2f}".format(
                costval, cval, vval
            ))
        logger.info("Deterministic estimator has " +
            "cost: {:.2f}, compute {:.2f}, variance {:.2f}".format(
                base_cost, base_c, base_v
            ))
        logger.info("Change factors are " +
            "cost {:.2f}, compute {:.2f}, variance {:.2f}.".format(
                costval/base_cost, cval/base_c, vval/base_v
            ))

    return idxs, q
