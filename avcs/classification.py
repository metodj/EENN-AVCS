import numpy as np
from typing import List


def eenn_avcs_classification(
    logits: np.ndarray,
    thresholds: np.ndarray,
    alpha: float,
    seed: int = 0,
) -> List[List[List[int]]]:
    np.random.seed(seed)
    L, N, _ = logits.shape

    logits_clip = np.where(logits < thresholds[:, np.newaxis, np.newaxis], 0, logits)

    C = []
    for i in range(N):
        R_i = np.ones_like(logits_clip[:, i, :]) * np.inf
        for l in range(L):
            dir_support = np.where(logits_clip[l, i] > 0.0)[0]
            dir_alphas = logits_clip[l, i][dir_support]
            R_support_denom = np.random.dirichlet(dir_alphas)
            R_support = (dir_alphas / dir_alphas.sum()) / R_support_denom
            R_i[l, dir_support] = R_support

        R_i = np.array(R_i)
        R_i = R_i.cumprod(axis=0)
        C.append([list(np.where(row <= (1 / alpha))[0]) for row in R_i])

    return C


def find_relu_thresholds(
    logits, targets, init_thresholds, alpha, thres_delta=0.5, seed=0
):
    L, _, _ = logits.shape

    relu_thres = init_thresholds
    for thres_l in range(L):
        while True:
            relu_thres[thres_l] += thres_delta

            c_avcs = eenn_avcs_classification(logits, relu_thres, alpha, seed=seed)
            coverage = marignal_coverage_classification(c_avcs, targets)

            if not (coverage > 1 - alpha).all():
                relu_thres[thres_l] -= thres_delta
                print(thres_l, relu_thres)
                c_avcs = eenn_avcs_classification(logits, relu_thres, alpha, seed=seed)
                break

    return relu_thres


def sizes_classification(c: List[List[List[int]]]) -> List[List[int]]:
    return [[len(c_l) for c_l in c_i] for c_i in c]


def marignal_coverage_classification(
    c: List[List[List[int]]], targets: np.ndarray
) -> List[List[float]]:
    N, L = len(c), len(c[0])
    coverage = [0.0 for _ in range(L)]
    for i in range(N):
        for l in range(L):
            if targets[i] in c[i][l]:
                coverage[l] += 1

    coverage = np.array(coverage) / N
    return coverage


def consistency_classifciation(sets: List[List[int]]) -> float:
    L = len(sets)
    sets_intersect = running_intersection_classification(sets)
    cons_arr = []
    for l in range(L):
        size_intersect = len(sets_intersect[l])
        size = len(sets[l])
        if size_intersect == 0 and size > 0:
            cons_arr.append(0.0)
        elif size_intersect == 0 and size == 0:
            cons_arr.append(1.0)
        elif size_intersect > 0 and size == 0:
            cons_arr.append(1.0)
        else:
            cons_arr.append(size_intersect / size)
    return cons_arr


def running_intersection_classification(sets: List[List[int]]) -> List[List[int]]:
    sets_intersect = []
    curr_intersect = set(sets[0])
    for sublist in sets:
        curr_intersect &= set(sublist)
        sets_intersect.append(list(curr_intersect))
    return sets_intersect
