import numpy as np

from typing import Tuple, List


def intersection(intervals: List[Tuple[float, float]]) -> Tuple[float, float]:
    return (max([x[0] for x in intervals]), min(x[1] for x in intervals))


def consistency(C_arr, consistency_type="all", start_index=0):
    assert consistency_type in ["t-1", "all"]
    D = len(C_arr[0])
    consist_arr_all = []
    for i in range(len(C_arr)):
        consist_arr = []
        for d in range(start_index + 1, D):
            C_d = C_arr[i][d][1] - C_arr[i][d][0]
            assert C_d >= 0, f"interval is negative: {C_d}"
            if C_d == 0:
                consist_arr.append(1.0)
            else:
                if consistency_type == "t-1":
                    intersect_interval = intersection([C_arr[i][d - 1], C_arr[i][d]])
                elif consistency_type == "all":
                    intersect_interval = intersection(C_arr[i][: d + 1])
                consist_arr.append(
                    (intersect_interval[1] - intersect_interval[0]) / C_d
                )
        consist_arr_all.append(consist_arr)
    return np.array(consist_arr_all)


def marginal_coverage(y_arr, intervals):
    assert len(y_arr) == len(intervals)
    depth = len(intervals[0])
    coverage = [0 for _ in range(depth)]
    for i in range(len(y_arr)):
        for d in range(depth):
            if intervals[i][d][0] <= y_arr[i] <= intervals[i][d][1]:
                coverage[d] += 1
    return [c / len(y_arr) for c in coverage]
