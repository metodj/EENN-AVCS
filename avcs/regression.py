import torch
import numpy as np

from typing import List, Tuple, Union

from utils.blr import BayesLinearRegressor
from utils.utils import set_all_seeds_torch

OOD_INTERVAL = (0.0, 0.0)


def solve_quadratic_convex(a: float, b: float, c: float) -> Tuple[float, float]:
    if a <= 0.0:
        return OOD_INTERVAL
    D = b**2 - 4 * a * c
    if D >= 0:
        return (
            (-b - np.sqrt(D)) / (2 * a),
            (-b + np.sqrt(D)) / (2 * a),
        )
    else:  # OOD, empty interval
        return OOD_INTERVAL


def eenn_avcs_regression(
    x_star: torch.Tensor,
    BLR_models: List[BayesLinearRegressor],
    alpha: float,
    seed: int = 0,
    S: int = 1,
) -> Tuple[List[List[Tuple[float, float]]], List[float], List[float]]:
    """
    Compute AVCS intervals on top of a pretrained Bayesian Early-Exit Neural Network.

    Parameters:
    x_star: The input values for prediction. Shape: (b, D) where b is the batch size and D is the input dimension
    BLR_models: A list of Bayesian Linear Regression models.
    alpha: The confidence level for interval prediction.
    seed:
    S: The number of weights' samples at each exit.
    pabee:

    Returns:
    A tuple containing the list of AVCS intervals, epistemic uncertainty values, and predictions.
    """
    set_all_seeds_torch(seed)

    albert = BLR_models[0].model is None
    b = x_star.shape[1] if albert else x_star.shape[0]

    c_x = [[] for _ in range(b)]
    epistemic_uncertainty, preds = [], []
    alpha_t, beta_t, gamma_t = (
        torch.zeros(b, 1),
        torch.zeros(b, 1),
        torch.ones(b, 1) * np.log(alpha),
    )
    for t in range(len(BLR_models)):
        assert BLR_models[t].fitted, "BLR model not fitted yet."

        # 1) compute params of the current posterior predictive and likehood
        mu_t, Sigma_t = BLR_models[t].post_mu, BLR_models[t].post_cov

        if albert:
            h_x = x_star[t]
        else:
            _, h_x = BLR_models[t].forward_with_features(x_star)

        h_x = h_x.detach().numpy()
        h_x = np.concatenate([h_x, torch.ones(b, 1)], axis=1)

        sigma_denom = BLR_models[t].sigma_likelihood
        v_star = np.einsum("ij,jk,ik->i", h_x, Sigma_t, h_x)[:, np.newaxis]
        pred_post_var = v_star + sigma_denom**2

        mu_num = np.dot(h_x, mu_t)

        # compute epistemic uncertainty and mean predictions
        epistemic_uncertainty.append(v_star)
        preds.append(mu_num)

        for _ in range(S):
            w_hat_t = np.random.multivariate_normal(mu_t.squeeze(), Sigma_t, size=1)
            mu_denom = np.dot(h_x, w_hat_t.mean(axis=0)[:, np.newaxis])

            # 2) Update the params of quadratic equation
            alpha_t += 0.5 * (1 / sigma_denom**2 - 1 / pred_post_var)
            beta_t += mu_num / pred_post_var - mu_denom / sigma_denom**2
            gamma_t += (
                mu_denom**2 / (2 * sigma_denom**2)
                - mu_num**2 / (2 * pred_post_var)
                + np.log(sigma_denom)
                - np.log(np.sqrt(pred_post_var))
            )

        # 3) solve the quadratic equation to get the interval C_t(x)
        for i in range(b):
            c_x[i].append(
                solve_quadratic_convex(
                    alpha_t[i].item(), beta_t[i].item(), gamma_t[i].item()
                )
            )

    return c_x, epistemic_uncertainty, preds


def parallel_eenn_avcs_regression(
    x_star: np.array, BLR_models: List[BayesLinearRegressor], alpha: float, S: int = 5
) -> Tuple[List[List[Tuple[float, float]]], List[float], List[float]]:
    c_avcs_parallel = []
    for s in range(S):
        c_avcs, epistem_uncer, preds = eenn_avcs_regression(
            x_star=x_star, BLR_models=BLR_models, alpha=alpha, seed=s
        )
        c_avcs_parallel.append(c_avcs)

    # take intersection between intervals from different seeds
    C_all_avcs = []
    for i in range(len(c_avcs_parallel[0])):
        C_i = []
        for l in range(len(c_avcs_parallel[0][0])):
            if OOD_INTERVAL in [
                c_avcs_parallel[s][i][l] for s in range(len(c_avcs_parallel))
            ]:
                C_i.append(OOD_INTERVAL)
            else:
                C_i.append(
                    (
                        np.max(
                            [
                                c_avcs_parallel[s][i][l][0]
                                for s in range(len(c_avcs_parallel))
                            ]
                        ),
                        np.min(
                            [
                                c_avcs_parallel[s][i][l][1]
                                for s in range(len(c_avcs_parallel))
                            ]
                        ),
                    )
                )
        C_all_avcs.append(C_i)

    return C_all_avcs, epistem_uncer, preds


def running_intersection(
    intervals: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    L = len(intervals)
    running_intersection = [intervals[0]]
    for i in range(1, L):
        y_L = max(running_intersection[i - 1][0], intervals[i][0])
        y_R = min(running_intersection[i - 1][1], intervals[i][1])
        if y_L > y_R:
            running_intersection.extend([OOD_INTERVAL for _ in range(i, L)])
            break
        running_intersection.append(
            (
                max(running_intersection[i - 1][0], intervals[i][0]),
                min(running_intersection[i - 1][1], intervals[i][1]),
            )
        )
    return running_intersection


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
            # assert C_d >= 0, f"interval is negative: {C_d}"
            if C_d < 0:
                print(f"interval is negative: {C_d}")
                consist_arr.append(0.)
            elif C_d == 0:
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
