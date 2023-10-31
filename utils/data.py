import numpy as np
import math
from sklearn.model_selection import train_test_split

from typing import Optional

from utils.utils import set_all_seeds_torch


def get_synthetic_data(data_type: str, seed: int = 42):
    """
    Data loader for regression synthetic data.

    Parameters
    ----------
    data_type : str
        Type of data to load. Options are:
        - "3_clusters_homoskedastic"
        - "3_clusters_heteroskedastic"
        - "wiggle"
        - "1_cluster_homoskedastic"
        - "bimodal"
    seed : int
        Random seed to use for generating data.
    -------
    """
    set_all_seeds_torch(seed)

    if data_type == "3_clusters_homoskedastic":
        X_train, y_train, X_test, y_test = gen_simple_1d(hetero=False)
    elif data_type == "3_clusters_heteroskedastic":
        X_train, y_train, X_test, y_test = gen_simple_1d(hetero=False)
    elif data_type == "wiggle":
        X_train, y_train, X_test, y_test, _, _ = gen_wiggle()
    elif data_type == "1_cluster_homoskedastic":
        X_train, y_train, X_test, y_test, _, _ = generate_regression_outputs(type="hsc")
    elif data_type == "bimodal":
        X_train, y_train, X_test, y_test, _, _ = generate_regression_outputs(
            type="bimodal"
        )
    else:
        raise NotImplementedError()

    # get data range for plotting
    eps = 0.5
    x_min, x_max = X_train.min() - eps, X_train.max() + eps
    y_min, y_max = y_train.min() - eps, y_train.max() + eps

    return X_train, y_train, X_test, y_test, y_max, y_min, x_min, x_max


def gt_function(x):
    return x - 0.1 * x**2 + np.cos(np.pi * x / 2)


"""
Functions below adapted from:  https://github.com/cambridge-mlg/DUN
"""


def gen_simple_1d(hetero=False):
    np.random.seed(0)
    Npoints = 1002
    x0 = np.random.uniform(-1, 0, size=int(Npoints / 3))
    x1 = np.random.uniform(1.7, 2.5, size=int(Npoints / 3))
    x2 = np.random.uniform(4, 5, size=int(Npoints / 3))
    x = np.concatenate([x0, x1, x2])

    y = gt_function(x)

    homo_noise_std = 0.25
    homo_noise = np.random.randn(*x.shape) * homo_noise_std
    y_homo = y + homo_noise

    hetero_noise_std = np.abs(0.1 * np.abs(x) ** 1.5)
    hetero_noise = np.random.randn(*x.shape) * hetero_noise_std
    y_hetero = y + hetero_noise

    X = x[:, np.newaxis]
    y_joint = np.stack([y_homo, y_hetero], axis=1)

    X_train, X_test, y_joint_train, y_joint_test = train_test_split(
        X, y_joint, test_size=0.5, random_state=42
    )
    y_hetero_train, y_hetero_test = (
        y_joint_train[:, 1, np.newaxis],
        y_joint_test[:, 1, np.newaxis],
    )
    y_homo_train, y_homo_test = (
        y_joint_train[:, 0, np.newaxis],
        y_joint_test[:, 0, np.newaxis],
    )

    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    y_hetero_means, y_hetero_stds = y_hetero_train.mean(axis=0), y_hetero_train.std(
        axis=0
    )
    y_homo_means, y_homo_stds = y_homo_test.mean(axis=0), y_homo_test.std(axis=0)

    X_train = ((X_train - x_means) / x_stds).astype(np.float32)
    X_test = ((X_test - x_means) / x_stds).astype(np.float32)

    y_hetero_train = ((y_hetero_train - y_hetero_means) / y_hetero_stds).astype(
        np.float32
    )
    y_hetero_test = ((y_hetero_test - y_hetero_means) / y_hetero_stds).astype(
        np.float32
    )

    y_homo_train = ((y_homo_train - y_homo_means) / y_homo_stds).astype(np.float32)
    y_homo_test = ((y_homo_test - y_homo_means) / y_homo_stds).astype(np.float32)

    if hetero:
        return X_train, y_hetero_train, X_test, y_hetero_test
    else:
        return X_train, y_homo_train, X_test, y_homo_test


def _zip_dataset(x, y):
    return list(zip(x, y))


def _normalise_by_train(train_data, test_data, valid_data):
    train_mean, train_std = train_data.mean(axis=0), train_data.std(axis=0)

    train_data_norm = ((train_data - train_mean) / train_std).astype(np.float32)
    valid_data_norm = ((valid_data - train_mean) / train_std).astype(np.float32)
    test_data_norm = ((test_data - train_mean) / train_std).astype(np.float32)
    return (train_data_norm, test_data_norm, valid_data_norm)


def _train_valid_test_split(x, y, seed):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=seed
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.1, random_state=seed
    )

    return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)


def gen_wiggle(
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = 42,
    noise_std: Optional[float] = None,
):
    if n_samples is None:
        n_samples = 900
    if random_seed is not None:
        np.random.seed(random_seed)
    if noise_std is None:
        noise_std = 0.25

    x = np.random.randn(n_samples) * 2.5 + 5

    def function(x):
        return np.sin(np.pi * x) + 0.2 * np.cos(np.pi * x * 4) - 0.3 * x

    y = function(x)

    noise = np.random.randn(*x.shape) * noise_std
    y = y + noise

    x = x[:, np.newaxis]
    y = y[:, np.newaxis]

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = _train_valid_test_split(
        x, y, random_seed
    )

    (x_train, x_test, x_valid) = _normalise_by_train(x_train, x_test, x_valid)
    (y_train, y_test, y_valid) = _normalise_by_train(y_train, y_test, y_valid)

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def sample_gaussian_mixture(N, means, covs, weights):
    """
    Copied from: https://github.com/2020fa-207-final-project/decomposition-of-uncertainty

    Sample data from a mixture of Gaussians

    Parameters:
        N : int - size of the sample
        means : list - mean value of each gaussian
        covs : covariances of each gaussian
        weights : mixture weights for each gaussian

    Returns:
        mixture_sample : N samples from the mixture of Gaussians
    """
    # Sample from a multinomial to find the N for each Gaussian
    Z = np.random.multinomial(n=N, pvals=weights)

    # Create samples from each mixture z
    samples = []
    for mean, cov, z in zip(means, covs, Z):
        samples.append(np.random.normal(mean, np.sqrt(cov), z))

    # Flatten the resulting list and return as np.array
    mixture_samples = np.array([sample for subset in samples for sample in subset])
    return mixture_samples


def generate_regression_outputs(type="hsc", N=None, X=None, random_seed=42):
    """
    Copied from: https://github.com/2020fa-207-final-project/decomposition-of-uncertainty

    Generate dummy regression data for testing the BNN+LV

    Parameters:
        type : string - 'hsc' or 'bimodal' (hsc = heteroscedastic)
        N : number of samples - defaults to 750 or len(X) if an X array is passed
        X : np.array - X data used to generate Y data, if not provided defaults
                       to those provided in the paper for each type

    Returns:
        data_tuple : (Y, X)
    """
    # Set up N
    if N is None:
        if X is None:
            N = 960
        else:
            N = X.shape[0]

    if type == "hsc":
        # Functional form of the true relationship
        eqn = lambda x: 7 * np.sin(x) + 3 * abs(np.cos(x / 2)) * np.random.normal(0, 1)
        eqn = np.vectorize(eqn)

        # If no X data is passed then generate it
        if X is None:
            X = sample_gaussian_mixture(
                N,
                means=[-4, 0, 4],
                covs=[(2 / 5) ** 2, 0.9**2, (2 / 5) ** 2],
                weights=[1 / 3, 1 / 3, 1 / 3],
            )

        # Sample the Y data
        Y = eqn(X)

    elif type == "bimodal":
        # Functional form of the true relationship and a bernoulli variable to determine mode
        eqn1 = lambda x, z: z * (10 * np.cos(x) + np.random.normal(0, 1)) + (1 - z) * (
            10 * np.sin(x) + np.random.normal(0, 1)
        )
        eqn1 = np.vectorize(eqn1)
        Z = np.random.binomial(1, 0.5, size=N)

        # If no X data is passed then generate it
        if X is None:
            X = np.random.exponential(1 / 2, size=N)

            # Rescale to bound between [-0.5, 2] (exponential is from 0 to inf)
            X = (X / (np.max(X) / 2.5)) - 0.5

        Y = eqn1(X, Z)

    else:
        raise ValueError("Error: type must be one of 'hsc' or 'bimodal'")

    # convert to float32
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = _train_valid_test_split(
        X, Y, random_seed
    )

    return (
        x_train.reshape(-1, 1),
        y_train.reshape(-1, 1),
        x_test.reshape(-1, 1),
        y_test.reshape(-1, 1),
        x_valid.reshape(-1, 1),
        y_valid.reshape(-1, 1),
    )
