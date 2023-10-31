"""
Code modified based on this blog post:     
    http://krasserm.github.io/2019/02/23/bayesian-linear-regression/
"""
from laplace.utils import FeatureExtractor
import numpy as np
import torch.nn as nn
import torch

from typing import Optional, List, Tuple


class BayesLinearRegressor(FeatureExtractor):
    def __init__(self, model: Optional[nn.Module], mu_prior: Optional[np.array] = None):
        if model:
            super().__init__(model)
        else:
            self.model = None
        self.sigma_likelihood = None
        self.sigma_prior = None

        self.mu_prior = mu_prior

        self.post_mu = None
        self.post_cov = None

        self.fitted = False

    def posterior(self, X, y, alpha, beta, return_inverse=False):
        """Computes mean and covariance matrix of the posterior distribution."""
        S_N_inv = alpha * np.eye(X.shape[1]) + beta * X.T.dot(X)
        S_N = np.linalg.inv(S_N_inv)
        if self.mu_prior is None:
            m_N = beta * S_N.dot(X.T).dot(y)
        else:
            m_N = beta * S_N.dot(X.T).dot(y) + alpha * np.matmul(
                S_N, self.mu_prior
            ).reshape(-1, 1)

        if return_inverse:
            return m_N, S_N, S_N_inv
        else:
            return m_N, S_N

    def posterior_predictive(self, X_test):
        """Computes mean and variances of the posterior predictive distribution."""
        assert self.fitted, "Model not fitted yet!"
        if self.model:
            _, X_test = self.forward_with_features(torch.tensor(X_test))
        X_test = torch.cat([X_test, torch.ones((X_test.shape[0], 1))], dim=1)
        X_test = X_test.detach().numpy()
        y = X_test.dot(self.post_mu)
        # Only compute variances (diagonal elements of covariance matrix)
        y_var = self.sigma_likelihood**2 + np.sum(
            X_test.dot(self.post_cov) * X_test, axis=1
        )

        return y, y_var

    def fit(
        self,
        X: np.array,
        y: np.array,
        alpha_0=1e-5,
        beta_0=1e-5,
        max_iter=1000,
        rtol=1e-5,
        verbose=False,
    ):
        """
        Jointly infers the posterior sufficient statistics and optimal values
        for alpha and beta by maximizing the log marginal likelihood.

        Args:
            X: Design matrix (N x M).
            y: Target value array (N x 1).
            alpha_0: Initial value for alpha.
            beta_0: Initial value for beta.
            max_iter: Maximum number of iterations.
            rtol: Convergence criterion.

        Returns:
        """

        if self.model:
            # pass X through the NN and get the features h(X)
            _, X = self.forward_with_features(torch.tensor(X))
        X = torch.cat([X, torch.ones((X.shape[0], 1))], dim=1)
        X = X.detach().numpy()

        N, M = X.shape

        eigenvalues_0 = np.linalg.eigvalsh(X.T.dot(X))

        beta = beta_0
        alpha = alpha_0

        for i in range(max_iter):
            beta_prev = beta
            alpha_prev = alpha

            eigenvalues = eigenvalues_0 * beta

            m_N, S_N, _ = self.posterior(X, y, alpha, beta, return_inverse=True)

            gamma = np.sum(eigenvalues / (eigenvalues + alpha))
            if self.mu_prior is None:
                alpha = gamma / np.sum(m_N**2)
            else:
                alpha = gamma / np.sum((m_N - self.mu_prior) ** 2)

            beta_inv = 1 / (N - gamma) * (np.sum((y - X.dot(m_N)) ** 2))
            beta = 1 / beta_inv

            if np.isclose(alpha_prev, alpha, rtol=rtol) and np.isclose(
                beta_prev, beta, rtol=rtol
            ):
                if verbose:
                    print(f"Convergence after {i + 1} iterations.")
                break

        if verbose:
            print(f"Stopped after {max_iter} iterations.")

        self.post_mu = m_N
        self.post_cov = S_N
        self.sigma_likelihood = 1 / np.sqrt(beta)
        self.sigma_prior = 1 / np.sqrt(alpha)
        self.fitted = True

        assert self.sigma_likelihood >= 0.0
        assert self.sigma_prior >= 0.0


def eenn_bayes_intervals(
    x_star: float,
    BLR_models: List[BayesLinearRegressor],
    n_std: int = 2,
) -> List[Tuple[float, float]]:
    """
    y_L = mu_pred - n_std * std_pred
    y_R = mu_pred + n_std * std_pred
    """
    albert = BLR_models[0].model is None
    b = x_star.shape[1] if albert else x_star.shape[0]
    c_x = []
    for t, blr_d in enumerate(BLR_models):
        if albert:
            y_mu, y_var = blr_d.posterior_predictive(x_star[t])
        else:
            y_mu, y_var = blr_d.posterior_predictive(x_star)
        y_mu = y_mu.squeeze()
        y_var = y_var.squeeze()
        y_L = y_mu - n_std * np.sqrt(y_var)
        y_R = y_mu + n_std * np.sqrt(y_var)
        c_x.append((y_L, y_R))

    # transpose
    c_x = [
        [(c_x[t][0][i], c_x[t][1][i]) for t in range(len(BLR_models))] for i in range(b)
    ]
    return c_x
