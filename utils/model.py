import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from typing import List

from utils.blr import BayesLinearRegressor


class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.1),
        )

    def forward(self, x):
        return x + self.block(x)


class MLP(nn.Module):
    def __init__(self, input_dim, width, output_dim, depth):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(self.input_dim, width)
        self.depth = depth
        self.width = width

        self.layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        self.output_layers.append(nn.Linear(width, self.output_dim))
        for d in range(self.depth):
            self.layers.append(ResBlock(width))
            self.output_layers.append(nn.Linear(width, self.output_dim))

    def forward(self, x):
        act_vec = torch.zeros(self.depth + 1, x.shape[0], self.output_dim).type(
            x.type()
        )
        x = self.input_layer(x)
        act_vec[0] = self.output_layers[0](x)
        for i in range(self.depth):
            x = self.layers[i](x)
            act_vec[i + 1] = self.output_layers[i + 1](x)
        return act_vec


class MLP_l(nn.Module):
    def __init__(self, input_dim, width, output_dim, depth):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(self.input_dim, width)
        self.output_layer = nn.Linear(width, self.output_dim)
        self.depth = depth
        self.width = width

        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            self.layers.append(ResBlock(width))

    def forward(self, x):
        x = self.input_layer(x)
        for i in range(self.depth):
            x = self.layers[i](x)
        return self.output_layer(x)


def get_params_at_depth_l(model, l):
    params = OrderedDict()
    for name, param in model.state_dict().items():
        if f"output_layers.{l}." in name:
            params[name.replace(f"s.{l}", "")] = param

        block_bool = [f".{_l}.block" in name for _l in range(l)]
        if "input_layer" in name or any(block_bool):
            params[name] = param

    return params


def fit_eenn_blr(
    width: int,
    depth: int,
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    device: str,
    blr_mu_prior_MLP: bool = False,
    lr: float = 1e-3,
    momentum: float = 0.9,
    wd: float = 1e-4,
    epochs: int = 500,
    print_fit_every: int = 100,
    plot_loss: bool = False,
) -> List[BayesLinearRegressor]:
    # init EENN
    model = MLP(1, width, 1, depth).to(device)
    print(f"Nr. params: {sum(p.numel() for p in model.parameters())}")

    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_dataloader = DataLoader(train_data, batch_size=501, shuffle=True)
    N = len(X_train)

    optim = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=wd
    )
    loss_func = torch.nn.MSELoss()

    # train EENN
    for epoch in range(epochs):
        e_loss = 0.0
        model.train()
        for x, y in train_dataloader:
            optim.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            b_loss = 0.0
            for l in range(model.depth + 1):
                b_loss_l = loss_func(y_pred[l], y)
                b_loss += b_loss_l

            b_loss /= model.depth
            b_loss.backward()
            optim.step()

            e_loss += b_loss * x.shape[0]
        e_loss /= N

        if epoch % print_fit_every == 0:
            print(f"Epoch {epoch} | Loss: {e_loss:.4f}")
            if plot_loss:
                model.eval()
                xs = torch.range(-3, 3, step=0.01)
                ys_pred = model(xs.view(-1, 1).to(device))

                xs = xs.numpy()
                ys_pred = ys_pred.cpu().detach().numpy()[:, :, 0]

                fig, ax = plt.subplots(1, 1, figsize=(15, 5))
                # plot train and validation loss per depth

                test_pred = model(torch.Tensor(X_test))
                loss_depth_test = [
                    loss_func(test_pred[l], torch.Tensor(y_test)).item()
                    for l in range(depth + 1)
                ]
                train_pred = model(torch.Tensor(X_train))
                loss_depth_train = [
                    loss_func(train_pred[l], torch.Tensor(y_train)).item()
                    for l in range(depth + 1)
                ]
                ax.plot(range(1, depth + 1), loss_depth_test[1:], label="test loss")
                ax.plot(range(1, depth + 1), loss_depth_train[1:], label="train loss")
                plt.legend()

                plt.show()

    model.eval()

    if blr_mu_prior_MLP:
        mu_priors = []
        for name, param in model.named_parameters():
            if "output" in name:
                mu_priors.append((name, param.data.flatten().cpu().numpy()))
        mu_priors = [
            np.concatenate((mu_priors[l][1], mu_priors[l + 1][1]))
            for l in range(0, len(mu_priors), 2)
        ]

    # define MLP_l and initialize them using fitted MLP
    MLPs = []
    for l in range(depth + 1):
        model_l = MLP_l(1, width, 1, l).to(device)
        model_l.load_state_dict(get_params_at_depth_l(model, l=l))
        model_l.eval()
        MLPs.append(model_l)

    # fit BLR at each exit
    BLRs = []
    for d in range(depth + 1):
        if blr_mu_prior_MLP:
            BLR_d = BayesLinearRegressor(MLPs[d], mu_prior=mu_priors[d])
        else:
            BLR_d = BayesLinearRegressor(MLPs[d])
        BLR_d.fit(X_train, y_train)
        BLRs.append(BLR_d)

    return BLRs
