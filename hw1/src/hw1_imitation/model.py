"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim
        output_dim = chunk_size * action_dim
        dims = (input_dim, *hidden_dims, output_dim)

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))

            # ReLU after every layer except the output layer
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_action_chunk = self.sample_actions(state=state)
        # Eq. (1): squared L2 norm per chunk, averaged over the batch
        return (action_chunk - pred_action_chunk).pow(2).sum(dim=(1, 2)).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        actions = self.network(state)
        return actions.view(-1, self.chunk_size, self.action_dim)  # (B, N, A)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        input_dim = state_dim + chunk_size * action_dim + 1
        output_dim = chunk_size * action_dim
        dims = (input_dim, *hidden_dims, output_dim)

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))

            # ReLU after every layer except the output layer
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        a1 = action_chunk.reshape(B, self.chunk_size * self.action_dim)
        a0 = torch.randn_like(a1)
        tau = torch.rand((B, 1), device=state.device, dtype=state.dtype)
        a_tau = tau * a1 + (1 - tau) * a0
        v = self._velocity(state, a_tau, tau)
        return (v - (a1 - a0)).pow(2).sum(dim=1).mean()  # Eq (2)

    def _velocity(self, state, a_tau, tau):
        return self.network(torch.cat([state, a_tau, tau], dim=1))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        actions = torch.randn(B, self.chunk_size * self.action_dim,
                        device=state.device, dtype=state.dtype)
        for k in range(num_steps):
            tau = torch.full((B, 1), k / num_steps,
                             device=state.device, dtype=state.dtype)
            actions = actions + self._velocity(state, actions, tau) / num_steps  # Eq (3)
        return actions.view(B, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
