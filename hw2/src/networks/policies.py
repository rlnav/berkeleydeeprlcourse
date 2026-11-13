import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        obs = ptu.from_numpy(obs[np.newaxis])
        distribution = self.forward(obs)
        actions = distribution.sample()
        return ptu.to_numpy(actions.squeeze(0))

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            distribution = D.Categorical(logits=logits)
            return distribution
        else:
            # define the forward pass for a policy with a continuous action space.
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            distribution = D.Normal(mean, std)
            return distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        r"""Implements the policy gradient actor update.

        Performs one gradient step on the surrogate loss, whose gradient is the policy gradient estimate:

            L(theta) = -(1/N) * sum_i A_i * log pi_theta(a_i | s_i)

            grad_theta J(theta) ~= (1/N) * sum_i grad_theta log pi_theta(a_i | s_i) * A_i = -grad_theta L(theta)

        where i indexes every (s_t, a_t) pair in the batch, N = sum_i T_i is the total number of transitions, and
        A_i is the advantage (e.g. the discounted return or reward-to-go, optionally minus a baseline). The advantages
        are treated as constants, so no gradient flows through them.

        Args:
            obs: observations s_i, shape (N, ob_dim).
            actions: actions a_i taken in the rollouts, shape (N,). Integer indices for discrete action spaces.
            advantages: advantage estimates A_i, shape (N,).

        Returns:
            A dict with the scalar loss under the key "Actor Loss".
        """
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # compute the policy gradient actor loss
        self.optimizer.zero_grad()
        distribution = self.forward(obs)
        log_probs = distribution.log_prob(actions)
        loss = -(log_probs * advantages).mean()

        # perform an optimizer step
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
