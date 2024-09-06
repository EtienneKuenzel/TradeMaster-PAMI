from .builder import NETS
from .custom import Net
from torch import Tensor
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional, Any
import torch.utils.checkpoint as checkpoint
"""@NETS.register_module()
class QNet(Net):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, explore_rate = 0.25):
        super().__init__()
        print("aaa")
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_rate = explore_rate
        self.action_dim = action_dim

        # init weights
        self.net.apply(self.init_weights)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # Q values for multiple actions

    def get_action(self, state: Tensor) -> Tensor:  # return the index [int] of discrete action for exploration
        if self.explore_rate < torch.rand(1):
            action = self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action

    def init_weights(self, m):
        # init linear
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform(m.weight)
            m.bias.data.zero_()"""
@NETS.register_module()
class QNet(Net):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, explore_rate = 0.25):
        super(QNet, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Create the input layer
        self.layers.append(nn.Linear(82, 64))
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('linear'))
        self.layers.append(nn.Linear(64, 32))
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('linear'))
        # Create the output layer (Q-values for actions)
        self.layers.append(nn.Linear(32, 3))
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('linear'))

        # Create PAU (Parameterized Activation Units) activation functions
        a = torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983), dtype=torch.float)
        for weights in [a] * 2:
            self.activations.append(PAU(weights=weights, cuda=torch.cuda.is_available()).requires_grad_(True))

    def forward(self, state: Tensor) -> Tensor:
        # Pass the input through layers with PAU activations
        x = state
        for i in range(len(self.activations)):
            x = self.activations[i](self.layers[i](x))

        # Final layer without activation (output Q-values for all actions)
        x = self.layers[-1](x)
        return x

    def get_action(self, state: Tensor) -> Tensor:  # return the index [int] of discrete action for exploration
        if 0.25 < torch.rand(1):
            x = state
            for i in range(len(self.activations)):
                x = self.activations[i](self.layers[i](x))

            # Final layer without activation (output Q-values for all actions)
            x = self.layers[-1](x)
            action = x.argmax(dim=1)
            action = action.view(1, 1)
        else:
            action = torch.randint(3, size=(state.shape[0], 1))
        return action
class PAU(nn.Module):
    """
    This class implements the Pade Activation Unit proposed in:
    https://arxiv.org/pdf/1907.06732.pdf
    """

    def __init__(
            self,
            weights,
            m: int = 5,
            n: int = 4,
            initial_shape: Optional[str] = "relu",
            efficient: bool = True,
            eps: float = 1e-08,
            activation_unit = 5 ,
            **kwargs: Any
    ) -> None:
        """
        Constructor method
        :param m (int): Size of nominator polynomial. Default 5.
        :param n (int): Size of denominator polynomial. Default 4.
        :param initial_shape (Optional[str]): Initial shape of PAU, if None random shape is used, also if m and n are
        not the default value (5 and 4) a random shape is utilized. Default "leaky_relu_0_2".
        :param efficient (bool): If true efficient variant with checkpointing is used. Default True.
        :param eps (float): Constant for numerical stability. Default 1e-08.
        :param **kwargs (Any): Unused
        """
        # Call super constructor
        super(PAU, self).__init__()
        # Save parameters
        self.efficient: bool = efficient
        self.m: int = m
        self.n: int = n
        self.eps: float = eps
        self.initial_weights = weights
        # Init weights
        weights_nominator, weights_denominator = self.initial_weights
        self.weights_nominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(weights_nominator.view(1, -1))
        self.weights_denominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(weights_denominator.view(1, -1))
    def freeze(self) -> None:
        """
        Function freezes the PAU weights by converting them to fixed model parameters.
        """

        if isinstance(self.weights_nominator, nn.Parameter):
            weights_nominator = self.weights_nominator.data.clone()
            del self.weights_nominator
            self.register_buffer("weights_nominator", weights_nominator)
        if isinstance(self.weights_denominator, nn.Parameter):
            weights_denominator = self.weights_denominator.data.clone()
            del self.weights_denominator
            self.register_buffer("weights_denominator", weights_denominator)
    def _forward(self,input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input (torch.Tensor): Input tensor of the shape [*]
        :return (torch.Tensor): Output tensor of the shape [*]
        """
        # Save original shape
        shape: Tuple[int, ...] = input.shape
        # Flatten input tensor
        input: torch.Tensor = input.view(-1)
        if self.efficient:
            # Init nominator and denominator
            nominator: torch.Tensor = torch.ones_like(input=input) * self.weights_nominator[..., 0]
            denominator: torch.Tensor = torch.zeros_like(input=input)
            # Compute nominator and denominator iteratively
            for index in range(1, self.m + 1):
                x: torch.Tensor = (input ** index)
                nominator: torch.Tensor = nominator + x * self.weights_nominator[..., index]
                if index < (self.n + 1):
                    denominator: torch.Tensor = denominator + x * self.weights_denominator[..., index - 1]
            denominator: torch.Tensor = denominator + 1.
        else:
            # Get Vandermonde matrix
            vander_matrix: torch.Tensor = torch.vander(x=input, N=self.m + 1, increasing=True)
            # Compute nominator
            nominator: torch.Tensor = (vander_matrix * self.weights_nominator).sum(-1)
            # Compute denominator
            denominator: torch.Tensor = 1. + torch.abs((vander_matrix[:, 1:self.n + 1]
                                                        * self.weights_denominator).sum(-1))
        # Compute output and reshape
        output: torch.Tensor = (nominator / denominator.clamp(min=self.eps)).view(shape)
        return output

    def forward(self,input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input (torch.Tensor): Input tensor of the shape [batch size, *]
        :return (torch.Tensor): Output tensor of the shape [batch size, *]
        """
        # Make input contiguous if needed
        input: torch.Tensor = input if input.is_contiguous() else input.contiguous()
        if self.efficient:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input=input)
