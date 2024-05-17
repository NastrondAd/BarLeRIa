from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()
        self.affine_transform = AffineTransform(dim)
        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            PlanarFlowLogDetJacobian(t) for t in self.transforms
        ))
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
    def forward(self, z):

        log_jacobians = []
        z, sum_log_abs_det_jacobians = self.affine_transform(z)
        log_jacobians.append(self.relu(sum_log_abs_det_jacobians))
        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            z = transform(z)
        zk = z

        return zk, sum(log_jacobians)

class AffineTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.logsigma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        z = self.mu + self.logsigma.exp() * x
        sum_log_abs_det_jacobians = self.logsigma.sum()
        return z, sum_log_abs_det_jacobians

class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)

def safe_log(z):
    return torch.log(z + 1e-7)

class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight

        det_grad = 1 + torch.matmul(psi, self.scale.t().unsqueeze(dim=0).repeat(psi.shape[0],1,1))
        # print(det_grad.shape)
        return safe_log(det_grad.abs())