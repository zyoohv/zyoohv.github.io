import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torchvision import models, transforms


class linear(Function):

    @staticmethod
    def forward(self, input, weight, bias):
        self.save_for_backward(input, weight)
        output = input.mm(weight) + bias
        return output

    @staticmethod
    def backward(self, grad_output):
        input, weight = self.saved_variables
        return grad_output.mm(weight.t()), input.t().mm(grad_output), torch.sum(grad_output)


class model(nn.Module):

    def __init__(self, config):
        super(model, self).__init__()
        
        self.config = config

        self.weight1 = nn.Parameter(torch.randn(self.config['n'], self.config['hidden']))
        self.bias1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(self.config['hidden'], 1))
        self.bias2 = nn.Parameter(torch.randn(1))

        self.linear1 = linear.apply
        self.linear2 = linear.apply

    def forward(self, x):

        x = F.tanh(self.linear1(x, self.weight1, self.bias1))
        x = self.linear2(x, self.weight2, self.bias2)

        return x