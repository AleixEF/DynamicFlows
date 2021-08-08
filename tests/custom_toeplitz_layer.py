#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:03:26 2021

@author: aleix
"""

import torch
from torch import nn 
from torch.autograd import Function


"""
In this test we create a custom layer that introduces a toeplitz matrix 
constraint.

Achievements: 
    The layer only contains the toeplitz parameters.
    The gradient respect to the parameters is calculated correctly

Problems encountered:
    The topepliz matrix is not updated by the optmimizer. This is because
    the toeplitz matrix is built as an independent copy of toeplitz_params.

Possible solutions:
    reate the new toeplitz matrix every time params are updated (slow)
    Unknown: Try to create a toeplitz matrix that is built as a pointer that points to 
    the parameters (optimal)

"""


class LinearToeplitz(nn.Module):
    def __init__(self, input_features, output_features):
        super(LinearToeplitz, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # a toeplitz matrix has n_rows + n_cols -1 independent params
        self.toeplitz_params =  nn.Parameter(
            torch.randn((output_features+input_features-1)))
        # TODO: change to He initialization        
        self.bias = nn.Parameter(torch.randn((output_features)))

        self.weight = create_toeplitz_matrix(self.toeplitz_params, 
                                             (output_features, input_features))

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


def create_toeplitz_matrix(parameters, matrix_shape):
    n_rows, n_cols = matrix_shape             
    toep_matrix = torch.tensor([])
    for i in range(n_rows):
        i_start = n_rows - 1 - i
        i_end = i_start + n_cols
        row = parameters[i_start: i_end]
        toep_matrix = torch.cat((toep_matrix, row))
    return toep_matrix.view(matrix_shape)



lin_toeplitz = LinearToeplitz(input_features=3, output_features=3)

for name, param in lin_toeplitz.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param} \n")


x_input = torch.randn((1, 3))

# Analytical gradient computation
A = torch.zeros((3, 3))
A[0, 1] = 1
A[1, 2] = 1

B = torch.zeros((3, 3))
B[0, 2] = 1

analytical_gradient = torch.tensor([torch.sum(B.T @ x_input[0]),  # t4
                                   torch.sum(A.T @ x_input[0]),  # t3
                                   x_input.sum(),  # t0
                                   torch.sum(A @ x_input[0]),  # t1 
                                   torch.sum(B @ x_input[0])])  # t2
# pytorch gradient computation
output = lin_toeplitz(x_input)

optim = torch.optim.SGD(lin_toeplitz.parameters(), lr=1)
optim.zero_grad()

loss = output.sum()
loss.backward()
print("Analytical gradient computation:\n",
      analytical_gradient)

print("Pytoch gradient computation:\n",
      lin_toeplitz.toeplitz_params.grad)

print("All close?", torch.allclose(analytical_gradient, lin_toeplitz.toeplitz_params.grad))


print()
print("weights before optimizer")
print(lin_toeplitz.weight)

print("weight after optimizer")
optim.step()
print(lin_toeplitz.weight)

print("problem encoutered, weights do not change. The weight matrix is a copy"
      "of the params, but we need a pointer")



