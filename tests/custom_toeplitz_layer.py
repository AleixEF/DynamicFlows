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
    The topepliz matrix can not be updated by the optmimizer. This is because
    the toeplitz matrix is built as an independent copy of toeplitz_params.

Solution implemented:
    Create the new toeplitz matrix every time params are updated (slow)
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

    def forward(self, x_input):
        weight = create_toeplitz_matrix(self.toeplitz_params, 
                                       (self.output_features, 
                                        self.input_features))                                    
        return LinearFunction.apply(x_input, weight, self.bias)

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
    toep_matrix = torch.zeros(matrix_shape)
    i_start = matrix_shape[0] 
    i_end = i_start + matrix_shape[1] 
    for i in range(matrix_shape[0]):
        i_start -= 1
        i_end -= 1
        toep_matrix[i] = parameters[i_start: i_end]
    return toep_matrix


def main():
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
    analytical_gradient = torch.tensor([torch.sum(B.T @ x_input[0]), 
                                       torch.sum(A.T @ x_input[0]),  
                                       x_input.sum(),
                                       torch.sum(A @ x_input[0]),   
                                       torch.sum(B @ x_input[0])]) 
    
    # pytorch gradient computation
    optim = torch.optim.SGD(lin_toeplitz.parameters(), lr=1)
    optim.zero_grad()
    
    output = lin_toeplitz(x_input)
    loss = output.sum()
    loss.backward()
    print("Analytical gradient computation:\n",
          analytical_gradient)
    
    print("Pytoch gradient computation:\n",
          lin_toeplitz.toeplitz_params.grad)
    
    print("All close?", torch.allclose(analytical_gradient, 
                                       lin_toeplitz.toeplitz_params.grad))
    return


if __name__ == '__main__':
    main()




