import torch.nn as nn
import torch
from device import device
import math

class DecoderBayes(nn.Module):
    def __init__(self, mask):
        super(DecoderBayes, self).__init__()
        self.sparse_layer = BayesianSparseLayer(mask)

    def forward(self, x):
        return self.sparse_layer(x.to(device))

    def weights(self):
        w = self.sparse_layer.weight_mu * self.sparse_layer.mask
        #w = self.spa
        return w

    def kl(self):
        return self.sparse_layer.kl





######### You don't need to understand this part of the code in detail #########
class SparseLayerFunction(torch.autograd.Function):
    """
    We define our own autograd function which masks it's weights by 'mask'.
    For more details, see https://pytorch.org/docs/stable/notes/extending.html
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias, mask):

        weight = weight * mask # change weight to 0 where mask == 0
        #calculate the output
        output = input.mm(weight.t())
        output += bias.unsqueeze(0).expand_as(output) # Add bias to all values in output
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output): # define the gradient formula
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and only to improve efficiency
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            # change grad_weight to 0 where mask == 0
            grad_weight = grad_weight * mask
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class BayesianSparseLayer(nn.Module):
    def __init__(self, mask, prior_std=1.0):
        """
        Bayesian sparse layer: weights are sampled from a Gaussian.
        Mask determines the sparse connections.
        """
        super(BayesianSparseLayer, self).__init__()

        self.mask = nn.Parameter(torch.tensor(mask, dtype=torch.float).t(), requires_grad=False)

        # Learnable mean and log std of weights
        self.weight_mu = nn.Parameter(torch.Tensor(mask.shape[1], mask.shape[0]))
        self.weight_logvar = nn.Parameter(torch.Tensor(mask.shape[1], mask.shape[0]))

        self.bias = nn.Parameter(torch.Tensor(mask.shape[1]))
        self.reset_parameters()

        self.prior_std = prior_std  # used for KL divergence

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_logvar.data.fill_(-5.0)  # small initial variance
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # Sample weights from posterior
        std = torch.exp(0.5 * self.weight_logvar)
        eps = torch.randn_like(std)
        weight_sample = self.weight_mu + eps * std

        # Apply mask
        weight_sample = weight_sample * self.mask

        # Linear forward pass
        output = input.mm(weight_sample.t()) + self.bias

        # Store KL divergence for the layer
        self.kl = self.kl_divergence()
        return output

    def kl_divergence(self):
        """
        KL divergence between posterior and prior N(0, prior_std^2)
        """
        std = torch.exp(0.5 * self.weight_logvar)
        var = std ** 2
        prior_var = self.prior_std ** 2

        kl = 0.5 * ((var + self.weight_mu ** 2) / prior_var - 1 - self.weight_logvar + math.log(prior_var))
        kl = kl * self.mask  # Only count masked weights
        return kl.sum()