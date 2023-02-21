import torch
import torch.nn as nn

class GaussianDropout(nn.Module):
    def __init__(self, alpha=0.5):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        if self.train():
            # randn returns N(0,1), changing variance to alpha and mean to 1
            epsilon = torch.randn(x.size()) * self.alpha + 1
            epsilon = torch.autograd.Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()
            return x * epsilon
        else:
            return x

class InformationDropout(nn.Module):
    def __init__(self, layer, prior='log-uniform', max_alpha=0.7):
        super(InformationDropout, self).__init__()
        self.max_alpha = max_alpha
        self.alpha = torch.tensor([max_alpha], requires_grad=True).cuda()
        self.layer = layer
        self.prior = prior

    def kl(self):
        if self.prior == 'log-uniform':
            # the KL term is zero for alpha == max_alpha
            kl = - torch.log(self.alpha)
        else:
            mu1 = torch.zeros_like(self.alpha)
            sigma1 = torch.ones_like(self.alpha)
            kl = 0.5 * ((self.alpha / sigma1) ** 2 + (self.mu - mu1) ** 2 / sigma1 ** 2 - 1 + 2 * (torch.log(sigma1) - torch.log(self.alpha)))
        sum_dim = len(kl.shape)
        return torch.sum(kl, dim=list(range(1, sum_dim)))

    def forward(self, x, activation, network_layer):
        # Computes the noise parameter alpha for the new layer based on the input
        # Rescale alpha in the allowed range and add a small value for numerical stability
        self.alpha = 0.001 + self.max_alpha * torch.sigmoid(self.layer(x))
        network_activation = activation(network_layer(x))
        self.mu = torch.log(torch.maximum(network_activation, torch.Tensor([1e-4]).cuda()))

        if self.train():
            # sample from N(0,1) of size network_activation
            e = torch.normal(mean=torch.zeros_like(network_activation), std=torch.ones_like(network_activation))
            e = torch.autograd.Variable(e)
            if x.is_cuda:
                e = e.cuda()
            # final sample from the log-normal distribution
            #TODO do we even need to add the zero array? or is it a replacement for a different mean?
            epsilon = torch.exp(torch.zeros_like(network_activation) + self.alpha * e)
            return network_activation * epsilon
        else:
            return network_activation

def dropout(p, method='standard', layer=None, prior=None):
    if method == 'standard':
        return nn.Dropout(p)
    elif method == 'gaussian':
        return GaussianDropout(p/(1-p))
    elif method == 'information':
        return InformationDropout(layer, prior=prior)