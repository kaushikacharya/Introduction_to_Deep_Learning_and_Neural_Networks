import torch
import torch.nn as nn

seed = 172
torch.manual_seed(seed)

# reconstructed and input are tensors of the same size
# mu and logvar are vectors of the same size

def elbo(reconstructed, input, mu, logvar):
    """
    Args:
        `reconstructed`: The reconstructed input of size [B, C, W, H],
        `input`: The original input of size [B, C, W, H],
        `mu`: The mean of the Gaussian of size [N], where N is the latent dimension
        `logvar`: The log of the variance of the Gaussian of size [N], where N is the latent dimension

    Returns:
        a scalar
    """
    criterion = nn.BCELoss(reduction="sum")
    negative_reconstruction_error = criterion(reconstructed, input)
    kl_divergence = -0.5*torch.sum(torch.ones(logvar.size()) + logvar - mu*mu - torch.exp(logvar))
    loss = negative_reconstruction_error + kl_divergence

    return loss

