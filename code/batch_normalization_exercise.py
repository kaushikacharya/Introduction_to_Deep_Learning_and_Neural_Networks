import torch

# Gamma and beta are provided as 1d tensors. 

def batchnorm(X, gamma, beta):
    print(f"X.shape: {X.shape}")
    print(f"gamma: {gamma}")
    print(f"beta: {beta}")
    N,C,H,W = list(X.shape)
    mu = torch.mean(X, dim=(0,2,3))
    sigma = torch.std(X, dim=(0,2,3))

    # bn = torch.div(torch.sub(X, mu[None,:,None,None]), sigma[None,:,None,None])
    bn = torch.mul(gamma[None,:,None,None], torch.div(torch.sub(X, mu[None,:,None,None]), sigma[None,:,None,None])) + beta[None,:,None,None]

    # bn = torch.mul( gamma.reshape((1,C,1,1)), torch.div(torch.sub(X, mu.reshape((1,C,1,1))), sigma.reshape((1,C,1,1))) ) + beta.reshape((1,C,1,1))

    return bn
