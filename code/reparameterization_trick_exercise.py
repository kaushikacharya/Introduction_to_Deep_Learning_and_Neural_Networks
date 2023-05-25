import torch

seed = 172
torch.manual_seed(seed)

def reparameterize(mu, log_var):
    """
        Args:
            `mu`: mean from the encoder's latent space
            `log_var`: log variance from the encoder's latent space

        Returns:
            the reparameterized latent vector z
    """
    # standard deviation (Both the equations should work)
    # sigma = torch.sqrt(torch.exp(log_var))
    sigma = torch.exp(0.5*log_var)
    epsilon = torch.normal(mean=torch.zeros(mu.size()), std=torch.ones(mu.size()))
    # epsilon = torch.rand_like(mu)

    z = mu + sigma * epsilon

    return z

if __name__ == "__main__":
    mu = torch.tensor([0., 0., 0., 0., 0.])
    log_var = torch.tensor([0., 0., 0., 0., 0.])
    z = reparameterize(mu=mu, log_var=log_var)
    print(z)
