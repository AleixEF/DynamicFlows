# Template adapted from: https://github.com/senya-ashukha/real-nvp-pytorch

class RealNVP(nn.Module):
    
    def __init__(self, n_I, n_H, n_O, n_layers):  
        """ Initialising the attributes of the class. Define a shallow neural 
        network for the scale function. You can also choose to define a single 
        network and split the outputs accordingly. Only point to remember is 
        that the last activation function is different for the scale and the 
        translation parameters.
        ----
        Args:

        - n_I: no. of input dimensions 
        - n_H: no. of hidden dimensions
        - n_O: no. of output dimensions
        - n_layers: no. of coupling layers

        """      
        super(RealNVP, self).__init__()

        # Define a shallow neural network for the scale function
        net_s = lambda: nn.Sequential(nn.Linear(n_I, n_H), 
                              nn.LeakyReLU(), 
                              nn.Linear(n_H, n_H), 
                              nn.LeakyReLU(), 
                              nn.Linear(n_H, n_O), 
                              nn.Tanh())

        # Define a shallow neural network for the translation function
        net_t = lambda: nn.Sequential(nn.Linear(n_I, n_H), 
                              nn.LeakyReLU(), 
                              nn.Linear(n_H, n_H), 
                              nn.LeakyReLU(), 
                              nn.Linear(n_H, n_O))
        
        # Define a group of simple "cross" pattern masks. The number of these masks 
        # is equal to the number of affine coupling layers
        masks = torch.from_numpy(np.array([[1, 0], [0, 1]] * int(n_layers)).astype(np.float32))
        self.mask = nn.Parameter(masks, requires_grad = False)

        # Define the prior for defining a multivariate Gaussian distribution
        self.prior = distributions.MultivariateNormal(torch.zeros(n_I), torch.eye(n_I))
        
        # Define the translation and scale parameters for each coupling layer
        self.t = torch.nn.ModuleList([net_t() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([net_s() for _ in range(len(masks))])
    

    def f(self, x):
        """ This function is used for implementing the normalizing function 
        f:X --> Z. 
        """
        log_det_J = torch.zeros(x.shape[0]).to(device)
        z = x
        #print(z.shape)
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z 
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = z_ + (1 - self.mask[i]) * (z - t) * torch.exp(-s)
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def g(self, z):
        """ This function is used for implementing the normalizing function 
        f:X --> Z. 
        """
        x = copy.deepcopy(z)
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def log_prob(self, x):
        """ Computing the log prob p(x) from p(z) using the change of variable
        formula (inference)
        """
        z, logdet = self.f(x)
        pz = self.prior.log_prob(z.cpu()).to(device)
        return pz + logdet
    
    def sample(self, batchSize):
        """ Generate samples drawn from the given batch of data from the latent
        space Z (generate samples)
        """
        z = self.prior.sample((batchSize,1))
        logpx = self.prior.log_prob(z)
        x = self.g(z.to(device))
        return x
