#!/usr/bin/env python3

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)

class VAE(nn.Module):
    def __init__(self, LATENT_SPACE_DIM=320, num_imgs=1):
        super().__init__()

        self.size_after_conv = [256, 16*num_imgs, 16]
        self.inter_size = 512

        self.fc_e1 = nn.Conv2d(3, 32, kernel_size=4, padding=1, stride=2)
        self.be1 = nn.BatchNorm2d(32)
        self.fc_e2 = nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2)
        self.be2 = nn.BatchNorm2d(64)
        self.fc_e3 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.be3 = nn.BatchNorm2d(128)
        self.fc_e4 = nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2)
        self.be4 = nn.BatchNorm2d(256)
        self.fc_e61 = nn.Linear(np.prod(self.size_after_conv), self.inter_size)
        self.fc_e71 = nn.Linear(self.inter_size, LATENT_SPACE_DIM)
        self.fc_e62 = nn.Linear(np.prod(self.size_after_conv), self.inter_size)
        self.fc_e72 = nn.Linear(self.inter_size, LATENT_SPACE_DIM)

        self.fc_d11 = nn.Linear(LATENT_SPACE_DIM, self.inter_size)
        self.fc_d12 = nn.Linear(self.inter_size, np.prod(self.size_after_conv))
        self.fc_d2 = nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2)
        self.bd1 = nn.BatchNorm2d(128)
        self.fc_d3 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
        self.bd2 = nn.BatchNorm2d(64)
        self.fc_d4 = nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2)
        self.bd3 = nn.BatchNorm2d(32)
        self.fc_d5 = nn.ConvTranspose2d(32,  3, kernel_size=4, padding=1, stride=2)
        self.bd4 = nn.BatchNorm2d(3)

    def encode(self, x):
        x = F.relu(self.fc_e1(x))
        x = self.be1(x)
        x = F.relu(self.fc_e2(x))
        x = self.be2(x)
        x = F.relu(self.fc_e3(x))
        x = self.be3(x)
        x = F.relu(self.fc_e4(x))
        x = self.be4(x)

        # logger.info(x.shape)
        x = x.view([x.size()[0], -1])
        
        mu = F.relu(self.fc_e61(x))
        mu = self.fc_e71(mu)

        logvar = F.relu(self.fc_e62(x))
        logvar = self.fc_e72(logvar)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        nb = z.size()[0]
        z = F.relu(self.fc_d11(z))
        z = self.fc_d12(z)
        z = z.view([nb]+self.size_after_conv)
        
        z = F.relu(self.fc_d2(z))
        z = self.bd1(z)
        z = F.relu(self.fc_d3(z))
        z = self.bd2(z)
        z = F.relu(self.fc_d4(z))
        z = self.bd3(z)
        z = F.relu(self.fc_d5(z))
        z = self.bd4(z)
        z = torch.sigmoid(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z

# model is a torch.nn.Module that contains the model definition.
global model, VAE_BCE_LOSS
model = VAE()
VAE_BCE_LOSS = 1.

# Use MSE loss as distance from input to output:
reconstruction_function = lambda recon_x, x: torch.sum((recon_x - x) ** 2)
def vae_loss_fn(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = torch.sum(reconstruction_function(recon_x, x))
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return VAE_BCE_LOSS * BCE + KLD, BCE, KLD

def loss(Ypred, Yactual, X):
    """loss function for learning problem

    Arguments:
        Ypred {Model output type} -- predicted output
        Yactual {Data output type} -- output from data
        X {torch.Variable[input]} -- input

    Returns:
        Tuple[nn.Variable] -- Parts of the loss function; the first element is passed to the optimizer
        nn.Variable -- the loss to optimize
    """
    reconstructed, mu, logvar, z = Ypred
    return vae_loss_fn(reconstructed, Yactual, mu, logvar)

def loss_labels():
    """Keys corresponding to the components of the loss
    
    Returns:
        Tuple[str] -- Tuple of loss 
    """
    return ("vae_loss", "reconstruction_loss", "prior_loss")

def configure(props):
    global model

    try:
        lsd = props["latent_space_dim"]
        if "num_imgs" in props:
            num_imgs = props["num_imgs"]
        model = VAE(lsd, num_imgs)
    except KeyError:
        pass
    logger.info(f"Latent space is 1x{lsd}.")

    try:
        global VAE_BCE_LOSS
        VAE_BCE_LOSS = 10**float(props["bce"])
    except KeyError:
        pass
    logger.info(f"BCE to KLD ratio {VAE_BCE_LOSS}.")
