# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:55:08 2018

@author: herminarto.nugroho
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
from torchvision.utils import save_image
#from torchsummary import summary

#from pushover import notify
#from utils import makegif
from random import randint

from IPython.display import Image
from IPython.core.display import Image, display

from custom_datasets import CustomDatasetFromImages, CustomSplitLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100
log_interval = 10
epochs = 50
latent_dim = 20

dataset = CustomDatasetFromImages('../data/fire_labels.csv')
train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
#train_loader, test_loader, valid_loader = CustomSplitLoader(dataset, batch_size, 60, 20, 20)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        kernel_size = 4
        stride = 1
        padding = 0
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 5, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size, stride, padding),
            nn.ReLU())
        self.conv1 = nn.Conv2d(40, latent_dim, 16, stride, padding)
        self.conv2 = nn.Conv2d(40, latent_dim, 16, stride, padding)
        self.conv3 = nn.ConvTranspose2d(latent_dim, 40, 1, stride, padding)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 20, 7, stride, padding),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 10, 8, stride, padding),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 5, 8, stride, padding),
            nn.ReLU(),
            nn.ConvTranspose2d(5, 3, 8, stride, padding),
            nn.Sigmoid())
    
    def bottleneck(self, h):
        mu, logvar = self.conv1(h), self.conv2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.conv3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
           
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(valid_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 3, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
for epoch in range(epochs):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, latent_dim, 1, 1).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 3, 28, 28),
                   'results/sample_' + str(epoch) + '.png')