# Core Python
import os
import io
import csv
import ast
import random
import fnmatch
import pickle

# Numerical and Data Handling
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  # Redundant — remove one (see note below)

# ASE (Atomic Simulation Environment)
from ase import Atoms
from ase.io import read, write
from ase.build import molecule

# Pymatgen
from pymatgen.core import Lattice, Structure, Molecule, Element
from pymatgen.transformations.standard_transformations import RotationTransformation

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, png_dim1, png_dim2, input_dim):
        super(Generator, self).__init__()

        self.png_dim1 = png_dim1
        self.png_dim2 = png_dim2
        self.input_dim = input_dim

        self.fc = nn.Linear(input_dim, png_dim1 * png_dim2 * input_dim)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        # Conv layers
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lrelu4 = nn.LeakyReLU(0.3, inplace=True)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.lrelu5 = nn.LeakyReLU(0.5, inplace=True)

        self.conv_out = nn.Conv2d(128, 1, kernel_size=4, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Fully connected
        x = self.fc(x)
        x = self.lrelu1(x)

        # Reshape to (batch, channels, height, width)
        x = x.view(-1, self.input_dim, self.png_dim1, self.png_dim2)

        # Conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu2(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu3(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu4(x)

        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, png_dim1, png_dim2):
        super(Discriminator, self).__init__()

        # Convolution layers with padding='valid' equivalent
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)  # Keras default is 'valid'
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop1 = nn.Dropout2d(0.4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.drop2 = nn.Dropout2d(0.4)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.drop3 = nn.Dropout2d(0.4)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.drop4 = nn.Dropout2d(0.4)

        # Compute the size after conv layers
        out_h = self._conv_output_size(png_dim1, [4,4,3,3])
        out_w = self._conv_output_size(png_dim2, [4,4,3,3])

        self.flatten = nn.Flatten()
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)
        self.drop5 = nn.Dropout(0.4)

        self.fc = nn.Linear(256 * out_h * out_w, 1)

    def _conv_output_size(self, size, kernel_sizes, strides=None, paddings=None):
        """Compute output size after sequence of conv layers (like Keras 'valid')."""
        if strides is None:
            strides = [1] * len(kernel_sizes)
        if paddings is None:
            paddings = [0] * len(kernel_sizes)
        for k, s, p in zip(kernel_sizes, strides, paddings):
            size = (size - k + 2 * p)//s + 1
        return size

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.lrelu3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.lrelu4(x)
        x = self.drop4(x)

        x = self.flatten(x)
        x = self.lrelu5(x)
        x = self.drop5(x)

        x = self.fc(x)
        return x

class GANS(nn.Module):
    def __init__(self, generator, discriminator, input_dim):
        super(GANS, self).__init__()
        self.input_dim = input_dim
        self.batch_size = None
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Placeholders for optimizers and loss functions
        self.g_opt = None
        self.d_opt = None
        self.g_loss_fn = None
        self.d_loss_fn = None

    def compile(self, g_opt, d_opt, g_loss, d_loss):
        """
        Set the optimizers and loss functions for generator and discriminator.
        """
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss_fn = g_loss
        self.d_loss_fn = d_loss

    def batch_data(self, data, batch_size):
        """
        Prepare data in batches and reshape to add channel dimension.
        """
        data = np.array(data)
        data_size = (len(data) // batch_size) * batch_size
        batched_data = data[:data_size].reshape((data_size, data.shape[1], data.shape[2], 1))
        self.batch_size = batch_size
        return torch.tensor(batched_data)
    def wasserstein_distance_loss(self,real_output, fake_output):
        """
        Wasserstein GAN loss: D(fake) - D(real)
        real_output: discriminator outputs on real images
        fake_output: discriminator outputs on generated images
        """
        return torch.mean(fake_output) - torch.mean(real_output)
    def generator_loss(self,fake_output):
        """
        WGAN generator loss: -mean(D(G(z)))
        fake_output: discriminator outputs on generated images
        """
        return -torch.mean(fake_output)
    def gradient_penalty(self,real_output, fake_output, max_pixel_value=1.0):
        """
        Compute WGAN-GP gradient penalty.
        
        real_output: real images tensor (batch, channels, height, width)
        fake_output: generated images tensor (batch, channels, height, width)
        discriminator: discriminator/critic model
        max_pixel_value: scale factor (like in TF code)
        """
        batch_size = real_output.size(0)

        # Sample alpha
        alpha = torch.randn(batch_size, 1, 1, 1, device=self.device)

        # Scale real images
        real_scaled = real_output

        # Interpolation
        diff = fake_output - real_scaled
        interpolated = real_scaled + alpha * diff
        interpolated.requires_grad_(True)

        # Pass interpolated samples through discriminator
        pred = self.discriminator(interpolated)

        # Compute gradients w.r.t. interpolated
        grads = torch.autograd.grad(
            outputs=pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Flatten per sample and compute norm
        grads_flat = grads.view(batch_size, -1)
        norm = torch.sqrt(torch.sum(grads_flat ** 2, dim=1) + 1e-12)  # add epsilon for numerical stability

        # Gradient penalty
        gp = torch.mean((norm - 1.0) ** 2)

        return gp
    def train_step(self, real_images):

        batch_size = real_images.size(0)

        # Move real images to device
        real_images = real_images.to(self.device)

        real = real_images.view(real_images.size(0), real_images.size(3), real_images.size(1), real_images.size(2)).to(torch.float32)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.d_opt.zero_grad()
        z = torch.randn(batch_size, 1, self.input_dim, device=self.device)
        fake_images = self.generator(z)

        # Optional noise injection
        noise_real = 0.3 * torch.randn_like(fake_images)
        noise_fake = -0.3 * torch.randn_like(fake_images)

        yhat_real = self.discriminator(real) #+ noise_real
        yhat_fake = self.discriminator(fake_images.detach())# + noise_fake
        d_loss = self.wasserstein_distance_loss(yhat_real, yhat_fake)
        gp = self.gradient_penalty(real, fake_images.detach())
        total_d_loss = d_loss + gp
        total_d_loss.backward()
        self.d_opt.step()

        # ---------------------
        # Train Generator
        # ---------------------
        self.g_opt.zero_grad()
        z = torch.randn(batch_size, 1, self.input_dim,device=self.device)
        #print(z.shape)
        gen_images = self.generator(z)
        #print(gen_images.shape)
        #gen = gen_images.view(gen_images.size(0), gen_images.size(3), gen_images.size(1), gen_images.size(2)).to(torch.float32)
        #print(gen_images.shape)
        predicted_labels = self.discriminator(gen_images)
        total_g_loss = self.generator_loss(predicted_labels)
        total_g_loss.backward()
        self.g_opt.step()

        return {"d_loss": total_d_loss.item(), "g_loss": total_g_loss.item()}

                                                                                                                                                         
