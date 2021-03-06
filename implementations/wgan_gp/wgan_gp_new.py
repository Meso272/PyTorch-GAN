import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import sys
sys.path.append("..")
from cesm import CLDHGH
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=4, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=16, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--hidden_dims', nargs='*', type=int)
parser.add_argument("--lm", type=float, default=10, help="lambda")
parser.add_argument("--save", type=str,  help="saving path")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.in_channels=opt.channels
        self.latent_dim = opt.latent_dim
        
        
        modules = []
        if opt.hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        else:
            hidden_dims=opt.hidden_dims
        #print(hidden_dims)
        self.last_fm_nums=hidden_dims[-1]
        self.last_fm_size=int( opt.img_size/(2**len(hidden_dims)) )
        # Build Encoder
        in_channels=opt.channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=in_channels,
                              kernel_size= 3, stride= 1, padding  = 1),###added layer
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_z = nn.Linear(hidden_dims[-1]*self.last_fm_size*self.last_fm_size, self.latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1] * self.last_fm_size*self.last_fm_size)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i],
                                       kernel_size=3,
                                       stride = 1,
                                       padding=1,
                                       output_padding=0),##added layer
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer_1 = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                       hidden_dims[-1],
                                       kernel_size=3,
                                       stride = 1,
                                       padding=1,
                                       output_padding=0),##added layer
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            )
        self.final_layer_2=nn.Sequential(nn.Conv2d(hidden_dims[-1], out_channels= opt.channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input) :
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        z = self.fc_z(result)
        return z

    def decode(self, z) :
        result = self.decoder_input(z)
        result = result.view(-1, self.last_fm_nums, self.last_fm_size, self.last_fm_size)
        result = self.decoder(result)
        result = self.final_layer_1(result)
        result= self.final_layer_2(result)
        
        return result

    def forward(self, input, **kwargs) :
        z = self.encode(input)
        return  self.decode(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.in_channels=opt.channels
        self.latent_dim = opt.latent_dim
        
        
        modules = []
        if opt.hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        else:
            hidden_dims=opt.hidden_dims
        self.last_fm_nums=hidden_dims[-1]
        self.last_fm_size=int( opt.img_size/(2**len(hidden_dims)) )
        # Build Encoder
        in_channels=opt.channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=in_channels,
                              kernel_size= 3, stride= 1, padding  = 1),###added layer
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.model = nn.Sequential(*modules)

        self.adv_layer = nn.Sequential( nn.Linear(hidden_dims[-1]*self.last_fm_size*self.last_fm_size, 1),nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
datapath="/home/jliu447/lossycompression/multisnapshot-data-cleaned/CLDHGH"
dataloader = torch.utils.data.DataLoader(
    dataset=CLDHGH(path=datapath,start=0,end=50,size=opt.img_size,gan_scale=True),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(real_imgs)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(real_imgs)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)+opt.lm*F.mse_loss(real_imgs,fake_imgs)

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            #if batches_done % opt.sample_interval == 0:
            #    save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic


torch.save(generator, opt.save)
torch.save(generator.state_dict(),opt.save+'.params.pt')
