import torch
from torch import nn
from torch.nn import functional as F
import yaml
import os
from typing import List, Callable, Union, Any, TypeVar, Tuple

from torch import tensor as Tensor
TRAIN = True

from torch import nn
from abc import abstractmethod

from typing import List, Callable, Union, Any, TypeVar, Tuple

from torch import tensor as Tensor

Tensor = TypeVar("torch.tensor")


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

Tensor = TypeVar("torch.tensor")
checkpoint_path =  "/home/chenyu/Desktop/DDRL_project/DipVAE/logs/DIPVAE/dipvae_ckpt_full/checkpoints/epoch=19-step=19919.ckpt"

class DIPVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        hidden_dims: List = None,
        lambda_diag: float = 0.05,
        lambda_offdiag: float = 0.1,
        **kwargs
    ) -> None:
        super(DIPVAE, self).__init__()

        self.latent_dim = latent_dim
        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]


        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 8 * 8, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 8 * 8)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['state_dict']

        # Fix the keys in the state_dict
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.', '')  # remove the 'model.' prefix
            new_state_dict[new_key] = value

        # Load the modified state_dict
        self.load_state_dict(new_state_dict)
        print("Model loaded successfully from {}".format(checkpoint_path))

    def encode(self, input: Tensor) -> List[Tensor]:
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
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 8, 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # print("<< input", input.shape)
        input = F.interpolate(input, size=(256, 256), mode='bilinear', align_corners=False)
        x = input
        # is_seq = x.dim() == 5
        # if is_seq:
        #     n = x.shape[0]
        #     t = x.shape[1]
        #     x = rearrange(x, "n t c h w -> (n t) c h w")

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        # print("<<< z shape", z.shape)
        return mu

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input, reduction="sum")

        kld_loss = torch.sum(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        # DIP Loss
        centered_mu = mu - mu.mean(dim=1, keepdim=True)  # [B x D]
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze()  # [D X D]

        # Add Variance for DIP Loss II
        cov_z = cov_mu + torch.mean(
            torch.diagonal((2.0 * log_var).exp(), dim1=0), dim=0
        )  # [D x D]
        # For DIp Loss I
        # cov_z = cov_mu

        cov_diag = torch.diag(cov_z)  # [D]
        cov_offdiag = cov_z - torch.diag(cov_diag)  # [D x D]
        dip_loss = self.lambda_offdiag * torch.sum(
            cov_offdiag**2
        ) + self.lambda_diag * torch.sum((cov_diag - 1) ** 2)

        loss = recons_loss + kld_weight * kld_loss + dip_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": -kld_loss,
            "DIP_Loss": dip_loss,
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


# ckpt = torch.load(
#     "/home/chenyu/Desktop/visuotactile_representations/DipVAE/logs/DIPVAE/Train_openteach_66/checkpoints/epoch=23-step=10271.ckpt"
# )
# experiment = VAEXperiment(model, config["exp_params"])
# experiment.load_state_dict(ckpt["state_dict"])



# from PIL import Image
# import torchvision.transforms as transforms

# # Path to your image file
# image_path = '/home/chenyu/Desktop/DDRL_project/DipVAE/dataset/train/0_0000.jpg'
# image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
# print(image.size)
# # Define the transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor()

# ])
# input_image = transform(image)
# model = DIPVAE()
# print("Image", input_image.shape)
# input_image = input_image.unsqueeze(0)
# # ckpt = torch.load(checkpoint_path)
# # model.load_state_dict(ckpt["state_dict"])
# model.eval()
# with torch.no_grad():
#     reconstructed_image, _, _, _ = model(input_image)
# print(reconstructed_image)