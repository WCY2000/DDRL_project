import os
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
TRAIN = True

class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params["retain_first_backpass"]
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params["kld_weight"],  # al_img.shape[0]/ self.num_train_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True
        )

    if TRAIN:
        def on_validation_end(self) -> None:
            self.sample_images()

    def visualize_output(self, images, outputs, path):
        plt.figure(figsize=(20, 5))
        for i in range(0,10):
            plt.subplot(2, len(images), i + 1)
            plt.imshow(
                images[i].permute(1, 2, 0)
            )  # Convert from CHW to HWC format for matplotlib
            plt.title("Original")
            plt.axis("off")

            plt.subplot(2, len(images), len(images) + i + 1)
            plt.imshow(
                outputs[i].detach().permute(1, 2, 0)
            )  # Convert and detach from gradients for visualization
            plt.title("Output")
            plt.axis("off")
        plt.savefig(path)

    def mse(self, imageA, imageB):
        err = torch.mean((imageA - imageB) ** 2)
        return err.item()

    def psnr(self, imageA, imageB):
        mse_val = self.mse(imageA, imageB)
        if mse_val == 0:
            return float("inf")
        max_pixel = 1.0
        return 20 * torch.log10(max_pixel / torch.sqrt(torch.tensor(mse_val))).item()

    def visualize_output_with_metrics(self, images, outputs, path):
        plt.figure(figsize=(30, 5))
        for i in range(10):
            # Original Image
            plt.subplot(2, 10, i + 1)
            plt.imshow(images[i].cpu().permute(1, 2, 0))
            plt.title("Original")
            plt.axis("off")

            # Output Image
            plt.subplot(2, 10, 10 + i + 1)
            output_image = outputs[i].detach().cpu().permute(1, 2, 0)
            plt.imshow(output_image)

            # Calculate PSNR and MSE
            psnr_value = self.psnr(images[i], outputs[i])
            mse_value = self.mse(images[i], outputs[i])

            # Add PSNR and MSE as text on the Output image plot
            plt.text(
                5,
                -5,
                f"PSNR: {psnr_value:.3f} dB",
                color="white",
                fontsize=12,
                backgroundcolor="black",
            )
            plt.text(
                5,
                output_image.shape[0] - 5,
                f"MSE: {mse_value:.3f}",
                color="white",
                fontsize=12,
                backgroundcolor="black",
            )

            # plt.title("Output")
            plt.axis("off")

        plt.savefig(path)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        # test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        # print("<< recons", recons.shape)
        path = os.path.join(
            self.logger.log_dir,
            "Reconstructions",
            f"{self.logger.name}_Epoch_{self.current_epoch}.png",
        )
        print(path)
        self.visualize_output_with_metrics(test_input.cpu(), recons.cpu(), path)

        # vutils.save_image(# `recons` is a variable that stores the reconstructed images generated by
        # the VAE model for a given set of input images (`test_input`). These
        # reconstructed images are then saved as an image file using
        # `vutils.save_image` function. The purpose of reconstructing images is to
        # evaluate how well the VAE model is able to reconstruct the input images,
        # which is a common evaluation metric for generative models like VAEs.
        # `recons` is a variable that stores the reconstructed images generated by
        # the VAE model using the test input images and labels. These reconstructed
        # images are then saved as a PNG file in the "Reconstructions" directory
        # with a specific naming convention that includes the logger name and the
        # current epoch number. The `vutils.save_image` function is used to save the
        # reconstructed images.
        # recons.data,
        #                   os.path.join(self.logger.log_dir ,
        #                                "Reconstructions",
        #                                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                   normalize=True,
        #                   nrow=12)

        # try:
        #     samples = self.model.sample(144,
        #                                 self.curr_device,
        #                                 labels = test_label)
        #     vutils.save_image(samples.cpu().data,
        #                       os.path.join(self.logger.log_dir ,
        #                                    "Samples",
        #                                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                       normalize=True,
        #                       nrow=12)
        # except Warning:
        #     pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params["LR"],
            weight_decay=self.params["weight_decay"],
        )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params["LR_2"] is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model, self.params["submodel"]).parameters(),
                    lr=self.params["LR_2"],
                )
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params["scheduler_gamma"] is not None:
                gamma = self.params["scheduler_gamma"]
                assert (
                    0 < gamma < 1
                ), "Gamma for ExponentialLR should be between 0 and 1"

                scheduler = {
                    "scheduler": optim.lr_scheduler.ExponentialLR(
                        optimizer, gamma=gamma
                    ),
                    "interval": "epoch",  # Typically, LR schedulers are stepped each epoch
                    "frequency": 1,
                }
                scheds = [scheduler]
                return optims, scheds
        except:
            return optims
