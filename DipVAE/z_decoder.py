import os
import torch
import yaml
from torchvision import transforms
from models import DIPVAE  # Adjust this import according to your project structure
from experiment import VAEXperiment
import matplotlib.pyplot as plt


def read_z_latent(file_path):
    z_latent_vectors = []
    with open(file_path, "r") as f:
        for line in f:
            z_latent_vector = [float(num) for num in line.split()]
            z_latent_vectors.append(torch.tensor(z_latent_vector))
    z_latent_vectors = torch.stack(z_latent_vectors)
    return z_latent_vectors


def decode_z_latent(model, z_latent_vectors, device):
    model.eval()
    with torch.no_grad():
        z_latent_vectors = z_latent_vectors.to(device)
        reconstructed_images = model.decode(z_latent_vectors)
    return reconstructed_images


def save_reconstructed_images(reconstructed_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    length = reconstructed_images.shape[0]

    for i in range(length):
        # Move the tensor to CPU and convert it to a NumPy array for plotting
        image = reconstructed_images[i].cpu().permute(1, 2, 0).numpy()

        plt.figure(figsize=(3, 3))  # You can adjust the figure size as needed
        plt.imshow(image)  # No need to permute again as it's already in HWC format
        plt.axis("off")

        # Save each image as a separate file
        plt.savefig(
            os.path.join(output_dir, f"{i:04d}.png"), bbox_inches="tight", pad_inches=0
        )
        plt.close()  #

    # for i, image in enumerate(reconstructed_images):
    #     pil_image = transforms.ToPILImage()(image.cpu())
    #     pil_image.save(os.path.join(output_dir, f'reconstructed_{i}.png'))


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def visualize_output(self, images, outputs, path):
    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
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


def main():
    z_latent_file_path = "/home/chenyu/Desktop/visuotactile_representations/hiss/dataset/0404_lstm_kinova/pred_0.txt"
    output_dir = "./dataset/reconstructed_LSTM_kinova_0404"
    config_path = "./configs/dip_vae.yaml"
    checkpoint_path = "/home/chenyu/Desktop/visuotactile_representations/DipVAE/logs/DIPVAE/Train_openteach_66/checkpoints/epoch=23-step=10271.ckpt"
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # Load the model configuration from YAML file
    config = load_config(config_path)

    # Initialize the DIPVAE model with parameters from the config
    model = DIPVAE(**config["model_params"])
    model = model.to(device)

    # Initialize the VAEXperiment with the model and experiment parameters from the config
    experiment = VAEXperiment(model, config["exp_params"])
    experiment = experiment.to(device)

    # Load the checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Update the VAEXperiment instance with the loaded state dict
    experiment.load_state_dict(ckpt["state_dict"])

    # Read z-latent vectors from file
    z_latent_vectors = read_z_latent(z_latent_file_path)
    file_path = "/home/chenyu/Desktop/visuotactile_representations/hiss/hiss/tasks/target_tensors.pt"
    data = torch.load(file_path)
    target_mean = data["target_mean"]
    target_std = data["target_std"]

    z_latent_vectors = z_latent_vectors * target_std + target_mean

    # Decode z-latent vectors to get reconstructed images using the experiment's model
    reconstructed_images = decode_z_latent(experiment.model, z_latent_vectors, device)

    # Save the reconstructed images
    save_reconstructed_images(reconstructed_images, output_dir)


if __name__ == "__main__":
    main()
