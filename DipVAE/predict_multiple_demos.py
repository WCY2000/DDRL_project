import yaml
import subprocess

# Path to the YAML configuration file
config_file_path = "configs/dip_vae.yaml"

# Iterate over the specified range of demo numbers
for demo_num in range(0, 67):
    # Load the current configuration
    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    # Update the data_path parameter
    config["data_params"]["data_path"] = f"./dataset/demo_{demo_num}"

    # Save the modified configuration back to the original file
    with open(config_file_path, "w") as file:
        yaml.safe_dump(config, file)

    # Run the predict.py script with the updated configuration
    subprocess.run(["python", "predict.py", "-c", config_file_path])
