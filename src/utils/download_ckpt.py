import wandb

# Initialize the Weights & Biases API
api = wandb.Api()

# Get the run
run_id = "thfq9er6"
run = api.run(f"ship_segmentation/ship-segmentation/{run_id}")

# File you want to download
file_path = "logs/train/runs/2023-07-24_06-38-17/checkpoints/epoch_059.ckpt"

# Location to save the downloaded file
location = '/workspace/ckpt/'

# Download 
run.file(file_path).download(root=location)
print(f'Successfully downloaded file "{file_path}" to {location}')
