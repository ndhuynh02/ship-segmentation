import wandb

# Initialize the Weights & Biases API
api = wandb.Api()

# Get the run
run_id = "rqfgjw1r"
run = api.run(f"ship_segmentation/ship-segmentation/{run_id}")

# Location of file you want to upload
ckpt_path = "/workspace/logs/train/runs/2023-07-15_13-11-08/checkpoints/epoch_049.ckpt"

# Upload
run.upload_file(ckpt_path)
print(f"Successfully upload {ckpt_path}")
