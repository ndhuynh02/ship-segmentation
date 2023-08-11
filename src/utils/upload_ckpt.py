import wandb

api = wandb.Api()

run_id = "rqfgjw1r"
run = api.run(f"ship_segmentation/ship-segmentation/{run_id}")

ckpt_path = "/workspace/logs/train/runs/2023-07-15_13-11-08/checkpoints/epoch_049.ckpt"
run.upload_file(ckpt_path)
