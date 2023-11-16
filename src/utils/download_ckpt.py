import argparse
import shutil
from pathlib import Path

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download checkpoint from Wandb")
    parser.add_argument(
        "--url",
        default="https://api.wandb.ai/files/ship_segmentation/ship-segmentation/t61uuwps/logs/train/runs/2023-08-08_17-15-09/checkpoints/epoch_048.ckpt",
        help="Example: https://api.wandb.ai/files/ship_segmentation/ship-segmentation/t61uuwps/logs/train/runs/2023-08-08_17-15-09/checkpoints/epoch_048.ckpt",
    )
    parser.add_argument("--directory", default=None, help="download checkpoint file directory")
    parser.add_argument("--ckpt", default="checkpoint.ckpt", help="file checkpoint name")

    # Initialize the Weights & Biases API
    api = wandb.Api()

    args = parser.parse_args()
    redundant = "https://api.wandb.ai/files/"

    url = args.url[len(redundant) :]

    # Get the run
    run = api.run("/".join(url.split("/")[:3]))

    # File you want to download
    file_path = "/".join(url.split("/")[3:])

    # # Location to save the downloaded file
    location = Path(args.directory) if args.directory is not None else Path().resolve()

    # Download
    run.file(file_path).download(root=location)

    # move checkpoint to propricate location
    Path(location / file_path).rename(location / args.ckpt)
    # remove logs/... folder
    shutil.rmtree(location / file_path.split("/")[0])

    print(f'Successfully downloaded file "{args.ckpt}" to {location}')
