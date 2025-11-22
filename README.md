This is "Ship Instance Segmentation in Satellite Imagery" project. (Segment ships from images taken by satellite)

It is implemented using [Ligtning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

## 🚀  Quickstart

It is recommend to build a Docker image instead of install requirements with `pip`.

```bash
# Build Docker image
docker build -t satellite .

# Start Docker container
docker run --gpus all --shm-size=4gb -it --name ship_segment -d -p 7860:7860 -t satellite
```

Docker volume may be used with tag `-v satellite_volume:/workspace/data` to store data, which will be downloaded if necessary

However, if `pip` is preferred, please install using the `requirements.txt` file and install [MMRotate](https://mmrotate.readthedocs.io/en/v0.1.1/install.html) with precaution of **PyTorch and CUDA version**

```bash
pip install -r requirements.txt

pip install openmim
mim install mmrotate
```

## 🚄  How to train

```bash
python src/train.py
```

Please note that all configurations (e.g `batch_size` or `num_workers`) can be overridden with [Hydra](https://hydra.cc/)

## ✨  Inference

There are 2 ways to infer using pre-trained model

1. With `src/infer.py`

```bash
python src/infer.py --input input.jpg --output output.png
```

This will create an instance segmentation image on which the mask is already overlaid. Checkpoint path can also be passed as arguments

2. With [Gradio](https://www.gradio.app/)

```bash
python gradio/app.py
```

The model will be deployed on Gradio for 72 hours. This is more user friendly since input image can easily dragged into for inference.
