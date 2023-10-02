# Triton
Hơw to run: 
- Add torchscript model to model_repository:
    - Unet: add "model.pt" to folder "model_repository/unet/1"
- Run Triton: docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/triton/model_repository:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models
- Run:
    - Unet: python triton/client.py
