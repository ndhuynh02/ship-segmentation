FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Copy requirements.txt and install dependencies
COPY requirements_for_docker.txt .

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements_for_docker.txt

# Install dependencies for cv2
RUN apt-get install ffmpeg libsm6 libxext6  -y

# set PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/workspace/src"

# Copy the rest of the files
COPY . .

RUN git config --global --add safe.directory /workspace

# Run bash by default
CMD ["bash"]
