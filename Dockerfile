FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Copy requirements.txt and install dependencies
COPY requirements_for_docker.txt .

RUN apt-get update && pip install --upgrade pip 
RUN apt-get install -y git

RUN pip install --no-cache-dir -r requirements_for_docker.txt

# Copy the rest of the files
COPY . .

RUN git config --global --add safe.directory /workspace

# Run bash by default
CMD ["bash"]
