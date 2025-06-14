FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Ensure python points to python3 (only if needed; Ubuntu 20.04 default is python3)
RUN ln -sf $(which python3) /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PaddlePaddle GPU (for CUDA 11.7)
RUN pip install paddlepaddle-gpu==2.6.2.post117 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/

# Set working directory
WORKDIR /app

# Copy your local code to the container
COPY . .

# Install other Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Run model download script if you have one
# RUN bash scripts/download_models.sh

# Expose Gradio/Flask/your app port (adjust if needed)
# EXPOSE 9999

# Start the application 
# CMD ["python", "app/main.py"]

