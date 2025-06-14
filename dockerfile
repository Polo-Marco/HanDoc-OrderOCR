FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Non iterative
ENV DEBIAN_FRONTEND=noninteractive

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

# Set Preprocessing directory
WORKDIR /app
# Copy your local code to the container
COPY . /app

# Clone PaddleOCR repo
RUN git clone https://github.com/PaddlePaddle/PaddleOCR.git

# Environment Path setting for PaddleOCR
RUN ln -sf /lib/x86_64-linux-gnu/libcudnn.so.8 /lib/x86_64-linux-gnu/libcudnn.so
RUN ln -s /usr/local/cuda/lib64/libcublas.so.11 /usr/local/cuda/lib64/libcublas.so
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install PaddleOCR dependencies
RUN pip install --no-cache-dir -r /app/PaddleOCR/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Run model download script if you have one
# RUN bash scripts/download_models.sh

# Set main directory
WORKDIR /app/app

# Expose Gradio/Flask/your app port (adjust if needed)
EXPOSE 9999

# Start the application 
CMD ["python", "main.py"]

