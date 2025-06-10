# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app
 
# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
# Upgrade pip and install Python package management tools
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy==1.23.5 \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 \
    opencv-python==4.6.0.66 \
    face-recognition==1.3.0 \
    tqdm==4.65.0 \
    pillow==9.5.0 \
    scikit-learn==1.2.2 \
    dlib==19.24.0

# Copy application code into the container (adjust if necessary)
COPY . /app

# Run the application (adjust this as needed)
CMD ["python3", "encodingvideos.py"]