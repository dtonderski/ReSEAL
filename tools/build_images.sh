#!/bin/bash

# Build the CUDA-enabled image
echo "Building docker image..."
docker build \
  --file "./docker/Dockerfile.cuda" \
  --tag reseal_dev_image:cuda-v1.0 \
  "./"

# Display a message indicating that the build process is complete
echo "Build complete!"

