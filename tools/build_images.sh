#!/bin/bash

# Build the CUDA-enabled image
echo "Building CUDA image..."
docker build \
  --file "./docker/Dockerfile.cuda" \
  --tag reseal_dev_image:cuda-v1.0 \
  "./"

# Build the non-CUDA image
# echo "Building non-CUDA image..."
# docker build \
#   --file "${BASE_DIR}/Dockerfile" \
#   --build-arg BASE_IMAGE=ubuntu:20.04 \
#   --tag reseal_dev_image:cpu-v1.0 \
#   "${BASE_DIR}"

# Display a message indicating that the build process is complete
echo "Build complete!"

