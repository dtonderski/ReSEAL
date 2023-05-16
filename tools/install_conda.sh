#/bin/bash

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda

# Add conda to path
echo "export PATH=$HOME/miniconda/bin:$PATH" >> ~/.bashrc
source ~/.bashrc

# Initialize conda
conda init bash

# Create habitat environment
conda create -y -n habitat python=3.9 cmake==3.14.0

# Activate habitat environment
echo "conda activate habitat" >> ~/.bashrc
source ~/.bashrc

