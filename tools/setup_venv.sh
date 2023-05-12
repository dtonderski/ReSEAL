# /usr/bin/bash

# Install habitat-sim
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat -y

# Install requirements
pip install --default-timeout=1000 -r requirements.txt
pip install -r requirements-dev.txt

