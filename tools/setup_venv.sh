# /usr/bin/bash

# Setup venv
python3 -m venv .venv
source .venv/bin/activate

# Update pip and setuptool
pip install --upgrade pip
pip install setuptools==65.6.3

# Install habitat-sim
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout v0.2.3
pip install -e .

# Install requirements
pip install --default-timeout=1000 -r requirements.txt
pip install -r requirements-dev.txt

