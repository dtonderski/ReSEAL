#!/bin/bash

conda init
source ~/.bashrc
source activate habitat

python3 -m pip install --user --default-timeout=100 --use-deprecated=legacy-resolver -r .devcontainer/requirements-dev.txt
python3 -m pip install --user --default-timeout=100 --use-deprecated=legacy-resolver -r requirements.txt
python3 -m pip install -e .