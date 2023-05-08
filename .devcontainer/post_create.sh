#!/bin/bash

conda init
source ~/.bashrc
source activate habitat

python3 -m pip install --user --default-timeout=100 --no-warn-script-location -r requirements-dev.txt
python3 -m pip install --user --default-timeout=100 --no-warn-script-location -r requirements.txt
python3 -m pip install -e .
