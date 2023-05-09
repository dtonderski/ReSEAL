# /usr/bin/bash

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install setuptools==65.6.3
pip install --default-timeout=1000 -r requirements.txt
pip install -r requirements-dev.txt

