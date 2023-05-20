#!/bin/bash

# API-TOKEN-ID= # replace with your API token ID
# API-TOKEN-SECRET= # replace with your API token secret

# Default value
DATA_PATH="./data/raw"

# If a command line argument is provided, use it as the data path
if [ ! -z $1 ]; then
  DATA_PATH=$1
fi

echo "Data will be downloaded to $DATA_PATH"

echo "Please enter your API token ID";
read api_token_id;

echo "Please enter your secret API token";
api_token_secret=$(python -c "import getpass; passwd=getpass.getpass(); print(passwd)")

for split in train val minival
do
    echo "Downloading ${split} split";
    python -m habitat_sim.utils.datasets_download --username $api_token_id \
    --password $api_token_secret --uids hm3d_${split}_v0.2 \
    --data-path $DATA_PATH/${split}/
done
