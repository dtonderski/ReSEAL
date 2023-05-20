# ReSEAL: Rethinking SEAL in the context of ObjectNav
This repository is an implementation of SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency [[arxiv]](https://arxiv.org/abs/2112.01001).
SEAL is an algorithm to enable embodied agents to explore its environment and improve it perception model in a self-supervised manner.
Exploration is driven by policy learned with reinforcement learning.
Using self-supervised learning, SEAL improves the perception model, a semantic segmentation model.

# Setup
We use the `devcontainer` feature is `vscode` for the development environment.
For more information about devloping inside a container, we refer to [[link]](https://code.visualstudio.com/docs/devcontainers/containers#_create-a-devcontainerjson-file).
The only requirement is installing the `ms-vscode-remote.remote-containers` extension in `vscode`.

~~The development container is built on the [[AI Habitat Challenge 2023]](https://aihabitat.org/challenge/2023/) Docker image [[Docker Hub]](https://aihabitat.org/challenge/2023/)~~

We define our own Docker image in `docker/Dockerfile.cuda`, which is built on Nvidia's [CudaGL containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cudagl).

To build the image, run the following `tools/build_images.sh` script.
To download the image, run the following command in the terminal:
```bash
docker pull dominic4810/reseal:cuda-v1.0
```

Run the following command in `vscode` to build the container and start developing in the `devcontainer`:
```
Dev Containers: Rebuild and Reopen in Container
```
# Data
## Downloading HM3D dataset
To download the data, run:
```
conda activate habitat
sh tools/get_data.sh
```
This will prompt you for your Matterport API token ID and secret API token (the latter will not be shown on screen),
and download the data to the ./data/raw/ directory. If you want to download your data to a non-default directory, run:
```
conda activate habitat
sh tools/get_data.sh [data-path]
```
## Generating trajctories
To generate trajectories, use the following script:
```
python src/scripts/generate_trajectories.py
```
* `--scene-name`: e.g. "train/00000-kfPV7w3FaU5"
* `--start-position`: Initial location of the agent, in (x, y, z). E.g. "[-0.6, 1.2, 0.0]"
* `--max-num-steps`: Maximum number of steps/commands in integer
* `--goal-position`: Coordinate of goal position in (x, y, z)
* `--use-random-policy`: flag for using random policy instead of goal position
* `--commands-file`: Path to `json` file with list of agent commands. E.g. "config/trajectories.json"

These will be saved to `data/interim/trajectories/{scene_name}`.

## Generate 3D map
Run the following command to generate, 3D semantic maps and top down view of the map
```
python src/scripts/generate_semantic_map_3d.py --scene-name train/00000-kfPV7w3FaU5 --num-steps 50
```

# Running on Euler Cluster
To run the scripts on ETHZ's Euler cluster, we use conda environments.

First, you have to define the modules:
```
module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
```

Before installing the conda environment, we have to comment line `15` in `requirements.txt`. This is because `open3d` is [not support on CentOs 7.x](https://github.com/isl-org/Open3D/issues/4706)

We provide scripts for setting up the conda environment:

```bash
bash tools/install_conda.sh
exec bash
bash tools/setup_venv.sh
```

In some cases, you might have to also reinstall torch to get the correct drivers. We recommend running:
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

If all goes well, you can try running some scripts to generate trajectories and 3D semantic maps
```bash
conda activate habitat
python src/scripts/generate_trajectories.py --use-random-policy
```

# Repository strucutre
The repository is structure according to the template in [[link]](https://towardsdatascience.com/structuring-machine-learning-projects-be473775a1b6) and inspired by [[Cookiecutter Data Science]](https://drivendata.github.io/cookiecutter-data-science/)

```
├── README.md          
│
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed. (e.g. virtual sensor output)
│   ├── processed      <- The final dataset for training, visualization, etc. (e.g. training
│   │                     images with self-supervise labels)
│   └── raw            <- 3D scene datasets (e.g. HM3D-Semantics)
│
├── models             <- Trained and serialized models, model configs, etc.
│
├── notebooks          <- Jupyter notebooks.
│
├── references         <- Markdown files, PDF, and other explanatory material
│
├── requirements.txt
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Modules for reading, downloading, and writing data
│   │
│   ├── features       <- Modules for transforming data into features or higher level
│   │                     representations
│   │
│   ├── models         <- Modules of model implementation, trainers, model evaluation, etc.
│   │
│   ├── scripts        <- Modules python scripts.
|   │
│   └── visualization  <- Scripts to create visualizations
│
└── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
```

This structure makes `src` a python module, which makes working with the code base as easy as importing a python package:

```python
from src.data import some_submodule
```
### Example for implementing a script
Furthermore, we use `Fire` so that python scripts implemented in `src.scripts` can be called in the terminal like a shell command.

In `src/scripts/example_script.py`:
```python
from fire import Fire

def main():
    Fire(main_func)


def main_func(
    first_var: str = "",
    second_var: int = 0,
):
    print(first_var, second_var)
```

In `setup.py`:
```python
...
'console_scripts': [
    'example_script = src.scripts.example_script:main',
]
```

In the terminal:
```bash
$ example_script --first-var foobar --second-var 42
foobar 42
```

# X11 Forwarding
To get GUIs running in docker containers to display, we need to setup X11 forwarding.
The `devcontainer` is already setup for X11 forwarding.

We mount the `.Xauthority` file in the devcontainer.
If this doesn't file doesn't exist, simply create it:
```bash
touch ~/.Xauthority
```
or copy the existing file:
```bash
cp /run/usr/1000/gdm/Xauthority ~/.Xauthority
```
The existing file can be found by running `xauth info`

If you encounter errors `Display not found` or `Could not initialize GLFW`, try running:
```bash
xhost +
````
This disables access control, allowing all clients to connect to the X server. To re-enable access control, run:
```bash
xhost -
```

### Status
Summary of which setups work. Tested using `habitat-viewer`
| Set up        | Status/Known issues   |
| ------------- | -------------         |
| `conda` installation on **ubuntu** machine | :white_check_mark:   |
| `conda` installation on MacOs | :white_check_mark: |
| `conda` installation on remote **ubuntu** machine, connected via SSH from MacOS | :x: Not working due to issues with OpenGL on MacOS  |
| `devcontainer` running on local **ubuntu** machine  | :white_check_mark:      |
| `devcontainer` running on remote **ubuntu** machine, connected via SSH from MacOS  | :x: Not working due to issues with OpenGL on MacOS|
| `devcontainer` running local MacOS  | :x: Not working due to issues with OpenGL on MacOS|
