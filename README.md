# ReSEAL: Rethinking SEAL in the context of ObjectNav
This repository is an implementation of SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency [[arxiv]](https://arxiv.org/abs/2112.01001).
SEAL is an algorithm to enable embodied agents to explore its environment and improve it perception model in a self-supervised manner.
Exploration is driven by policy learned with reinforcement learning.
Using self-supervised learning, SEAL improves the perception model, a semantic segmentation model.

# Setup
We use the `devcontainer` feature is `vscode` for the development environment.
For more information about devloping inside a container, we refer to [[link]](https://code.visualstudio.com/docs/devcontainers/containers#_create-a-devcontainerjson-file).
The only requirement is installing the `ms-vscode-remote.remote-containers` extension in `vscode`.

The development container is built on the [[AI Habitat Challenge 2023]](https://aihabitat.org/challenge/2023/) Docker image [[Docker Hub]](https://aihabitat.org/challenge/2023/).
To download the image, run the following command in the terminal:
```bash
docker pull fairembodied/habitat-challenge:habitat_navigation_2023_base_docker
```
For more information about working with this image, we refer to this [[guide]](https://github.com/facebookresearch/habitat-lab#docker-setup).

Run the following command in `vscode` to build the container and start developing in the `devcontainer`:
```
Dev Containers: Rebuild and Reopen in Container
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
