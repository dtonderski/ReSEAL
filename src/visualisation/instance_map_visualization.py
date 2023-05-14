from typing import Dict

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
from yacs.config import CfgNode

from ..utils.category_mapping import get_instance_index_to_reseal_name_dict, get_reseal_index_to_color_dict
from ..utils.datatypes import InstanceMap2DCategorical, InstanceMap3DCategorical, LabelMap3DCategorical, RGBImage


def get_instance_colormap(opacity: float = 0.5, base_cmap: str = 'viridis', data_paths_cfg: CfgNode = None,
                          max_instance_index = 256
                          ) -> LinearSegmentedColormap:
    cmap = get_cmap(base_cmap)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    unlabeled_color = get_reseal_index_to_color_dict(data_paths_cfg)[0]
    for i in range(int(cmap.N//max_instance_index/2)):
        cmaplist[i] = [*unlabeled_color, opacity]
    return LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)


def visualize_3d_categorical_instance_map(instance_map: InstanceMap3DCategorical,
                                          label_map: LabelMap3DCategorical,
                                          opacity: float = 0.5,
                                          base_cmap: str = 'viridis',
                                          data_paths_cfg: CfgNode = None):
    """Note that we need the label map to get the reseal names for the instance map values.

    Args:
        instance_map (InstanceMap3DCategorical): _description_
        label_map (LabelMap3DCategorical): _description_
        opacity (float, optional): _description_. Defaults to 0.5.
        base_cmap (str, optional): _description_. Defaults to 'viridis'.
        data_paths_cfg (CfgNode, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    map_for_plotting = instance_map.sum(axis = -1)
    dims = map_for_plotting.shape

    plotter = pv.Plotter()
    sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=True,
        n_labels=0,
        italic=True,
        fmt="%.1f",
        font_family="arial",
    )
    grid = pv.UniformGrid()

    grid.dimensions = tuple(np.array(dims) + 1) # type: ignore
    grid.cell_data["values"] = map_for_plotting.flatten(order="F")  # Flatten the array in column-major order
    grid = grid.threshold(0.001)

    map_index_to_reseal_name_dict = {
        k+1: v for k, v in get_instance_index_to_reseal_name_dict(instance_map, label_map, data_paths_cfg).items()}



    plotter.add_mesh(grid, scalars="values",
                     cmap=get_instance_colormap(opacity, base_cmap, data_paths_cfg, instance_map.max()),
                     annotations = map_index_to_reseal_name_dict, scalar_bar_args=sargs)
    plotter.set_background('#AAAAAA')
    return plotter

def visualize_2d_categorical_instance_map(instance_map_2d: InstanceMap2DCategorical,
                                          rgb_image: RGBImage,
                                          instance_label_to_reseal_name_dict: Dict[int, str]):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axs

    unique_indices = np.unique(instance_map_2d[..., 0])
    max_label = np.max(unique_indices)

    colormap = get_instance_colormap(0.5, max_instance_index=max_label)

    # First plot
    im = ax1.imshow(instance_map_2d, cmap = colormap, interpolation='none')
    cbar = fig.colorbar(im, ax=ax1, cmap=colormap)
    cbar.vmin = 0
    cbar.ax.get_yaxis().set_ticks([])
    unique_indices_reseal_names = [instance_label_to_reseal_name_dict[i] for i in unique_indices]

    for j, (index, name) in enumerate(zip(unique_indices, unique_indices_reseal_names)):
        cbar.ax.text(1.5, (index), name, va='center')
    # Second plot
    instance_map_2d_colored = colormap(plt.Normalize()(instance_map_2d[..., 0]))[...,:3]
    instance_map_2d_colored[instance_map_2d_colored == 1/3] = 0
    ax2.imshow(rgb_image * 0.3 + (rgb_image + 0.8) * instance_map_2d_colored)

    # Third plot
    ax3.imshow(rgb_image)
