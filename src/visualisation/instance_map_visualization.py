import numpy as np
import pyvista as pv
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
from yacs.config import CfgNode

from ..utils.category_mapping import get_instance_index_to_reseal_name_dict, get_reseal_color_dict
from ..utils.datatypes import InstanceMap3DCategorical, LabelMap3DCategorical


def get_instance_colormap(opacity: float = 0.5, base_cmap: str = 'viridis', data_paths_cfg: CfgNode = None
                          ) -> LinearSegmentedColormap:
    cmap = get_cmap(base_cmap)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = [*get_reseal_color_dict(data_paths_cfg)[0], opacity]
    return LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)


def visualize_categorical_instance_map(instance_map: InstanceMap3DCategorical,
                                       label_map: LabelMap3DCategorical,
                                       opacity: float = 0.5,
                                       base_cmap: str = 'viridis',
                                       data_paths_cfg: CfgNode = None):
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
    
    plotter.add_mesh(grid, scalars="values", cmap=get_instance_colormap(opacity, base_cmap, data_paths_cfg), 
                     annotations = map_index_to_reseal_name_dict, scalar_bar_args=sargs)
    return plotter
