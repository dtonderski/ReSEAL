from typing import Dict, Tuple, Union

import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from nptyping import Float, Int, NDArray, Shape
from plotly import graph_objects as go
from yacs.config import CfgNode

from ..utils.category_mapping import get_reseal_index_to_color_dict, get_reseal_name_to_reseal_index_dict
from ..utils.datatypes import LabelMap3DCategorical, SemanticMap3D


def visualize_semantic_map(semantic_map: SemanticMap3D, object_threshold:float = 0.5,
                           opacity:float = 0.5, data_paths_cfg: CfgNode = None) -> pv.Plotter:
    map_for_plotting = get_map_for_plotting_from_semantic_map(semantic_map, object_threshold)
    return visualize_map(map_for_plotting, opacity, data_paths_cfg)

def visualize_categorical_label_map(label_map: LabelMap3DCategorical, opacity:float = 0.5,
                                    data_paths_cfg: CfgNode = None) -> pv.Plotter:
    map_for_plotting = get_map_for_plotting_from_label_map(label_map)
    return visualize_map(map_for_plotting, opacity, data_paths_cfg)

def visualize_map(map_for_plotting: NDArray[Shape["X,Y,Z"], Int],
                  opacity:float = 0.5,
                  data_paths_cfg: CfgNode = None) -> pv.Plotter:

    color_map, reseal_map = get_plotting_dicts(map_for_plotting.max(), opacity, data_paths_cfg)
    dims = map_for_plotting.shape

    grid = pv.UniformGrid()
    grid.dimensions = tuple(np.array(dims) + 1) # type: ignore
    grid.cell_data["values"] = map_for_plotting.flatten(order="F")  # Flatten the array in column-major order
    grid = grid.threshold(0.001)

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
    plotter.set_background('#AAAAAA')
    plotter.add_mesh(grid, scalars="values", cmap=color_map, annotations = reseal_map, scalar_bar_args=sargs)
    return plotter

def get_map_for_plotting_from_semantic_map(semantic_map: SemanticMap3D, object_threshold:float = 0.5
                                           ) -> NDArray[Shape["X,Y,Z"], Int]:
    occupancy = semantic_map[:,:,:,0]
    map_at_index_semantics_thresholded = semantic_map[:,:,:,1:].copy()
    map_at_index_semantics_thresholded[object_threshold < 0.5] = 0

    # 0 if no semantic information, 1-6 if semantic information
    map_at_index_categorical = (np.argmax(map_at_index_semantics_thresholded, axis=-1)
                                + np.sum(map_at_index_semantics_thresholded, axis=-1))

    # 0 if no occupancy, 1 if occupied but no semantic information, 2-7 if occupied and semantic information
    map_for_plotting = (map_at_index_categorical + occupancy)
    return map_for_plotting

def get_map_for_plotting_from_label_map(label_map: LabelMap3DCategorical) -> NDArray[Shape["X,Y,Z"], Int]:
    return label_map.sum(axis=-1)

def get_plotting_dicts(max_present_label_index: int, opacity:float = 0.5,
                       data_paths_cfg: CfgNode = None) -> Tuple[ListedColormap, Dict[int, str]]:
    """Get color map and label names for plotting.

    Args:
        opacity (float, optional): Not sure why, but this sets the opacity of all voxels. Defaults to 0.4.
        data_paths_cfg (CfgNode, optional): data paths config. Defaults to None.

    Returns:
        Tuple[ListedColormap, Dict[int, str]]: color map and label names
    """
    color_dict = {k:v for k,v in get_reseal_index_to_color_dict(data_paths_cfg).items() if k < max_present_label_index}
    color_dict[0] = np.array([*color_dict[0], opacity])
    color_map = ListedColormap([*[color_dict[category] for category in color_dict]])

    reseal_map = {v+1:k for k,v in get_reseal_name_to_reseal_index_dict(data_paths_cfg).items()}
    return color_map, reseal_map

def get_point_cloud_representation_of_categorical_label_map(label_map, grid_index_of_origin, resolution, 
                                                            get_points_with_no_semantics: bool = True):
    """ Generates a (N, 6) point cloud representation of voxel indices with semantic information, which can be easily
    logged to WandB. The first 3 dimensions are the coordinate of the origin of the voxel (i.e. the bottom left corner),
    and the last 3 dimensions are the RGB values of the voxel, according to the color map defined in reseal.tsv.

    Args:
        label_map (_type_): _description_
        grid_index_of_origin (_type_): _description_
        resolution (_type_): _description_

    Returns:
        _type_: _description_
    """
    from src.utils.geometric_transformations import grid_indices_to_world_coordinates
    from src.visualisation.semantic_map_visualization import get_map_for_plotting_from_label_map, get_plotting_dicts

    map_for_plotting = get_map_for_plotting_from_label_map(label_map)
    color_map, _ = get_plotting_dicts(map_for_plotting.max(), 0.1)
    min_value = 1 - int(get_points_with_no_semantics) # 0 if get_points_with_no_semantics, 1 otherwise
    grid_indices = np.argwhere(map_for_plotting > min_value)
    coordinates = grid_indices_to_world_coordinates(grid_indices, grid_index_of_origin, resolution)
    # Need -1 here because unoccupied voxels are should be mapped to reseal index 0
    colours = color_map(map_for_plotting[map_for_plotting > min_value]-1)[..., :3]
    return np.concatenate([coordinates, colours], axis=-1)

def visualize_point_cloud_representation_of_categorical_label_map(
    point_cloud_with_colors: NDArray[Shape["N,6"], Float], scene_id: str, show_points_with_no_semantics: bool = True
    ) -> go.Figure:
    coordinates = point_cloud_with_colors[:, :3]
    colors = point_cloud_with_colors[:, 3:]  # Color values in [0, 1]

    interesting_mask = colors.sum(axis = -1) > 1

    data = [go.Scatter3d(
        x=coordinates[interesting_mask, 0],
        y=coordinates[interesting_mask, 1],
        z=coordinates[interesting_mask, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=colors[interesting_mask],  # set color to RGB values
            colorscale=None,  # Do not use a colorscale with RGB
            opacity= 1
            ),
        name='Points with semantics'
        )]

    if show_points_with_no_semantics:
        data.append(go.Scatter3d(
            x=coordinates[~interesting_mask, 0],
            y=coordinates[~interesting_mask, 1],
            z=coordinates[~interesting_mask, 2],
            mode='markers',
            marker=dict(
                size=0.1,
                color=colors[~interesting_mask],  # set color to RGB values
                colorscale=None,  # Do not use a colorscale with RGB
                opacity= 0.1
                ),
            name='Points without semantics'
            ))

    fig = go.Figure(data=data)

    fig.update_layout(
        title=scene_id,
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),  # Change the 'up' direction to the y-axis
            ),
    )
    return fig