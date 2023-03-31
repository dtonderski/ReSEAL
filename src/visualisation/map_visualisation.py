
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from yacs.config import CfgNode
from ..utils.geometric_transformations import grid_indices_to_world_coordinates
from ..utils.datatypes import SemanticMap3D

def calculate_vertices(xmin=0, ymin=0, zmin=0, xmax=None, ymax=None, zmax=None):
    xmax = xmin + 1 if xmax is None else xmax
    ymax = ymin + 1 if ymax is None else ymax
    zmax = zmin + 1 if zmax is None else zmax
    return {
        "x": [xmin, xmin, xmax, xmax, xmin, xmin, xmax, xmax],
        "y": [ymin, ymax, ymax, ymin, ymin, ymax, ymax, ymin],
        "z": [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax],
        "i": [7, 0, 0, 0, 4, 4, 6, 1, 4, 0, 3, 6],
        "j": [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        "k": [0, 7, 2, 3, 6, 7, 1, 6, 5, 5, 7, 2],
    }

def draw_cube(vertices, color = "gold", opacity = 0.5, name = "cube", colors_shown_in_legend = None,
              legendrank = 0):
    colors_shown_in_legend = set() if colors_shown_in_legend is None else colors_shown_in_legend
    if color in colors_shown_in_legend:
        showlegend = False
    else:
        showlegend = True
        colors_shown_in_legend.add(color)
    fig = go.Figure(data=[
        go.Mesh3d(
            # 8 vertices of a cube
            x=vertices['x'],
            y=vertices['y'],
            z=vertices['z'],
            colorscale=[[0, 'gold']],
            # i, j and k give the vertices of triangles
            i = vertices["i"],
            j = vertices["j"],
            k = vertices["k"],
            color = color,
            showscale=True,
            opacity = opacity,
            name=name,
            showlegend=showlegend,
            legendgroup = color,
            legendrank = legendrank
        )
    ])
    return fig, colors_shown_in_legend

def draw_voxels(semantic_3d_map: SemanticMap3D, cfg: CfgNode, colorscale: str = "viridis", return_colors = False):
    indices_of_occupied_voxels = np.array(np.where(semantic_3d_map[:,:,:,0])).transpose()
    semantic_classes_of_occupied_voxels = np.argmax(
        semantic_3d_map[np.where(semantic_3d_map[:,:,:,0])][:,1:], axis=1)
    cubes = []
    colors_shown_in_legend: set = set()
    colors = px.colors.sample_colorscale(colorscale,
                                        [n/(cfg.NUM_SEMANTIC_CLASSES -1) for n in range(cfg.NUM_SEMANTIC_CLASSES)])
    occupied_voxel_coordinates = grid_indices_to_world_coordinates(indices_of_occupied_voxels,
                                                                   cfg.EGOCENTRIC_MAP_ORIGIN_OFFSET,
                                                                   cfg.RESOLUTION)

    for i, (x,y,z) in enumerate(occupied_voxel_coordinates): # pylint: disable=invalid-name
        vertices = calculate_vertices(x,y,z)
        cube, colors_shown_in_legend = draw_cube(vertices, color = colors[semantic_classes_of_occupied_voxels[i]],
                                                opacity = 0.5, name = str(semantic_classes_of_occupied_voxels[i]),
                                                colors_shown_in_legend=colors_shown_in_legend,
                                                legendrank = semantic_classes_of_occupied_voxels[i])
        cubes.append(cube)

    if return_colors:
        return cubes, colors
    return cubes
