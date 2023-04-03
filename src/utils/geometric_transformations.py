import numpy as np
from typing import Union
from nptyping import NDArray, Shape, Int, Float
from ..utils.datatypes import GridIndex2D, GridIndex3D


def coordinates_to_grid_indices(coordinates: NDArray[Shape["NPixels, NDims"], Float],
                                grid_indices_of_origin: Union[GridIndex2D, GridIndex3D],
                                voxel_size: float) -> NDArray[Shape["NPixels, NDims"], Int]:
    return (np.floor(coordinates/voxel_size) + grid_indices_of_origin).astype(int)

def grid_indices_to_world_coordinates(grid_indices: NDArray[Shape["NPixels, NDims"], Int],
                                      grid_indices_of_origin: Union[GridIndex2D, GridIndex3D],
                                      voxel_size: float,
                                      voxel_center_coordinates: bool = False) \
                                      -> NDArray[Shape["NPixels, NDims"], Float]:
    # Returns the coordinates of the voxel in world coordinates
    voxel_origin_world_coordinates = (grid_indices - grid_indices_of_origin)*voxel_size
    if voxel_center_coordinates:
        voxel_origin_world_coordinates += voxel_size/2
    return voxel_origin_world_coordinates
