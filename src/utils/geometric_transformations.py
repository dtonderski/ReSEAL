import numpy as np
from nptyping import NDArray, Shape, Int, Float


def coordinates_to_grid_indices(coordinates: NDArray[Shape["NPixels, 3"], Float],
                                grid_index_of_origin: NDArray[Shape["3"], Int],
                                voxel_size: float) -> NDArray[Shape["NPixels, 3"], Int]:
    return (np.floor(coordinates/voxel_size) + grid_index_of_origin).astype(int)

def grid_indices_to_world_coordinates(grid_indices: NDArray[Shape["NPixels, 3"], Int],
                                      grid_index_of_origin: NDArray[Shape["3"], Int],
                                      voxel_size: float,
                                      voxel_center_coordinates: bool = False) -> NDArray[Shape["NPixels, 3"], Float]:
    # Returns the coordinates of the voxel in world coordinates
    voxel_origin_world_coordinates = (grid_indices - grid_index_of_origin)*voxel_size
    if voxel_center_coordinates:
        voxel_origin_world_coordinates += voxel_size/2
    return voxel_origin_world_coordinates
