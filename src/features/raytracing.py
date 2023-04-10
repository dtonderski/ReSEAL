from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from nptyping import Bool, Float, Int, NDArray, Shape
from numba import jit
from yacs.config import CfgNode

from ..utils.datatypes import Coordinate3D, GridIndex3D, SemanticMap3D
from ..utils.geometric_transformations import coordinates_to_grid_indices


@jit(nopython=False, forceobj=True)
def advance_rays(x_arr: NDArray[Shape["*"], Float],
                 y_arr: NDArray[Shape["*"], Float],
                 z_arr: NDArray[Shape["*"], Float],
                 i_arr: NDArray[Shape["*"], Int],
                 j_arr: NDArray[Shape["*"], Int],
                 k_arr: NDArray[Shape["*"], Int],
                 resolution: float,
                 grid_index_of_origin: GridIndex3D,
                 ray_directions: NDArray[Shape["NRays, 3"], Float],
                 active_ray_mask: NDArray[Shape["*"], Bool],
                 active_ray_indices: NDArray[Shape["*"], Int],
                 occupancy_grid: NDArray[Shape["NumPixelsX, NumPixelsY, NumPixelsZ"], Float]):
    """Advance the rays by one step, according to the algorithm in A Fast Voxel Traversal Algorithm for Ray Tracing by \
        John Amanatides and Andrew Woo (http://www.cse.yorku.ca/~amana/research/grid.pdf).

    Args:
        x_arr (NDArray[Shape["NRays"], Float]): x-coordinates of the origin of each ray.
        y_arr (NDArray[Shape["NRays"], Float]): y-coordinates of the origin of each ray.
        z_arr (NDArray[Shape["NRays"], Float]): z-coordinates of the origin of each ray.
        i_arr (NDArray[Shape["NRays"], Int]): i-indices of the origin of each ray.
        j_arr (NDArray[Shape["NRays"], Int]): j-indices of the origin of each ray.
        k_arr (NDArray[Shape["NRays"], Int]): k-indices of the origin of each ray.
        resolution (float): the size of a voxel (in the same coordinates as x_arr).
        grid_index_of_origin (GridIndex3D): grid index of the origin of the scene.
        ray_directions (NDArray[Shape["NRays, 3"], Float]): array containing the x,y,z directions of each ray.
        active_ray_mask (NDArray[Shape["NRays"], Bool]): boolean mask determining whether a ray should be advanced.
        active_ray_indices (NDArray[Shape["NActiveRays"], Int]): array containing the indices of the active rays.
        occupancy_grid (NDArray[Shape["NumPixelsX, NumPixelsY, NumPixelsZ"], Float]): first channel of the semantic \
            map, determines whether a voxel is occupied.

    Returns:
        x_arr (NDArray[Shape["NRays"], Float]): x-coordinates of the origin of each ray.
        y_arr (NDArray[Shape["NRays"], Float]): y-coordinates of the origin of each ray.
        z_arr (NDArray[Shape["NRays"], Float]): z-coordinates of the origin of each ray.
        i_arr (NDArray[Shape["NRays"], Int]): i-indices of the origin of each ray.
        j_arr (NDArray[Shape["NRays"], Int]): j-indices of the origin of each ray.
        k_arr (NDArray[Shape["NRays"], Int]): k-indices of the origin of each ray.
        active_ray_mask (NDArray[Shape["NRays"], Bool]): boolean mask determining whether a ray should be advanced.
        active_ray_indices (NDArray[Shape["NActiveRays"], Int]): array containing the indices of the active rays.
    """
    x_next = resolution*(
        i_arr[active_ray_mask] - grid_index_of_origin[0] + (ray_directions[0, active_ray_mask]>0).astype(int))
    y_next = resolution*(
        j_arr[active_ray_mask] - grid_index_of_origin[1] + (ray_directions[1, active_ray_mask]>0).astype(int))
    z_next = resolution*(
        k_arr[active_ray_mask] - grid_index_of_origin[2] + (ray_directions[2, active_ray_mask]>0).astype(int))

    # Calculate the 'time' it takes for each ray to hit the next grid wall in each dimension
    dt_x = (x_next - x_arr[active_ray_mask]) / ray_directions[0,active_ray_mask]
    dt_y = (y_next - y_arr[active_ray_mask]) / ray_directions[1,active_ray_mask]
    dt_z = (z_next - z_arr[active_ray_mask]) / ray_directions[2,active_ray_mask]

    # Get the indices of the rays that hit the grid wall in each dimension first
    dt_x_smallest = np.logical_and(dt_x <= dt_y, dt_x <= dt_z)
    dt_y_smallest = np.logical_and(dt_y <= dt_x, dt_y <= dt_z)
    dt_z_smallest = np.invert(np.logical_or(dt_x_smallest, dt_y_smallest))

    # Update the ray to hit the closest grid wall
    x_smallest = active_ray_indices[dt_x_smallest]
    i_arr[x_smallest] += np.sign(ray_directions[0, x_smallest]).astype(int)
    x_arr[x_smallest] = x_next[dt_x_smallest]
    y_arr[x_smallest] += ray_directions[1, x_smallest] * dt_x[dt_x_smallest]
    z_arr[x_smallest] += ray_directions[2, x_smallest] * dt_x[dt_x_smallest]

    y_smallest = active_ray_indices[dt_y_smallest]
    j_arr[y_smallest] += np.sign(ray_directions[1,y_smallest]).astype(int)
    x_arr[y_smallest] += ray_directions[0, y_smallest] * dt_y[dt_y_smallest]
    y_arr[y_smallest] = y_next[dt_y_smallest]
    z_arr[y_smallest] += ray_directions[2, y_smallest] * dt_y[dt_y_smallest]

    z_smallest = active_ray_indices[dt_z_smallest]
    k_arr[z_smallest] += np.sign(ray_directions[2,z_smallest]).astype(int)
    x_arr[z_smallest] += ray_directions[0,z_smallest] * dt_z[dt_z_smallest]
    y_arr[z_smallest] += ray_directions[1,z_smallest] * dt_z[dt_z_smallest]
    z_arr[z_smallest] = z_next[dt_z_smallest]

    # If ray has hit a voxel, deactivate it
    rays_to_deactivate_hit = np.where(occupancy_grid[i_arr,j_arr,k_arr])
    active_ray_mask[rays_to_deactivate_hit] = False

    # If ray has hit the limit of the grid, deactivate it
    rays_to_deactivate_limit = np.where(np.logical_or.reduce((i_arr == 0, j_arr == 0, k_arr == 0,
                                                                i_arr == occupancy_grid.shape[0]-1,
                                                                j_arr == occupancy_grid.shape[1]-1,
                                                                k_arr == occupancy_grid.shape[2]-1)))
    active_ray_mask[rays_to_deactivate_limit] = False
    active_ray_indices = np.flatnonzero(active_ray_mask)

    return x_arr, y_arr, z_arr, i_arr, j_arr, k_arr, active_ray_mask, active_ray_indices

def raytrace_3d(ray_directions: NDArray[Shape["3, NRays"], Float],
                semantic_map_3d: SemanticMap3D,
                ray_origin_coords: Union[Coordinate3D, NDArray[Shape["3, NRays"], Float]],
                grid_index_of_origin: Union[GridIndex3D, NDArray[Shape["3"], Int]],
                map_builder_cfg: CfgNode) \
                -> Tuple[NDArray[Shape["NRays, NChannels"], Int],                     # type: ignore[name-defined]
                         Dict[int, List[Tuple[float, float, float]]]]:
    """Traces rays through a 3D semantic map until they hit a voxel, and returns the class of the voxel they hit.

    Args:
        ray_directions (NDArray[Shape["NRays, 3"], Float]): array containing the x,y,z directions of each ray.
        semantic_map_3d (SemanticMap3D): semantic map of the scene in which to raytrace.
        ray_origin_coords (Union[Coordinate3D, NDArray[Shape["NRays, 3"], Float]]): either a tuple containing the \
            coordinates of the origin of all rays, or an array of shape (NRays, 3) containing the coordinates of the \
            origin of each ray.
        grid_index_of_origin (GridIndex3D): grid index of the origin of the scene.
        map_builder_cfg (CfgNode): configuration of the current map builder, must contain RESOLUTION.

    Returns:
        ray_labels (NDArray[Shape[NRays, NChannels], Int]): If ray i hits a voxel which in the semantic 3d map is \
            represented by a vector v, then ray_labels[i] = v. If it hits no voxel, then ray_labels[i,j] = 0 for all j.
        intersections (Dict[int, List[Tuple[float, float, float]]]): dictionary mapping from ray index to \
                                                                     list of containing first and final ray coordinate.
    """
    occupancy_grid = semantic_map_3d[:,:,:,0]

    # normalize ray_directions
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis = 0)

    x_arr, y_arr, z_arr, i_arr, j_arr, k_arr = get_ray_origin_coords_and_indices(ray_origin_coords,
                                                                                 grid_index_of_origin,
                                                                                 ray_directions,
                                                                                 map_builder_cfg)

    intersections = {ray_index: [(x,y,z)] for ray_index,(x,y,z) in enumerate(zip(x_arr, y_arr, z_arr))}

    # Store indices of active rays so we don't have to iterate over all of them
    active_ray_mask = np.ones_like(x_arr, dtype=bool)
    # Deactivate rays that have already hit a voxel
    rays_to_deactivate = np.where(occupancy_grid[i_arr,j_arr,k_arr])
    active_ray_mask[rays_to_deactivate] = False
    # This creates a list of indices of active rays
    active_ray_indices = np.flatnonzero(active_ray_mask)

    # Iterate until all rays have hit a voxel
    while active_ray_indices.size:
        x_arr, y_arr, z_arr, i_arr, j_arr, k_arr, active_ray_mask, active_ray_indices = (
            advance_rays(x_arr, y_arr, z_arr, i_arr, j_arr, k_arr, map_builder_cfg.RESOLUTION, grid_index_of_origin, 
                         ray_directions, active_ray_mask, active_ray_indices, occupancy_grid))

    # Store the intersection points if requested
    for ray_index, (x,y,z) in enumerate(zip(x_arr, y_arr, z_arr)):
        intersections[ray_index].append((x,y,z))

    # ray_labels is the vector representing the grid cell at the end of each ray in the semantic 3d map.
    ray_labels = semantic_map_3d[i_arr,j_arr,k_arr]

    return ray_labels, intersections



def get_ray_origin_coords_and_indices(ray_origin_coords: Union[Coordinate3D, NDArray[Shape["3, NRays"], Float]],
                                      grid_index_of_origin,
                                      ray_directions_normalized,
                                      map_builder_cfg: CfgNode) -> Tuple[NDArray[Shape["*"], Float],
                                                                 NDArray[Shape["*"], Float],
                                                                 NDArray[Shape["*"], Float],
                                                                 NDArray[Shape["*"], Int],
                                                                 NDArray[Shape["*"], Int],
                                                                 NDArray[Shape["*"], Int]]:
    """ Builds the arrays containing the coordinates and grid indices of the origin of each ray.

    Args:
        ray_origin_coords (Union[Coordinate3D, NDArray[Shape["3, NRays"], Float]]): either a tuple containing the \
            coordinates of the origin of all rays, or an array of shape (NRays, 3) containing the coordinates of the \
            origin of each ray.
        grid_index_of_origin (GridIndex3D): grid index of the origin of the scene.
        ray_directions_normalized (NDArray[Shape["NRays, 3"], Float]): array containing the normalized x,y,z directions
            of each ray.
        map_builder_cfg (CfgNode): configuration of the current map builder, must contain RESOLUTION.

    Returns:
        x_arr (NDArray[Shape["NRays"], Float]): x-coordinates of the origin of each ray.
        y_arr (NDArray[Shape["NRays"], Float]): y-coordinates of the origin of each ray.
        z_arr (NDArray[Shape["NRays"], Float]): z-coordinates of the origin of each ray.
        i_arr (NDArray[Shape["NRays"], Int]): i-indices of the origin of each ray.
        j_arr (NDArray[Shape["NRays"], Int]): j-indices of the origin of each ray.
        k_arr (NDArray[Shape["NRays"], Int]): k-indices of the origin of each ray.
    """

    if isinstance(ray_origin_coords, tuple):
        ray_origin_coords_array = np.zeros(ray_directions_normalized.shape)
        ray_origin_coords_array[0] = ray_origin_coords[0]
        ray_origin_coords_array[1] = ray_origin_coords[1]
        ray_origin_coords_array[2] = ray_origin_coords[2]
    else:
        ray_origin_coords_array = ray_origin_coords

    x_arr = ray_origin_coords_array[0]
    y_arr = ray_origin_coords_array[1]
    z_arr = ray_origin_coords_array[2]

    ray_origin_indices = coordinates_to_grid_indices(ray_origin_coords_array.T, grid_index_of_origin,
                                                     map_builder_cfg.RESOLUTION).T

    i_arr = ray_origin_indices[0]
    j_arr = ray_origin_indices[1]
    k_arr = ray_origin_indices[2]

    # If the ray goes backwards in a given dimension, we need to offset the starting grid index by 1
    i_arr -= (ray_directions_normalized[0]<0).astype(int)
    j_arr -= (ray_directions_normalized[1]<0).astype(int)
    k_arr -= (ray_directions_normalized[2]<0).astype(int)

    return x_arr, y_arr, z_arr, i_arr, j_arr, k_arr


def raytrace_3d_from_angles(theta: NDArray[Shape["NRays"], Float],                          # type: ignore[name-defined]
                            phi: NDArray[Shape["NRays"], Float],                            # type: ignore[name-defined]
                            semantic_map_3d: SemanticMap3D,
                            ray_origin_coords: Union[Coordinate3D, NDArray[Shape["3, NRays"], Float]],
                            grid_index_of_origin: GridIndex3D,
                            map_builder_cfg: CfgNode) \
                            -> Tuple[NDArray[Shape["NRays, NChannels"], Int],               # type: ignore[name-defined]
                                     Dict[int, List[Tuple[float, float, float]]]]:
    """Traces rays through a 3D semantic map until they hit a voxel, and returns the class of the voxel they hit.

    Args:
        ray_directions (NDArray[Shape["NRays, 3"], Float]): array containing the x,y,z directions of each ray.
        semantic_map_3d (SemanticMap3D): semantic map of the scene in which to raytrace.
        ray_origin_coords (Union[Coordinate3D, NDArray[Shape["NRays, 3"], Float]]): either a tuple containing the \
            coordinates of the origin of all rays, or an array of shape (NRays, 3) containing the coordinates of the \
            origin of each ray.
        grid_index_of_origin (GridIndex3D): grid index of the origin of the scene.
        map_builder_cfg (CfgNode): configuration of the current map_builder, must contain RESOLUTION.

    Returns:
        ray_classes (NDArray[Shape[NRays], Int]): The class of each ray, equal to the class of the first voxel it hits \
                                                  or -1 if it doesn't hit anything.
        intersections (Dict[int, List[Tuple[float, float, float]]]): dictionary mapping from ray index to \
                                                                     list of containing first and final ray coordinate.
    """

    ray_x_direction = np.sin(theta)*np.cos(phi)
    ray_y_direction = np.sin(theta)*np.sin(phi)
    ray_z_direction = np.cos(theta)

    ray_directions = np.stack((ray_x_direction, ray_y_direction, ray_z_direction), axis = 0)

    return raytrace_3d(ray_directions, semantic_map_3d, ray_origin_coords, grid_index_of_origin, map_builder_cfg)
