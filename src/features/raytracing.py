from typing import Dict, List, Tuple, Union

import numpy as np
from nptyping import Float, Int, NDArray, Shape
from yacs.config import CfgNode

from ..utils.datatypes import Coordinate3D, GridIndex3D, SemanticMap3D
from ..utils.geometric_transformations import coordinates_to_grid_indices


def raytrace_3d(ray_directions: NDArray[Shape["3, NRays"], Float],
                semantic_map_3d: SemanticMap3D,
                ray_origin_coords: Union[Coordinate3D, NDArray[Shape["3, NRays"], Float]],
                grid_index_of_origin: GridIndex3D,
                cfg: CfgNode,
                return_intersections = False) \
                -> Union[Tuple[NDArray[Shape["NRays"], Int],                                # type: ignore[name-defined]
                               Dict[int, List[Tuple[float, float, float]]]],
                         NDArray[Shape["NRays"], Int]]:
    """Traces rays through a 3D semantic map until they hit a voxel, and returns the class of the voxel they hit.

    Args:
        ray_directions (NDArray[Shape["NRays, 3"], Float]): array containing the x,y,z directions of each ray.
        semantic_map_3d (SemanticMap3D): semantic map of the scene in which to raytrace.
        ray_origin_coords (Union[Coordinate3D, NDArray[Shape["NRays, 3"], Float]]): either a tuple containing the \
            coordinates of the origin of all rays, or an array of shape (NRays, 3) containing the coordinates of the \
            origin of each ray.
        grid_index_of_origin (GridIndex3D): grid index of the origin of the scene.
        cfg (CfgNode): configuration of the current scene.
        return_intersections (bool, optional): determines whether to return intersections. Defaults to False.

    Returns:
        ray_classes (NDArray[Shape[NRays], Int]): The class of each ray, equal to the class of the first voxel it hits \
                                                  or -1 if it doesn't hit anything.
        intersections (Dict[int, List[Tuple[float, float, float]]], optional): dictionary mapping from ray index to \
                                                                               list of intersections.
    """
    grid = semantic_map_3d[:,:,:,0]
    grid_classes = (np.argmax(semantic_map_3d[:,:,:,1:], axis = -1))

    # normalize ray_directions
    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis = 0)

    x_arr, y_arr, z_arr, i_arr, j_arr, k_arr = get_ray_origin_coords_and_indices(ray_origin_coords,
                                                                                 grid_index_of_origin,
                                                                                 ray_directions,
                                                                                 cfg)

    if return_intersections:
        intersections = {ray_index: [(x,y,z)] for ray_index,(x,y,z) in enumerate(zip(x_arr, y_arr, z_arr))}

    # Store indices of active rays so we don't have to iterate over all of them
    active_ray_mask = np.ones_like(x_arr, dtype=bool)
    # Deactivate rays that have already hit a voxel
    rays_to_deactivate = np.where(grid[i_arr,j_arr,k_arr])
    active_ray_mask[rays_to_deactivate] = False
    # This creates a list of indices of active rays
    active_ray_indices = np.flatnonzero(active_ray_mask)

    # Iterate until all rays have hit a voxel
    while active_ray_indices.size:
        # Get the coordinate of the next grid 'wall' (wall of a potential voxel) that each ray will hit in each
        # dimension. The sign term is there because the next voxel wall depends on the direction of the ray.
        x_next = cfg.RESOLUTION*(
            i_arr[active_ray_mask] - grid_index_of_origin[0] + (ray_directions[0, active_ray_mask]>0).astype(int))
        y_next = cfg.RESOLUTION*(
            j_arr[active_ray_mask] - grid_index_of_origin[1] + (ray_directions[1, active_ray_mask]>0).astype(int))
        z_next = cfg.RESOLUTION*(
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

        # Store the intersection points if requested
        if return_intersections:
            for ray_index, (x,y,z,active) in enumerate( # pylint: disable = invalid-name
                zip(x_arr, y_arr, z_arr, active_ray_mask)):
                if active:
                    intersections[ray_index].append((x,y,z))

        # If ray has hit a voxel, deactivate it
        rays_to_deactivate_hit = np.where(grid[i_arr,j_arr,k_arr])
        active_ray_mask[rays_to_deactivate_hit] = False

        # If ray has hit the limit of the grid, deactivate it
        rays_to_deactivate_limit = np.where(np.logical_or.reduce((i_arr == 0, j_arr == 0, k_arr == 0,
                                                                    i_arr == grid.shape[0]-1,
                                                                    j_arr == grid.shape[1]-1,
                                                                    k_arr == grid.shape[2]-1)))
        active_ray_mask[rays_to_deactivate_limit] = False
        active_ray_indices = np.flatnonzero(active_ray_mask)

    # Classify rays based on the voxel they hit, or -1 if they hit the limit of the grid
    ray_classes = grid_classes[i_arr,j_arr,k_arr]
    ray_classes[rays_to_deactivate_limit] = -1

    if return_intersections:
        return ray_classes, intersections
    return ray_classes


def get_ray_origin_coords_and_indices(ray_origin_coords: Union[Coordinate3D, NDArray[Shape["3, NRays"], Float]],
                                      grid_index_of_origin,
                                      ray_directions_normalized,
                                      cfg: CfgNode):
    """ Builds the arrays containing the coordinates and grid indices of the origin of each ray.

    Args:
        ray_origin_coords (Union[Coordinate3D, NDArray[Shape["3, NRays"], Float]]): either a tuple containing the \
            coordinates of the origin of all rays, or an array of shape (NRays, 3) containing the coordinates of the \
            origin of each ray.
        grid_index_of_origin (GridIndex3D): grid index of the origin of the scene.
        ray_directions_normalized (NDArray[Shape["NRays, 3"], Float]): array containing the normalized x,y,z directions
            of each ray.
        cfg (CfgNode): configuration of the current scene.

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

    ray_origin_indices = coordinates_to_grid_indices(ray_origin_coords_array.T, grid_index_of_origin, cfg.RESOLUTION).T

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
                            cfg: CfgNode,
                            return_intersections = False) \
                            -> Union[Tuple[NDArray[Shape["NRays"], Int],                    # type: ignore[name-defined]
                                           Dict[int, List[Tuple[float, float, float]]]],
                                     NDArray[Shape["NRays"], Int]]:
    """Traces rays through a 3D semantic map until they hit a voxel, and returns the class of the voxel they hit.

    Args:
        theta (NDArray[Shape["NRays"], Float]): flat array of theta angles of rays.
        phi (NDArray[Shape["NRays"], Float]): flat array of phi angles of rays.
        semantic_map_3d (SemanticMap3D): semantic map of the scene in which to raytrace.
        ray_origin_coords (Union[Coordinate3D, NDArray[Shape["NRays, 3"], Float]]): either a tuple containing the \
            coordinates of the origin of all rays, or an array of shape (NRays, 3) containing the coordinates of the \
            origin of each ray.
        grid_index_of_origin (GridIndex3D): grid index of the origin of the scene.
        cfg (CfgNode): configuration of the current scene.
        return_intersections (bool, optional): determines whether to return intersections. Defaults to False.

    Returns:
        ray_classes (NDArray[Shape[NRays], Int]): The class of each ray, equal to the class of the first voxel it hits \
                                                  or -1 if it doesn't hit anything.
        intersections (Dict[int, List[Tuple[float, float, float]]], optional): dictionary mapping from ray index to \
                                                                               list of intersections.
    """

    ray_x_direction = np.sin(theta)*np.cos(phi)
    ray_y_direction = np.sin(theta)*np.sin(phi)
    ray_z_direction = np.cos(theta)

    ray_directions = np.stack((ray_x_direction, ray_y_direction, ray_z_direction), axis = 0)

    return raytrace_3d(ray_directions, semantic_map_3d, ray_origin_coords, grid_index_of_origin, cfg,
                       return_intersections)
