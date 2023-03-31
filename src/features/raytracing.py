from typing import Dict, List, Tuple, Union
import numpy as np
from yacs.config import CfgNode
from nptyping import NDArray, Shape, Float, Int
from ..utils.datatypes import SemanticMap3D


def raytrace_3d(theta: NDArray[Shape["NRays"], Float],                          # type: ignore[name-defined]
                phi: NDArray[Shape["NRays"], Float],                            # type: ignore[name-defined]
                semantic_map_3d: SemanticMap3D,
                cfg: CfgNode,
                return_intersections = False) \
                -> Union[                                                       # type: ignore[name-defined]
                    Tuple[NDArray[Shape["NRays"], Int],
                          Dict[int, List[Tuple[float, float, float]]]],
                    NDArray[Shape["NRays"], Int]]:
    """_summary_

    Args:
        theta (NDArray[Shape["*"]): flat array of theta angles of rays.
        phi (NDArray[Shape["*"]): flat array of phi angles of rays.
        semantic_map_3d (SemanticMap3D): semantic map of the scene in which to raytrace.
        cfg (CfgNode): config of the current scene.
        return_intersections (bool, optional): determines whether to return intersections. Defaults to False.

    Returns:
        ray_classes (NDArray[Shape[NRays], Int]): The class of each ray, equal to the class of the first voxel it hits
                                                  or -1 if it doesn't hit anything
        intersections (Dict[int, List[Tuple[float, float, float]]], optional): dictionary mapping from ray index to
                                                                               list of intersections
    """

    # TODO: this shouldn't be egocentric right?
    origin = cfg.EGOCENTRIC_MAP_ORIGIN_OFFSET

    grid: NDArray[Shape["NumPixelsX, NumPixelsY, NumPixelsZ"], Float] = semantic_map_3d[:,:,:,0]
    grid_classes: NDArray[Shape["NumPixelsX, NumPixelsY, NumPixelsZ, NumChannels-1"], Float] = (
        np.argmax(semantic_map_3d[:,:,:,1:], axis = -1))

    # Store some angles for optimization so we don't have to recalculate in each iteration
    # x = r*sin(theta)*cos(phi)
    sincos = np.sin(theta)*np.cos(phi)
    # y = r*sin(theta)*sin(phi)
    sinsin = np.sin(theta)*np.sin(phi)
    # z = r*cos(theta)
    cos = np.cos(theta)
    cosecsec = 1/(sincos)
    coseccosec = 1/(sinsin)
    sec = 1/(cos)

    sgnsincos = np.sign(sincos).astype(int)
    sgnsinsin = np.sign(sinsin).astype(int)
    sgncos = np.sign(cos).astype(int)

    # Initialize arrays to store the current position of each ray
    x_arr, y_arr, z_arr = np.zeros_like(theta), np.zeros_like(theta), np.zeros_like(theta)
    i,j,k = (np.ones_like(theta, dtype=int)*origin[0], np.ones_like(theta, dtype=int)*origin[1],
            np.ones_like(theta, dtype=int)*origin[2])

    i -= (sgnsincos<0).astype(int)
    j -= (sgnsinsin<0).astype(int)
    k -= (sgncos<0).astype(int)

    if return_intersections:
        intersections: Dict[int, List[Tuple[float, float, float]]] = {
            ray_index: [(x,y,z)] for ray_index,(x,y,z) in enumerate(zip(x_arr, y_arr, z_arr))}

    # Store indices of active rays so we don't have to iterate over all of them
    active_ray_mask = np.ones_like(theta, dtype=bool)
    # Deactivate rays that have already hit a voxel
    rays_to_deactivate = np.where(grid[i,j,k])
    active_ray_mask[rays_to_deactivate] = False
    # This creates a list of indices of active rays
    active_ray_indices = np.flatnonzero(active_ray_mask)

    # Iterate until all rays have hit a voxel
    while active_ray_indices.size:
        # Get the coordinate of the next grid 'wall' (wall of a potential voxel) that each ray will hit in each
        # dimension. The sign term is there because the next voxel wall depends on the direction of the ray.
        x_next = cfg.RESOLUTION*(i[active_ray_mask] - origin[0] + (sgnsincos[active_ray_mask]>0).astype(int))
        y_next = cfg.RESOLUTION*(j[active_ray_mask] - origin[1] + (sgnsinsin[active_ray_mask]>0).astype(int))
        z_next = cfg.RESOLUTION*(k[active_ray_mask] - origin[2] + (sgncos[active_ray_mask]>0).astype(int))

        # Calculate the 'time' it takes for each ray to hit the next grid wall in each dimension
        dt_x = (x_next - x_arr[active_ray_mask]) * cosecsec[active_ray_mask]
        dt_y = (y_next - y_arr[active_ray_mask]) * coseccosec[active_ray_mask]
        dt_z = (z_next - z_arr[active_ray_mask]) * sec[active_ray_mask]

        # Get the indices of the rays that hit the grid wall in each dimension first
        dt_x_smallest = np.logical_and(dt_x <= dt_y, dt_x <= dt_z)
        dt_y_smallest = np.logical_and(dt_y <= dt_x, dt_y <= dt_z)
        dt_z_smallest = np.invert(np.logical_or(dt_x_smallest, dt_y_smallest))

        # Update the ray to hit the closest grid wall
        x_smallest = active_ray_indices[dt_x_smallest]
        i[x_smallest] += sgnsincos[x_smallest]
        x_arr[x_smallest] = x_next[dt_x_smallest]
        y_arr[x_smallest] += sinsin[x_smallest] * dt_x[dt_x_smallest]
        z_arr[x_smallest] += cos[x_smallest] * dt_x[dt_x_smallest]

        y_smallest = active_ray_indices[dt_y_smallest]
        j[y_smallest] += sgnsinsin[y_smallest]
        x_arr[y_smallest] += sincos[y_smallest] * dt_y[dt_y_smallest]
        y_arr[y_smallest] = y_next[dt_y_smallest]
        z_arr[y_smallest] += cos[y_smallest] * dt_y[dt_y_smallest]

        z_smallest = active_ray_indices[dt_z_smallest]
        k[z_smallest] += sgncos[z_smallest]
        x_arr[z_smallest] += sincos[z_smallest] * dt_z[dt_z_smallest]
        y_arr[z_smallest] += sinsin[z_smallest] * dt_z[dt_z_smallest]
        z_arr[z_smallest] = z_next[dt_z_smallest]

        # Store the intersection points if requested
        if return_intersections:
            for ray_index, (x,y,z,active) in enumerate( # pylint: disable = invalid-name
                zip(x_arr, y_arr, z_arr, active_ray_mask)):
                if active:
                    intersections[ray_index].append((x,y,z))

        # If ray has hit a voxel, deactivate it
        rays_to_deactivate_hit = np.where(grid[i,j,k])
        active_ray_mask[rays_to_deactivate_hit] = False

        # If ray has hit the limit of the grid, deactivate it
        rays_to_deactivate_limit = np.where(np.logical_or.reduce((i == 0, j == 0, k == 0,
                                                                i == grid.shape[0]-1,
                                                                j == grid.shape[1]-1,
                                                                k == grid.shape[2]-1)))
        active_ray_mask[rays_to_deactivate_limit] = False
        active_ray_indices = np.flatnonzero(active_ray_mask)

    # Classify rays based on the voxel they hit, or -1 if they hit the limit of the grid
    ray_classes: NDArray[Shape["NRays"], Int] = grid_classes[i,j,k]
    ray_classes[rays_to_deactivate_limit] = -1

    if return_intersections:
        return ray_classes, intersections
    return ray_classes
