import numpy as np
import quaternion
from nptyping import Float, Int, NDArray, Shape
from yacs.config import CfgNode

from ..features.raytracing import raytrace_3d
from ..utils.datatypes import Coordinate3D, GridIndex3D, SemanticMap2D, SemanticMap3D


def get_ray_directions_sensor_coords(sensor_cfg: CfgNode) -> NDArray[Shape["3, Height, Width"], Float]:
    """ Get the directions of rays in camera coordinates for a given sensor config. \
        www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html

    Args:
        sensor_config (CfgNode): sensor config, must contain WIDTH, HEIGHT, HFOV

    Returns:
        NDArray[Shape["3, Height, Width"], Float]: element at index [:,i,j] is the direction of the ray corresponding \
            to the pixel at index [i,j] in the image.
    """

    pixel_ndc_x = (np.arange(sensor_cfg.WIDTH)+0.5) / sensor_cfg.WIDTH
    pixel_ndc_y = (np.arange(sensor_cfg.HEIGHT)+0.5) / sensor_cfg.HEIGHT

    pixel_screen_x = 2 * pixel_ndc_x - 1
    pixel_screen_y = 1 - 2*pixel_ndc_y

    pixel_camera_x = pixel_screen_x * np.tan(sensor_cfg.HFOV/2*np.pi/180)
    # Here, we need to divide by the image aspect ratio because pixels have to be square in camera coordinates
    pixel_camera_y = pixel_screen_y/(sensor_cfg.WIDTH/sensor_cfg.HEIGHT)* np.tan(sensor_cfg.HFOV/2*np.pi/180)

    ray_directions_x, ray_directions_y = np.meshgrid(pixel_camera_x, pixel_camera_y)
    ray_directions_z = -np.ones_like(ray_directions_x)

    ray_directions = np.stack([ray_directions_x, ray_directions_y, ray_directions_z], axis=0)
    ray_directions_normalized = ray_directions / np.linalg.norm(ray_directions, axis=0, keepdims=True)
    return ray_directions_normalized

def get_ray_directions_world_coords(sensor_rotation: np.quaternion, #type: ignore[name-defined]
                                    sensor_cfg: CfgNode) -> NDArray[Shape["3, Width, Height"], Float]:
    """ Get the directions of rays in world coordinates for a given sensor resolution, horizontal field of view and \
        camera rotation. Note that since we are only getting the direction of the rays, the camera position is not \
        needed.

        For the formula for the calculation of the rotation matrix from camera coordinates to world coordinates, see \
        https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle.

        Since the y axis of the camera is always identical to the y axis of the world, we must always rotate around it.
        Therefore, we have u = (0,1,0), so we get:
            ┌                    ┐
            | cos(θ)  0   sin(θ) |
        R = |   0     1     0    | ,
            |-sin(θ)  0   cos(θ) |
            └                    ┘
        where θ is the angle of rotation. Next, we have that the z axis of the camera in world coordiantes is given by \
        (sin(θ), 0, cos(θ)), which can easily be verified by R * (0,0,1). The first three elements of the quaternion \
        are the negative of that, so we have q = (-sin(θ), 0, -cos(θ), 0). Therefore:
            ┌                    ┐   ┌             ┐
            | cos(θ)  0   sin(θ) |   | -q2  0  -q0 |
        R = |   0     1     0    | = |  0   1   0  |.
            |-sin(θ)  0   cos(θ) |   | q0   0  -q2 |
            └                    ┘   └             ┘

    Args:
        sensor_rotation (np.quaternion): quaternion where the first three elements represent the z direction of the \
            camera, which is the negative of the direction the camera is facing.
        sensor_cfg (CfgNode): sensor config, must contain WIDTH, HEIGHT, HFOV

    Returns:
        ray_directions_world (NDArray[Shape["3, Width, Height"], Float]): array where element at index [:, i, j] is \
            the direction of the ray corresponding to the pixel at index [i,j] in the image.
    """
    ray_directions_sensor = get_ray_directions_sensor_coords(sensor_cfg)
    q_float = quaternion.as_float_array(sensor_rotation)

    rotation_matrix = np.eye(3)
    rotation_matrix[0,0] = -q_float[2]
    rotation_matrix[0,2] = -q_float[0]
    rotation_matrix[2,0] = q_float[0]
    rotation_matrix[2,2] = -q_float[2]

    ray_directions_world = np.einsum('ij, jkl ->ikl', rotation_matrix, ray_directions_sensor)

    return ray_directions_world


def propagate_labels(sensor_rotation: np.quaternion,                                        #type: ignore[name-defined]
                     sensor_position: Coordinate3D,
                     semantic_map_3d: SemanticMap3D,
                     grid_index_of_origin: GridIndex3D,
                     sim_cfg: CfgNode) -> SemanticMap2D:
    """_summary_

    Args:
        sensor_rotation (np.quaternion): quaternion where the first three elements represent the z direction of the \
            camera, which is the negative of the direction the camera is facing.
        sensor_position (Coordinate3D): 3d coordinates of the sensor position, which will be the origin of the rays
        semantic_map_3d (SemanticMap3D): semantic map of the scene.
        grid_index_of_origin (GridIndex3D): grid index of the origin of the scene.
        sim_cfg (CfgNode): simulation config, must contain RESOLUTION and SENSOR_CFG, and sim_cfg.SENSOR_CFG \
            must contain WIDTH, HEIGHT, HFOV

    Returns:
        SemanticMap2D: semantic 2d map of the scene from the viewpoint of the sensor.
    """
    ray_directions = get_ray_directions_world_coords(sensor_rotation, sim_cfg.SENSOR_CFG)

    ray_directions_flat = ray_directions.reshape(3, -1)

    ray_labels, _ = raytrace_3d(ray_directions_flat, semantic_map_3d, sensor_position, grid_index_of_origin,
                              sim_cfg, False)

    return ray_labels_to_semantic_map_2d(ray_labels, sim_cfg.SENSOR_CFG)

def ray_labels_to_semantic_map_2d(ray_labels: NDArray[Shape["NRays, NChannels"], Int],
                                   sensor_cfg: CfgNode) -> SemanticMap2D:
    """ Transforms ray_labels returned by raytrace_3d into SemanticMap2D by reshaping nrays to width, height and \
        removing the first channel, which is the occupancy channel.

    Args:
        ray_labels (NDArray[Shape[NRays, NChannels], Int]): If ray i hits a voxel which in the semantic 3d map is \
            represented by a vector v, then ray_labels[i] = v. If it hits no voxel, then ray_labels[i,j] = 0 for all j.
        sensor_cfg (CfgNode): sensor config, must contain WIDTH, HEIGHT

    Returns:
        SemanticMap2D: semantic 2d map of the scene from the viewpoint of the sensor.
    """
    return ray_labels.reshape(sensor_cfg.HEIGHT, sensor_cfg.WIDTH, -1)[:,:,1:]
