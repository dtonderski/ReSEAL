import numpy as np
import quaternion
from nptyping import Float, Int, NDArray, Shape


def get_ray_directions_camera_coords(sensor_resolution: NDArray[Shape["2"], Int],
                                     sensor_hfov_deg: int) -> NDArray[Shape["3, Width, Height"], Float]:
    """ Get the directions of rays in camera coordinates for a given sensor resolution and horizontal field of view. \
        www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html

    Args:
        sensor_resolution (NDArray[Shape["2"], Int]): [width, height] of the sensor in pixels.
        sensor_hfov_deg (int): horizontal field of view of the sensor in degrees.

    Returns:
        NDArray[Shape["Width, Height, 3"], Float]: element at index [i,j] is the direction of the ray corresponding \
            to the pixel at index [i,j] in the image.
    """

    width, height = sensor_resolution

    pixel_ndc_x = (np.arange(width)+0.5) / width
    pixel_ndc_y = (np.arange(height)+0.5) / height

    pixel_screen_x = 2 * pixel_ndc_x - 1
    pixel_screen_y = 1 - 2*pixel_ndc_y

    pixel_camera_x = pixel_screen_x * np.tan(sensor_hfov_deg/2*np.pi/180)
    # Here, we need to divide by the image aspect ratio because pixels have to be square in camera coordinates
    pixel_camera_y = pixel_screen_y/(width/height)* np.tan(sensor_hfov_deg/2*np.pi/180)

    ray_directions_x, ray_directions_y = np.meshgrid(pixel_camera_x, pixel_camera_y, indexing = 'ij')
    ray_directions_z = -np.ones_like(ray_directions_x)

    ray_directions = np.stack([ray_directions_x, ray_directions_y, ray_directions_z], axis=0)
    ray_directions_normalized = ray_directions / np.linalg.norm(ray_directions, axis=0, keepdims=True)
    return ray_directions_normalized

def get_ray_directions_world_coords(camera_rotation: np.quaternion, #type: ignore[name-defined]
                                    sensor_resolution: NDArray[Shape["2"], Int],
                                    sensor_hfov_deg: int) -> NDArray[Shape["3, Width, Height"], Float]:
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
        sensor_resolution (NDArray[Shape["2"], Int]): _description_
        sensor_hfov_deg (int): _description_
        camera_rotation (np.quaternion): quaternion where the first three elements represent the z direction of the \
            camera, which is the negative of the direction the camera is facing.

    Returns:
        ray_directions_world (NDArray[Shape["3, Width, Height"], Float]): array where element at index [:, i, j] is \
            the direction of the ray corresponding to the pixel at index [i,j] in the image.
    """
    ray_directions_camera = get_ray_directions_camera_coords(sensor_resolution, sensor_hfov_deg)
    q_float = quaternion.as_float_array(camera_rotation)

    rotation_matrix = np.eye(3)
    rotation_matrix[0,0] = -q_float[2]
    rotation_matrix[0,2] = -q_float[0]
    rotation_matrix[2,0] = q_float[0]
    rotation_matrix[2,2] = -q_float[2]

    ray_directions_world = np.einsum('ij, jkl ->ikl', rotation_matrix, ray_directions_camera)

    return ray_directions_world
