from typing import Union

import numpy as np
import quaternion
from nptyping import Float, Int, NDArray, Shape

from ..utils.datatypes import GridIndex2D, GridIndex3D, HomogenousTransform, Pose, TranslationVector


def coordinates_to_grid_indices(
    coordinates: NDArray[Shape["NPixels, NDims"], Float],
    grid_indices_of_origin: Union[GridIndex2D, GridIndex3D],
    voxel_size: float,
) -> NDArray[Shape["NPixels, NDims"], Int]:
    return (np.floor(coordinates / voxel_size) + grid_indices_of_origin).astype(int)


def grid_indices_to_world_coordinates(
    grid_indices: NDArray[Shape["NPixels, NDims"], Int],
    grid_indices_of_origin: Union[GridIndex2D, GridIndex3D],
    voxel_size: float,
    voxel_center_coordinates: bool = False,
) -> NDArray[Shape["NPixels, NDims"], Float]:
    # Returns the coordinates of the voxel in world coordinates
    voxel_origin_world_coordinates = (grid_indices - grid_indices_of_origin) * voxel_size
    if voxel_center_coordinates:
        voxel_origin_world_coordinates += voxel_size / 2
    return voxel_origin_world_coordinates


class HomogenousTransformFactory:
    @staticmethod
    def from_pose(pose: Pose, translate_first: bool = True) -> HomogenousTransform:
        """Returns a 4x4 homogenous matrix from a pose consisting of a translation vector and a rotation quaternion"""
        translation_vector, rotation_quaternion = pose
        rotation_transformation = HomogenousTransformFactory.from_quaternion(rotation_quaternion)
        translation_transformation = HomogenousTransformFactory.from_translation(translation_vector)
        if translate_first:
            return np.matmul(translation_transformation, rotation_transformation)
        return np.matmul(rotation_transformation, translation_transformation)

    @staticmethod
    def from_translation(translation: TranslationVector) -> HomogenousTransform:
        """Returns a 4x4 homogenous matrix from a translation vector"""
        homogenous_matrix = np.eye(4)
        homogenous_matrix[:3, 3] = translation
        return homogenous_matrix

    @staticmethod
    def from_quaternion(rotation_quaternion: quaternion.quaternion) -> HomogenousTransform:
        """Returns a 4x4 homogenous matrix from a rotation quaternion"""
        rotation_matrix = quaternion.as_rotation_matrix(rotation_quaternion)
        homogenous_matrix = np.eye(4)
        homogenous_matrix[:3, :3] = rotation_matrix
        return homogenous_matrix
