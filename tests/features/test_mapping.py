import numpy as np
import pytest
import quaternion
from yacs.config import CfgNode

from src.features.mapping import Geocentric3DMapBuilder
from src.utils.datatypes import Coordinate3D, CoordinatesMapping3Dto3D, Pose, SemanticMap3D


class TestGeocentric3DMapBuilder:
    def test_calc_ego_to_geocentric_coordinate_mapping(
        self, geocentric_3d_map_builder, pose, expected_ego_to_geo_coord_mapping
    ):
        ego_to_geo_coord_mapping = geocentric_3d_map_builder._calc_ego_to_geocentric_coordinate_mapping(pose)
        for ego_coord, expected_geo_coord in expected_ego_to_geo_coord_mapping.items():
            geo_coord = ego_to_geo_coord_mapping[ego_coord]
            np.testing.assert_array_equal(geo_coord, expected_geo_coord)

    def test_reshape_geocentric_map(
        self, geocentric_3d_map_builder, pose, expected_ego_to_geo_coord_mapping_after_reshaping
    ):
        ego_to_geo_coord_mapping = geocentric_3d_map_builder._calc_ego_to_geocentric_coordinate_mapping(pose)
        ego_to_geo_coord_mapping = geocentric_3d_map_builder._reshape_geocentric_map(ego_to_geo_coord_mapping)
        for ego_coord, expected_geo_coord in expected_ego_to_geo_coord_mapping_after_reshaping.items():
            geo_coord = ego_to_geo_coord_mapping[ego_coord]
            np.testing.assert_array_equal(geo_coord, expected_geo_coord)
        assert geocentric_3d_map_builder._world_origin_in_geo == (3, 1, 0)
        assert geocentric_3d_map_builder._geocentric_map.shape == (4, 5, 3, 1)

    def test_update_geocentric_map(self, geocentric_3d_map_builder, pose, egocentric_map, expected_geocentric_map_z0):
        ego_to_geo_coord_mapping = geocentric_3d_map_builder._calc_ego_to_geocentric_coordinate_mapping(pose)
        ego_to_geo_coord_mapping = geocentric_3d_map_builder._reshape_geocentric_map(ego_to_geo_coord_mapping)
        geocentric_3d_map_builder._update_geocentric_map(egocentric_map, ego_to_geo_coord_mapping)
        np.testing.assert_array_equal(geocentric_3d_map_builder._geocentric_map[:, :, 0, 0], expected_geocentric_map_z0)


@pytest.fixture
def map_builder_cfg() -> CfgNode:
    cfg = CfgNode()
    cfg.RESOLUTION = 1
    cfg.EGOCENTRIC_MAP_SHAPE = (3, 3, 3)
    cfg.NUM_SEMANTIC_CLASSES = 1
    cfg.EGOCENTRIC_MAP_ORIGIN_OFFSET = (0, 0, 0)
    return cfg


@pytest.fixture
def world_origin_in_geo_frame() -> Coordinate3D:
    return 2, 1, 0


@pytest.fixture
def geocentric_3d_map_builder(
    map_builder_cfg: CfgNode, world_origin_in_geo_frame: Coordinate3D
) -> Geocentric3DMapBuilder:
    map_builder = Geocentric3DMapBuilder(None, map_builder_cfg)
    map_builder._world_origin_in_geo = world_origin_in_geo_frame
    map_builder._geocentric_map = np.zeros((3, 2, 1, 1))
    return map_builder


@pytest.fixture
def pose() -> Pose:
    translation = np.array([-1, 1, 0])
    rotation = quaternion.from_rotation_vector(np.pi / 2 * np.array([0, 0, 1]))
    return translation, rotation


@pytest.fixture
def expected_ego_to_geo_coord_mapping() -> CoordinatesMapping3Dto3D:
    return {
        (0, 0, 0): (1, 2, 0),
        (0, 1, 0): (0, 2, 0),
        (0, 2, 0): (-1, 2, 0),
        (1, 0, 0): (1, 3, 0),
        (1, 1, 0): (0, 3, 0),
        (1, 2, 0): (-1, 3, 0),
        (2, 0, 0): (1, 4, 0),
        (2, 1, 0): (0, 4, 0),
        (2, 2, 0): (-1, 4, 0),
    }


@pytest.fixture
def expected_ego_to_geo_coord_mapping_after_reshaping() -> CoordinatesMapping3Dto3D:
    return {
        (0, 0, 0): (2, 2, 0),
        (0, 1, 0): (1, 2, 0),
        (0, 2, 0): (0, 2, 0),
        (1, 0, 0): (2, 3, 0),
        (1, 1, 0): (1, 3, 0),
        (1, 2, 0): (0, 3, 0),
        (2, 0, 0): (2, 4, 0),
        (2, 1, 0): (1, 4, 0),
        (2, 2, 0): (0, 4, 0),
    }


@pytest.fixture
def egocentric_map() -> SemanticMap3D:
    egocentric_map = np.zeros((3, 3, 3, 1))
    egocentric_map[:, :, 0, 0] = np.arange(9).reshape((3, 3))
    return egocentric_map


@pytest.fixture
def expected_geocentric_map_z0() -> SemanticMap3D:
    geocentric_map = np.array(
        [
            [2, 5, 8],
            [1, 4, 7],
            [0, 3, 6],
        ]
    )
    geocentric_map = np.pad(geocentric_map, ((0, 1), (2, 0)), mode="constant")
    return geocentric_map
