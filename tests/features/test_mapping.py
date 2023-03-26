import numpy as np
import pytest
import quaternion
from yacs.config import CfgNode

from src.features.mapping import Geocentric3DMapBuilder
from src.utils.datatypes import Coordinate3D, CoordinatesMapping3Dto3D, Pose


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


@pytest.fixture
def map_builder_cfg() -> CfgNode:
    cfg = CfgNode()
    cfg.RESOLUTION = 1
    cfg.EGOCENTRIC_MAP_SHAPE = (3, 3, 3)
    cfg.NUM_SEMANTIC_CLASSES = 1
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
