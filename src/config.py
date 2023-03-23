from yacs.config import CfgNode


def default_map_builder_cfg():
    map_builder_cfg = CfgNode()
    map_builder_cfg.RESOLUTION = 1.0  # cm per pixel
    map_builder_cfg.EGOCENTRIC_MAP_SHAPE = (500, 500, 500)  # (x, y, z) in pixel
    map_builder_cfg.NUM_SEMANTIC_CLASSES = 10
    return map_builder_cfg
