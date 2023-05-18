from src.model.action.pipeline import ActionPipeline
from yacs.config import CfgNode
from typing import List, Tuple
from pathlib import PurePath, Path
from src.features.mapping import SemanticMap3DBuilder
from src.utils.datatypes import SemanticMap3D, Pose
from src.model.perception.labeler import LabelGenerator
from src.data import scene, filepath
from src.model.action.pipeline import create_action_pipeline
import _pickle as cPickle

class DataGenerator:
    def __init__(self, epoch_number: int, data_generator_cfg: CfgNode, data_paths_cfg: CfgNode, sim_cfg: CfgNode, 
                 map_builder_cfg: CfgNode, map_processor_cfg: CfgNode, action_module_cfg: CfgNode) -> None:
        """ This takes 

        Args:
            action_pipeline (ActionPipeline): _description_
            data_generator_cfg (CfgNode): needs the following:
                - NUM_SCENES: int - number of scenes to sample from
                - NUM_STEPS: int - number of steps to take in each scene
                - SPLIT: str - one of ['train', 'val', 'minval', 'test']
            data_paths_cfg (CfgNode): needs the following:
                - TRAJECTORIES_DIR: str
                - RAW_DATA_DIR: str
            sim_cfg (CfgNode): needs the following:
                - DEFAULT_POSITION: list[float]
                - SENSOR_CFG, needs the following:
                    - HEIGHT: int
                    - WIDTH: int
                    - HFOV: float
                    - DEFAULT_AGENT_ID: int
            map_builder_cfg (CfgNode): Semantic map builder cfguration, including: 
                - RESOLUTION: Resolution of the voxel grid, in meters per voxel 
                - NUM_SEMANTIC_CLASSES: Number of semantic classes, should be 6 for default ReSEAL cfg 

            map_processor_cfg (CfgNode): config of the map processor. Must include 
                - NO_OBJECT_CONFIDENCE_THRESHOLD (float)
                - HOLE_VOXEL_THRESHOLD (int)
                - OBJECT_VOXEL_THRESHOLD (int)
                - DILATE (bool)
            
            action_module_cfg (CfgNode): config of the action module. Must include
                - PREPROCESSOR (str) - should be "DummyPreprocessor" for now.
                - GLOBAL_POLICY (CfgNode) - config of the global policy. Must include
                    - NAME (str) - should be "RandomGlobalPolicy" for now.
                    - LR_SCHEDULE (CfgNode) - config of the learning rate scheduler. Must include
                        - NAME (str) - should be "ConstantLR" for now.
                        - INIT_LR (float) - initial learning rate. This is used for the action policy, so can be \
                            arbitrary.
                    - OBSERVATION_SPACE_SHAPE (list[int]) - unused here, can be arbitrary, e.g. to [100, 100, 100, 11].
                - LOCAL_POLICY (CfgNode) - config of the local policy. Must include
                    - DISTANCE_THRESHOLD (float) - this means that the position is accepted
                - ACTION_PIPELINE (CfgNode) - config of the action pipeline. Must include
                    - IS_DETERMINISTIC (bool) - whether the action pipeline is deterministic or not.
                    - GLOBAL_POLICY_POLLING_FREQUENCY (int) - how often the global policy is polled.


        """
        self._data_generator_cfg = data_generator_cfg
        self._data_paths_cfg = data_paths_cfg
        self._sim_cfg = sim_cfg
        self._map_builder_cfg = map_builder_cfg
        self._map_processor_cfg = map_processor_cfg
        self._action_module_cfg = action_module_cfg
        self._epoch_number = epoch_number
        pass
    
    def _make_dirs(self, data_paths: filepath.GenerateEpochTrajectoryFilepaths, use_semantic_sensor: bool) -> None:
        data_paths.trajectory_output_dir.mkdir(parents=True, exist_ok=True)
        data_paths.rgb_dir.mkdir(parents=True, exist_ok=True)
        data_paths.depth_dir.mkdir(parents=True, exist_ok=True)
        if use_semantic_sensor:
            data_paths.semantic_dir.mkdir(parents=True, exist_ok=True)
        data_paths.label_dict_dir.mkdir(parents=True, exist_ok=True)        
    
    def __call__(self) -> None:
        scene_ids = self._sample_scene_ids()
        for scene_id in scene_ids:
            data_paths = filepath.GenerateEpochTrajectoryFilepaths(self._data_paths_cfg, self._data_generator_cfg.SPLIT,
                                                                   scene_id, self._epoch_number)            
            sim = scene.initialize_sim(
                data_paths.scene_split, data_paths.scene_id, data_paths_cfg=self._data_paths_cfg, sim_cfg=self._sim_cfg)

            agent = sim.get_agent(self._sim_cfg.DEFAULT_AGENT_ID)
            use_semantic_sensor = scene.check_if_semantic_sensor_used(sim)
            self._make_dirs(data_paths, use_semantic_sensor)
            action_pipeline = create_action_pipeline(self._action_module_cfg, str(data_paths.navmesh_filepath), agent)

            map_builder, poses = self._step_through_trajectory(action_pipeline, sim)
            semantic_map = map_builder.semantic_map
            grid_index_of_origin = map_builder.get_grid_index_of_origin()
            
            label_generator = LabelGenerator(semantic_map, grid_index_of_origin, self._map_builder_cfg, 
                                             self._map_processor_cfg, self._sim_cfg.SENSOR_CFG)
            for t, pose in enumerate(poses):
                label_dict = label_generator.get_label_dict(pose)
                with open((data_paths.label_dict_dir / str(t)).with_suffix('.pickle'), 'wb') as fp:
                    cPickle.dump(label_dict, fp)

    def _sample_scene_ids(self) -> List[str]:
        """ This returns a sample of all available scene_ids for the given split. 

        Returns:
            List[PurePath]: _description_
        """
        pass
    
    def _step_through_trajectory(self, action_pipeline: ActionPipeline, scene_id: str) -> Tuple[SemanticMap3DBuilder, List[Pose]]:
        """ This steps through a scene, saving RGBD data and poses, and returns the built semantic map.

        Returns:
            SemanticMap3D: _description_
        """
        map_builder = SemanticMap3DBuilder()
        pass
    
    