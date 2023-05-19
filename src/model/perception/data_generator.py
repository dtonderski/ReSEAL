import os
import random
from pathlib import Path, PurePath
from typing import List, Tuple

import _pickle as cPickle
import numpy as np
import quaternion
from habitat_sim import Simulator
from habitat_sim.simulator import ObservationDict
from PIL import Image
from tqdm import trange, tqdm
from yacs.config import CfgNode

from src.config import (
    default_action_module_cfg,
    default_data_paths_cfg,
    default_map_builder_cfg,
    default_map_processor_cfg,
    default_perception_data_generator_cfg,
    default_sim_cfg,
    default_model_cfg
)
from src.data import filepath, scene
from src.features.mapping import SemanticMap3DBuilder
from src.model.action.pipeline import ActionPipeline, create_action_pipeline
from src.model.perception.labeler import LabelGenerator
from src.utils.datatypes import Pose
from src.model.perception.model_wrapper import ModelWrapper


class DataGenerator:
    def __init__(self, data_generator_cfg: CfgNode, data_paths_cfg: CfgNode, sim_cfg: CfgNode, 
                 map_builder_cfg: CfgNode, map_processor_cfg: CfgNode, action_module_cfg: CfgNode) -> None:
        """ This class is responsible for generating the training data for the perception module. It uses the action \
            module to generate the trajectories and the semantic map builder to generate the semantic maps. The data \
            saved is the RGBD data, the semantic data if available (this is only used for benchmarking), and the \
            corresponding labels.

        Args:
            data_generator_cfg (CfgNode): needs the following:
                - NUM_SCENES: int - number of scenes to sample from
                - NUM_STEPS: int - number of steps to take in each scene
                - SPLIT: str - one of ['train', 'val', 'minval', 'test']
                - SEED: int - specify the random selection of scenes. If None, will use current system time.
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
    
    def __call__(self, model: ModelWrapper, epoch_number: int) -> None:
        """ This does the following:
            1. Sample scene ids from the split given in data_generator_cfg,
            2. Step through them using the action policy defined in action_module_cfg, saving the RGBD data and \
                storing the poses and map_builder,
            3. Builds labels for each pose using the map_builder and saves them.

        Args:
            epoch_number (int): the epoch number, used for saving the data.
        """
        scene_ids = self._sample_scene_ids()
        for i, scene_id in enumerate(scene_ids):
            print(f"Generating data for scene {scene_id}: {i+1}/{len(scene_ids)}.")
            data_paths = filepath.GenerateEpochTrajectoryFilepaths(self._data_paths_cfg, 
                                                                   self._data_generator_cfg.SPLIT,
                                                                   scene_id, epoch_number)            
            
            map_builder, poses = self._step_through_trajectory(data_paths)
            semantic_map = map_builder.semantic_map
            grid_index_of_origin = map_builder.get_grid_index_of_origin()
            
            label_generator = LabelGenerator(semantic_map, grid_index_of_origin, self._map_builder_cfg, 
                                             self._map_processor_cfg, self._sim_cfg.SENSOR_CFG)
            for t, pose in tqdm(enumerate(poses)):
                label_dict = label_generator.get_label_dict(pose)
                with open((data_paths.label_dict_dir / str(t)).with_suffix('.pickle'), 'wb') as fp:
                    cPickle.dump(label_dict, fp)


    def _sample_scene_ids(self) -> List[str]:
        """ This returns a sample of all available scene_ids for the given split. 

        Returns:
            List[PurePath]: _description_
        """
        raw_split_path = filepath.get_raw_data_split_dir(self._data_paths_cfg, self._data_generator_cfg.SPLIT)
        scene_ids = list(sorted([f.name for f in os.scandir(raw_split_path) if f.is_dir()]))
        random.seed(self._data_generator_cfg.SEED)
        selected_scene_ids = random.sample(scene_ids, self._data_generator_cfg.NUM_SCENES)
        return selected_scene_ids
    
    def _step_through_trajectory(self,
                                 data_paths: filepath.GenerateEpochTrajectoryFilepaths,
                                 ) -> Tuple[SemanticMap3DBuilder, List[Pose]]:
        """ This steps through a scene, saving RGBD data and poses, and returns the built semantic map.

        Args:
            data_paths (filepath.GenerateEpochTrajectoryFilepaths): structure containing the data_paths to use.

        Returns:
            Tuple[SemanticMap3DBuilder, List[Pose]]: the map_builder, used to extract the semantic_map and 
                grid_index_of_origin for label_building, and the poses.
        """

        map_builder = SemanticMap3DBuilder(self._map_builder_cfg, self._sim_cfg)
        poses = []
        
        sim, action_pipeline = self._initialize_sim_and_action(data_paths)
        use_semantic_sensor = scene.check_if_semantic_sensor_used(sim)
        self._make_dirs(data_paths, use_semantic_sensor)

        for count in trange(self._data_generator_cfg.NUM_STEPS):
            print(f"Step {count}/{self._data_generator_cfg.NUM_STEPS}.")
            action = action_pipeline(None)  # type: ignore[arg-type]
            # This means we are withing the threshold
            while not action:
                action = action_pipeline(None)  # type: ignore[arg-type]

            observations: ObservationDict = sim.step(action)
            rgb = observations["color_sensor"]  # pylint: disable=unsubscriptable-object
            depth = observations["depth_sensor"]  # pylint: disable=unsubscriptable-object
            semantics = observations["semantic_sensor"] if use_semantic_sensor else None
            self._save_observations(count, rgb, depth, semantics, data_paths)
            
            pose = (sim.get_agent(0).state.position, sim.get_agent(0).state.rotation)
            poses.append(pose)
            
            map = model(rgb[..., :3])
            map_builder.update_point_cloud(map, depth, pose)
        
        print("Updating semantic map...")
        map_builder.update_semantic_map()
        print("Semantic map updated!")
        return map_builder, poses

    def _initialize_sim_and_action(
        self, data_paths: filepath.GenerateEpochTrajectoryFilepaths) -> Tuple[Simulator, ActionPipeline]:
        """ Initializes the sim, agent, and the action pipeline given a data_paths object.

        Args:
            data_paths (filepath.GenerateEpochTrajectoryFilepaths): the data_paths object to use.

        Returns:
            Tuple[Simulator, ActionPipeline]: the initialized sim and action_pipeline.
        """
        sim = scene.initialize_sim(
            data_paths.scene_split, data_paths.scene_id, data_paths_cfg=self._data_paths_cfg, sim_cfg=self._sim_cfg)

        agent = sim.get_agent(self._sim_cfg.DEFAULT_AGENT_ID)
        action_pipeline = create_action_pipeline(self._action_module_cfg, str(data_paths.navmesh_filepath), agent)
        
        return sim, action_pipeline
    
    def _save_observations(
        self, count: int, rgb, depth, semantics, data_paths: filepath.GenerateEpochTrajectoryFilepaths) -> None:
        """ Saves the observations according to the data_paths.

        Args:
            count (int): current iteration in the simulator
            rgb (_type_): the rgb image from the sensor
            depth (_type_): the depth image from the sensor
            semantics (_type_): the semantic image from the sensor, if available
            data_paths (filepath.GenerateEpochTrajectoryFilepaths): the data_paths object to use.
        """
        Image.fromarray(rgb[:, :, :3]).save(data_paths.rgb_dir / f"{count}.png")
        np.save(data_paths.depth_dir / f"{count}", depth)
        if semantics:
            np.save(data_paths.semantic_dir / f"{count}", semantics)            
    
    def _make_dirs(self, data_paths: filepath.GenerateEpochTrajectoryFilepaths, use_semantic_sensor: bool) -> None:
        """ Initializes the directories for saving the data.

        Args:
            data_paths (filepath.GenerateEpochTrajectoryFilepaths): the data_paths object to use.
            use_semantic_sensor (bool): whether or not the semantic sensor is used, determines if the semantic dir \
                is created.
        """
        data_paths.trajectory_output_dir.mkdir(parents=True, exist_ok=True)
        data_paths.rgb_dir.mkdir(parents=True, exist_ok=True)
        data_paths.depth_dir.mkdir(parents=True, exist_ok=True)
        if use_semantic_sensor:
            data_paths.semantic_dir.mkdir(parents=True, exist_ok=True)
        data_paths.label_dict_dir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':    
    data_generator = DataGenerator(default_perception_data_generator_cfg(), default_data_paths_cfg(), 
                                   default_sim_cfg(), default_map_builder_cfg(), default_map_processor_cfg(), 
                                   default_action_module_cfg())
    model = ModelWrapper(default_model_cfg())
    model.cuda()
    model.eval()
    data_generator(model, 0)