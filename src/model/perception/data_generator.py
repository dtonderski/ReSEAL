import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Tuple

import pickle
import numpy as np
from habitat_sim import Simulator
from habitat_sim.simulator import ObservationDict
from PIL import Image
from tqdm import tqdm, trange
from yacs.config import CfgNode

from src.data import filepath, scene
from src.data.filepath import GenerateEpochTrajectoryFilepaths
from src.features.mapping import SemanticMap3DBuilder
from src.model.action.pipeline import ActionPipeline, create_action_pipeline
from src.model.perception.labeler import LabelGenerator
from src.model.perception.model_wrapper import ModelWrapper
from src.model.perception.wandb_perception_logger import WandbPerceptionLogger
from src.utils.datatypes import Pose, SemanticMap3D, GridIndex3D


class DataGenerator:
    def __init__(self, perception_cfg: CfgNode, wandb_logger: Optional[WandbPerceptionLogger] = None) -> None:
        """ This class is responsible for generating the training data for the perception module. It uses the action \
            module to generate the trajectories and the semantic map builder to generate the semantic maps. The data \
            saved is the RGBD data, the semantic data if available (this is only used for benchmarking), and the \
            corresponding labels.

        Args:
            perception_cfg (CfgNode): perception module configuration. Needs the following:
                DATA_GENERATOR (CfgNode): needs the following:
                    - NUM_SCENES: int - number of scenes to sample from
                    - NUM_STEPS: int - number of steps to take in each scene
                    - SPLIT: str - one of ['train', 'val', 'minval', 'test']
                    - SEMANTIC_SCENES_ONLY: bool - whether to only use scenes with semantic maps
                    - SEED: int - specify the random selection of scenes. If None, will use current system time.
                DATA_PATHS (CfgNode): needs the following:
                    - ANNOTATED_SCENE_CONFIG_PATH_IN_SPLIT: str
                    - TRAJECTORIES_DIR: str
                    - RAW_DATA_DIR: str
                SIM (CfgNode): needs the following:
                    - DEFAULT_POSITION: list[float]
                    - SENSOR_CFG, needs the following:
                        - HEIGHT: int
                        - WIDTH: int
                        - HFOV: float
                        - DEFAULT_AGENT_ID: int
                MAP_BUILDER (CfgNode): Semantic map builder cfguration, including: 
                    - RESOLUTION: Resolution of the voxel grid, in meters per voxel 
                    - NUM_SEMANTIC_CLASSES: Number of semantic classes, should be 6 for default ReSEAL cfg 
                MAP_PROCESSOR (CfgNode): config of the map processor. Must include 
                    - NO_OBJECT_CONFIDENCE_THRESHOLD (float)
                    - HOLE_VOXEL_THRESHOLD (int)
                    - OBJECT_VOXEL_THRESHOLD (int)
                    - DILATE (bool)
                ACTION_MODULE (CfgNode): config of the action module. Must include
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
            wandb_logger (CfgNode): optional argument that can be passed to log the semantic maps to wandb.
        """
        self._perception_cfg = perception_cfg
        self._wandb_logger = wandb_logger
    
    def __call__(self, model: ModelWrapper, epoch: int) -> None:
        """ This does the following:
            1. Sample scene ids from the split given in data_generator_cfg,
            2. Step through them using the action policy defined in action_module_cfg, saving the RGBD data and \
                storing the poses and map_builder,
            3. Builds labels for each pose using the map_builder and saves them.

        Args:
            epoch_number (int): the epoch number, used for saving the data.
        """
        scene_ids = self._sample_scene_ids()
        if self._wandb_logger:
            self._wandb_logger.log_scene_ids(scene_ids)
        for i, scene_id in enumerate(scene_ids):
            print(f"Generating data for scene {scene_id}: {i+1}/{len(scene_ids)}, epoch {epoch}...")
            self._process_scene(model, scene_id, epoch)

    def _process_scene(self, model: ModelWrapper, scene_id: str, epoch: int) -> None:
        """ This does steps 2 and 3 from __call__.

        Args:
            scene_id (str): the scene id to process.
            epoch (int): the current epoch.
        """
        data_paths = GenerateEpochTrajectoryFilepaths(self._perception_cfg.DATA_PATHS,
                                                self._perception_cfg.DATA_GENERATOR.SPLIT,
                                                scene_id, epoch)

        semantic_map, grid_index_of_origin, poses = self._step_through_trajectory(model, data_paths)
        
        label_generator = LabelGenerator(semantic_map, grid_index_of_origin, self._perception_cfg.MAP_BUILDER, 
                                            self._perception_cfg.MAP_PROCESSOR, self._perception_cfg.SIM.SENSOR_CFG)
        self._save_categorical_label_map_to_wandb(label_generator, grid_index_of_origin, scene_id, epoch)
        self._generate_labels(poses, label_generator, data_paths)

    @staticmethod
    def _save_label_dict_from_pose(args:Tuple[Pose, int, LabelGenerator, GenerateEpochTrajectoryFilepaths]) -> None:
        pose, t, label_generator, data_paths = args
        label_dict = label_generator.get_label_dict(pose)
        with open((data_paths.label_dict_dir / str(t)).with_suffix('.pickle'), 'wb') as fp:
            pickle.dump(label_dict, fp)
        print(f"Saved label dict for step {t}.   ", end = '\r')

    def _generate_labels(
        self, poses: List[Pose], label_generator: LabelGenerator, data_paths: GenerateEpochTrajectoryFilepaths
        ) -> None:
        start_time = datetime.now()
        print(f"Starting generating labels at time {start_time.strftime('%H:%M:%S')}")
        with ThreadPoolExecutor() as executor:
            # Create an iterable of arguments for your save function
            args = [(pose, t, label_generator, data_paths) for t, pose in enumerate(poses)]

            executor.map(self._save_label_dict_from_pose, args)
        # Print finish time and number of seconds it took
        print(f"Finished generating labels at time {datetime.now().strftime('%H:%M:%S')}, "
              f"it took {datetime.now() - start_time} seconds.")
        
    def _step_through_trajectory(self, model: ModelWrapper, data_paths: GenerateEpochTrajectoryFilepaths
                                 ) -> Tuple[SemanticMap3D, GridIndex3D, List[Pose]]:
        """ This steps through a scene, saving RGBD data and poses, and returns the built semantic map.

        Args:
            data_paths (filepath.GenerateEpochTrajectoryFilepaths): structure containing the data_paths to use.

        Returns:
            Tuple[SemanticMap3DBuilder, List[Pose]]: the map_builder, used to extract the semantic_map and 
                grid_index_of_origin for label_building, and the poses.
        """

        map_builder = SemanticMap3DBuilder(self._perception_cfg.MAP_BUILDER, self._perception_cfg.SIM)
        poses: List = []
        sim, action_pipeline = self._initialize_sim_and_action(data_paths)
        use_semantic_sensor = scene.check_if_semantic_sensor_used(sim)
        self._make_dirs(data_paths, use_semantic_sensor)

        for count in trange(self._perception_cfg.DATA_GENERATOR.NUM_STEPS):
            action = action_pipeline(None)  # type: ignore[arg-type]
            # This means we are withing the threshold
            while not action:
                action = action_pipeline(None)  # type: ignore[arg-type]

            observations: ObservationDict = sim.step(action)
            rgb = observations["color_sensor"]  # pylint: disable=unsubscriptable-object
            depth = observations["depth_sensor"]  # pylint: disable=unsubscriptable-object
            semantics = observations["semantic_sensor"] if use_semantic_sensor else None # pylint: disable=unsubscriptable-object
            self._save_observations_and_poses(count, rgb, depth, semantics, data_paths)
            pose = (sim.get_agent(0).state.position, sim.get_agent(0).state.rotation)

            poses.append(pose)
            map = model(rgb[..., :3])
            map_builder.update_point_cloud(map, depth, pose) # type: ignore[arg-type]
        sim.close()
        print("Data generated! Updating semantic map...")
        map_builder.update_semantic_map()
        self._save_map_builder_and_poses(map_builder, poses, data_paths)
        print("Semantic map updated!")
        return map_builder.semantic_map, map_builder.get_grid_index_of_origin(), poses

    def _sample_scene_ids(self) -> List[str]:
        """ This returns a sample of all available scene_ids for the given split. 

        Returns:
            List[PurePath]: _description_
        """
        if self._perception_cfg.DATA_GENERATOR.SEMANTIC_SCENES_ONLY:
            scene_ids = [x.name for x in scene.get_annotated_scene_set(self._perception_cfg.DATA_PATHS) 
                         if x.parent.name == self._perception_cfg.DATA_GENERATOR.SPLIT]            
        else:
            raw_split_path = filepath.get_raw_data_split_dir(self._perception_cfg.DATA_PATHS, 
                                                            self._perception_cfg.DATA_GENERATOR.SPLIT)
            scene_ids = list(sorted([f.name for f in os.scandir(raw_split_path) if f.is_dir()]))
        random.seed(self._perception_cfg.DATA_GENERATOR.SEED)
        selected_scene_ids = random.sample(scene_ids, self._perception_cfg.DATA_GENERATOR.NUM_SCENES)
        return selected_scene_ids

    def _initialize_sim_and_action(
        self, data_paths: GenerateEpochTrajectoryFilepaths) -> Tuple[Simulator, ActionPipeline]:
        """ Initializes the sim, agent, and the action pipeline given a data_paths object.

        Args:
            data_paths (filepath.GenerateEpochTrajectoryFilepaths): the data_paths object to use.

        Returns:
            Tuple[Simulator, ActionPipeline]: the initialized sim and action_pipeline.
        """
        sim = scene.initialize_sim(
            data_paths.scene_split, data_paths.scene_id, data_paths_cfg=self._perception_cfg.DATA_PATHS, 
            sim_cfg=self._perception_cfg.SIM)

        agent = sim.get_agent(self._perception_cfg.SIM.DEFAULT_AGENT_ID)
        action_pipeline = create_action_pipeline(self._perception_cfg.ACTION_MODULE, 
                                                 str(data_paths.navmesh_filepath), agent)
        
        return sim, action_pipeline
    
    def _save_observations_and_poses(
        self, count: int, rgb, depth, semantics, data_paths: GenerateEpochTrajectoryFilepaths) -> None:
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
        if semantics is not None:
            np.save(data_paths.semantic_dir / f"{count}", semantics)                       
    
    def _save_map_builder_and_poses(
        self, map_builder: SemanticMap3DBuilder, poses: List[Pose], data_paths: GenerateEpochTrajectoryFilepaths
        ) -> None:
        """ Saves the semantic map and grid_index_of_origin to the data_paths for visualization purposes.
        """
        print(f"Saving semantic map to {data_paths.semantic_map_filepath}.")
        with open(data_paths.semantic_map_filepath, 'wb') as fp:
            pickle.dump(map_builder.semantic_map, fp)
        print(f"Saving grid index of origin to {data_paths.grid_index_of_origin_filepath}")
        with open(data_paths.grid_index_of_origin_filepath, 'wb') as fp:
            pickle.dump(map_builder.get_grid_index_of_origin(), fp)
        print(f"Saving poses to {data_paths.poses_filepath}")
        with open(data_paths.poses_filepath, 'wb') as fp:
            pickle.dump(poses, fp)
    
    def _save_categorical_label_map_to_wandb(self, label_generator: LabelGenerator, grid_index_of_origin,
                                             scene_id: str, epoch: int) -> None:
        if self._wandb_logger:
            print(f"Saving semantic map point cloud representation to wandb.")
            self._wandb_logger.log_categorical_label_map(label_generator.categorical_label_map,
                                                         grid_index_of_origin,
                                                         self._perception_cfg.MAP_BUILDER.RESOLUTION,
                                                         scene_id,
                                                         epoch)

    
    def _make_dirs(self, data_paths: GenerateEpochTrajectoryFilepaths, use_semantic_sensor: bool) -> None:
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

def main():
    from src.model.perception.perception_pipeline_config import get_perception_cfg
    perception_cfg = get_perception_cfg()
    wandb_logger = WandbPerceptionLogger(perception_cfg)

    data_generator = DataGenerator(perception_cfg, wandb_logger)
        
    model = ModelWrapper(perception_cfg.MODEL)
    model.cuda()
    model.eval()
    for epoch in range(2):
        data_generator(model, epoch)
        wandb_logger.on_epoch_end(epoch)

if __name__ == '__main__':
    main()
