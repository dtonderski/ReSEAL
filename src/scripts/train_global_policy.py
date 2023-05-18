from fire import Fire

from src import config as cfg
from src.model.action.env import HabitatEnv
from src.model.action.local_policy import GreedyLocalPolicy
from src.model.action.feature_extractor import SemanticMapFeatureExtractor
from src.features.mapping import SemanticMap3DBuilder
from src.model.perception.model_wrapper import ModelWrapper
from src.data import scene, filepath

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


def main(scene_name: str = "minival/00800-TEEsavR23oF"):
    env_cfg = cfg.default_env_cfg()
    sim_cfg = cfg.default_sim_cfg()
    action_module_cfg = cfg.default_action_module_cfg()
    map_builder_cfg = cfg.default_map_builder_cfg()
    perception_model_cfg =cfg.default_perception_model_cfg()

    data_paths_cfg = cfg.default_data_paths_cfg()
    data_paths = filepath.GenerateTrajectoryFilepaths(data_paths_cfg, scene_name)

    sim = scene.initialize_sim(
        data_paths.scene_split, data_paths.scene_id, data_paths_cfg=data_paths_cfg, sim_cfg=sim_cfg
    )
    local_policy = GreedyLocalPolicy(
        action_module_cfg.LOCAL_POLICY, str(data_paths.navmesh_filepath), sim.get_agent(sim_cfg.DEFAULT_AGENT_ID)
    )
    map_builder = SemanticMap3DBuilder(map_builder_cfg, sim_cfg)
    perception_model = ModelWrapper(perception_model_cfg, device='cuda')

    env = HabitatEnv(sim, local_policy, map_builder, perception_model, env_cfg, str(data_paths.navmesh_filepath))
    policy_kwargs = dict(
        features_extractor_class=SemanticMapFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256)
    )

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=2, n_epochs=1, n_steps=100, device='cuda')
    logger = configure(None, ["stdout"])
    model.set_logger(logger)
    model.learn(300)


if __name__ == "__main__":
    Fire(main)
