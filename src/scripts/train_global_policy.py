from fire import Fire

from src import config as cfg
from src.model.action.env import HabitatEnv
from src.model.action.local_policy import GreedyLocalPolicy
from src.model.action.global_policy import create_global_policy
from src.model.action.preprocessing import create_preprocessor
from src.features.mapping import SemanticMap3DBuilder
from src.model.perception.model_wrapper import ModelWrapper
from src.data import scene, filepath

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


def main(scene_name: str = "minival/00800-TEEsavR23oF"):
    env_cfg = cfg.default_env_cfg()
    sim_cfg = cfg.default_sim_cfg()
    action_module_cfg = cfg.default_action_module_cfg()
    action_module_cfg.GLOBAL_POLICY.NAME = "CnnPolicy"
    action_module_cfg.PREPROCESSOR.NAME = "IdentityPreprocessor"
    map_builder_cfg = cfg.default_map_builder_cfg()
    perception_model_cfg = cfg.default_perception_model_cfg()

    data_paths_cfg = cfg.default_data_paths_cfg()
    data_paths = filepath.GenerateTrajectoryFilepaths(data_paths_cfg, scene_name)

    sim = scene.initialize_sim(
        data_paths.scene_split, data_paths.scene_id, data_paths_cfg=data_paths_cfg, sim_cfg=sim_cfg
    )
    local_policy = GreedyLocalPolicy(
        action_module_cfg.LOCAL_POLICY, str(data_paths.navmesh_filepath), sim.get_agent(sim_cfg.DEFAULT_AGENT_ID)
    )
    map_builder = SemanticMap3DBuilder(map_builder_cfg, sim_cfg)
    perception_model = ModelWrapper(perception_model_cfg, device="cuda")
    preprocessor = create_preprocessor(action_module_cfg.PREPROCESSOR)
    env = HabitatEnv(
        sim,
        local_policy,
        map_builder,
        perception_model,
        preprocessor,
        env_cfg,
        str(data_paths.navmesh_filepath),
    )
    policy_kwargs = create_global_policy(
        action_module_cfg.GLOBAL_POLICY,
        observation_space=env.observation_space,
        action_space=env.action_space,
        return_kwargs=True,
    )

    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=2, n_epochs=1, n_steps=10, device="cuda")  # type: ignore[arg-type]
    logger = configure(None, ["stdout"])
    model.set_logger(logger)
    model.learn(10)

    model.policy.save(action_module_cfg.GLOBAL_POLICY.MODEL_PATH)


if __name__ == "__main__":
    Fire(main)
