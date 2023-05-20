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
import wandb
from wandb.integration.sb3 import WandbCallback


def main(scene_name: str = "minival/00800-TEEsavR23oF"):
    wandb.login()
    config = {
        "policy_type": "PPO",
        "env_name": "HabitatEnv",
        "total_timesteps": 10000,
    }
    run = wandb.init(
        project="reseal",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

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
    action_module_cfg.GLOBAL_POLICY.MAP_SHAPE = map_builder.semantic_map_at_pose_shape
    policy_kwargs = create_global_policy(
        action_module_cfg.GLOBAL_POLICY,
        return_kwargs=True,
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,  # type: ignore[arg-type]
        verbose=2,
        n_epochs=10,
        n_steps=100,
        device="cuda",
        tensorboard_log="runs/{run.id}",
    )
    model.learn(
        total_timesteps=config["total_timesteps"],  # type: ignore[arg-type]
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=action_module_cfg.GLOBAL_POLICY.MODEL_PATH,
            model_save_freq=100,
            verbose=2,
        )
    )
    model_save_path = action_module_cfg.GLOBAL_POLICY.MODEL_PATH + f"/{run.id}.pth"  # type: ignore[union-attr]
    model.policy.save(model_save_path)

    run.finish()  # type: ignore[union-attr]


if __name__ == "__main__":
    Fire(main)
