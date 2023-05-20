from typing import Optional
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
import wandb
from wandb.integration.sb3 import WandbCallback


def main(
    scene_name: str = "minival/00800-TEEsavR23oF",
    env_cfg_filepath: Optional[str] = None,
    training_cfg_filepath: Optional[str] = None,
):
    env_cfg = cfg.default_env_cfg()
    if env_cfg_filepath:
        env_cfg.merge_from_file(env_cfg_filepath)
    sim_cfg = cfg.default_sim_cfg()
    action_module_cfg = cfg.default_action_module_cfg()
    action_module_cfg.GLOBAL_POLICY.NAME = "CnnPolicy"
    action_module_cfg.PREPROCESSOR.NAME = "IdentityPreprocessor"
    map_builder_cfg = cfg.default_map_builder_cfg()
    perception_model_cfg = cfg.default_perception_model_cfg()

    data_paths_cfg = cfg.default_data_paths_cfg()
    data_paths = filepath.GenerateTrajectoryFilepaths(data_paths_cfg, scene_name)

    training_cfg = cfg.default_action_training_cfg()
    if training_cfg_filepath:
        training_cfg.merge_from_file(training_cfg_filepath)
    env_cfg.MAX_STEPS = training_cfg.NUM_STEPS_PER_EPISODE
    training_cfg.MODEL_PATH = f"models/{scene_name}/global_policy" 

    wandb.login()
    config = {
        "policy_type": "PPO",
        "env_name": "HabitatEnv",
        "total_timesteps": training_cfg.NUM_TOTAL_STEPS,
        "timesteps_per_episode": training_cfg.NUM_STEPS_PER_EPISODE,
        "num_epochs": training_cfg.NUM_EPOCHS,
        "batch_size": training_cfg.BATCH_SIZE,
        "model_path": training_cfg.MODEL_PATH,
    }
    run = wandb.init(
        project="reseal",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

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
        n_epochs=training_cfg.NUM_EPOCHS,
        n_steps=training_cfg.NUM_STEPS_PER_EPISODE,
        device="cuda",
        tensorboard_log="runs/{run.id}",
    )
    model.learn(
        total_timesteps=config["total_timesteps"],  # type: ignore[arg-type]
        callback=WandbCallback(
            model_save_path=training_cfg.MODEL_PATH,
            model_save_freq=1000,
            gradient_save_freq=1000,
            verbose=2,
        ),
    )
    model_save_path = training_cfg.MODEL_PATH + f"/{run.id}.pth"  # type: ignore[union-attr]
    model.policy.save(model_save_path)

    run.finish()  # type: ignore[union-attr]


if __name__ == "__main__":
    Fire(main)
