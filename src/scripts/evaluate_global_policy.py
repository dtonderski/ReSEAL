from fire import Fire
from src import config as cfg
from src.data import filepath, scene
from src.model.action.env import HabitatEnv
from src.model.action.local_policy import GreedyLocalPolicy
from src.model.action.preprocessing import create_preprocessor
from src.model.action.global_policy import create_global_policy
from src.features.mapping import SemanticMap3DBuilder
from src.model.perception.model_wrapper import ModelWrapper

from tqdm import trange
import torch
import numpy as np


def main(
    scene_name: str = "minival/00800-TEEsavR23oF",
    env_cfg_file: str = "config/env_cfg.yaml",
    action_pipeline_cfg_filepath: str = "config/action_pipeline_cfg.yaml",
):
    # Load configs
    env_cfg = cfg.default_env_cfg()
    env_cfg.merge_from_file(env_cfg_file)
    sim_cfg = cfg.default_sim_cfg()
    map_builder_cfg = cfg.default_map_builder_cfg()
    perception_model_cfg = cfg.default_perception_model_cfg()
    action_module_cfg = cfg.default_action_module_cfg()
    action_module_cfg.merge_from_file(action_pipeline_cfg_filepath)
    data_paths_cfg = cfg.default_data_paths_cfg()
    data_paths = filepath.GenerateTrajectoryFilepaths(data_paths_cfg, scene_name)

    # Initialize sim
    sim = scene.initialize_sim(
        data_paths.scene_split, data_paths.scene_id, data_paths_cfg=data_paths_cfg, sim_cfg=sim_cfg
    )

    # Initialize components for environment
    local_policy = GreedyLocalPolicy(
        action_module_cfg.LOCAL_POLICY, str(data_paths.navmesh_filepath), sim.get_agent(sim_cfg.DEFAULT_AGENT_ID)
    )
    map_builder = SemanticMap3DBuilder(map_builder_cfg, sim_cfg)
    perception_model = ModelWrapper(perception_model_cfg, device="cuda")
    preprocessor = create_preprocessor(action_module_cfg.PREPROCESSOR)

    # Initialize environment
    env = HabitatEnv(
        sim,
        local_policy,
        map_builder,
        perception_model,
        preprocessor,
        env_cfg,
        str(data_paths.navmesh_filepath),
    )

    # Initiatize global policy
    global_policy = create_global_policy(
        action_module_cfg.GLOBAL_POLICY, navmesh_filepath=str(data_paths.navmesh_filepath)
    ).to("cuda").eval()  # type: ignore[union-attr]

    rewards = []
    for _ in trange(20):
        episode_reward = 0.0
        obs, _ = env.reset()
        done = False
        while not done:
            for key in obs:
                obs[key] = torch.Tensor(obs[key]).to("cuda")
            with torch.no_grad():
                action, _, _ = global_policy.forward(obs, False) # type: ignore[arg-type]
            action = action.numpy(force = True)[0]
            obs, reward, done, _trunc, info = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)

    # Print average, std, min and max reward
    print(f"Average reward: {np.mean(rewards)}")
    print(f"Std reward: {np.std(rewards)}")
    print(f"Min reward: {np.min(rewards)}")
    print(f"Max reward: {np.max(rewards)}")

    print(rewards)


if __name__ == "__main__":
    Fire(main)
