from typing import Any, List, Optional, Dict

from yacs.config import CfgNode

import wandb
from src.model.perception.model_wrapper import LossDict
from src.utils.datatypes import GridIndex3D, LabelMap3DCategorical
from src.visualisation.semantic_map_visualization import visualize_categorical_label_map_plotly


def cfg_node_to_dict(cfg: CfgNode):
    """
    Recursively convert a yacs.CfgNode to a dictionary.

    :param cfg: The yacs.CfgNode to convert.
    :return: The converted dictionary.
    """
    if isinstance(cfg, dict):
        result = {}
        for key in cfg:
            result[key] = cfg_node_to_dict(cfg[key])
        return result
    else:
        return cfg

class WandbPerceptionLogger:
    """ This class is used to log various information about the run to wandb. Some of the information logged includes:
        - Training loss
        - Categorical label maps for each data generation epoch
        
        NOTE: because the run name is based on the unusual config values, you MUST call WandbPerceptionLogger.init()
        manually.
    """
    @property
    def config(self):
        return wandb.config
    
    def __init__(self, perception_cfg: CfgNode):
        """ Initializes the wandb run. """
        self._config = perception_cfg.WANDB

        if self._config.USE_WANDB:
            wandb.init(
                entity='davton',
                project='reseal',
                name=self._get_run_name(perception_cfg),
            )
            
            
            wandb.define_metric("batch")
            wandb.define_metric("epoch")
            self.add_config("perception_cfg", perception_cfg)
    
    @staticmethod
    def _get_run_name(perception_cfg: CfgNode):
        """ Returns a string that can be used as the run name. """
        if "run_name" in perception_cfg:
            return perception_cfg.run_name
        else:
            return (f"score_threshold: {perception_cfg.MODEL.SCORE_THRESHOLD}, "
                    f"num_scenes: {perception_cfg.DATA_GENERATOR.NUM_SCENES}, "
                    f"num_steps: {perception_cfg.DATA_GENERATOR.NUM_STEPS}")

    def on_epoch_end(self, epoch):
        """ Log the epoch and commit to server.
        """
        if self._config.USE_WANDB:
            wandb.log({"epoch": epoch})
        
    
    def log_scene_ids(self, scene_ids: List[str]) -> None:
        """ Logs the scene ids to wandb. """
        if self._config.LOG_SCENE_IDS and self._config.USE_WANDB:
            wandb.log({"scene_ids": wandb.Html("<br>".join(scene_ids), inject=False)}, commit=False)
    
    def log_categorical_label_map(self, label_map: LabelMap3DCategorical, grid_index_of_origin: GridIndex3D, 
                                  resolution: float, scene_id: str, epoch: int) -> None:
        """ Saves a categorical label map to log it to wandb. """
        if self._config.LOG_MAP and self._config.USE_WANDB:
            # I don't think this is necessary, but I am having memory issues, so I'll give it a shot.
            fig = visualize_categorical_label_map_plotly(label_map, grid_index_of_origin, resolution, scene_id, epoch)
            wandb_fig = wandb.Plotly(fig)
            del fig

            wandb.log({f"label_map_{scene_id}": wandb_fig}, commit=False)
        
        #self._label_map_figs.append(fig)
    
    def log_average_loss_dict(self, loss_dict: LossDict) -> None:
        """ Logs the average loss dictionary to wandb. """
        if self._config.LOG_LOSS and self._config.USE_WANDB:
            wandb.log(loss_dict, commit=False)

    def add_configs(self, cfg_dict_names: List[CfgNode], cfg_nodes: List[CfgNode]) -> None:
        """ Adds a list of configuration dictionaries to the wandb config. """
        for cfg_dict_name, cfg_node in zip(cfg_dict_names, cfg_nodes):
            self.add_config(cfg_dict_name, cfg_node)
    
    def add_config(self, cfg_dict_name: str, cfg_node: CfgNode):
        """ Adds a configuration dictionary to the wandb config. """
        if self._config.USE_WANDB:
            wandb.config[cfg_dict_name] = cfg_node_to_dict(cfg_node)
    
    def add_unusual_config(self, config_dict: Dict[str, Any]) -> None:
        """ This adds a config dictionary to the "unusual" wandb config key. This is used because our configuration 
            has ~100 elements, so figuring out which ones are non-default would be very hard without this. We store 
            both this and the normal config dictionary in case the default config changes in the future.

        Args:
            key (str): the key, for example NUM_EPOCHS
            value (Any): the value, for example 100.
        """
        if self._config.USE_WANDB:
            if 'unusual' not in wandb.config:
                wandb.config['unusual'] = {}
            wandb.config['unusual'].update(config_dict)

# def main():
#     wandb_logger = WandbPerceptionLogger("test2")

# if __name__ == '__main__':
#     main()