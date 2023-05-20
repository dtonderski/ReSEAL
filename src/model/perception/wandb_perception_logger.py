from typing import Any, List, Optional

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
    def __init__(self, run_name: Optional[str]):
        """ Initializes the wandb run. """
        wandb.init(
            entity='davton',
            project='reseal',
            name=run_name
        )

        wandb.define_metric("batch")
        wandb.define_metric("epoch")

    def on_epoch_end(self, epoch):
        """ Log the epoch and commit to server.
        """
        wandb.log({"epoch": epoch})
        
    
    def log_scene_ids(self, scene_ids: List[str], epoch: int) -> None:
        """ Logs the scene ids to wandb. """
        wandb.log({"scene_ids": wandb.Html("<br>".join(scene_ids), inject=False)}, commit=False)
    
    def log_categorical_label_map(self, label_map: LabelMap3DCategorical, grid_index_of_origin: GridIndex3D, 
                                  resolution: float, scene_id: str, epoch: int) -> None:
        """ Saves a categorical label map to log it to wandb. """
        fig = wandb.Plotly(
            visualize_categorical_label_map_plotly(label_map, grid_index_of_origin, resolution, scene_id, epoch))

        wandb.log({f"label_map_{scene_id}": fig}, commit=False)
        
        #self._label_map_figs.append(fig)
    
    def log_average_loss_dict(self, loss_dict: LossDict) -> None:
        """ Logs the average loss dictionary to wandb. """
        wandb.log(loss_dict, commit=False)

    def add_configs(self, cfg_dict_names: List[CfgNode], cfg_nodes: List[CfgNode]) -> None:
        """ Adds a list of configuration dictionaries to the wandb config. """
        for cfg_dict_name, cfg_node in zip(cfg_dict_names, cfg_nodes):
            self.add_config(cfg_dict_name, cfg_node)
    
    def add_config(self, cfg_dict_name: str, cfg_node: CfgNode):
        """ Adds a configuration dictionary to the wandb config. """
        wandb.config[cfg_dict_name] = cfg_node_to_dict(cfg_node)
    
    def add_unusual_config(self, key: str, value: Any) -> None:
        """ This adds a key-value pair to the "unusual" wandb config dictionary. This is used because our configuration 
            has ~100 elements, so figuring out which ones are non-default would be very hard without this. We store 
            both this and the normal config dictionary in case the default config changes in the future.

        Args:
            key (str): the key, for example NUM_EPOCHS
            value (Any): the value, for example 100.
        """
        if 'unusual' not in wandb.config:
            wandb.config['unusual'] = {}
        wandb.config['unusual'][key] = value

# def main():
#     wandb_logger = WandbPerceptionLogger("test2")

# if __name__ == '__main__':
#     main()