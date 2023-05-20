import pathlib
from collections import defaultdict
from typing import Optional

import torch
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from src.data.MaskRCNNDataset import MaskRCNNDataset, collate_fn
from src.model.perception.data_generator import DataGenerator
from src.model.perception.model_wrapper import ModelWrapper
from src.model.perception.wandb_perception_logger import WandbPerceptionLogger
from src.model.perception.perception_pipeline_config import get_perception_cfg



def train_perception_model_for_one_epoch(
    model: ModelWrapper, epoch_number:int, wandb_logger: WandbPerceptionLogger, perception_config: CfgNode
    ) -> None:
    if epoch_number > 0:
        data_generator = DataGenerator(data_generator_cfg(), data_paths_cfg(), 
                                       sim_cfg(), map_builder_cfg(), map_processor_cfg(), 
                                       action_module_cfg())
        data_generator(model, epoch_number)
    training_dataset = MaskRCNNDataset(data_paths_cfg(), data_generator_cfg().SPLIT, 
                                       epoch_number)

    training_dataloader = DataLoader(training_dataset,
                                    batch_size=training_cfg().BATCH_SIZE,
                                    shuffle=training_cfg().SHUFFLE,
                                    num_workers=training_cfg().NUM_WORKERS,
                                    collate_fn=collate_fn)
    
    params =  [p for p in model.maskrcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=training_cfg().LEARNING_RATE,
                                momentum=training_cfg().OPTIM_MOMENTUM, 
                                weight_decay=training_cfg().OPTIM_WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                step_size=training_cfg().OPTIM_STEP_SIZE,
                                gamma=training_cfg().OPTIM_GAMMA)
    model.train()
    print("Training model...")
    losses_dict = defaultdict(list)
    for image, target in tqdm(training_dataloader):
        loss_dict = model(model_input=image, labels=target)
        for k, v in loss_dict.items():
            losses_dict[k].append(v.item())
        loss= sum(v for k, v in loss_dict.items() if k != "loss_box_reg" and k != "loss_rpn_box_reg")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
    average_loss_dict = {k: sum(v)/len(v) for k, v in losses_dict.items()}
    wandb_logger.log_average_loss_dict(average_loss_dict)
    model.eval()
    model.save(pathlib.Path(data_paths_cfg().MODEL_DIR) / "MaskRCNN" / f"model_ep{epoch_number}.pth")
    wandb_logger.on_epoch_end(epoch_number)

def load_kwargs_to_config(kwargs, config: CfgNode) -> CfgNode:
    for k,v in kwargs.items():
        val = config
        for key in k.split("."):
            key = key.upper()
            if not isinstance(val[key], CfgNode):
                val[key] = v
                break
            val = val[key]
    return config

def train_perception_model_with_action_policy(**kwargs) -> None:
    perception_cfg = get_perception_cfg()
    perception_cfg = load_kwargs_to_config(kwargs, perception_cfg)
    print(WandbPerceptionLogger._get_run_name(perception_cfg))
    wandb_logger = WandbPerceptionLogger(perception_cfg)
    wandb_logger.add_unusual_config(kwargs)
    print(wandb_logger.config["unusual"])
    # model_config = model_cfg()
    
    # num_epochs = training_cfg().NUM_EPOCHS
    # # initialize perception model
    # model = ModelWrapper(model_config=perception_cfg.MODEL_CONFIG)
    # model.cuda()
    # for epoch in range(0, num_epochs):
    #     train_perception_model_for_one_epoch(model, epoch, wandb_logger)

if __name__ == '__main__':
    Fire(train_perception_model_with_action_policy)