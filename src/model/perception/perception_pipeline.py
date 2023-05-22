import pathlib
from collections import defaultdict
from typing import List, Optional

import torch
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from src.data.MaskRCNNDataset import MaskRCNNDataset, collate_fn
from src.model.perception.data_generator import DataGenerator
from src.model.perception.model_wrapper import LossDict, ModelWrapper
from src.model.perception.wandb_perception_logger import WandbPerceptionLogger
from src.model.perception.perception_pipeline_config import get_perception_cfg



def train_perception_model_for_one_epoch(
    model: ModelWrapper, epoch_number:int, wandb_logger: WandbPerceptionLogger, perception_config: CfgNode
    ) -> None:
    if epoch_number >= 0:
        data_generator = DataGenerator(perception_config, wandb_logger)
        data_generator(model, epoch_number)
    training_dataset = MaskRCNNDataset(perception_config.DATA_PATHS, perception_config.DATA_GENERATOR.SPLIT,
                                       epoch_number)

    training_dataloader = DataLoader(training_dataset,
                                    batch_size=perception_config.TRAINING.BATCH_SIZE,
                                    shuffle=perception_config.TRAINING.SHUFFLE,
                                    num_workers=perception_config.TRAINING.NUM_WORKERS,
                                    collate_fn=collate_fn)
    
    params =  [p for p in model.maskrcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=perception_config.TRAINING.LEARNING_RATE,
                                momentum=perception_config.TRAINING.OPTIM_MOMENTUM, 
                                weight_decay=perception_config.TRAINING.OPTIM_WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                step_size=perception_config.TRAINING.OPTIM_STEP_SIZE,
                                gamma=perception_config.TRAINING.OPTIM_GAMMA)
    model.train()
    print("Training model...")
    losses_dict = defaultdict(list)
    for image, target in tqdm(training_dataloader):
        loss_dict: LossDict = model(model_input=image, labels=target) # type: ignore
        for k, v in loss_dict.items():
            losses_dict[k].append(v.item()) # type: ignore
        loss = sum(v for k, v in loss_dict.items() if k != "loss_box_reg" and k != "loss_rpn_box_reg") # type: ignore
        optimizer.zero_grad()
        loss.backward() # type: ignore
        optimizer.step()
        lr_scheduler.step()
    average_loss_dict = {k: sum(v)/len(v) for k, v in losses_dict.items()}
    wandb_logger.log_average_loss_dict(average_loss_dict) # type: ignore
    model.eval()
    model.save(pathlib.Path(perception_config.DATA_PATHS.MODEL_DIR) / "MaskRCNN" / f"model_ep{epoch_number}.pth")
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
    wandb_logger = WandbPerceptionLogger(perception_cfg)
    wandb_logger.add_unusual_config(kwargs)
    
    num_epochs = perception_cfg.TRAINING.NUM_EPOCHS
    # initialize perception model
    model = ModelWrapper(model_config=perception_cfg.MODEL)
    model.cuda()
    for epoch in range(0, num_epochs):
        train_perception_model_for_one_epoch(model, epoch, wandb_logger, perception_cfg)

if __name__ == '__main__':
    Fire(train_perception_model_with_action_policy)
