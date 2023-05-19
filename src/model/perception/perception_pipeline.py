import torch
from torch.utils.data import DataLoader
from src.model.perception.data_generator import DataGenerator
from src.data.MaskRCNNDataset import MaskRCNNDataset, collate_fn
from src.model.perception.model_wrapper import ModelWrapper
from tqdm import tqdm

from typing import Optional
from src.config import (
	CfgNode,
    default_action_module_cfg,
    default_data_paths_cfg,
    default_map_builder_cfg,
    default_map_processor_cfg,
    default_perception_data_generator_cfg,
    default_sim_cfg,
    default_model_cfg,
	train_maskrcnn_cfg
)

def train_perception_model_for_one_epoch(model, epoch_number:int) -> None:
	data_generator = DataGenerator(default_perception_data_generator_cfg(), default_data_paths_cfg(), 
                                   default_sim_cfg(), default_map_builder_cfg(), default_map_processor_cfg(), 
                                   default_action_module_cfg())
	data_generator(model, epoch_number)
	training_dataset = MaskRCNNDataset(default_data_paths_cfg(), default_perception_data_generator_cfg().SPLIT, epoch_number)
	training_dataloader = DataLoader(training_dataset,
									batch_size=train_maskrcnn_cfg().BATCH_SIZE,
									shuffle=train_maskrcnn_cfg().SHUFFLE,
									num_workers=train_maskrcnn_cfg().NUM_WORKERS,
									collate_fn=collate_fn)
	
	
	
	params =  [p for p in model.maskrcnn.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=train_maskrcnn_cfg().LEARNING_RATE,
								momentum=train_maskrcnn_cfg().OPTIM_MOMENTUM, 
								weight_decay=train_maskrcnn_cfg().OPTIM_WEIGHT_DECAY)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
								step_size=train_maskrcnn_cfg().OPTIM_STEP_SIZE,
								gamma=train_maskrcnn_cfg().OPTIM_GAMMA)
	model.train()
	for image, target in tqdm(training_dataloader):
		loss_dict = model(model_input=image, labels=target)
		loss= sum(l for l in loss_dict.values())
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		lr_scheduler.step()
	model.eval()
	torch.save(model._maskrcnn.state_dict(), "./models/MaskRCNN/model_"+str(epoch_number)+"ep")

def train_perception_model_with_action_policy()->None:
	model_config = default_model_cfg()
	num_epochs = train_maskrcnn_cfg().NUM_EPOCHS
	# initialize perception model
	model = ModelWrapper(model_config=model_config)
	model.cuda()
	for epoch in range(num_epochs):
		train_perception_model_for_one_epoch(model, epoch_number=epoch)

if __name__ == '__main__':    
    train_perception_model_with_action_policy()