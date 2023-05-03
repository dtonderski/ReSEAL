import pandas as pd
import numpy as np
from yacs.config import CfgNode



# Method for creating a pandas Dataframe containing the mapping between hm3d raw_categories to the selected MaskRCNN categories
# 1. load two tsv files containing the mappings (files under config folder)
# 2. remove unnecessary columns
# 3. join the mappings together
def load_hm3d_to_maskcat_mapping() -> pd.DataFrame:
	# load mapping from hm3d to mpcat40 and drop unnecessary columns
	hm3d_to_mpcat_mapping = pd.read_csv("./../config/category_mapping.tsv", sep='\t')
	hm3d_to_mpcat_mapping = hm3d_to_mpcat_mapping[{'raw_category', 'mpcat40'}].set_index('raw_category')
	
	# load mapping from mpcat40 to specified maskrcnn categories (bed, chair, couch, potted plant, toilet, tv)
	mpcat_to_maskcat_mapping = pd.read_csv("./../config/MaskRCNN_category_mapping.tsv", sep='\t').set_index('mpcat40')

	# join both maps to optain mapping from hm3d to maskrcnn categories
	hm3d_to_maskcat_mapping = hm3d_to_mpcat_mapping.join(mpcat_to_maskcat_mapping, on='mpcat40').drop(columns=['mpcat40'])
	return hm3d_to_maskcat_mapping

# Method for loading scene specific mapping from instance to hm3d category
# need to specify the corresponding scene semantic file path
def load_scene_instance_to_hm3d_mapping(scene_semantic_file_path) -> pd.DataFrame:
	# load mapping from instance number to hm3d category
	instance_to_hm3d_mapping = pd.read_csv(scene_semantic_file_path, sep=',', skiprows=[0], names=['instance','color','hm3d', 'room']).set_index('instance')
	# drop unnecessary columns
	instance_to_hm3d_mapping = instance_to_hm3d_mapping[{'instance', 'hm3d'}]
	return instance_to_hm3d_mapping

# create a mapping from category number to category name
def load_mask_instance_to_maskcat(cfg: CfgNode) ->pd.DataFrame:
	### To Do: include tracked categories into the MaskRCNN config file
	# tracked_categories = cfg.TRACKED_CATEGORIES
	tracked_categories_names = {(65, 'bed'), (62, 'chair'), (63,'couch'), (64,'potted plant'), (70,'toilet'), (72,'tv')}
	mask_instance_to_mascat = pd.DataFrame(tracked_categories_names, columns={'maskInstNum', 'maskCatName'}).sort_values('maskCatName', ignore_index=True).set_index('maskInstNum')
	mask_instance_to_mascat = mask_instance_to_mascat.assign(maskcat = range(len(mask_instance_to_mascat)))
	return mask_instance_to_mascat
