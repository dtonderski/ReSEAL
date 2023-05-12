from pathlib import PurePath
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float, Float32, Int64, UInt8
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from typing_extensions import TypedDict
from yacs.config import CfgNode

from src.utils.category_mapping import get_maskrcnn_index_to_reseal_index_dict, get_reseal_index_to_maskrcnn_index_dict
from src.utils.datatypes import SemanticMap2D


class LossDict(TypedDict):
    loss_classifier: Float32[torch.Tensor, ""]
    loss_box_reg: Float32[torch.Tensor, ""]
    loss_mask: Float32[torch.Tensor, ""]
    loss_objectness: Float32[torch.Tensor, ""]
    loss_rpn_box_reg: Float32[torch.Tensor, ""]

class PredictionDict(TypedDict):
    boxes: Float32[torch.Tensor, "N 4"]
    labels: Int64[torch.Tensor, "N"]
    scores: Float32[torch.Tensor, "N"]
    masks: Float32[torch.Tensor, "N 1 H W"]

class LabelDict(TypedDict):
    boxes: Float32[torch.Tensor, "N 4"]
    labels: Int64[torch.Tensor, "N"]
    # The masks are binary
    masks: UInt8[torch.Tensor, "N H W"]

class ModelWrapper():
    """ 
    """
    @property
    def maskrcnn(self) -> nn.Module:
        return self._maskrcnn
    
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def device(self) -> str:
        return self._device

    def __init__(self,
                 model_config: CfgNode,
                 weights: Optional[Union[PurePath, str, MaskRCNN_ResNet50_FPN_Weights]] \
                     = MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
                 mode='eval',
                 device = 'cpu'):
        """_summary_

        Args:
            model_config (CfgNode): Must contain:
                USE_INITIAL_TRANSFORMS (bool): decides whether to use the built-in initial transformations.
                SCORE_THRESHOLD (float): detemines when to discard a prediction.
                MASK_THRESHOLD (float): determines the threshold for setting a mask element to 0. Usually 0.5.
            weights (Optional[Union[PurePath, str, MaskRCNN_ResNet50_FPN_Weights]], optional): the weights of the \
                model. If None, does not load weights, if PurePath or str, loads state_dict from file, and if \
                MaskRCNN_ResNet50_FPN_Weights, loads weights from torchvision. Defaults to \
                MaskRCNN_ResNet50_FPN_Weights.COCO_V1.
            mode (str, optional): starting mode of the model. Defaults to 'eval'.
        """
        super().__init__()
        self._model_config = model_config
        self._load_model(weights)
        self._update_mode(mode)
        self._update_device(device)
        self._load_initial_transforms()
        self._load_dictionaries()

    def _load_model(self, weights: Optional[Union[PurePath, str, MaskRCNN_ResNet50_FPN_Weights]]):
        if isinstance(weights, (PurePath, str)):
            self._maskrcnn = maskrcnn_resnet50_fpn()
            self._maskrcnn.load_state_dict(torch.load(weights))
        elif isinstance(weights, MaskRCNN_ResNet50_FPN_Weights):
            self._maskrcnn = maskrcnn_resnet50_fpn(weights = weights)
        elif weights is None:
            self._maskrcnn = maskrcnn_resnet50_fpn()
        else:
            raise ValueError(f"Unknown weights type: {type(weights)}")

    def _update_mode(self, mode):
        if mode == 'train':
            self._maskrcnn.train()
            self._mode = 'train'
        elif mode == 'eval':
            self._maskrcnn.eval()
            self._mode = 'eval'
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def _update_device(self, device):
        if device == 'cpu':
            self._maskrcnn.cpu()
            self._device = 'cpu'
        elif device == 'cuda':
            self._maskrcnn.cuda()
            self._device = 'cuda'
        else:
            raise ValueError(f"Unknown device: {device}! Must be one of cpu, cuda.")

    def _load_initial_transforms(self):
        if self._model_config.USE_INITIAL_TRANSFORMS:
            self._initial_transforms = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
        else:
            self._initial_transforms = None

    def _load_dictionaries(self):
        self._reseal_index_to_maskrcnn_index_dict = get_reseal_index_to_maskrcnn_index_dict()
        self._maskrcnn_index_to_reseal_index_dict = get_maskrcnn_index_to_reseal_index_dict()
        self._num_semantic_classes = max(self._reseal_index_to_maskrcnn_index_dict.keys())

    def train(self):
        self._update_mode('train')

    def eval(self):
        self._update_mode('eval')

    def cuda(self):
        self._update_device('cuda')
        
    def cpu(self):
        self._update_device('cpu')

    def __call__(self,
                 model_input: Union[Float[torch.Tensor, "B C H W"],
                              Float[np.ndarray, "H W C"]],
                 labels: Optional[Union[List[LabelDict], LabelDict]] = None,
                 label_indices_in_reseal_space = True) -> Union[List[SemanticMap2D], SemanticMap2D, List[LossDict]]:
        """ If the model is in train mode, labels must be provided, and it returns the loss dictionary. \
            If the model is in eval mode, labels must not be provided, and it returns a SemanticMap2D.

        Args:
            image (Union[str, int]): image can be a numpy array of shape (H, W, C) or a torch tensor of shape \
                (B, C, H, W). In both cases, the image is assumed to be in RGB format and with values in [0, 1].
            labels (Optional[List[LabelDict], LabelDict], optional): the label dict or list of label dicts used to \
                calculate the loss. Only use if model is in train mode. Defaults to None.
            label_indices_in_reseal_space (bool, optional): If True, the label indices are assumed to be in reseal \
                index space, and are converted to maskrcnn indices. If False, they are not converted.

        Raises:
            ValueError: If model is in train mode and labels is None.
            ValueError: If model is in eval mode and labels is not None.

        Returns:
            Union[PredictionDict, List[LossDict]]: _description_
        """
        # This shouldn't happen if the user follows typing but it's nice for convenience
        labels = self._preprocess_labels(labels, label_indices_in_reseal_space)
        if labels is not None:
            self._labels_to_device(labels)
        if self._mode == 'eval' and labels is not None:
            raise ValueError("In eval mode, labels must not be provided.")

        if self._mode == 'train' and labels is None:
            raise ValueError("In train mode, instance_map must be provided.")

        model_input_preprocessed = self._preprocess_image(model_input)
        model_input_preprocessed = model_input_preprocessed.to(self._device)

        if self._mode == 'train':
            return self._maskrcnn(model_input_preprocessed, labels)

        semantic_maps = self._predictions_to_semantic_maps(self._maskrcnn(model_input_preprocessed))

        if len(model_input.shape) == 3:
            return semantic_maps[0]

        return semantic_maps

    def _preprocess_labels(self,
                           labels: Optional[Union[List[LabelDict], LabelDict]],
                           label_indices_in_reseal_space: bool) -> Optional[List[LabelDict]]:
        if labels is None:
            return None
        if not isinstance(labels, list):
            labels = [labels]
        if label_indices_in_reseal_space:
            for label_dict in labels:
                label_dict['labels'].apply_(get_reseal_index_to_maskrcnn_index_dict().get)
        return labels

    def _labels_to_device(self, labels):
        """Move all tensors in all label dicts in labels to the device of the model.
        """
        for label_dict in labels:
            for key, value in label_dict.items():
                label_dict[key] = value.to(self._device)
        

    def _preprocess_image(self, image: Union[Float[torch.Tensor, "B C H W"], Float[np.ndarray, "H W C"]]
                          ) -> Float[torch.Tensor, "B C H W"]:
        if isinstance(image, np.ndarray):
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"Unknown image shape: {image.shape} for numpy array! Must be H,W,C.")
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) != 4 or image.shape[1] != 3:
                raise ValueError(f"Unknown image shape: {image.shape} for torch tensor! Must be B,C,H,W.")
        else:
            raise ValueError(f"Unknown image type: {type(image)}")

        return self._initial_transforms(image) if self._initial_transforms else image

    def _predictions_to_semantic_maps(self, predictions: List[PredictionDict]) -> List[SemanticMap2D]:
        semantic_maps: List[SemanticMap2D] = []
        for prediction in predictions:
            masks = prediction['masks']
            labels = prediction['labels']
            scores = prediction['scores']

            score_over_threshold = scores > self._model_config.SCORE_THRESHOLD
            # Threshold the relevant masks and multiply by the score
            masks_thresholded = ((masks[score_over_threshold] > self._model_config.MASK_THRESHOLD)
                                 *scores[score_over_threshold].view(-1,1,1,1))

            semantic_map = torch.zeros((masks.shape[2],masks.shape[3],self._num_semantic_classes),
                                    dtype=torch.float64, device = self._device)

            labels = labels.cpu().apply_(get_maskrcnn_index_to_reseal_index_dict().get).to(self._device)
            for mask, label in zip(masks_thresholded, labels[score_over_threshold]):
                if label > 0:
                    semantic_map[..., label-1] = mask.squeeze()
            semantic_maps.append(semantic_map.detach().cpu().numpy())
        return semantic_maps
