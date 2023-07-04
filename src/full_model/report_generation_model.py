from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import sys
sys.path.append("/home/hermione/Documents/VLP/rgrg_edit/")
from src.language_model.language_model import LanguageModel

class ReportGenerationModel(nn.Module):
    """
    Full model consisting of:
        - image encoder
        - language model decoder
    """

    def __init__(self):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # since we have grayscale images, we need to change the first conv layer to accept 1 in_channel (instead of 3)
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # use only the feature extractor of the pre-trained classification model
        # (i.e. use all children but the last 2, which are AdaptiveAvgPool2d and Linear)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])#[:-2]

        # FasterRCNN needs to know the number of output channels of the backbone
        # for ResNet-50, it's 2048 (with feature maps of size 16x16)
        self.backbone.out_channels = 2048

        # Load the best object detector from the 1st training stage here when starting the 2nd training stage
        # path_to_best_object_detector_weights = "/u/home/tanida/runs/object_detector/run_10/weights/val_loss_13.482_epoch_6.pth"
        # self.object_detector.load_state_dict(torch.load(path_to_best_object_detector_weights))

        #self.binary_classifier_region_selection = BinaryClassifierRegionSelection()
        #self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal()
        self.language_model = LanguageModel()

    def forward(
        self,
        images: torch.FloatTensor,  # images is of shape [batch_size x 1 x 512 x 512] (whole gray-scale images of size 512 x 512)
        #image_targets: List[Dict],  # contains a dict for every image with keys "boxes" and "labels"
        input_ids: torch.LongTensor,  # shape [(batch_size * 29) x seq_len], 1 sentence for every region for every image (sentence can be empty, i.e. "")
        attention_mask: torch.FloatTensor,  # shape [(batch_size * 29) x seq_len]
        return_loss: bool = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
    ):
        """
        Forward method is used for training and evaluation of model.
        Generate method is used for inference.
        """

        # top_region_features of shape [batch_size x 29 x 1024] (i.e. 1 feature vector for every region for every image in batch)
        # class_detected is a boolean tensor of shape [batch_size x 29]. Its value is True for a class if the object detector detected the class/region in the image
        features = self.backbone(images)
        print("Features shape: ", features.shape)
        del images

        language_model_loss = self.language_model(
            input_ids,
            attention_mask,
            features,
            return_loss,
            past_key_values,
            position_ids,
            use_cache,
        )

        del input_ids
        del attention_mask
        del features

        if self.training:
            return language_model_loss
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.FloatTensor,  # images is of shape [batch_size x 1 x 512 x 512] (whole gray-scale images of size 512 x 512)
        max_length: int = None,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        early_stopping: bool = False,
    ):
        """
        In inference mode, we usually input 1 image (with 29 regions) at a time.

        The object detector first finds the region features for all 29 regions.

        The binary classifier takes the region_features of shape [batch_size=1, 29, 1024] and returns:
            - selected_region_features: shape [num_regions_selected_in_batch, 1024],
            all region_features which were selected by the classifier to get a sentence generated (and which were also detected by the object detector)

            - selected_regions: shape [batch_size x 29], boolean matrix that indicates which regions were selected to get a sentences generated
            (these regions must also have been detected by the object detector).
            This is needed in case we want to find the corresponding reference sentences to compute scores for metrics such as BertScore or BLEU.

        The decoder then takes the selected_region_features and generates output ids for the batch.
        These output ids can then be decoded by the tokenizer to get the generated sentences.

        We also return selected_regions, such that we can map each generated sentence to a selected region.
        We also return detections, such that we can map each generated sentence to a bounding box.
        We also return class_detected to know which regions were not detected by the object detector (can be plotted).
        """
        # top_region_features of shape [batch_size x 29 x 1024]
        features = self.backbone(images)

        del images
        
        # output_ids of shape (num_regions_selected_in_batch x longest_generated_sequence_length)
        output_ids = self.language_model.generate(
            features,
            max_length,
            num_beams,
            num_beam_groups,
            do_sample,
            num_return_sequences,
            early_stopping,
        )

        return output_ids
