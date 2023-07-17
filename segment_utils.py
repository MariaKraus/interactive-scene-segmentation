import time
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2


class SegmentAnything:
    def __init__(self,
                 checkpoint: str,
                 model_type: str,
                 device: str) -> None:

        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        # initialize segment anything
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        # run the model on cuda
        self.sam.to(device=device)
        # link that explains the configurations:
        # https://github.com/facebookresearch/segment-anything/blob/9e1eb9fdbc4bca4cd0d948b8ae7fe505d9f4ebc7/segment_anything/automatic_mask_generator.py#L35
        # This is what we want to change with the

        points_per_side, pred_iou_thresh, stability_score_thresh = self.sample_parameters()
        self.parameters = points_per_side, pred_iou_thresh, stability_score_thresh
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

    def segment_image(self, image):
        start_time = time.time()
        # create the masks for the image
        masks = self.mask_generator.generate(image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=False)
        end_time = time.time()
        print(f"image segmentation: {end_time - start_time} seconds")
        return masks

    def segment_finer(self, image, parameters):
        print("segment finer with ", parameters[0], " points per side")
        parameters = list(parameters)
        parameters[0] = parameters[0] + 2
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=parameters[0],
            pred_iou_thresh=0.6,
            stability_score_thresh=0.5,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        masks = self.segment_image(image)
        return masks, parameters


    def segment_coarser(self, image, parameters):
        print("segment coarser with ", parameters[0], " points per side")
        parameters = list(parameters)
        parameters[0]= parameters[0] - 2
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=parameters[0] - 2,
            pred_iou_thresh=0.6,
            stability_score_thresh=0.5,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
        masks = self.segment_image(image)
        return masks, parameters

    def show_masks(self, image, masks):
        """
        Depicts the mask in various colors.

        :param masks:
        :param image:
        :param anns: list of masks, each mask is a dictionary containing:
                    - segmentation : the mask
                    - area : the area of the mask in pixels
                    - bbox : the boundary box of the mask in XYWH format
                    - predicted_iou : the model's own prediction for the quality of the mask
                    - point_coords : the sampled input point that generated this mask
                    - stability_score : an additional measure of mask quality
                    - crop_box : the crop of the image used to generate this mask in XYWH format
        """
        if len(masks) == 0:
            return
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

        for ann in sorted_anns:
            m = ann['segmentation']
            mask = image.copy()
            try:
                color_mask = ann['color']
                break
            except:
                ann['color'] = list(np.random.choice(range(256), size=3))
                color_mask = ann['color']

            mask[m] = color_mask
            image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
        return image

    def sample_parameters(self):
        points_per_side = int(np.random.normal(15, 15))
        pred_iou_thresh = np.random.normal(0.6, 0.2)
        stability_score_thresh = np.random.normal(0.5, 0.1)

        # Ensure parameters are within valid range
        points_per_side = min(max(points_per_side, 2), 40)
        pred_iou_thresh = max(min(pred_iou_thresh, 1.0), 0.0)
        stability_score_thresh = max(min(stability_score_thresh, 1.0), 0.0)

        print("points_per_side", points_per_side)
        #print("pred_iou_tresh", pred_iou_thresh)
        #print("stability_score_thresh", stability_score_thresh)

        return points_per_side, pred_iou_thresh, stability_score_thresh
