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

        self.parameters = [12]  # 12 as initial value
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.parameters[0],
            pred_iou_thresh=0.6,
            stability_score_thresh=0.5,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

    def segment_image(self, image):
        start_time = time.time()
        print(" ", self.parameters[0])
        # create the masks for the image
        masks = self.mask_generator.generate(image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=False)
        end_time = time.time()
        print(f"image segmentation: {end_time - start_time} seconds")
        return masks

    def segment_finer(self, image):
        self.parameters[0] = self.parameters[0] + 2
        print("segment finer with ", self.parameters[0], " points per side")
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.parameters[0],
            pred_iou_thresh=0.6,
            stability_score_thresh=0.5,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,

        )
        masks = self.segment_image(image)
        return masks, self.parameters[0]


    def segment_coarser(self, image):
        if self.parameters[0] > 4:
            self.parameters[0] = self.parameters[0] - 2
            print("segment coarser with ", self.parameters[0], " points per side")
        else:
            print("already at coarsest level")

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.parameters[0],
            pred_iou_thresh=0.6,
            stability_score_thresh=0.5,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        masks = self.segment_image(image)
        return masks, self.parameters[0]

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