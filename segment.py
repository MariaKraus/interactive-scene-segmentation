import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

from segment_utils import segment_image

sys.path.append("..")
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import time


def segment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", "-path_to_image", type=str, required=True)
    args = parser.parse_args()

    image = cv2.imread(args.i)
    # convert image into RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # load the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

    # set the model, there are 3 models available on the SAM github page
    start_time = time.time()
    masks = segment_image(image)

    print("--- %s seconds for segmentation ---" % (time.time() - start_time))

    # print the number of masks/segmented elements found in the image
    print("Number of masks: ", len(masks))

    def show_anns(anns):
        """
        Depicts the mask in various colors.

        :param anns: list of masks, each mask is a dictionary containing:
                    - segmentation : the mask
                    - area : the area of the mask in pixels
                    - bbox : the boundary box of the mask in XYWH format
                    - predicted_iou : the model's own prediction for the quality of the mask
                    - point_coords : the sampled input point that generated this mask
                    - stability_score : an additional measure of mask quality
                    - crop_box : the crop of the image used to generate this mask in XYWH format
        """
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.35)))

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')

    plt.show()


segment()
