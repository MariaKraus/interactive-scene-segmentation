import cv2
import numpy as np

selected_masks = []


def is_dictionary_in_list(dictionary, dictionary_list):
    for d in dictionary_list:
        if np.array_equal(d['segmentation'], dictionary['segmentation']):
            return True
    return False


def remove_mask(mask):
    global selected_masks
    for i, d in enumerate(selected_masks):
        if np.array_equal(d['segmentation'], mask['segmentation']):
            np.delete(selected_masks, i)
            break


def point_selection(event, selected_points, x, y, image, masks):
    selected_points.append((x, y))
    print(f"Selected point: ({x}, {y})")

    for mask in masks:
        if mask['segmentation'][y][x] == 1:
            if is_dictionary_in_list(mask, selected_masks):
                remove_mask(mask)
            else:
                selected_mask = mask
                selected_mask['color'] = list(np.random.choice(range(256), size=3))
                selected_masks.append(selected_mask)
            break

    return selected_masks