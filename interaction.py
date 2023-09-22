import os
import cv2
import numpy as np
from image_container import ImageContainer

drawing = False


def mouse_callback(event, x, y, flags, param):
    """
        Callback function to handle mouse events.

        Parameters:
            :param event: The type of mouse event.
            :param x: The x-coordinate of the mouse event.
            :param y: The y-coordinate of the mouse event.
            :param flags: Additional flags for the mouse event.
            :param param: Tuple containing the selection type and selected points.

        Returns:
            None
    """
    (selection_type, selected_points) = param
    global drawing

    if selection_type == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            selected_points.pop()

    elif selection_type == "area":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.clear()
            drawing = True
            selected_points.append((x, y))
        if event == cv2.EVENT_LBUTTONUP:
            if drawing is True:
                selected_points.append((x, y))
                drawing = False
        if drawing is True:
            selected_points.append((x, y))
    elif selection_type == "polygon":
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_points) > 2 and np.linalg.norm(np.array(selected_points[0]) - np.array((x, y))) < 25:
                new_point = selected_points[0]
            else:
                new_point = (x, y)
            selected_points.append(new_point)
        if event == cv2.EVENT_RBUTTONDOWN:
            selected_points.pop()


def keyboard_callback(event, param):
    """
    Method to handle keyboard events
    :param event: the event that was triggered
    :param param: the parameters that were passed to the callback
    """
    (model, curr_img, new) = param

    if event == 13:  # Check if the key is the "Enter" key (key code 13)
        print("pressed Enter")

    if event == ord('q'):
        cv2.destroyAllWindows()
        exit(0)

    if event == 49:  # if 1 was pressed
        print("point selection: Select masks you want to resegment")
        curr_img.selection_type = "point"
        # empty the selected points
        curr_img.set_selected_points([])
        # clear the selected masks
        curr_img.set_selected_masks([])

    if event == 50:  # if 2 was pressed
        print("area selection: Select an area that you want to re-segment "
              "by clicking and dragging the mouse over the picture.")
        curr_img.selection_type = "area"
        # empty the selected points
        curr_img.set_selected_points([])
        # clear the selected masks
        curr_img.set_selected_masks([])

    if event == 51:  # if 3 was pressed
        print("Press Left Mouse Button to add a point. Press Right Mouse Button to remove a point.")
        curr_img.selection_type = "polygon"
        # empty the selected points
        curr_img.set_selected_points([])
        # clear the selected masks
        curr_img.set_selected_masks([])

    if event == 100:  # d = arrow
        print("New image")
        new = True

    if event == 119:  # w = arrow up, segment finer
        selected_area_image = get_selected_area_pixels(curr_img)
        # if an area was selected return new masks
        if len(selected_area_image) != 0:
            masks, model_parameters = model.segment_finer(selected_area_image)
            masked_image = model.show_masks(selected_area_image, masks)
            curr_img.set_image(selected_area_image)
            curr_img.set_file_name(curr_img.file_name)
            curr_img.set_masked_image(masked_image)
            curr_img.set_masks(masks)
            curr_img.set_model_parameters(model_parameters)
            # empty the selected points
            curr_img.set_selected_points([])
            # clear the selected masks
            curr_img.set_selected_masks([])

    if event == 115:  # s = arrow down, segment coarser
        selected_area_image = get_selected_area_pixels(curr_img)
        # if an area was selected return new masks
        if len(selected_area_image) != 0:
            masks, model_parameters = model.segment_coarser(selected_area_image)
            masked_image = model.show_masks(selected_area_image, masks)
            curr_img.set_image(selected_area_image)
            curr_img.set_file_name(curr_img.file_name)
            curr_img.set_masked_image(masked_image)
            curr_img.set_masks(masks)
            curr_img.set_model_parameters(model_parameters)
            # empty the selected points
            curr_img.set_selected_points([])
            # clear the selected masks
            curr_img.set_selected_masks([])

    return model, curr_img, new


# Function to handle keyboard events
def get_selected_area_pixels(curr_img):
    selected_area = curr_img.image.copy()
    bounding_box = []

    # if no region was specified, return the whole image
    if not curr_img.selected_points:
        print("Whole image selected for resegmentation")
        return selected_area

    if curr_img.selection_type == "point":
        combined_masks = np.zeros(selected_area.shape)

        # if no region was specified, return the whole image
        if not curr_img.selected_masks:
            print("No masks was selected, whole image selected for resegmentation")
            return selected_area

        for mask in curr_img.selected_masks:
            converted_mask = np.array(mask['segmentation'], dtype=np.uint8)
            converted_mask = np.repeat(converted_mask[:, :, np.newaxis], 3, axis=2)
            combined_masks += converted_mask

        selected_area = np.where(combined_masks > 0, selected_area, [0, 0, 0])
        # Find the indices of non-black pixels
        non_black_indices = np.argwhere(np.any(selected_area != [0, 0, 0], axis=-1))
        # Find the top-left and bottom-right points
        top_left = non_black_indices.min(axis=0)
        bottom_right = non_black_indices.max(axis=0)
        bounding_box = [[top_left[1], top_left[0]], [bottom_right[1], bottom_right[0]]]

    if curr_img.selection_type == "area":
        start_point = curr_img.selected_points[0]
        end_point = curr_img.selected_points[-1]
        # if only one point was selected return the whole image
        if start_point == end_point:
            print("Please select a complete area for resegmentation")
            return []
        bounding_box = [start_point, end_point]

    if curr_img.selection_type == "polygon":
        polygon_points = np.array(curr_img.selected_points)
        if (polygon_points[0][0] != polygon_points[-1][0]) | (polygon_points[0][1] != polygon_points[-1][1]):
            print("Please selected a closed area for resegmentation")
            return []
        mask = np.zeros((curr_img.image.shape[0], curr_img.image.shape[1], 3), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], [1, 1, 1])
        selected_area = np.where(mask > 0, selected_area, [0, 0, 0])
        bounding_box = [np.min(polygon_points, axis=0), np.max(polygon_points, axis=0)]

    # extract region of interest based on bounding box
    p1, p2 = bounding_box

    cv2.imwrite("selected_area_raw.png", selected_area)
    # Determine the x and y coordinates of the bounding box
    x1 = min(p1[0], p2[0])
    x2 = max(p1[0], p2[0])
    y1 = min(p1[1], p2[1])
    y2 = max(p1[1], p2[1])

    # Select the area from the image using the bounding box coordinates
    selected_area = selected_area[y1:y2, x1:x2]
    cv2.imwrite("selected_area_scaled.png", selected_area)
    selected_area = np.array(selected_area, dtype=np.uint8)
    return selected_area


def is_dictionary_in_list(dictionary, dictionary_list):
    for d in dictionary_list:
        if np.array_equal(d['segmentation'], dictionary['segmentation']):
            return True
    return False


def remove_mask(mask, selected_masks):
    for i, d in enumerate(selected_masks):
        if np.array_equal(d['segmentation'], mask['segmentation']):
            np.delete(selected_masks, i)
            break


def select_masks(x, y, masks, selected_masks):
    """
        Select masks based on the given coordinates (x, y).

        Parameters:
            :param x: The x-coordinate.
            :param y: The y-coordinate.
            :param masks: List of masks to select from.
            :param selected_masks: List of selected masks.

        Returns:
            :return list: Updated list of selected masks.
        """
    for mask in masks:
        if mask['segmentation'][y][x] == 1:
            if is_dictionary_in_list(mask, selected_masks):
                remove_mask(mask, selected_masks)
            else:
                selected_mask = mask
                # if there is already a color, keep the old one
                try:
                    print(selected_mask['color'])
                except KeyError:
                    selected_mask['color'] = list(np.random.choice(range(256), size=3))
                selected_masks.append(selected_mask)
            break
    return selected_masks
