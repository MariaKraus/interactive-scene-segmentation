import os
import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

drawing = False

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    (selection_type, selected_points) = param
    global drawing

    if selection_type == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
            return selected_points
        if event == cv2.EVENT_RBUTTONDOWN:
            selected_points.pop()

    elif selection_type == "area":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.clear()
            drawing = True
            selected_points.append((x, y))
            return selected_points
        if event == cv2.EVENT_LBUTTONUP:
            if drawing is True:
                selected_points.append((x, y))
                drawing = False
                return selected_points
        if drawing is True:
            selected_points.append((x, y))
            return selected_points

    elif selection_type == "polygon":
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_points) > 2 and np.linalg.norm(np.array(selected_points[0]) - np.array((x, y))) < 25:
                new_point = selected_points[0]
            else:
                new_point = (x, y)
            selected_points.append(new_point)
        if event == cv2.EVENT_RBUTTONDOWN:
            selected_points.pop()

def write_number_to_file(filename, number, name):
    with open(filename, 'a') as file:
        file.write(name + "," + str(number) + "\n")

def keyboard_callback(event, param):
    """
    Method to handle keyboard events
    :param event: The keyboard event (pressed key)
    :param param: param = sam, base_image, masked_image, masks, selection_type, selected_points, selected_masks, interactive_trainer
    :return: param = sam, base_image, masked_image, masks, selection_type, selected_points, selected_masks, interactive_trainer
    """
    (model, base_image, _, _, _, _, _, model_parameters, interactive_trainer, name) = param
    param = list(param)

    if event == 13:  # Check if the key is the "Enter" key (key code 13)
        print("pressed Enter")

    if event == ord('q'):
        cv2.destroyAllWindows()
        exit(0)

    if event == 49:  # if 1 was pressed
        print("point selection: Select masks you want to resegment")
        param[4] = "point"
        # empty the selected points
        param[5].clear()
        # clear the selected masks
        param[6].clear()

    if event == 50:  # if 2 was pressed
        print("area selection: Select an area that you want to re-segment "
              "by clicking and dragging the mouse over the picture.")
        param[4] = "area"
        # empty the selected points
        param[5].clear()
        # clear the selected masks
        param[6].clear()

    if event == 51:  # if 3 was pressed
        print("Press Left Mouse Button to add a point. Press Right Mouse Button to remove a point.")
        param[4] = "polygon"
        # empty the selected points
        param[5].clear()
        # clear the selected masks
        param[6].clear()

    if event == 100:  # d = arrow
        print("New image")
        param[1] = None

        points_per_side, _, _ = model.parameters
        #convert base image to grayscale
        greyscaled = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        print("image entropy:", entropy(greyscaled, disk(5)).mean())


        image_resized = cv2.resize(base_image, (100, 100))
        # cv2.imwrite("segmentation_image.png", base_image)
        write_number_to_file('label_images.txt', points_per_side, name)
        interactive_trainer.update(image_resized, points_per_side)

    if event == 119:  # w = arrow up, segment finer
        selected_area_image = get_selected_area_pixels(param)
        # if an area was selected return new masks
        if len(selected_area_image) != 0:
            masks, model_parameters = model.segment_finer(selected_area_image, model_parameters)
            masked_image = model.show_masks(selected_area_image, masks)
            param[1] = selected_area_image
            param[2] = masked_image
            param[3] = masks
            # empty the selected points
            param[5].clear()
            # clear the selected masks
            param[6].clear()
            param[7] = model_parameters

    if event == 115:  # s = arrow down, segment coarser
        selected_area_image = get_selected_area_pixels(param)
        # if an area was selected return new masks
        if len(selected_area_image) != 0:
            masks, model_parameters = model.segment_coarser(selected_area_image, model_parameters)
            masked_image = model.show_masks(selected_area_image, masks)
            param[1] = selected_area_image
            param[2] = masked_image
            param[3] = masks
            # empty the selected points
            param[5].clear()
            # clear the selected masks
            param[6].clear()
            param[7] = model_parameters

    return param


# Function to handle keyboard events
def get_selected_area_pixels(param):
    _, base_image, _, _, selection_type, selected_points, selected_masks, _, _, _ = param
    selected_area = base_image.copy()
    bounding_box = []

    # if no region was specified, return the whole image
    if not selected_points:
        print("Whole image selected for resegmentation")
        return selected_area

    if selection_type == "point":
        combined_masks = np.zeros(selected_area.shape)
        for mask in selected_masks:
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

    if selection_type == "area":
        start_point = selected_points[0]
        end_point = selected_points[-1]
        # if only one point was selected return the whole image
        if start_point == end_point:
            print("Please select a complete area for resegmentation")
            return []
        bounding_box = [start_point, end_point]

    if selection_type == "polygon":
        polygon_points = np.array(selected_points)
        if (polygon_points[0][0] != polygon_points[-1][0]) | (polygon_points[0][1] != polygon_points[-1][1]):
            print("Please selected a closed area for resegmentation")
            return []
        mask = np.zeros((base_image.shape[0], base_image.shape[1], 3), dtype=np.uint8)
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


def load_images(directory: str):
    # Read image from user input
    directory_path = ""
    images = []
    file_names = []
    # directory exists
    while not images:
        for filename in sorted(os.listdir(directory)):
            img = cv2.imread(os.path.join(directory, filename))
            file_names.append(filename)
            if img is not None:
                images.append(img)
    return images, file_names
