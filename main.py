import time

import cv2

# Global variables
import numpy as np

from interaction import *
from segment_utils import segment_image, show_anns

selection_type = "point"
selected_points = []
selected_rect = None
start_point = None
end_point = None
drawing = False
selected_masks = []
mask_color = dict()

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global selected_points, selected_rect, start_point, drawing, end_point

    if selection_type == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_mask, color = point_selection(event, selected_points, x, y, image, masks)
            print(type(selected_mask))
            selected_mask['color'] = color
            selected_masks.append(selected_mask)

    elif selection_type == "area":
        draw_rect(event, x, y, flags, param)

    if event == ord('q'):
        cv2.destroyAllWindows()
        exit(0)


# Function to handle menu selection
def select_selection_type():
    global selection_type
    while True:
        print("Select a selection type:")
        print("1. Point")
        print("2. Area")
        choice = input("Enter your choice (1/2): ")
        if choice == "1":
            selection_type = "point"
            break
        elif choice == "2":
            selection_type = "area"
            break
        else:
            print("Invalid choice. Try again.\n")


# Read image from user input
image_path = input("Enter the path of the image: ")
#  /Users/danielbosch/Downloads/tools.jpg
# E:\Projects\interactive-scene-segmentation\test\desk.jpg
image = cv2.imread(image_path)
#convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a named window and set mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Select the type of selection
select_selection_type()

start_time = time.time()
masks = segment_image(image)
end_time = time.time()
print(f"image segmentation: {end_time - start_time} seconds")


# Display the image
while True:

    cv2.imshow("Image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if len(selected_masks) > 0:
        for mask in selected_masks:
            m = mask['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.array(mask['color'])
            for i in range(3):
                img[:, :, i] = color_mask[i]
            cv2.imshow("Image", np.dstack((img, m * 0.35)))
            cv2.waitKey(0)


# Cleanup
cv2.destroyAllWindows()
