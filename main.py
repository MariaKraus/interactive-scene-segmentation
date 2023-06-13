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
    global selected_points, selected_rect, start_point, drawing, end_point, selected_masks

    if selection_type == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_masks = point_selection(event, selected_points, x, y, image, masks)

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
baseImage = image.copy()
#convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a named window and set mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Select the type of selection
select_selection_type()

start_time = time.time()
masks = segment_image(image)
masks = sorted(masks, key=lambda x: x['area'], reverse=False)
end_time = time.time()
print(f"image segmentation: {end_time - start_time} seconds")


# Display the image
while True:

    image = baseImage.copy()

    if len(selected_masks) == 0:
        cv2.imshow("Image", image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if len(selected_masks) > 0:
        image = show_anns(image, selected_masks)
        cv2.imshow("Image", image)



# Cleanup
cv2.destroyAllWindows()
