import time

import cv2
import tkinter as tk
from tkinter import messagebox

# Global variables
import numpy as np
from PIL import Image, ImageTk

from interaction import *
from segment_utils import segment_image, show_anns

selection_type = "point"
selected_points = []
selected_rect = None
drag_start = None
drag_end = None
drawing = False
selected_masks = []
mask_color = dict()
polygon_closed = False


global image

def segment_finer(root):
    print("segment finer")
    root.destroy()

def segment_coarser(root):
    print("segment coarser")
    root.destroy()

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global selected_points, selected_rect, drag_start, drawing, drag_end, selected_masks, polygon_closed

    if selection_type == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_masks = point_selection(event, selected_points, x, y, image, masks)

    elif selection_type == "area":
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drag_start = (x, y)
            selected_rect = (x, y, 0, 0)
        if drawing is True:
            drag_end = (x, y)
            selected_rect = (selected_rect[0], selected_rect[1], x - selected_rect[0], y - selected_rect[1])
        if event == cv2.EVENT_LBUTTONUP:
            drawing = False
            drag_end = (x, y)
            selected_rect = (selected_rect[0], selected_rect[1], x - selected_rect[0], y - selected_rect[1])

    elif selection_type == "polygon":
        if event == cv2.EVENT_LBUTTONDOWN:
            if polygon_closed is False:

                if len(selected_points) > 2 and np.linalg.norm(np.array(selected_points[0]) - np.array((x, y))) < 25:
                    new_point = selected_points[0]
                    polygon_closed = True
                else:
                    new_point = (x, y)
                selected_points.append(new_point)


        if event == cv2.EVENT_RBUTTONDOWN:
            selected_points.pop()
            polygon_closed = False

    if event == ord('q'):
        cv2.destroyAllWindows()
        exit(0)

    if event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button click event
        menu = tk.Tk()
        menu.title("Menu")
        menu.rowconfigure(0, minsize=50, weight=1)
        menu.columnconfigure([0, 1], minsize=50, weight=1)
        btn_finer = tk.Button(master=menu, text="segment finer", command=lambda:segment_finer(menu))
        btn_finer.grid(row=0, column=0, sticky="nsew")
        btn_coarser = tk.Button(master=menu, text="segment coarser", command=lambda:segment_coarser(menu))
        btn_coarser.grid(row=0, column=1, sticky="nsew")
        menu.attributes('-topmost', True)
        menu.mainloop()

# Function to handle menu selection
def select_selection_type():
    global selection_type
    while True:
        print("Select a selection type:")
        print("1. Point")
        print("2. Area")
        print("3. Polygon")
        choice = input("Enter your choice (1/2): ")
        if choice == "1":
            selection_type = "point"
            break
        elif choice == "2":
            selection_type = "area"
            break
        elif choice == "3":
            selection_type = "polygon"
            print("Press Left Mouse Button to add a point. Press Right Mouse Button to remove a point.")
            break
        else:
            print("Invalid choice. Try again.\n")


# Read image from user input
image_path = input("Enter the path of the image: ")
#  /Users/danielbosch/Downloads/tools.jpg
# E:\Projects\interactive-scene-segmentation\test\desk.jpg
#/home/maria/interactive-scene-segmentation/test/desk.jpg
image = cv2.imread(image_path)
baseImage = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a named window and set mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)


# Select the type of selection
select_selection_type()

start_time = time.time()
# masks = segment_image(image)
# masks = sorted(masks, key=lambda x: x['area'], reverse=False)
end_time = time.time()
print(f"image segmentation: {end_time - start_time} seconds")

# Display the image
while True:
    image = baseImage.copy()

    if selection_type == "polygon":
        for i in range(len(selected_points) - 1):
            cv2.line(image, selected_points[i], selected_points[i + 1], (0, 255, 0), 2)


    if selection_type == "area" and drawing is True:
        cv2.rectangle(image, drag_start, drag_end, (255, 0, 0), 2, )


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
