import cv2
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox


import numpy as np
from PIL import Image, ImageTk

from segment_utils import segment_finer, segment_coarser

drawing = False
polygon_closed = False

# Function to handle menu selection
def select_selection_type():

    while True:
        print("Select a selection type:")
        print("1. Point")
        print("2. Area")
        print("3. Polygon")
        choice = input("Enter your choice (1/2/3): ")
        if choice == "1":
            return "point"
            break
        elif choice == "2":
            return "area"
            break
        elif choice == "3":
            print("Press Left Mouse Button to add a point. Press Right Mouse Button to remove a point.")
            return "polygon"
            break
        else:
            print("Invalid choice. Try again.\n")

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    (selection_type, selected_points) = param
    global drawing

    if selection_type == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x,y))
            return selected_points
        if event == cv2.EVENT_RBUTTONDOWN:
            selected_points.pop()

    elif selection_type == "area":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.clear()
            drawing = True
            selected_points.append((x,y))
            return selected_points
        if event == cv2.EVENT_LBUTTONUP:
            if drawing is True:
                selected_points.append((x,y))
                drawing = False
                return selected_points
        if drawing is True:
            selected_points.append((x,y))
            print(selected_points)
            return selected_points


    elif selection_type == "polygon":
        global polygon_closed
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




# Function to handle keyboard events
def keyboard_callback(event):
    if event == 13:  # Check if the key is the "Enter" key (key code 13)
        menu = tk.Tk()
        menu.title("Menu")
        menu.rowconfigure(0, minsize=50, weight=1)
        menu.columnconfigure([0, 1], minsize=50, weight=1)
        btn_finer = tk.Button(master=menu, text="segment finer", command=lambda: segment_finer(menu))
        btn_finer.grid(row=0, column=0, sticky="nsew")
        btn_coarser = tk.Button(master=menu, text="segment coarser", command=lambda: segment_coarser(menu))
        btn_coarser.grid(row=0, column=1, sticky="nsew")
        menu.attributes('-topmost', True)
        menu.mainloop()

    if event == ord('q'):
        cv2.destroyAllWindows()
        exit(0)


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
                remove_mask(mask,selected_masks)
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

