import cv2
import numpy as np

selected_masks = []

def point_selection(event, selected_points, x, y, image, masks):
    selected_mask = None
    selected_points.append((x, y))
    print(f"Selected point: ({x}, {y})")

    for mask in masks:
        if mask['segmentation'][y][x] == 1:
            if mask in selected_masks:
                selected_masks.remove(mask)
            else:
                selected_mask = mask
                selected_mask['color'] = list(np.random.choice(range(256), size=3))
                selected_masks.append(selected_mask)
            break

    return selected_masks





def draw_rect(event, x, y, flags, param):
    global x1, y1, drawing, radius, num, img, img2, end_point
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
        cv2.rectangle(img, (x1, y1), (x1, y1), (255, 0, 0), 1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            a, b = x, y
            if a != x & b != y:
                img = img2.copy()
                end_point = (a, b)
                cv2.rectangle(img, (x1, y1), end_point, (255, 0, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.rectangle(img, (x1, y1), end_point, (255, 0, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '_'.join(['label', str(num)]), (x + 20, y + 20), font, 1, (200, 255, 155), 1, cv2.LINE_AA)
        img2 = img.copy()