import time
from interaction import *
from segment_utils import segment_image, show_masks

selection_type = "point"


# Read image from user input
image_path = input("Enter the path of the image: ")

#  /Users/danielbosch/Downloads/tools.jpg
# E:\Projects\interactive-scene-segmentation\test\desk.jpg
# /home/maria/interactive-scene-segmentation/test/desk.jpg

image = cv2.imread(image_path)
baseImage = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Select the type of selection
selection_type = select_selection_type()

# Arrays for user selection
selected_points = []
selected_masks = []

# Create a named window and set mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback, param=(selection_type,selected_points))

start_time = time.time()
masks = segment_image(image)
masks = sorted(masks, key=lambda x: x['area'], reverse=False)
end_time = time.time()
print(f"image segmentation: {end_time - start_time} seconds")

# Display the image
while True:
    image = baseImage.copy()

    # draw the user selections on the canvas
    if selection_type == "point":
        for i in range(len(selected_points)):
            cv2.circle(image, selected_points[i], radius=3, color=(0, 0, 255), thickness=-1)
            selected_masks = select_masks(selected_points[len(selected_points)-1][0], selected_points[len(selected_points)-1][1], masks, selected_masks)

    if selection_type == "polygon":
        for i in range(len(selected_points) - 1):
            cv2.line(image, selected_points[i], selected_points[i + 1], (0, 255, 0), 2)

    if (selection_type == "area") & (len(selected_points) > 1 ):
        cv2.rectangle(image, selected_points[0], selected_points[len(selected_points) - 1], (255, 0, 0), 2, )

    if len(selected_masks) == 0:
        cv2.imshow("Image", image)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    if key != -1:  # Check if any key is pressed
        keyboard_callback(key)  # Call the keyboard callback function

    if len(selected_masks) > 0:
        image = show_masks(image, selected_masks)

    cv2.imshow("Image", image)

# Cleanup
cv2.destroyAllWindows()
