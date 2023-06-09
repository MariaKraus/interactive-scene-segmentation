import cv2

# Global variables
selection_type = "point"
selected_points = []
selected_rect = None
start_point = None
end_point = None
drawing = False

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global selected_points, selected_rect, start_point, drawing, end_point

    if selection_type == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
            print(f"Selected point: ({x}, {y})")

    elif selection_type == "area":
        draw_rect(event, x, y, flags, param)

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
image = cv2.imread(image_path)

# Create a named window and set mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Select the type of selection
select_selection_type()

# Display the image
while True:
    if selection_type == "point":
        # Draw selected points on the image
        for point in selected_points:
            cv2.circle(image, point, 3, (0, 0, 255), -1)
    elif selection_type == "area" and selected_rect is not None:
        temp_image = image.copy()
        cv2.rectangle(temp_image, start_point, end_point, (0, 0, 255), 2)
        cv2.imshow("Image", temp_image)

    cv2.imshow("Image", image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
