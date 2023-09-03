import argparse

from interaction import *
from segment_utils import SegmentAnything
from interactive_learning import cnn
from skimage.filters.rank import entropy

#  /Users/danielbosch/Downloads/tools.jpg
# E:\Projects\interactive-scene-segmentation\test\desk.jpg
# /home/maria/interactive-scene-segmentation/test/desk.jpg

def main(directory: str, selection_type: str):
    # Select the type of selection
    images, file_names = load_images(directory)
    # Arrays for user selection
    selected_points = []
    selected_masks = []
    masks = []
    masked_image = None
    base_image = None
    name = None

    # Create a named window and set mouse callback
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", mouse_callback, param=(selection_type, selected_points))

    # Create the segmentation model
    sam = SegmentAnything(checkpoint="trained_models/sam_vit_b_01ec64.pth", model_type="vit_b", device="cuda")
    model_parameters = sam.parameters

    interactive_trainer = cnn.CNNTrainer()

    # Display the image
    while True:
        # Load the base image and the masks
        if base_image is None:
            # load the next image
            try:
                base_image = images.pop(0)
                name = file_names.pop(0)
                masks = sam.segment_image(base_image)
                masked_image = sam.show_masks(base_image, masks)
            except Exception as e:
                print(e)
                print("Last image in directory, training is over")
                cv2.destroyAllWindows()
                exit(0)
            # show the masked image
            temp_image = masked_image.copy()
        else:
            # show the masked image
            temp_image = masked_image.copy()

        # draw the user selections on the canvas
        if selection_type == "point":
            for i in range(len(selected_points)):
                cv2.circle(temp_image, selected_points[i], radius=3, color=(0, 0, 255), thickness=-1)
                selected_masks = select_masks(selected_points[len(selected_points) - 1][0],
                                              selected_points[len(selected_points) - 1][1], masks, selected_masks)

        if selection_type == "polygon":
            for i in range(len(selected_points) - 1):
                cv2.line(temp_image, selected_points[i], selected_points[i + 1], (0, 255, 0), 2)

        if (selection_type == "area") & (len(selected_points) > 1):
            cv2.rectangle(temp_image, selected_points[0], selected_points[len(selected_points) - 1], (255, 0, 0), 2, )

        # Wait for a key press
        key = cv2.waitKey(1)
        if key != -1:  # Check if any key is pressed
            param = keyboard_callback(key, param=(sam, base_image, masked_image, masks, selection_type, selected_points,
                                                  selected_masks, model_parameters,
                                                  interactive_trainer, name))  # Call the keyboard callback function
            _, base_image, masked_image, masks, selection_type, selected_points, selected_masks, model_parametersm, _n, _ = param
            # reset mouse callback with new interaction type
            cv2.setMouseCallback("Image", mouse_callback, param=(selection_type, selected_points))

        cv2.imshow("Image", temp_image)

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Scene Segmentation")
    parser.add_argument("--dir", type=str, default=os.getcwd() + "/train/images/",
                        help="The directory with the training images")
    parser.add_argument("--interaction", type=str, default="point",
                        help="The interaction type: point, area or polygon")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print("The given directory: ", args.dir, " does not exist.")
        exit(0)

    main(directory=args.dir, selection_type=args.interaction)