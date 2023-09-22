import argparse

from interaction import *
from segment_utils import SegmentAnything
from interactive_learning import cnn


#  /Users/danielbosch/Downloads/tools.jpg
# E:\Projects\interactive-scene-segmentation\test\desk.jpg
# /home/maria/interactive-scene-segmentation/test/desk.jpg
def write_number_to_file(filename, name, number):
    with open(filename, 'a') as file:
        file.write(name + "," + str(number) + "\n")


def load_images(directory: str):
    # Read image from user input
    images = []
    # directory exists
    while not images:
        for filename in sorted(os.listdir(directory)):
            img = cv2.imread(os.path.join(directory, filename))
            image = ImageContainer(image=img, file_name=filename)
            images.append(image)
    print("Loaded {} images".format(len(images)))
    return images


def main(directory: str, selec_t: str, model: str):
    # Create the segmentation model
    sam = SegmentAnything(checkpoint=model, model_type="vit_b", device="cuda")
    interactive_trainer = cnn.CNNTrainer()
    # Load the images
    images = load_images(directory)

    # Create the first image
    curr_img = images.pop(0)
    curr_img.set_masks(sam.segment_image(curr_img.image))
    curr_img.set_masked_image(sam.show_masks(curr_img.image, curr_img.masks))
    curr_img.set_selection_type(selec_t)
    curr_img.set_model_parameters(sam.parameters)

    training_set = []
    new = False

    # Check if the folder already exists
    if not os.path.exists(os.path.join(os.getcwd() + "/labeled_images/")):
        # If it doesn't exist, create the folder
        os.makedirs(os.path.join(os.getcwd() + "/labeled_images/"))

    # Create a named window and set mouse callback
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", mouse_callback, param=(curr_img.selection_type, curr_img.selected_points))

    # Display the image
    while True:
        # Load the base image and the masks
        if new is True:
            # load the next image
            # interactive_trainer.update(image_resized, curr_img.image)
            curr_img.set_model_parameters(sam.parameters)
            # save the selection type to transfer it to the new image
            selec_t = curr_img.selection_type
            training_set.append(curr_img)
            # save the image and the labels
            cv2.imwrite(os.path.join(os.getcwd() + "/labeled_images/" + curr_img.file_name), curr_img.image)
            write_number_to_file('labeled_images/label.txt', curr_img.file_name, curr_img.model_parameters[0])
            try:
                print(len(images), " images remaining")
                curr_img = images.pop(0)
                curr_img.set_masks(sam.segment_image(curr_img.image))
                curr_img.set_masked_image(sam.show_masks(curr_img.image, curr_img.masks))
                # set the selection type to the one of the previous image
                curr_img.set_selection_type(selec_t)
                new = False
            except:
                print("Last image in directory, training is over")
                cv2.destroyAllWindows()
                exit(0)

        # show the masked image
        temp_image = curr_img.masked_image.copy()

        # draw the user selections on the canvas
        if curr_img.selection_type == "point":
            for i in range(len(curr_img.selected_points)):
                cv2.circle(temp_image, curr_img.selected_points[i], radius=3, color=(0, 0, 255), thickness=-1)
                curr_img.set_selected_masks(select_masks(curr_img.selected_points[len(curr_img.selected_points) - 1][0],
                                                         curr_img.selected_points[len(curr_img.selected_points) - 1][1],
                                                         curr_img.masks, curr_img.selected_masks))
        # draw the user selections on the canvas
        if curr_img.selection_type == "polygon":
            for i in range(len(curr_img.selected_points) - 1):
                cv2.line(temp_image, curr_img.selected_points[i],
                         curr_img.selected_points[i + 1], (0, 255, 0), 2)

        # draw the user selections on the canvas
        if (curr_img.selection_type == "area") & (len(curr_img.selected_points) > 1):
            cv2.rectangle(temp_image, curr_img.selected_points[0],
                          curr_img.selected_points[len(curr_img.selected_points) - 1],
                          (255, 0, 0), 2, )

        # Wait for a key press
        key = cv2.waitKey(1)
        if key != -1:  # Check if any key is pressed
            param = keyboard_callback(key, param=(sam, curr_img, new))  # Call the keyboard callback function
            _, curr_img, new = param

        # set mouse callback
        cv2.setMouseCallback("Image", mouse_callback, param=(curr_img.selection_type, curr_img.selected_points))

        # show image
        cv2.imshow("Image", temp_image)

    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Train the model, adjust the following parameters as needed
    """
    parser = argparse.ArgumentParser(description="Interactive Scene Segmentation")
    parser.add_argument("--dir", type=str, default=os.getcwd() + "/train/coco/test2014", help="The directory with the training images")
    parser.add_argument("--model", type=str, default=os.getcwd() + "/trained_models/sam_vit_b_01ec64.pth",
                        help="Path to the SAM model")
    parser.add_argument("--interaction", type=str, default="point",
                        help="The interaction type: point, area or polygon")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print("The given directory: ", args.dir, " does not exist.")
        exit(0)

    main(directory=args.dir, selec_t=args.interaction, model=args.model)
