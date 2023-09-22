
class ImageContainer:
    def __init__(self, image, file_name):
        self.image = image
        self.file_name = file_name
        self.masks = None
        self.masked_image = None
        self.model_parameters = []
        self.selected_points = []
        self.selected_masks = []
        self.selection_type = None

    def set_image(self, image):
        self.image = image

    def set_file_name(self, file_name):
        self.file_name = file_name

    def set_masks(self, masks):
        self.masks = masks

    def set_masked_image(self, masked_image):
        self.masked_image = masked_image

    def set_selected_masks(self, selected_masks):
        self.selected_masks = selected_masks

    def set_model_parameters(self, model_parameters):
        self.model_parameters = model_parameters

    def set_selection_type(self, selection_type):
        self.selection_type = selection_type

    def set_selected_points(self, selected_points):
        self.selected_points = selected_points


