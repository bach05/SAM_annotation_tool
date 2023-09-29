import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import os
import matplotlib.pyplot as plt
import json

class ImageProcessor:
    def __init__(self, image, sam, label_info):
        self.input_point = []
        self.input_label = []
        self.input_boxes = []
        self.color_list = []
        self.image = image
        self.show_image = image.copy()
        self.mask = []
        self.label_info = label_info
        self.label_ids = list(label_info.keys())
        for i in self.label_ids:
            self.mask.append(np.zeros_like(image[:,:,0]))

        self.mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=int((image.shape[1] * image.shape[0])*0.0002))
        self.predictor = SamPredictor(sam)
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.selected_class = None
        self.toggle_new_obj = True

        # Fixed size for the display window
        self.fixed_window_size = (1000, 800)
        self.button_height = image.shape[1] // 10
        self.point_size = int((max(image.shape[1], image.shape[0]))*0.005)

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < self.button_height:  # Check if the click is within the button area for class selection
                # Calculate the class ID based on the x-coordinate of the click
                button_width = self.image.shape[1] // len(self.label_info)
                selected_class = x // button_width   # Adding 1 to match your class IDs

                # Check if the selected class exists in label_info
                if selected_class in self.label_info:
                    self.selected_class = selected_class
                    print(f"Selected class {selected_class}: {self.label_info[selected_class]['name']}")
                else:
                    print("Invalid class selection")
            else:
                # Add a positive point if the click is outside the button area
                self.input_point.append((x, y-self.button_height))
                self.input_label.append(1)
                self.color_list.append((0, 0, 255))
                print(f"[POSITIVE] Point clicked at ({x}, {y-self.button_height})")
                self.draw_points()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.input_point.append((x, y-self.button_height))
            self.input_label.append(0)
            self.color_list.append((255, 0, 0))
            print(f"[NEGATIVE] Point clicked at ({x}, {y-self.button_height})")
            self.draw_points()

    def show_mask(self, mask, class_id=None):

        class_id = self.selected_class

        if class_id is not None:
            color = self.label_info[class_id]['color']
        else:
            color = np.array([0, 0, 128], dtype=np.uint8)  # Default color (red)

        # Create a mask image with the specified color for true pixels
        # equal color where mask, else image
        # this would paint your object silhouette entirely with `color`
        masked_img = np.where(mask[..., None], color, self.image).astype(np.uint8)

        # use `addWeighted` to blend the two images
        # the object will be tinted toward `color`
        self.show_image = cv2.addWeighted(self.image, 0.6, masked_img, 0.4, 0)
        self.mask[class_id] = np.where(mask, class_id, np.zeros_like(mask))
        print("Updated mask!")

    def draw_points(self):
        for i,point in enumerate(self.input_point):
            color = self.color_list[i]
            cv2.circle(self.show_image, point, self.point_size+int(max(self.point_size*0.1, 2)), (255,255,255), -1)
            cv2.circle(self.show_image, point, self.point_size, color, -1)

    def execute_prediction(self):
        if len(self.input_label) > 0:
            if len(self.input_boxes) > 0:
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(self.input_point),
                    point_labels=np.array(self.input_label),
                    box=np.array(self.input_boxes),
                    multimask_output=False,
                )
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(self.input_point),
                    point_labels=np.array(self.input_label),
                    multimask_output=False,
                )
        else:
            masks, scores, logits = self.predictor.predict(
                box=np.array(self.input_boxes),
                multimask_output=False,
            )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            self.show_mask(mask)

    def enter_class_id(self):
        class_id = input("Enter class number ID: ")
        return int(class_id)

    def process_image(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.resizeWindow('image', self.fixed_window_size[0], self.fixed_window_size[1])  # Set fixed window size
        cv2.setMouseCallback('image', self.click_event)

        # Calculate the height for each button
        button_height = self.button_height
        button_width = self.image.shape[1] // len(self.label_info)

        # Calculate text size and thickness based on button height
        font_scale = button_width / 300  # You can adjust the divisor to control text size
        thickness = max(int(2*font_scale), 1)  # Ensure thickness is at least 1

        # Create an image to display buttons and the original image
        button_image = np.zeros((button_height, self.image.shape[1], 3), dtype=np.uint8)

        # Create buttons in the upper area for each entry in label_info
        for i, (class_id, info) in enumerate(self.label_info.items()):
            button_x_start = i * button_width
            button_x_end = (i + 1) * button_width
            cv2.rectangle(button_image, (button_x_start, 0), (button_x_end, button_height), info['color'], -1)
            cv2.putText(button_image, f'{class_id}: {info["name"]}', (button_x_start + 5, button_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)


        while True:
            # instructions = f"[current label {self.selected_class}] 's'=show temporary mask, 'q'=save and proceed"
            # cv2.putText(self.show_image, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            combined_image = np.vstack((button_image, self.show_image))
            cv2.imshow('image', combined_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Execute the prediction function with the stored inputs
                self.execute_prediction()
                cv2.imshow('image', combined_image)  # Display the updated image
            elif key == ord('a'):
                print("Adding a new object")
                self.selected_class = None
                self.input_point = []
                self.input_label = []
            elif key == ord('q'):
                self.mask = np.array(self.mask)
                if self.mask.sum()==0:
                    print("No mask set")
                    self.mask = None
                    break
                else:
                    self.mask = self.mask.max(axis=0)
                    self.mask = np.clip(self.mask, 0, len(self.label_ids)-1)
                    self.mask = self.mask.astype(np.uint8)
                    break

        cv2.destroyAllWindows()

        return self.image, self.mask



if __name__ == "__main__":

    # Read configuration from JSON file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    sam_model_name = config["sam_model"]["name"]
    checkpoint_path = config["sam_model"]["checkpoint_path"]
    sam = sam_model_registry[sam_model_name](checkpoint=checkpoint_path).cuda()

    # Convert string keys to integers for label_info dictionary
    label_info = {int(key): value for key, value in config["label_info"].items()}

    raw_data = config["raw_data_path"]
    out_path = config["output_path"]["root"]
    img_out_path = config["output_path"]["img_subpath"]
    label_out_path = config["output_path"]["label_subpath"]

    max_dimension = int(config["max_image_dimension"])

    # Get a list of all files in the folder
    files = os.listdir(raw_data)

    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    # Create a list to store the paths of image files
    image_paths = []

    # Iterate through the files and add image file paths to the list
    for file in files:
        # Check if the file has a valid image extension
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(raw_data, file))

    # Check if the folder exists
    if not os.path.exists(out_path):
        # If it doesn't exist, create it
        os.mkdir(out_path)
        os.mkdir(img_out_path)
        os.mkdir(label_out_path)
        print(f"Folder '{out_path}' created successfully.")
        index = 0
    else:
        print(f"Folder '{out_path}' already exists.")
        # Ask the user if they want to resume a previous labeling
        resume_labeling = input("Do you want to resume a previous labeling? (y/n) ").strip().lower()

        if resume_labeling == 'y':
            index = int(input("Please insert the index of the last labelled image: ").strip())
            print(f"Restarting from image {image_paths[index+1]}")
        else:
            index = 0
            print("Starting a new labeling session.")
            print("\033[91m***[WARNING]*** CURRENT LABELS WILL BE OVERWRITTEN.\033[0m")

    print("Image Processor Usage Instructions:")
    print("1. Click top buttons to select a label.")
    print("2a. Click on the image to set points of interest. Left click on target object (positive point), right click on background (negative point)")
    print("2b. Press 'a' to add a new object to the current mask")
    print("3. Press 's' to process the image and display the annotated mask.")
    print("4. Repeat step 2 and 3 until you are satified.")
    print("4. Press 'q' to save and proceed to the next image.")
    print()

    # Now you have a list of image file paths that you can open and process one by one
    for idx, image_path in enumerate(image_paths[index+1:]):
        print(f"Processing image: {image_path}, {idx+1} of {len(image_paths)}")

        out_name = image_path.split("/")[-1].split(".")[0]
        image = cv2.imread(image_path)

        # Resize the image if either dimension is larger than 1024
        if image.shape[0] > max_dimension or image.shape[1] > max_dimension:
            # Calculate new dimensions while maintaining aspect ratio
            aspect_ratio = image.shape[1] / image.shape[0]
            new_height = min(max_dimension, int(max_dimension / aspect_ratio))
            new_width = min(max_dimension, int(aspect_ratio * max_dimension))
            image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

        img_proc = ImageProcessor(image, sam, label_info)
        image, mask = img_proc.process_image()

        if mask is None:
            continue

        # Save the data
        image_filename = out_name+".png"  # Specify the filename for the image
        image_save_path = os.path.join(img_out_path, image_filename)
        cv2.imwrite(image_save_path, image)
        label_save_path = os.path.join(label_out_path, image_filename)
        cv2.imwrite(label_save_path, mask)

        # Concatenate the image and mask horizontally
        combined_image = np.hstack((image, cv2.cvtColor(mask*25, cv2.COLOR_GRAY2BGR)))

        # Display the combined image in a single window
        cv2.imshow("SAVED Image and Mask", combined_image)
        cv2.waitKey(1000)  # Wait for 1 second (1000 milliseconds)
        cv2.destroyAllWindows()






