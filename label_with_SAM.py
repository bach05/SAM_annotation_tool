import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import os
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False, wrong=False, correct=False, empty=False):
    if correct:
        color = np.array([0 / 255, 0 / 255, 128 / 255, 0.6])
    elif wrong:
        color = np.array([128 / 255, 0 / 255, 0 / 255, 0.6])
    elif empty:
        color = np.array([0 / 255, 128 / 255, 0 / 255, 0.6])
    elif random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(image,(x,y),5,(255,0,0), -1)
        mouseX,mouseY = x,y


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


class ImageProcessor:
    def __init__(self, image, sam, label_info):
        self.input_point = []
        self.input_label = []
        self.input_boxes = []
        self.color_list = []
        self.image = image
        self.show_image = image.copy()
        self.mask = None
        self.label_info = label_info
        self.label_ids = list(label_info.keys())

        self.mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=100)
        self.predictor = SamPredictor(sam)
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.selected_class = None

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if y < 30:  # Check if the click is within the button area for class selection
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
                self.input_point.append((x, y-30))
                self.input_label.append(1)
                self.color_list.append((0, 0, 255))
                print(f"[POSITIVE] Point clicked at ({x}, {y-30})")
                self.draw_points()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.input_point.append((x, y-30))
            self.input_label.append(0)
            self.color_list.append((255, 0, 0))
            print(f"[NEGATIVE] Point clicked at ({x}, {y-30})")
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
        if self.mask is None:
            self.mask = np.where(mask, class_id, np.zeros_like(mask))
        else:
            self.mask[mask] = class_id
        print("Updated mask!")

    def draw_points(self):
        for i,point in enumerate(self.input_point):
            color = self.color_list[i]
            cv2.circle(self.show_image, point, 7, (255,255,255), -1)
            cv2.circle(self.show_image, point, 5, color, -1)

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
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.click_event)

        # Calculate the height for each button
        button_height = 30
        button_width = self.image.shape[1] // len(self.label_info)

        # Create an image to display buttons and the original image
        button_image = np.zeros((button_height, self.image.shape[1], 3), dtype=np.uint8)

        # Create buttons in the upper area for each entry in label_info
        for i, (class_id, info) in enumerate(self.label_info.items()):
            button_x_start = i * button_width
            button_x_end = (i + 1) * button_width
            cv2.rectangle(button_image, (button_x_start, 0), (button_x_end, button_height), info['color'], -1)
            cv2.putText(button_image, f'{class_id}: {info["name"]}', (button_x_start + 5, button_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


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
                if self.mask is None:
                    print("No mask set")
                    break
                else:
                    self.mask = self.mask.astype(np.uint8)
                    break

        cv2.destroyAllWindows()

        return self.image, self.mask




if __name__ == "__main__":

    sam = sam_model_registry["vit_l"](checkpoint="./sam_vit_l_0b3195.pth").cuda()

    # label_info = {
    #     0:{"name":"background", "color":(0, 0, 255)},
    #     1:{"name":"big_object",  "color":(0, 255, 0)},
    #     2:{"name":"small_object", "color":(255, 0, 0)},
    #     3:{"name":"long_object", "color":(0, 255, 255)},
    #     4:{"name":"flat_object", "color":(255, 255, 0)}
    # }

    label_info = {
        0:{"name":"background", "color":(0, 0, 255)},
        1:{"name":"dog",  "color":(0, 255, 0)},
        2:{"name":"cat", "color":(255, 0, 0)},
    }

    raw_data = "example"
    out_path = "dataset_example"
    img_out_path = os.path.join(out_path, "imgs")
    label_out_path = os.path.join(out_path, "labels")

    # Check if the folder exists
    if not os.path.exists(out_path):
        # If it doesn't exist, create it
        os.mkdir(out_path)
        os.mkdir(img_out_path)
        os.mkdir(label_out_path)
        print(f"Folder '{out_path}' created successfully.")
    else:
        print(f"Folder '{out_path}' already exists.")

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

    print("Image Processor Usage Instructions:")
    print("1. Click top buttons to select a label.")
    print("2a. Click on the image to set points of interest. Left click on target object (positive point), right click on background (negative point)")
    print("2b. Press 'a' to add a new object to the current mask")
    print("3. Press 's' to process the image and display the annotated mask.")
    print("4. Repeat step 2 and 3 until you are satified.")
    print("4. Press 'q' to save and proceed to the next image.")
    print()

    # Now you have a list of image file paths that you can open and process one by one
    for idx, image_path in enumerate(image_paths):
        print(f"Processing image: {image_path}, {idx+1} of {len(image_paths)}")

        out_name = image_path.split("/")[-1].split(".")[0]
        image = cv2.imread(image_path)
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
        cv2.waitKey(10000)  # Wait for 1 second (1000 milliseconds)
        cv2.destroyAllWindows()






