!pip install ultralytics
from ultralytics import YOLO
from IPython.display import display
import matplotlib.pyplot as plt
# Load the best trained model from the training run
model = YOLO('best.pt')
results = model('chess_image_here')
from IPython.display import display
import matplotlib.pyplot as plt



# ... your existing code ...
results = model('8-6P1-8-7r-2k5-R7-2p2K2-8.png')
result = results[0]
annotated_image = result.plot()  # Get the annotated image as a NumPy array
display(plt.imshow(annotated_image))  # Display the image in the notebook
from PIL import Image
import numpy as np

def crop_detected_objects(image_path, results):
    """
    Crops detected objects from an image based on bounding box coordinates.

    Args:
        image_path (str): Path to the original image
        results: Detection results containing bounding boxes

    Returns:
        list: List of cropped image arrays
    """
    # Load the original image
    original_image = Image.open(image_path)

    cropped_images = []

    # Assuming results[0] contains detections with .boxes attribute
    # Modify according to your model's output format
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format

    # Crop each detected object
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])  # Convert coordinates to integers

        # Crop the region
        cropped = original_image.crop((x1, y1, x2, y2))
        cropped_images.append(np.array(cropped))

    return cropped_images

# Example usage:
image_path = 'crop_image.JPG'
results = model(image_path)  # Your model inference
cropped_objects = crop_detected_objects(image_path, results)

# Display or save the cropped images
for i, cropped in enumerate(cropped_objects):
    # Display
    plt.figure(figsize=(5, 5))
    plt.imshow(cropped)
    plt.axis('off')
    plt.show()

    # Optionally save the cropped images
    Image.fromarray(cropped).save(f'cropped_object_{i}.jpg')
