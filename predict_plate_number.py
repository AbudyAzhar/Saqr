import torch
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E', 'G', 'H', 'J', 'K', 'L', 'N', 'R', 'S', 'T', 'U', 'V', 'X', 'Z']

# Define the NMS function
def apply_nms(predictions, iou_threshold=0.7):
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=iou_threshold)
    if isinstance(indices, tuple) and len(indices) == 0:
        return []  # Return an empty list if no boxes are left after NMS
    else:
        return [predictions[i] for i in indices]

# Function to convert predictions to string
def predictions_to_string(predictions):
    # Sort predictions by 'x' coordinate to maintain left-to-right order
    sorted_predictions = predictions.sort_values(by='x', ascending=True).reset_index(drop=True)
    # Extract class indices
    class_indices = sorted_predictions['class'].astype(int).tolist() 
    # Map class indices to class names
    class_names_string = ''.join([class_names[i] for i in class_indices])
    
    return class_names_string

# Load the YOLOv8 model
#/////////////////////////////////////CHANGE PATH TO BEST MODEL//////////////////////////////////////////
model = YOLO('best.pt')

# Function to process an image and get the result string
def process_image(input):
    # Read the image
    if(type(input) == str):
        image = cv2.imread(input)
    else:
        image = input
    
    if image is None:
        raise ValueError(f"Image at path {input} could not be loaded.")
    
    # Run inference
    results = model(image)
    
    # Process results
    PBOX = pd.DataFrame(columns=range(6))
    for result in results:
        if len(result.boxes.data) > 0:
            raw_predictions = result.boxes.data.cpu().numpy().astype(float)
            filtered_predictions = apply_nms(raw_predictions, iou_threshold=0.7)
            arri = pd.DataFrame(filtered_predictions)
            if not arri.empty:
                PBOX = pd.concat([PBOX, arri], axis=0)
    
    # Add column names
    if not PBOX.empty:
        PBOX.columns = ['x', 'y', 'x2', 'y2', 'confidence', 'class']
        # Get the result string
        result_string = predictions_to_string(PBOX)
        return result_string
    else:
        return ""

# Example usage

def get_license_plate(input):
    result_string = process_image(input)
    return result_string
    
#path_to_plate = input("Enter the path of the image: ")
#get_license_plate(path_to_plate)
