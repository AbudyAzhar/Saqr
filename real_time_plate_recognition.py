import numpy as np
import pandas as pd
import predict_plate  # Ensure this module is accessible
import cv2
import os
import time
from tensorflow.lite.python.interpreter import Interpreter
from matplotlib import pyplot as plt

# Set paths and parameters
modelpath = 'detect.tflite'
lblpath = 'labelmap.txt'
min_conf = 0.85  # Adjusted minimum confidence to 0.85
cap = cv2.VideoCapture("demo_4.mp4")
crop_output_dir = 'cropped_plates'
os.makedirs(crop_output_dir, exist_ok=True)

# Load the TFLite model and allocate tensors
interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
float_input = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Load label map
with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
# Create a CSV file with headers if it doesn't exist
csv_file = 'detected.csv'
if not os.path.isfile(csv_file):
    df = pd.DataFrame(columns=['plate_number', 'detection_time', 'location'])
    df.to_csv(csv_file, index=False)

# Function to crop the English part of the license plate
def crop_english_part(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    middle_line = None
    if lines is not None:
        middle_y = image.shape[0] // 2
        min_distance = float('inf')
        for line in lines:
            _, y1, _, y2 = line[0]
            line_middle_y = (y1 + y2) // 2
            distance = abs(line_middle_y - middle_y)
            if distance < min_distance:
                min_distance = distance
                middle_line = line

    if middle_line is not None:
        _, y1, _, y2 = middle_line[0]
        cropped_image = image[y2:, :]
        return cropped_image
    else:
        print("No middle horizontal line detected.")
        return image  # Return the original image if no line is found


# Function to crop and return the license plate
def crop_and_return(image, box):
    ymin, xmin, ymax, xmax = box
    cropped_image = image[ymin:ymax, xmin:xmax]
    english_cropped_image = crop_english_part(cropped_image)
    return english_cropped_image


index = 0
last_frame_had_plate = False
last_process_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence scores

    current_frame_has_plate = False

    for i in range(len(scores)):
        if (scores[i] > min_conf) and (scores[i] <= 1.0):
            current_frame_has_plate = True
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                          (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Check if it's time to process the frame
            if current_time - last_process_time >= 0.05:
                cropped_plate = crop_and_return(frame, (ymin, xmin, ymax, xmax))
                plate_characters = predict_plate.get_license_plate(cropped_plate)

                # Check if the detected plate matches the specified patterns
                if plate_characters in ["4565KAA", "4552KAA", "7158GGA", "8485LTR"]:
                    print(f'!!!!!!!Detected Plate: {plate_characters}')
                    # Append the detected plate, time, and location to the CSV file
                    detection_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                    df = pd.DataFrame([[plate_characters, detection_time, 'Tuwaiq Academy']],
                                      columns=['plate_number', 'detection_time', 'location'])
                    df.to_csv(csv_file, mode='a', header=False, index=False)

                last_process_time = current_time  # Update the last process time
                index += 1

    # Update last_frame_had_plate
    last_frame_had_plate = current_frame_has_plate

    cv2.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()