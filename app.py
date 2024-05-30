import requests
import streamlit as st
import pandas as pd
import os
import cv2
import time
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import predict_plate

# Define the path to the CSV file
csv_file_path = 'stolen_vehicles.csv'
detected_plates_csv = 'detected_plates.csv'


# Function to load the CSV file or create an empty one if it doesn't exist
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=['License Plate', 'Location', 'Time'])


# Function to save the data to the CSV file
def save_data(data, file_path):
    data.to_csv(file_path, index=False)


# Load existing data
data = load_data(csv_file_path)
detected_data = load_data(detected_plates_csv)

# Streamlit app
st.title('SAQR')

# Navigation
pages = ['Register License Plate', 'View Registered Vehicles', 'Remove License Plate', 'Live License Plate Detection',
         'View Detected Plates']
page = st.selectbox('Select a Page', pages)

if page == 'Register License Plate':
    # Form for entering license plate, location, and time
    with st.form('license_plate_form'):
        license_plate = st.text_input('Enter License Plate')
        location = st.text_input('Enter Location')
        time_of_detection = st.text_input('Enter Time of Detection')
        submitted = st.form_submit_button('Submit')

        if submitted:
            # Add the new license plate, location, and time to the data
            new_row = pd.DataFrame(
                {'License Plate': [license_plate], 'Location': [location], 'Time': [time_of_detection]})
            data = pd.concat([data, new_row], ignore_index=True)

            # Save the updated data to the CSV file
            save_data(data, csv_file_path)

            st.success(f'License plate {license_plate} added.')

elif page == 'View Registered Vehicles':
    # Display the current data
    st.subheader('Registered Illegal Vehicles')
    st.write(data)

elif page == 'Remove License Plate':
    # Form for selecting and removing a license plate
    st.subheader('Remove a License Plate')
    with st.form('remove_license_plate_form'):
        license_plate_to_remove = st.selectbox('Select License Plate to Remove', data['License Plate'].unique())
        remove_submitted = st.form_submit_button('Remove')

        if remove_submitted:
            # Remove the selected license plate from the data
            data = data[data['License Plate'] != license_plate_to_remove]

            # Save the updated data to the CSV file
            save_data(data, csv_file_path)

            st.success(f'License plate {license_plate_to_remove} removed.')
            st.write(data)

elif page == 'Live License Plate Detection':
    # Live license plate detection
    st.subheader('Live License Plate Detection')

    modelpath = 'detect.tflite'
    lblpath = 'labelmap.txt'
    min_conf = 0.85  # Adjusted minimum confidence to 0.85

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


    # Function to process a frame and detect license plates
    def process_frame(frame, detected_data):
        frame_resized = cv2.resize(frame, (width, height))
        if float_input:
            frame_resized = (np.float32(frame_resized) - input_mean) / input_std
        else:
            frame_resized = np.uint8(frame_resized)
        input_data = np.expand_dims(frame_resized, axis=0)

        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output data
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        for i in range(len(scores)):
            if (scores[i] > min_conf) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * frame.shape[0])))
                xmin = int(max(1, (boxes[i][1] * frame.shape[1])))
                ymax = int(min(frame.shape[0], (boxes[i][2] * frame.shape[0])))
                xmax = int(min(frame.shape[1], (boxes[i][3] * frame.shape[1])))

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Crop and process the detected plate
                cropped_plate = frame[ymin:ymax, xmin:xmax]
                plate_characters = predict_plate.get_license_plate(cropped_plate)
                detection_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

                # Save detected plate information
                detected_row = pd.DataFrame(
                    {'License Plate': [plate_characters], 'Location': ['Live Detection'], 'Time': [detection_time]})
                detected_data = pd.concat([detected_data, detected_row], ignore_index=True)
                save_data(detected_data, detected_plates_csv)

                # Check if the detected plate matches the registered stolen vehicles
                if plate_characters in data['License Plate'].values:
                    st.warning(f'Detected Illegal Vehicle: {plate_characters}')

                # Draw label
                label = f'{plate_characters}: {int(scores[i] * 100)}%'
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame, detected_data


    # Initialize session state for video capture
    if 'video_capture' not in st.session_state:
        st.session_state.video_capture = None
    if 'running' not in st.session_state:
        st.session_state.running = False


    # Function to start video capture
    def start_video():
        if st.session_state.video_capture is None or not st.session_state.video_capture.isOpened():
            st.session_state.video_capture = cv2.VideoCapture(0)
        st.session_state.running = True


    # Function to stop video capture
    def stop_video():
        if st.session_state.video_capture is not None and st.session_state.video_capture.isOpened():
            st.session_state.video_capture.release()
        st.session_state.running = False


    # Buttons for controlling video capture
    start_button = st.button('Start', on_click=start_video, key='start_button')
    stop_button = st.button('Stop', on_click=stop_video, key='stop_button')

    # Display the video stream in Streamlit
    frame_placeholder = st.empty()

    while st.session_state.running:
        ret, frame = st.session_state.video_capture.read()
        if not ret:
            st.error('Error: Could not read frame from video stream.')
            break

        # Process the frame for license plate detection
        frame, detected_data = process_frame(frame, detected_data)

        # Display the processed frame
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels='RGB')

        # Check if the stop button is pressed
        if not st.session_state.running:
            break

    cv2.destroyAllWindows()

    # Button for restarting the app
    if st.button('Restart', key='restart_button'):
        st.experimental_rerun()

elif page == 'View Detected Plates':
    st.subheader('View Detected Plates')
    detected_data['Flagged'] = detected_data['License Plate'].isin(data['License Plate']).apply(
        lambda x: 'Yes' if x else 'No')
    st.write(detected_data)

    if st.button('Clear Detected Plates'):
        detected_data = pd.DataFrame(columns=['License Plate', 'Location', 'Time'])
        save_data(detected_data, detected_plates_csv)
        st.success('Detected plates cleared.')
        st.write(detected_data)
