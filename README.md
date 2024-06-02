
# Saqr - Real-time License Plate Recognition System

## Overview
Saqr is a real-time license plate recognition system designed to detect stolen or wanted cars. Utilizing a YOLOv8 model deployed on a Raspberry Pi with a camera, the system captures live video feeds and accurately recognizes license plates. This project supports both Arabic and English characters, making it particularly suitable for Saudi plates.

## Features
- **Real-time License Plate Detection**: Capture and process live video feeds to detect license plates.
- **Character Recognition**: Recognize license plates with both Arabic and English characters.
- **Data Management**: Register, view, and remove license plates from the database.
- **Detection Logging**: Log detected license plates with timestamp and location.
- **Web Interface**: User-friendly interface built with Streamlit for easy interaction and management.

## Requirements
### System Packages
- **libgl1**: OpenGL library used by OpenCV.

### Python Packages
- matplotlib==3.8.4
- numpy==1.24.3
- opencv_python_headless==4.9.0.80
- pandas==2.2.2
- Requests==2.32.2
- streamlit==1.35.0
- tensorflow==2.16.1
- torch==2.3.0
- ultralytics==8.2.22

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/Saqr.git
    cd Saqr
    ```

2. **Install system packages**:
    ```bash
    sudo apt-get install libgl1
    ```

3. **Create a virtual environment and activate it**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4. **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2. **Navigate through the web interface**:
    - **Register License Plate**: Enter the license plate, location, and detection time.
    - **View Registered Vehicles**: View the list of registered illegal vehicles.
    - **Remove License Plate**: Remove a license plate from the registered list.
    - **Live License Plate Detection**: Start real-time detection using the video feed.
    - **View Detected Plates**: View and manage detected plates.


## Data Files
- **detected.csv**: Logs detected license plates with columns: `plate_number`, `detection_time`, `location`.
- **detected_plates.csv**: Detailed detected plates log with columns: `License Plate`, `Location`, `Time`.
- **stolen_vehicles.csv**: Database of stolen vehicles with columns: `License Plate`, `Location`, and `Time`.
