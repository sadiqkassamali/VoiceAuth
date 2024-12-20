import cv2
import tensorflow as tf
import torch
import numpy as np
import os
import shutil
import uuid
import logging
from datetime import datetime
import json

# Initialize logging
logging.basicConfig(level=logging.INFO,  format="%(asctime)s - %(levelname)s - %(message)s",)

# Load YOLOv5 model for object detection (PyTorch)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load TensorFlow SSD model for object detection
ssd_model = tf.saved_model.load('ssd_mobilenet_v2_coco/saved_model')  # Ensure model is downloaded

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load MobileNetV2 for classification (TensorFlow)
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Load DeepFake Detection Model (assumed to be pre-trained)
# Placeholder for DeepFake detection model (adjust accordingly for actual model loading)
deepfake_model = tf.keras.models.load_model('path_to_deepfake_model')  # Replace with your model path


# Utility function to log video file details and predictions
def log_video_details(temp_file_path, file_format, file_size, audio_length, bitrate, result_text, selected_models):
    log_message = (
        f"File Path: {temp_file_path}\n"
        f"Format: {file_format}\n"
        f"Size (MB): {file_size:.2f}\n"
        f"Audio Length (s): {audio_length:.2f}\n"
        f"Bitrate (Mbps): {bitrate:.2f}\n"
        f"Result: {result_text}\n"
    )

    # Add DeepFake Detection prediction if selected
    try:
        if "DeepFake Detection" in selected_models:
            deepfake_is_fake, deepfake_confidence = detect_deepfake(temp_file_path)
            log_message += f"DeepFake Detection Prediction: {'Fake' if deepfake_is_fake else 'Real'} (Confidence: {deepfake_confidence:.2f})\n"
    except Exception as e:
        log_message += f"Error in DeepFake Detection model: {str(e)}\n"

    # Add Object Detection prediction using YOLOv5
    try:
        if "YOLOv5" in selected_models:
            yolo_results = detect_objects_yolo(temp_file_path)
            log_message += f"YOLOv5 Predictions: {yolo_results}\n"
    except Exception as e:
        log_message += f"Error in YOLOv5 model: {str(e)}\n"

    # Add Object Detection prediction using SSD
    try:
        if "SSD" in selected_models:
            ssd_results = detect_objects_ssd(temp_file_path)
            log_message += f"SSD Predictions: {ssd_results}\n"
    except Exception as e:
        log_message += f"Error in SSD model: {str(e)}\n"

    # Add Face Detection prediction using Haar Cascade
    try:
        if "Face Detection" in selected_models:
            face_results = detect_faces_haar(temp_file_path)
            log_message += f"Face Detection: Detected {len(face_results)} faces\n"
    except Exception as e:
        log_message += f"Error in Face Detection model: {str(e)}\n"

    # Add Image Classification prediction using MobileNetV2
    try:
        if "MobileNetV2" in selected_models:
            mobilenet_results = classify_image_mobilenet(temp_file_path)
            log_message += f"MobileNetV2 Classification: {mobilenet_results}\n"
    except Exception as e:
        log_message += f"Error in MobileNetV2 model: {str(e)}\n"

    # Log to the console or a file
    logging.info(log_message)
    return log_message


# Placeholder function for deepfake detection
def detect_deepfake(video_path, frame_skip=5):
    video_capture = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frames.append(frame)
        count += 1
    video_capture.release()

    # Preprocess frames and run deepfake detection as before
    frames_resized = cv2.resize(np.array(frames), (224, 224))  # Resize frames
    frames_resized = frames_resized / 255.0  # Normalize the pixel values
    prediction = deepfake_model.predict(frames_resized)
    deepfake_confidence = prediction[0]
    deepfake_is_fake = deepfake_confidence > 0.5
    return deepfake_is_fake, deepfake_confidence


# Object Detection using YOLOv5 (PyTorch)
def detect_objects_yolo(video_path):
    video_capture = cv2.VideoCapture(video_path)
    results = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        # Use YOLOv5 to detect objects in each frame
        yolo_results = yolo_model(frame)
        results.append(yolo_results.pandas().xywh[0])  # Collecting the results for each frame
    video_capture.release()
    return results


# Object Detection using SSD (TensorFlow)
def detect_objects_ssd(video_path):
    video_capture = cv2.VideoCapture(video_path)
    results = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        # Pre-process image for SSD model
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = ssd_model(input_tensor)
        results.append(detections)  # Collecting the results for each frame
    video_capture.release()
    return results


# Face Detection using Haar Cascade (OpenCV)
def detect_faces_haar(video_path):
    video_capture = cv2.VideoCapture(video_path)
    face_results = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_results.append(faces)
    video_capture.release()
    return face_results


# Image Classification using MobileNetV2 (TensorFlow)
def classify_image_mobilenet(video_path):
    video_capture = cv2.VideoCapture(video_path)
    results = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        img_array = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        predictions = mobilenet_model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
        results.append(decoded_predictions[0])
    video_capture.release()
    return results


# Utility function to handle video file analysis
def analyze_video(video_path, selected_models=["DeepFake Detection"]):
    try:
        # Dummy values for video properties, you can extract these using tools like ffmpeg
        file_format = "MP4"
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        audio_length = 120.0  # seconds (example)
        bitrate = 5.0  # Mbps (example)
        result_text = "Video processed successfully"

        # Log video details and model predictions
        log_message = log_video_details(video_path, file_format, file_size, audio_length, bitrate, result_text,
                                        selected_models)

        # Additional processing if needed (e.g., saving metadata)
        file_uuid = str(uuid.uuid4())
        save_metadata(file_uuid, video_path, result_text)

        # Example result label update or GUI interaction
        result_label = f"Analysis complete for {video_path}: {result_text}"
        logging.info(result_label)
        return log_message

    except Exception as e:
        logging.error(f"Error during video analysis: {str(e)}")
        return str(e)


# Save metadata function (example, adjust as necessary)
def save_metadata(file_uuid, video_path, result_text):
    metadata = {
        "file_uuid": file_uuid,
        "file_path": video_path,
        "result_text": result_text,
        "timestamp": str(datetime.now())
    }
    metadata_dir = 'metadata'
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_path = os.path.join(metadata_dir, f"{file_uuid}.json")
    with open(metadata_path, 'w', encoding="utf-8") as f:
        json.dump(metadata, f)


def get_video_length(video_path):
    """Get the duration of the video in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration


def extract_video_frames(video_path):
    """Extract frames from the video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
