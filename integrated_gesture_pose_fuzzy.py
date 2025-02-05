#!/usr/bin/env python3
"""
integrated_gesture_pose_fuzzy.py

This script integrates a live gesture-control system with several advanced features:
  - Hand and face landmark detection (for gesture recognition)
  - Robust pose detection (tracking arms and upper torso key points)
  - Automatic training data capture with human-in-the-loop prompts
  - Key press buffering with saving of the last 5 seconds of frames upon key events
  - Fuzzy distortion data augmentation to simulate small variations in hand/finger movements
  - A Flask web interface to toggle between training and production modes

Run this script on macOS with a supported webcam.
"""

import os
# Disable Continuity Cameras – use only the built-in (or non-continuity) webcam.
os.environ["OPENCV_AVFOUNDATION_IGNORE_CONTINUITY"] = "1"

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pickle
import random
from collections import deque
from flask import Flask, render_template, request, jsonify
from pynput import keyboard
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# ------------------------
# Global Shared State for Gesture Control
# ------------------------
shared_state = {
    "mode": "production",  # "training" or "production"
    "current_prompt": "",
    "next_prompt": "",
    "predicted_action": "none",
    "confidence": 0.0,
    "sample_count": 0
}

# ------------------------
# Global Buffers and Configurations for New Features
# ------------------------
BUFFER_SECONDS = 5       # seconds of frame history
FPS = 30                 # assumed frames per second
FRAME_BUFFER_SIZE = BUFFER_SECONDS * FPS
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)  # stores tuples: (frame, timestamp)

key_press_buffer = []    # stores (key_name, timestamp) for designated key events
motion_data = []         # collects motion features for clustering user approaches

# ------------------------
# Constants & Prompts for Gestures
# ------------------------
SAMPLE_THRESHOLD = 10         # number of training samples to collect per gesture
CONFIDENCE_THRESHOLD = 0.1    # threshold for accepting predictions in production mode
MOTION_WINDOW = 3             # number of consecutive frames used for a prediction sample
PAUSE_DURATION = 5            # pause (in seconds) between gesture training prompts

gesture_prompts = [
    "point_upper_left",
    "point_upper_right",
    "stop",
    "swipe_left",
    "swipe_right",
    "volume_adjust",
    "brightness_adjust",
    "mute",
    "back"
]

# Define designated keys for additional training (keyboard actions)
KEY_ACTIONS = {
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "next_song": "next_song",
    "pause": "pause"
}

# ------------------------
# Fuzzy Distortion for Data Augmentation
# ------------------------
def fuzzy_distortion(image, strength=5, kernel_size=3):
    """
    Applies a slight fuzzy distortion to the input image.

    Args:
        image (np.array): The input image.
        strength (int): Maximum displacement in pixels.
        kernel_size (int): Size of Gaussian blur kernel (must be odd). A larger kernel adds more blur.

    Returns:
        np.array: The distorted image.
    """
    rows, cols = image.shape[:2]

    # Create random displacement maps for x and y directions.
    dx = np.random.uniform(-strength, strength, size=(rows, cols))
    dy = np.random.uniform(-strength, strength, size=(rows, cols))

    # Create a mesh grid of (x,y) coordinates.
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = (map_x.astype(np.float32) + dx).clip(0, cols - 1)
    map_y = (map_y.astype(np.float32) + dy).clip(0, rows - 1)

    # Remap the image using the displacement maps.
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Optionally apply a Gaussian blur to further smooth the image.
    if kernel_size > 1:
        distorted_image = cv2.GaussianBlur(distorted_image, (kernel_size, kernel_size), 0)

    return distorted_image

def save_augmented_sample(image, label, count):
    """
    Generates a fuzzy-distorted version of the image and saves it to disk.

    Args:
        image (np.array): The original image.
        label (str): The gesture label.
        count (int): A count used to create a unique filename.
    """
    aug_image = fuzzy_distortion(image, strength=5, kernel_size=3)
    filename = f"aug_sample_{label}_{count}.jpg"
    cv2.imwrite(filename, aug_image)
    print(f"Saved augmented sample image: {filename}")

# ------------------------
# Helper: Beep Function
# ------------------------
def beep():
    """Plays a short beep sound (using macOS afplay)."""
    os.system("afplay /System/Library/Sounds/Ping.aiff")

# ------------------------
# Helper: Flatten Landmark Functions
# ------------------------
def flatten_hand_landmarks(hand_landmarks, expected_count=21):
    """
    Flattens hand landmarks into a list of floats.

    Args:
        hand_landmarks: A MediaPipe hand landmarks object.
        expected_count (int): Expected number of landmarks (default 21).

    Returns:
        List of floats (length = expected_count * 3). Returns zeros if no landmarks.
    """
    if hand_landmarks is None:
        return [0.0] * (expected_count * 3)
    vec = []
    for lm in hand_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

def flatten_face_landmarks(face_landmarks, expected_count=468):
    """
    Flattens face landmarks into a list of floats.

    Args:
        face_landmarks: A MediaPipe face landmarks object.
        expected_count (int): Expected number of landmarks (default 468).

    Returns:
        List of floats (length = expected_count * 3). Returns zeros if no landmarks.
    """
    if face_landmarks is None:
        return [0.0] * (expected_count * 3)
    vec = []
    for lm in face_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

# ------------------------
# Feature Extraction for Gesture Classification
# ------------------------
def extract_all_features(hand_results, face_results):
    """
    Extracts a combined feature vector from detected hand and face landmarks.
    """
    left_hand_vec = [0.0] * (21 * 3)
    right_hand_vec = [0.0] * (21 * 3)
    if hand_results.multi_hand_landmarks:
        hands = list(hand_results.multi_hand_landmarks)
        hands = sorted(hands, key=lambda hand: hand.landmark[mp.solutions.hands.HandLandmark.WRIST].x)
        if len(hands) >= 2:
            left_hand_vec = flatten_hand_landmarks(hands[0])
            right_hand_vec = flatten_hand_landmarks(hands[1])
        else:
            wrist_x = hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST].x
            if wrist_x < 0.5:
                left_hand_vec = flatten_hand_landmarks(hands[0])
            else:
                right_hand_vec = flatten_hand_landmarks(hands[0])
    face_vec = [0.0] * (468 * 3)
    if face_results.multi_face_landmarks:
        face_vec = flatten_face_landmarks(face_results.multi_face_landmarks[0])
    return left_hand_vec + right_hand_vec + face_vec

# ------------------------
# Classifier Definition for Gestures
# ------------------------
class MyGestureClassifier:
    def __init__(self):
        self.samples = []  # List of feature vectors
        self.labels = []   # Corresponding gesture labels
        self.model = None

    def train(self):
        if len(self.samples) == 0:
            print("No samples to train on.")
            return
        X = np.array(self.samples)
        y = np.array(self.labels)
        self.model = SVC(probability=True, kernel='linear')
        self.model.fit(X, y)
        print("Classifier trained on {} samples.".format(len(self.samples)))

    def update(self, new_samples):
        for feat, label in new_samples:
            self.samples.append(feat)
            self.labels.append(label)
        unique_labels = set(self.labels)
        if len(unique_labels) < 2:
            print("Skipping training: only one class present:", unique_labels)
            return
        else:
            self.train()

    def predict(self, features):
        if self.model is None:
            return ("none", 0.0)
        feat_array = np.array(features).reshape(1, -1)
        pred = self.model.predict(feat_array)[0]
        prob = self.model.predict_proba(feat_array).max()
        return (pred, prob)

# ------------------------
# Helper Functions for Classifier
# ------------------------
def initialize_classifier():
    return MyGestureClassifier()

def update_classifier(classifier, samples):
    classifier.update(samples)
    return classifier

def execute_system_action(action):
    """
    Executes a system command based on the recognized gesture or keyboard action.
    """
    print("Executing system action:", action)
    if action == "stop":
        print("Action: Stop executed.")
    elif action == "point_upper_left":
        os.system("osascript -e 'tell application \"System Events\" to set the position of the first window of process \"Finder\" to {0, 0}'")
    elif action == "point_upper_right":
        os.system("osascript -e 'tell application \"System Events\" to set the position of the first window of process \"Finder\" to {1000, 0}'")
    elif action == "swipe_left":
        print("Action: Swipe Left executed.")
    elif action == "swipe_right":
        print("Action: Swipe Right executed.")
    elif action == "volume_adjust":
        print("Action: Volume Adjust executed.")
    elif action == "brightness_adjust":
        print("Action: Brightness Adjust executed.")
    elif action == "mute":
        os.system("osascript -e 'set volume output muted true'")
        print("Action: Mute executed.")
    elif action == "back":
        print("Action: Back executed.")
    elif action in KEY_ACTIONS.values():
        if action == "volume_up":
            os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) + 10)'")
            print("Action: Volume Up executed.")
        elif action == "volume_down":
            os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) - 10)'")
            print("Action: Volume Down executed.")
        elif action == "next_song":
            print("Action: Next Song executed.")
        elif action == "pause":
            print("Action: Pause executed.")
    else:
        print("No valid action to execute.")

# ------------------------
# Pose Drawing Function (Arms & Upper Torso)
# ------------------------
def draw_pose_info(frame, pose_results, mp_pose):
    """
    Overlays key points on detected arms and upper torso from pose detection.

    Draws circles at the shoulders, elbows, wrists, and hips and connects them with lines.
    """
    if pose_results.pose_landmarks:
        h, w, _ = frame.shape
        # Define the key landmarks to track.
        landmarks_to_draw = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP
        ]
        pts = {}
        for lm_enum in landmarks_to_draw:
            lm = pose_results.pose_landmarks.landmark[lm_enum]
            cx, cy = int(lm.x * w), int(lm.y * h)
            pts[lm_enum] = (cx, cy)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Yellow circles

        # Draw left arm.
        if (mp_pose.PoseLandmark.LEFT_SHOULDER in pts and
            mp_pose.PoseLandmark.LEFT_ELBOW in pts and
            mp_pose.PoseLandmark.LEFT_WRIST in pts):
            cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_SHOULDER], pts[mp_pose.PoseLandmark.LEFT_ELBOW], (0,255,255), 2)
            cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_ELBOW], pts[mp_pose.PoseLandmark.LEFT_WRIST], (0,255,255), 2)
        # Draw right arm.
        if (mp_pose.PoseLandmark.RIGHT_SHOULDER in pts and
            mp_pose.PoseLandmark.RIGHT_ELBOW in pts and
            mp_pose.PoseLandmark.RIGHT_WRIST in pts):
            cv2.line(frame, pts[mp_pose.PoseLandmark.RIGHT_SHOULDER], pts[mp_pose.PoseLandmark.RIGHT_ELBOW], (0,255,255), 2)
            cv2.line(frame, pts[mp_pose.PoseLandmark.RIGHT_ELBOW], pts[mp_pose.PoseLandmark.RIGHT_WRIST], (0,255,255), 2)
        # Draw upper torso connections.
        if (mp_pose.PoseLandmark.LEFT_SHOULDER in pts and mp_pose.PoseLandmark.RIGHT_SHOULDER in pts):
            cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_SHOULDER], pts[mp_pose.PoseLandmark.RIGHT_SHOULDER], (0,255,255), 2)
        if (mp_pose.PoseLandmark.LEFT_SHOULDER in pts and mp_pose.PoseLandmark.LEFT_HIP in pts):
            cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_SHOULDER], pts[mp_pose.PoseLandmark.LEFT_HIP], (0,255,255), 2)
        if (mp_pose.PoseLandmark.RIGHT_SHOULDER in pts and mp_pose.PoseLandmark.RIGHT_HIP in pts):
            cv2.line(frame, pts[mp_pose.PoseLandmark.RIGHT_SHOULDER], pts[mp_pose.PoseLandmark.RIGHT_HIP], (0,255,255), 2)
    return frame

# ------------------------
# Debug Overlay Drawing Function
# ------------------------
def draw_debug_info(frame, hand_results, face_results, pose_results, state, next_prompt_text="", mp_pose=None):
    """
    Draws various debug overlays on the frame including mode, current gesture prompt,
    predicted action/confidence, and sample count. Also overlays pose landmarks.
    """
    h, w, _ = frame.shape
    cv2.putText(frame, f"Mode: {state['mode']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Current: {state['current_prompt']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if next_prompt_text:
        cv2.putText(frame, f"Next Up: {next_prompt_text}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Predicted: {state['predicted_action']} ({state['confidence']:.2f})", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if state["mode"] == "training":
        cv2.putText(frame, f"Samples: {state.get('sample_count', 0)}/{SAMPLE_THRESHOLD}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Overlay pose landmarks if available.
    if pose_results and mp_pose is not None:
        frame = draw_pose_info(frame, pose_results, mp_pose)
    return frame

# ------------------------
# Keyboard Listener Callback for Key Press Capture
# ------------------------
def on_key_press(key):
    """
    Captures key presses and, if the key is in our designated KEY_ACTIONS list,
    appends it with a timestamp to the key_press_buffer.
    """
    try:
        key_name = key.char if hasattr(key, 'char') and key.char is not None else str(key)
        if key_name in KEY_ACTIONS:
            timestamp = time.time()
            key_press_buffer.append((key_name, timestamp))
            print(f"Captured key press: {key_name} at {timestamp}")
    except Exception as e:
        print("Error capturing key press:", e)

keyboard_listener = keyboard.Listener(on_press=on_key_press)
keyboard_listener.daemon = True
keyboard_listener.start()

# ------------------------
# Clustering of User Motion
# ------------------------
def cluster_user_movements(motion_data):
    """
    Clusters different user movement approaches using K-Means.
    """
    if len(motion_data) < 10:
        print("Not enough data for clustering.")
        return None
    X = np.array(motion_data)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    print("User movement clusters:", labels)
    return labels

def extract_motion_features(hand_results, face_results, pose_results, mp_pose):
    """
    Extracts a feature vector combining hand, face, and pose information.
    For pose, we use the left and right shoulders.
    """
    feature_vector = []
    # Hand features: use wrist positions.
    if hand_results.multi_hand_landmarks:
        hands = list(hand_results.multi_hand_landmarks)
        for hand in hands:
            wrist = hand.landmark[mp.solutions.hands.HandLandmark.WRIST]
            feature_vector.extend([wrist.x, wrist.y, wrist.z])
    # Face feature: use the nose tip.
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        if len(face.landmark) > 1:
            nose_tip = face.landmark[1]
            feature_vector.extend([nose_tip.x, nose_tip.y, nose_tip.z])
    # Pose features: use left and right shoulders.
    if pose_results and pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        feature_vector.extend([left_shoulder.x, left_shoulder.y, left_shoulder.z])
        feature_vector.extend([right_shoulder.x, right_shoulder.y, right_shoulder.z])
    return feature_vector if feature_vector else None

# ------------------------
# Flask Web App Setup for Gesture Control
# ------------------------
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html', state=shared_state)

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    new_mode = request.form.get("mode")
    shared_state["mode"] = new_mode
    print("Switched mode to", new_mode)
    return jsonify(success=True)

@app.route('/confirm_action', methods=['POST'])
def confirm_action():
    action = shared_state.get("predicted_action")
    execute_system_action(action)
    return jsonify(success=True)

def run_flask():
    app.run(debug=False, use_reloader=False)

# ------------------------
# Main Application Loop
# ------------------------
def main():
    # Start the Flask server in a separate thread.
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Open the webcam.
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    # Initialize MediaPipe detectors.
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    face_detector = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    gesture_classifier = initialize_classifier()

    # Set up gesture prompts.
    prompt_index = 0
    current_prompt = gesture_prompts[prompt_index]
    next_prompt = gesture_prompts[(prompt_index + 1) % len(gesture_prompts)]

    samples_buffer = []     # Holds confirmed training samples as (feature_vector, label)
    recording_buffer = []   # Temporary buffer for current sample capture
    sample_count = 0        # Count of samples collected for current gesture
    waiting_for_pause = False
    pause_start_time = 0

    pred_buffer = []        # Buffer for predictions (using a window of frames)
    last_action_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a mirror effect.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe.
        hand_results = hands_detector.process(rgb_frame)
        face_results = face_detector.process(rgb_frame)
        pose_results = pose_detector.process(rgb_frame)

        # Save the current frame with timestamp into the frame buffer.
        frame_buffer.append((frame.copy(), time.time()))

        # Extract gesture features (from hands and face) for classifier.
        features = extract_all_features(hand_results, face_results)

        # Update the prediction buffer for live preview.
        if features is not None:
            pred_buffer.append(features)
            if len(pred_buffer) > MOTION_WINDOW:
                pred_buffer.pop(0)
            if len(pred_buffer) == MOTION_WINDOW:
                flat_pred = []
                for feat in pred_buffer:
                    flat_pred.extend(feat)
                predicted_action, confidence = gesture_classifier.predict(flat_pred)
                shared_state["predicted_action"] = predicted_action
                shared_state["confidence"] = confidence

        # -------------------------
        # Training Mode: Auto Capture of Gesture Samples
        # -------------------------
        if shared_state["mode"] == "training":
            shared_state["current_prompt"] = current_prompt
            shared_state["next_prompt"] = next_prompt

            if not waiting_for_pause:
                # Capture training samples.
                if features is not None:
                    recording_buffer.append(features)
                if len(recording_buffer) >= MOTION_WINDOW:
                    flat_sample = []
                    for feat in recording_buffer:
                        flat_sample.extend(feat)
                    samples_buffer.append((flat_sample, current_prompt))
                    sample_count += 1
                    print(f"Collected sample {sample_count} for '{current_prompt}'")
                    # Also save an augmented version of the current frame.
                    save_augmented_sample(frame, current_prompt, sample_count)
                    recording_buffer = []
                    shared_state["sample_count"] = sample_count

                # Once enough samples are collected, update the classifier and pause.
                if sample_count >= SAMPLE_THRESHOLD:
                    print(f"Collected {SAMPLE_THRESHOLD} samples for '{current_prompt}'. Updating classifier...")
                    gesture_classifier = update_classifier(gesture_classifier, samples_buffer)
                    samples_buffer = []
                    sample_count = 0
                    shared_state["sample_count"] = sample_count
                    beep()
                    # Save classifier progress.
                    with open("gesture_classifier_progress.pkl", "wb") as f:
                        pickle.dump(gesture_classifier, f)
                    waiting_for_pause = True
                    pause_start_time = time.time()
                    print("Pausing for 5 seconds before switching gesture...")
            else:
                # Pause period between gestures.
                if time.time() - pause_start_time >= PAUSE_DURATION:
                    prompt_index = (prompt_index + 1) % len(gesture_prompts)
                    current_prompt = gesture_prompts[prompt_index]
                    next_prompt = gesture_prompts[(prompt_index + 1) % len(gesture_prompts)]
                    waiting_for_pause = False
                    print("Switching to next gesture:", current_prompt)
                    beep()
                    with open("gesture_classifier_progress.pkl", "wb") as f:
                        pickle.dump(gesture_classifier, f)

        # -------------------------
        # Production Mode: Execute Actions Based on Predictions
        # -------------------------
        elif shared_state["mode"] == "production":
            if len(pred_buffer) == MOTION_WINDOW:
                if shared_state["confidence"] > CONFIDENCE_THRESHOLD and (time.time() - last_action_time) > 2:
                    execute_system_action(shared_state["predicted_action"])
                    last_action_time = time.time()

        # -------------------------
        # Process Key Presses: Save last 5 seconds of frames for designated keyboard actions
        # -------------------------
        if key_press_buffer:
            for key_name, press_time in key_press_buffer:
                relevant_frames = [frm for frm, t in frame_buffer if abs(t - press_time) < 0.5]
                if relevant_frames:
                    filename = f"training_data_{key_name}_{int(press_time)}.pkl"
                    with open(filename, "wb") as f:
                        pickle.dump(relevant_frames, f)
                    print(f"Saved {len(relevant_frames)} frames for key '{key_name}' to {filename}")
            key_press_buffer.clear()

        # -------------------------
        # Cluster Motion Data for Predictive Approaches (Optional)
        # -------------------------
        motion_features = extract_motion_features(hand_results, face_results, pose_results, mp_pose)
        if motion_features:
            motion_data.append(motion_features)
        if len(motion_data) > 50:
            cluster_user_movements(motion_data)
            motion_data.clear()

        # -------------------------
        # Draw Debug Information Overlay
        # -------------------------
        debug_frame = draw_debug_info(frame, hand_results, face_results, pose_results, shared_state, next_prompt_text=next_prompt, mp_pose=mp_pose)
        cv2.imshow('Gesture Control', debug_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()