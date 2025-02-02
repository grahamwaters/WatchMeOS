#!/usr/bin/env python3
"""
main.py – A live gesture-control application that uses the MacBook’s webcam
to detect gestures, update a classifier with human-in-the-loop training,
and execute (or preview) system actions via a Flask dashboard.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from flask import Flask, render_template, request, jsonify
import os
from sklearn.svm import SVC

# ------------------------
# Global Shared State
# ------------------------
shared_state = {
    "mode": "training",  # "training" or "production"
    "current_prompt": "",
    "predicted_action": "none",
    "confidence": 0.0,
}

# ------------------------
# Constants & Prompts
# ------------------------
SAMPLE_THRESHOLD = 20         # Number of samples per prompt before updating the model
CONFIDENCE_THRESHOLD = 0.7    # Confidence above which to trigger an action in production mode

# List of gestures that you want to train.
gesture_prompts = [
    "point_upper_left",
    "point_upper_right",
    "stop",
    "swipe_left",
    "swipe_right",
    "volume_adjust",
    "brightness_adjust"
]

# ------------------------
# Classifier Definition
# ------------------------
class MyGestureClassifier:
    def __init__(self):
        self.samples = []  # List of feature vectors (lists of floats)
        self.labels = []   # Corresponding gesture labels (strings)
        self.model = None

    def train(self):
        if len(self.samples) == 0:
            print("No samples to train on.")
            return
        X = np.array(self.samples)
        y = np.array(self.labels)
        # Create and train an SVC classifier with probability estimates enabled.
        self.model = SVC(probability=True, kernel='linear')
        self.model.fit(X, y)
        print("Classifier trained on {} samples.".format(len(self.samples)))

    def update(self, new_samples):
        # Append new samples.
        for feat, label in new_samples:
            self.samples.append(feat)
            self.labels.append(label)
        # Check if we have at least 2 different gesture classes before training.
        unique_labels = set(self.labels)
        if len(unique_labels) < 2:
            print("Skipping training: not enough classes (only found: {})".format(unique_labels))
            return  # Do not train yet.
        else:
            self.train()

    def predict(self, features):
        if self.model is None:
            return ("none", 0.0)
        # Reshape features for prediction (as a 2D array)
        feat_array = np.array(features).reshape(1, -1)
        pred = self.model.predict(feat_array)[0]
        prob = self.model.predict_proba(feat_array).max()
        return (pred, prob)

# ------------------------
# Helper Functions
# ------------------------
def initialize_classifier():
    return MyGestureClassifier()

def extract_features(hand_results, face_results):
    """
    Extract a simple feature vector from the detected hand landmarks.
    This example uses the first hand’s landmarks (if available)
    by flattening all (x, y, z) coordinates.
    """
    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        feature_vector = []
        for lm in hand_landmarks.landmark:
            feature_vector.extend([lm.x, lm.y, lm.z])
        return feature_vector
    else:
        return None

def update_classifier(classifier, samples):
    # Update the classifier with new samples.
    classifier.update(samples)
    return classifier

def execute_system_action(action):
    """
    Execute a system command based on the recognized gesture.
    (For demonstration purposes, many actions are placeholders.
     Some use AppleScript commands on macOS.)
    """
    print("Executing system action:", action)
    if action == "stop":
        print("Action: Stop executed.")
    elif action == "point_upper_left":
        # Example: Move a Finder window to the upper-left corner using AppleScript
        os.system("osascript -e 'tell application \"System Events\" to set the position of the first window of process \"Finder\" to {0, 0}'")
    elif action == "point_upper_right":
        # Example: Move a Finder window to the upper-right corner
        os.system("osascript -e 'tell application \"System Events\" to set the position of the first window of process \"Finder\" to {1000, 0}'")
    elif action == "swipe_left":
        print("Action: Swipe Left executed.")
    elif action == "swipe_right":
        print("Action: Swipe Right executed.")
    elif action == "volume_adjust":
        print("Action: Volume Adjust executed.")
    elif action == "brightness_adjust":
        print("Action: Brightness Adjust executed.")
    else:
        print("No valid action to execute.")

def draw_debug_info(frame, hand_results, face_results, state):
    """
    Draw text overlays on the frame showing the mode, current prompt,
    predicted action, and its confidence.
    Optionally, also draw circles at each hand landmark.
    """
    cv2.putText(frame, f"Mode: {state['mode']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Prompt: {state['current_prompt']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {state['predicted_action']} ({state['confidence']:.2f})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            for lm in handLms.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
    return frame

# ------------------------
# Flask Web App Setup
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
    # Run Flask on port 5000; disable reloader to avoid duplicate threads.
    app.run(debug=False, use_reloader=False)

# ------------------------
# Main Application Loop
# ------------------------
def main():
    # Start the Flask server in a separate thread.
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Open the webcam video stream.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    # Initialize MediaPipe solutions.
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    face_detector = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    # Initialize the gesture classifier.
    gesture_classifier = initialize_classifier()

    prompt_index = 0
    samples_buffer = []  # Buffer to hold training samples.
    last_action_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip and convert the frame.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe.
        hand_results = hands_detector.process(rgb_frame)
        face_results = face_detector.process(rgb_frame)

        # Extract features (from hand landmarks).
        features = extract_features(hand_results, face_results)
        current_mode = shared_state["mode"]

        if current_mode == "training":
            current_prompt = gesture_prompts[prompt_index]
            shared_state["current_prompt"] = current_prompt

            if features is not None:
                # Append the new sample as (features, label).
                samples_buffer.append((features, current_prompt))

            # When enough samples have been collected, update the classifier.
            # Note: The classifier will not actually train if only one class is present.
            if len(samples_buffer) >= SAMPLE_THRESHOLD:
                gesture_classifier = update_classifier(gesture_classifier, samples_buffer)
                samples_buffer = []  # Reset buffer for the next gesture.
                # Cycle to the next prompt.
                prompt_index = (prompt_index + 1) % len(gesture_prompts)
                print("Collected samples for prompt. Next prompt:", gesture_prompts[prompt_index])
                print("Make sure you perform at least two different gestures to train the model.")

            # Always predict (for preview purposes) if we have features.
            if features is not None:
                predicted_action, confidence = gesture_classifier.predict(features)
                shared_state["predicted_action"] = predicted_action
                shared_state["confidence"] = confidence

        elif current_mode == "production":
            # In production mode, automatically predict and (if confident) execute actions.
            if features is not None:
                predicted_action, confidence = gesture_classifier.predict(features)
                shared_state["predicted_action"] = predicted_action
                shared_state["confidence"] = confidence

                if confidence > CONFIDENCE_THRESHOLD and (time.time() - last_action_time) > 2:
                    execute_system_action(predicted_action)
                    last_action_time = time.time()

        # Draw debug information on the frame.
        debug_frame = draw_debug_info(frame, hand_results, face_results, shared_state)
        cv2.imshow('Gesture Control', debug_frame)

        # Exit on pressing the ESC key.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
