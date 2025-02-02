#!/usr/bin/env python3
"""
main.py – A live gesture-control application for macOS that uses the webcam
to detect gestures. It supports human-in-the-loop training, plays a beep
between prompts, saves progress to a pickle file, and uses a motion window
across keyframes so that actions (such as swiping) can be learned.
New actions added: "mute" and "back".
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from flask import Flask, render_template, request, jsonify
import os
import pickle
from sklearn.svm import SVC

# ------------------------
# Global Shared State
# ------------------------
shared_state = {
    "mode": "training",  # "training" or "production"
    "current_prompt": "",
    "predicted_action": "none",
    "confidence": 0.0,
    "sample_count": 0
}

# ------------------------
# Constants & Prompts
# ------------------------
SAMPLE_THRESHOLD = 10         # Reduced: Number of motion samples to collect per prompt
CONFIDENCE_THRESHOLD = 0.7    # Confidence above which to trigger an action in production mode
MOTION_WINDOW = 3             # Reduced: Number of consecutive frames to record for one sample

# List of gesture prompts to be trained.
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

# ------------------------
# Helper: Beep function
# ------------------------
def beep():
    """
    Play a short beep sound.
    On macOS, we can use afplay to play a system sound.
    """
    os.system("afplay /System/Library/Sounds/Ping.aiff")

# ------------------------
# Helper: Save Progress
# ------------------------
def save_progress(classifier):
    """
    Save the current classifier progress to a pickle file.
    This function is called each time a beep is played.
    """
    with open("gesture_classifier_progress.pkl", "wb") as f:
        pickle.dump(classifier, f)
    print("Progress saved to gesture_classifier_progress.pkl")

# ------------------------
# Classifier Definition
# ------------------------
class MyGestureClassifier:
    def __init__(self):
        self.samples = []  # List of feature vectors (each is a list of floats)
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
        # Check that we have at least 2 different gesture classes before training.
        unique_labels = set(self.labels)
        if len(unique_labels) < 2:
            print("Skipping training: only one class present:", unique_labels)
            return  # Do not train until more than one class exists.
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
    Extract a feature vector from detected hand landmarks.
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
    classifier.update(samples)
    return classifier

def execute_system_action(action):
    """
    Execute a system command based on the recognized gesture.
    (For demonstration, many actions are placeholders and some use AppleScript on macOS.)
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
    else:
        print("No valid action to execute.")

def draw_debug_info(frame, hand_results, face_results, state):
    """
    Draw overlays on the frame showing the current mode, prompt,
    predicted action/confidence, and the number of samples collected.
    Also draws circles at each hand landmark.
    """
    cv2.putText(frame, f"Mode: {state['mode']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Prompt: {state['current_prompt']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {state['predicted_action']} ({state['confidence']:.2f})", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if state["mode"] == "training":
        cv2.putText(frame, f"Samples: {state.get('sample_count', 0)}/{SAMPLE_THRESHOLD}", (10, 120),
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
    # Set the initial prompt.
    current_prompt = gesture_prompts[prompt_index]

    # Buffers for training samples.
    samples_buffer = []     # Holds (flattened motion sample, label) tuples.
    recording_buffer = []   # Holds consecutive frame feature vectors for one training sample.
    sample_count = 0        # Count of samples collected for current prompt.

    # Flag to wait for gesture change (space bar required)
    waiting_for_change = False

    # We'll also maintain a sliding prediction buffer for preview
    pred_buffer = []

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

        # Update the prediction buffer (for preview) if features are available.
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

        # Check for key presses.
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit.
            break

        # -------------------------
        # Training Mode (Human-in-the-Loop)
        # -------------------------
        if shared_state["mode"] == "training":
            shared_state["current_prompt"] = current_prompt

            # If not waiting for a gesture change, record training samples continuously.
            if not waiting_for_change:
                if features is not None:
                    recording_buffer.append(features)
                if len(recording_buffer) >= MOTION_WINDOW:
                    new_sample = []
                    for feat in recording_buffer:
                        new_sample.extend(feat)
                    samples_buffer.append((new_sample, current_prompt))
                    sample_count += 1
                    print(f"Collected sample {sample_count} for '{current_prompt}'")
                    recording_buffer = []  # Clear the recording buffer.
                    shared_state["sample_count"] = sample_count

            # Once enough samples have been collected, update the classifier and wait for space bar to change prompt.
            if sample_count >= SAMPLE_THRESHOLD and not waiting_for_change:
                print(f"Collected {SAMPLE_THRESHOLD} samples for '{current_prompt}'. Updating classifier...")
                gesture_classifier = update_classifier(gesture_classifier, samples_buffer)
                samples_buffer = []  # Reset the samples buffer.
                sample_count = 0
                shared_state["sample_count"] = sample_count
                beep()
                save_progress(gesture_classifier)
                waiting_for_change = True
                print("Press space bar to switch to the next gesture prompt.")

            # When waiting for a gesture change, only switch prompt on space bar press.
            if waiting_for_change and key == 32:
                waiting_for_change = False
                prompt_index = (prompt_index + 1) % len(gesture_prompts)
                current_prompt = gesture_prompts[prompt_index]
                print("Switching prompt to", current_prompt)
                beep()
                save_progress(gesture_classifier)

        # -------------------------
        # Production Mode
        # -------------------------
        elif shared_state["mode"] == "production":
            # In production mode, we rely on the sliding prediction buffer already updated above.
            if len(pred_buffer) == MOTION_WINDOW:
                # If confidence is high and a cooldown period has passed, execute the action.
                if shared_state["confidence"] > CONFIDENCE_THRESHOLD and (time.time() - last_action_time) > 2:
                    execute_system_action(shared_state["predicted_action"])
                    last_action_time = time.time()

        # Draw debug information on the frame.
        debug_frame = draw_debug_info(frame, hand_results, face_results, shared_state)
        cv2.imshow('Gesture Control', debug_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
