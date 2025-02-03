#!/usr/bin/env python3
"""
main.py – A live gesture-control application for macOS that uses the webcam
to detect gestures. It supports human-in-the-loop training by automatically
capturing training data. For each pose, the system collects a fixed number of
samples, updates the classifier, plays a beep, pauses for 5 seconds, and then
automatically switches to the next pose. The video overlay displays the current
pose and the "next up" pose.

New actions added: "mute" and "back".

This version tracks both of your hands and your face.
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
    "next_prompt": "",
    "predicted_action": "none",
    "confidence": 0.0,
    "sample_count": 0
}

# ------------------------
# Constants & Prompts
# ------------------------
SAMPLE_THRESHOLD = 10         # Number of training samples to collect per pose
CONFIDENCE_THRESHOLD = 0.7    # Confidence threshold for production mode
MOTION_WINDOW = 3             # Number of consecutive frames to record for one sample
PAUSE_DURATION = 5            # Seconds to pause between poses

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
    """Play a short beep sound (using macOS afplay)."""
    os.system("afplay /System/Library/Sounds/Ping.aiff")

# ------------------------
# Helper: Save Progress
# ------------------------
def save_progress(classifier):
    """Save the current classifier progress to a pickle file."""
    with open("gesture_classifier_progress.pkl", "wb") as f:
        pickle.dump(classifier, f)
    print("Progress saved to gesture_classifier_progress.pkl")

# ------------------------
# Helper: Flatten Landmark Functions
# ------------------------
def flatten_hand_landmarks(hand_landmarks, expected_count=21):
    """
    Flatten the landmarks of one hand.
    Returns a list of expected_count*3 values.
    If hand_landmarks is None, returns zeros.
    """
    if hand_landmarks is None:
        return [0.0] * (expected_count * 3)
    vec = []
    for lm in hand_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

def flatten_face_landmarks(face_landmarks, expected_count=468):
    """
    Flatten the landmarks of the face.
    Returns a list of expected_count*3 values.
    If face_landmarks is None, returns zeros.
    """
    if face_landmarks is None:
        return [0.0] * (expected_count * 3)
    vec = []
    for lm in face_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

# ------------------------
# New Feature Extraction Function
# ------------------------
def extract_all_features(hand_results, face_results):
    """
    Extract features from both hands and the face.

    For hands:
      - If two hands are detected, sort them by wrist x-coordinate.
      - If only one hand is detected, assign it to left if its wrist x < 0.5, else right.
      - Missing hand features are replaced by zeros.

    For face:
      - If detected, flatten all 468 landmarks.
      - Otherwise, return a zero vector.

    Returns a fixed-length vector of length 63 (left) + 63 (right) + 1404 (face) = 1530.
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
# Classifier Definition
# ------------------------
class MyGestureClassifier:
    def __init__(self):
        self.samples = []  # List of feature vectors
        self.labels = []   # Corresponding labels (strings)
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
# Helper Functions
# ------------------------
def initialize_classifier():
    return MyGestureClassifier()

def update_classifier(classifier, samples):
    classifier.update(samples)
    return classifier

def execute_system_action(action):
    """Execute a system command based on the recognized gesture."""
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

def draw_debug_info(frame, hand_results, face_results, state, next_prompt_text=""):
    """
    Draw overlays on the frame:
      - Display the mode, current prompt, next up pose, predicted action/confidence, and sample count.
      - Draw left-hand landmarks in red, right-hand landmarks in blue.
      - Draw face landmarks in faint green, mark the face center with a green circle,
        and draw a line from the face center to each hand's wrist.
    """
    h, w, _ = frame.shape
    cv2.putText(frame, f"Mode: {state['mode']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Current: {state['current_prompt']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if next_prompt_text:
        cv2.putText(frame, f"Next Up: {next_prompt_text}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # yellow text for next pose
    cv2.putText(frame, f"Predicted: {state['predicted_action']} ({state['confidence']:.2f})", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if state["mode"] == "training":
        cv2.putText(frame, f"Samples: {state.get('sample_count', 0)}/{SAMPLE_THRESHOLD}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Draw face landmarks and center.
    face_center = None
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        face_coords = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]
        for (cx, cy) in face_coords:
            cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
        face_center = (int(sum(x for x, y in face_coords)/len(face_coords)),
                       int(sum(y for x, y in face_coords)/len(face_coords)))
        cv2.circle(frame, face_center, 5, (0, 255, 0), -1)
    # Draw hand landmarks.
    if hand_results.multi_hand_landmarks:
        hands = list(hand_results.multi_hand_landmarks)
        hands = sorted(hands, key=lambda hand: hand.landmark[mp.solutions.hands.HandLandmark.WRIST].x)
        left_hand = None
        right_hand = None
        if len(hands) >= 2:
            left_hand = hands[0]
            right_hand = hands[1]
        elif len(hands) == 1:
            wrist_x = hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST].x
            if wrist_x < 0.5:
                left_hand = hands[0]
            else:
                right_hand = hands[0]
        if left_hand is not None:
            for lm in left_hand.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)  # red for left hand
            if face_center is not None:
                wrist = left_hand.landmark[mp.solutions.hands.HandLandmark.WRIST]
                wrist_pt = (int(wrist.x * w), int(wrist.y * h))
                cv2.line(frame, face_center, wrist_pt, (0, 0, 255), 2)
        if right_hand is not None:
            for lm in right_hand.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)  # blue for right hand
            if face_center is not None:
                wrist = right_hand.landmark[mp.solutions.hands.HandLandmark.WRIST]
                wrist_pt = (int(wrist.x * w), int(wrist.y * h))
                cv2.line(frame, face_center, wrist_pt, (255, 0, 0), 2)
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
    app.run(debug=False, use_reloader=False)

# ------------------------
# Main Application Loop
# ------------------------
def main():
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    face_detector = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    gesture_classifier = initialize_classifier()

    prompt_index = 0
    current_prompt = gesture_prompts[prompt_index]
    next_prompt = gesture_prompts[(prompt_index + 1) % len(gesture_prompts)]

    samples_buffer = []     # Holds confirmed training samples: list of (sample, label)
    recording_buffer = []   # Temporary buffer for current sample capture.
    sample_count = 0        # Count of samples collected for current pose.
    waiting_for_pause = False
    pause_start_time = 0

    pred_buffer = []        # Prediction buffer for live preview.

    last_action_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands_detector.process(rgb_frame)
        face_results = face_detector.process(rgb_frame)

        features = extract_all_features(hand_results, face_results)

        # Update prediction buffer for live preview.
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

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit.
            break

        # -------------------------
        # Training Mode (Auto Capture)
        # -------------------------
        if shared_state["mode"] == "training":
            shared_state["current_prompt"] = current_prompt
            shared_state["next_prompt"] = next_prompt

            if not waiting_for_pause:
                # Automatically capture training samples.
                if features is not None:
                    recording_buffer.append(features)
                if len(recording_buffer) >= MOTION_WINDOW:
                    flat_sample = []
                    for feat in recording_buffer:
                        flat_sample.extend(feat)
                    samples_buffer.append((flat_sample, current_prompt))
                    sample_count += 1
                    print(f"Collected sample {sample_count} for '{current_prompt}'")
                    recording_buffer = []
                    shared_state["sample_count"] = sample_count

                # When enough samples have been collected, update classifier and pause.
                if sample_count >= SAMPLE_THRESHOLD:
                    print(f"Collected {SAMPLE_THRESHOLD} samples for '{current_prompt}'. Updating classifier...")
                    gesture_classifier = update_classifier(gesture_classifier, samples_buffer)
                    samples_buffer = []
                    sample_count = 0
                    shared_state["sample_count"] = sample_count
                    beep()
                    save_progress(gesture_classifier)
                    waiting_for_pause = True
                    pause_start_time = time.time()
                    print("Pausing for 5 seconds before switching pose...")
            else:
                # During pause, do not capture new samples.
                if time.time() - pause_start_time >= PAUSE_DURATION:
                    # Automatically switch to the next pose.
                    prompt_index = (prompt_index + 1) % len(gesture_prompts)
                    current_prompt = gesture_prompts[prompt_index]
                    next_prompt = gesture_prompts[(prompt_index + 1) % len(gesture_prompts)]
                    waiting_for_pause = False
                    print("Switching to next prompt:", current_prompt)
                    beep()
                    save_progress(gesture_classifier)

        # -------------------------
        # Production Mode
        # -------------------------
        elif shared_state["mode"] == "production":
            if len(pred_buffer) == MOTION_WINDOW:
                if shared_state["confidence"] > CONFIDENCE_THRESHOLD and (time.time() - last_action_time) > 2:
                    execute_system_action(shared_state["predicted_action"])
                    last_action_time = time.time()

        debug_frame = draw_debug_info(frame, hand_results, face_results, shared_state, next_prompt_text=next_prompt)
        cv2.imshow('Gesture Control', debug_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
