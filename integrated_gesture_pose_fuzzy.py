#!/usr/bin/env python3
"""
integrated_keypress_gesture_pose_fuzzy.py

Workflow:
1. DATA COLLECTION & TRAINING:
   - The system continuously monitors your video (hands, face, and body pose).
   - When you press a designated key (e.g. "volume_up", "volume_down", "next_song", "pause"),
     the current frame’s extracted features (from hands, face, and pose) are saved as a training sample
     with that key as its label. Additionally, a fuzzy-distorted (augmented) image is saved.
   - Once enough samples have been collected, the SVM classifier is retrained.
   - Training metrics (e.g. training accuracy) are computed and saved to file.

2. TESTING / PRODUCTION:
   - In production mode the classifier predicts the key based on live features.
   - When you press a key, the actual key is compared against the prediction and performance metrics
     (including overall accuracy) are logged.
   - If the model’s accuracy and confidence are high enough, the system can automatically execute actions.

3. POSE DETECTION:
   - Uses MediaPipe to detect and track hands, face, and key points on the upper body (arms/torso).
   - (A stub is provided for integration with a HuggingFace ViTPose model if desired.)

4. RESEARCH METRICS:
   - All prediction events (with timestamp, actual key, predicted key, confidence, and correctness)
     are logged and saved to a JSON file for later descriptive statistical analysis.

5. WEB INTERFACE:
   - A Flask app (running in a separate thread) lets you toggle between training and production modes
     and manually confirm actions.
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
import json
from collections import deque
from flask import Flask, render_template, request, jsonify
from pynput import keyboard
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# ------------------------
# GLOBAL SHARED STATE & BUFFERS
# ------------------------
shared_state = {
    "mode": "production",  # "training" or "production"
    "predicted_action": "none",
    "confidence": 0.0
}

# Frame buffer: stores the last 5 seconds of frames (assumed FPS = 30)
BUFFER_SECONDS = 5
FPS = 30
FRAME_BUFFER_SIZE = BUFFER_SECONDS * FPS
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)  # Each entry: (frame, timestamp)

# Key press events: each entry is a tuple (key_name, timestamp)
key_press_buffer = []

# Training samples: list of tuples (feature_vector, key_label)
training_samples = []

# A temporary buffer for samples collected in the current training cycle.
samples_buffer = []

# Motion data for clustering (for research)
motion_data = []

# Performance metrics for logging (for research)
performance_metrics = {
    "total_predictions": 0,
    "correct_predictions": 0,
    "prediction_history": []  # Each entry: {timestamp, actual, predicted, confidence, correct}
}

# ------------------------
# CONFIGURABLE CONSTANTS
# ------------------------
SAMPLE_THRESHOLD = 10         # Number of samples to collect before updating the classifier
CONFIDENCE_THRESHOLD = 0.7    # Confidence threshold for production suggestions / auto-act
MOTION_WINDOW = 3             # Number of consecutive frames to aggregate for a prediction

# Designated keys for key-based training and actions.
KEY_ACTIONS = {
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "next_song": "next_song",
    "pause": "pause"
}

# ------------------------
# DATA AUGMENTATION: FUZZY DISTORTION
# ------------------------
def fuzzy_distortion(image, strength=5, kernel_size=3):
    """
    Applies a slight fuzzy distortion to the input image.
    This simulates small variations in hand/finger appearance.
    """
    rows, cols = image.shape[:2]
    dx = np.random.uniform(-strength, strength, size=(rows, cols))
    dy = np.random.uniform(-strength, strength, size=(rows, cols))
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = (map_x.astype(np.float32) + dx).clip(0, cols - 1)
    map_y = (map_y.astype(np.float32) + dy).clip(0, rows - 1)
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    if kernel_size > 1:
        distorted_image = cv2.GaussianBlur(distorted_image, (kernel_size, kernel_size), 0)
    return distorted_image

def save_augmented_sample(image, label, count):
    """
    Saves an augmented (fuzzy-distorted) version of the image to disk.
    """
    aug_image = fuzzy_distortion(image, strength=5, kernel_size=3)
    filename = f"aug_sample_{label}_{count}.jpg"
    cv2.imwrite(filename, aug_image)
    print(f"Saved augmented sample image: {filename}")

# ------------------------
# HELPER: BEEP
# ------------------------
def beep():
    """Plays a short beep sound (using macOS afplay)."""
    os.system("afplay /System/Library/Sounds/Ping.aiff")

# ------------------------
# HELPER: FLATTEN LANDMARKS
# ------------------------
def flatten_hand_landmarks(hand_landmarks, expected_count=21):
    if hand_landmarks is None:
        return [0.0] * (expected_count * 3)
    vec = []
    for lm in hand_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

def flatten_face_landmarks(face_landmarks, expected_count=468):
    if face_landmarks is None:
        return [0.0] * (expected_count * 3)
    vec = []
    for lm in face_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

# ------------------------
# FEATURE EXTRACTION FOR TRAINING (Hands, Face, and Pose)
# ------------------------
def extract_all_features(hand_results, face_results):
    """
    Extracts and concatenates hand and face features.
    Returns a fixed-length vector (hand: 2x21x3, face: 468x3) if available.
    (Note: This is used for the SVM classifier.)
    """
    left_hand_vec = [0.0] * (21 * 3)
    right_hand_vec = [0.0] * (21 * 3)
    if hand_results.multi_hand_landmarks:
        hands = list(hand_results.multi_hand_landmarks)
        hands = sorted(hands, key=lambda h: h.landmark[mp.solutions.hands.HandLandmark.WRIST].x)
        if len(hands) >= 2:
            left_hand_vec = flatten_hand_landmarks(hands[0])
            right_hand_vec = flatten_hand_landmarks(hands[1])
        else:
            wrist = hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST]
            if wrist.x < 0.5:
                left_hand_vec = flatten_hand_landmarks(hands[0])
            else:
                right_hand_vec = flatten_hand_landmarks(hands[0])
    face_vec = [0.0] * (468 * 3)
    if face_results.multi_face_landmarks:
        face_vec = flatten_face_landmarks(face_results.multi_face_landmarks[0])
    return left_hand_vec + right_hand_vec + face_vec

def extract_pose_features(pose_results, mp_pose):
    """
    Extracts pose features from selected landmarks (shoulders, elbows, wrists, hips).
    Returns a flattened vector.
    """
    if pose_results and pose_results.pose_landmarks:
        landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
        ]
        feat = []
        for lm_enum in landmarks:
            point = pose_results.pose_landmarks.landmark[lm_enum]
            feat.extend([point.x, point.y, point.z])
        return feat
    else:
        return [0.0] * (8 * 3)

def extract_full_features(hand_results, face_results, pose_results, mp_pose):
    """
    Combines hand, face, and pose features into one feature vector.
    (For classifier training, this vector is typically high-dimensional.)
    """
    features1 = extract_all_features(hand_results, face_results)
    features2 = extract_pose_features(pose_results, mp_pose)
    return features1 + features2

# ------------------------
# FIXED-LENGTH MOTION FEATURES FOR CLUSTERING
# ------------------------
def extract_motion_features(hand_results, face_results, pose_results, mp_pose):
    """
    Extracts a fixed-length 15-dimensional feature vector for clustering.
    The vector includes:
      - Hands: left and right wrist positions (3 coordinates each; if a hand is missing, zeros are used)
      - Face: nose tip position (3 coordinates; zeros if missing)
      - Pose: left and right shoulder positions (3 coordinates each; zeros if missing)
    Total length: 3 + 3 + 3 + 6 = 15.
    """
    # Hands: fixed length 6
    left_hand = [0.0, 0.0, 0.0]
    right_hand = [0.0, 0.0, 0.0]
    if hand_results.multi_hand_landmarks:
        hands = list(hand_results.multi_hand_landmarks)
        if len(hands) >= 2:
            hands = sorted(hands, key=lambda hand: hand.landmark[mp.solutions.hands.HandLandmark.WRIST].x)
            left_hand = [hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                         hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                         hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST].z]
            right_hand = [hands[1].landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                          hands[1].landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                          hands[1].landmark[mp.solutions.hands.HandLandmark.WRIST].z]
        else:
            wrist = hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST]
            if wrist.x < 0.5:
                left_hand = [wrist.x, wrist.y, wrist.z]
            else:
                right_hand = [wrist.x, wrist.y, wrist.z]
    # Face: fixed length 3
    face_feat = [0.0, 0.0, 0.0]
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        if len(face.landmark) > 1:
            nose_tip = face.landmark[1]
            face_feat = [nose_tip.x, nose_tip.y, nose_tip.z]
    # Pose: fixed length 6 (left and right shoulders)
    pose_feat = [0.0] * 6
    if pose_results and pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        pose_feat = [left_shoulder.x, left_shoulder.y, left_shoulder.z,
                     right_shoulder.x, right_shoulder.y, right_shoulder.z]
    return left_hand + right_hand + face_feat + pose_feat

# ------------------------
# KEY CLASSIFIER (SVM)
# ------------------------
class MyGestureClassifier:
    def __init__(self):
        self.samples = []  # list of feature vectors
        self.labels = []   # corresponding key labels
        self.model = None

    def train(self):
        if not self.samples:
            print("No samples to train on.")
            return
        X = np.array(self.samples)
        y = np.array(self.labels)
        self.model = SVC(probability=True, kernel="linear")
        self.model.fit(X, y)
        print(f"Classifier trained on {len(self.samples)} samples.")

    def update(self, new_samples):
        for feat, label in new_samples:
            self.samples.append(feat)
            self.labels.append(label)
        if len(set(self.labels)) < 2:
            print("Not enough classes to train. Waiting for more samples...")
            return
        self.train()

    def predict(self, features):
        if self.model is None:
            return ("none", 0.0)
        feat_array = np.array(features).reshape(1, -1)
        pred = self.model.predict(feat_array)[0]
        prob = self.model.predict_proba(feat_array).max()
        return (pred, prob)

def initialize_classifier():
    return MyGestureClassifier()

def update_classifier(classifier, samples):
    classifier.update(samples)
    return classifier

# ------------------------
# PERFORMANCE METRICS HELPERS (for research logging)
# ------------------------
def update_performance_metrics(actual, predicted, confidence, timestamp):
    global performance_metrics
    performance_metrics["total_predictions"] += 1
    correct = (actual == predicted)
    if correct:
        performance_metrics["correct_predictions"] += 1
    performance_metrics["prediction_history"].append({
        "timestamp": timestamp,
        "actual": actual,
        "predicted": predicted,
        "confidence": confidence,
        "correct": correct
    })

def compute_training_accuracy(classifier):
    if classifier.model is None or not classifier.samples:
        return 0.0
    X = np.array(classifier.samples)
    y = np.array(classifier.labels)
    preds = classifier.model.predict(X)
    correct = sum(1 for p, a in zip(preds, y) if p == a)
    return correct / len(y)

def save_performance_metrics():
    with open("performance_metrics.json", "w") as f:
        json.dump(performance_metrics, f, indent=4)
    print("Performance metrics saved.")

# ------------------------
# SYSTEM ACTION EXECUTION
# ------------------------
def execute_system_action(action):
    print(f"Executing system action: {action}")
    if action == "volume_up":
        os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) + 10)'")
    elif action == "volume_down":
        os.system("osascript -e 'set volume output volume ((output volume of (get volume settings)) - 10)'")
    elif action == "next_song":
        print("Action: Next Song executed.")
    elif action == "pause":
        print("Action: Pause executed.")
    else:
        print("No valid action to execute.")

# ------------------------
# POSE VISUALIZATION (MediaPipe-based; can be replaced with ViTPose integration)
# ------------------------
def draw_pose_info(frame, pose_results, mp_pose):
    if pose_results.pose_landmarks:
        h, w, _ = frame.shape
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
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        # Draw connections for arms and torso.
        if (mp_pose.PoseLandmark.LEFT_SHOULDER in pts and
            mp_pose.PoseLandmark.LEFT_ELBOW in pts and
            mp_pose.PoseLandmark.LEFT_WRIST in pts):
            cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_SHOULDER], pts[mp_pose.PoseLandmark.LEFT_ELBOW], (0,255,255), 2)
            cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_ELBOW], pts[mp_pose.PoseLandmark.LEFT_WRIST], (0,255,255), 2)
        if (mp_pose.PoseLandmark.RIGHT_SHOULDER in pts and
            mp_pose.PoseLandmark.RIGHT_ELBOW in pts and
            mp_pose.PoseLandmark.RIGHT_WRIST in pts):
            cv2.line(frame, pts[mp_pose.PoseLandmark.RIGHT_SHOULDER], pts[mp_pose.PoseLandmark.RIGHT_ELBOW], (0,255,255), 2)
            cv2.line(frame, pts[mp_pose.PoseLandmark.RIGHT_ELBOW], pts[mp_pose.PoseLandmark.RIGHT_WRIST], (0,255,255), 2)
        if (mp_pose.PoseLandmark.LEFT_SHOULDER in pts and mp_pose.PoseLandmark.RIGHT_SHOULDER in pts):
            cv2.line(frame, pts[mp_pose.PoseLandmark.LEFT_SHOULDER], pts[mp_pose.PoseLandmark.RIGHT_SHOULDER], (0,255,255), 2)
    return frame

# ------------------------
# DEBUG & OVERLAY DRAWING
# ------------------------
def draw_debug_info(frame, hand_results, face_results, pose_results, state, mp_pose=None):
    h, w, _ = frame.shape
    cv2.putText(frame, f"Mode: {state['mode']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Predicted: {state['predicted_action']} ({state['confidence']:.2f})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    if pose_results and mp_pose is not None:
        frame = draw_pose_info(frame, pose_results, mp_pose)
    return frame

# ------------------------
# KEYBOARD LISTENER (captures key press events)
# ------------------------
def on_key_press(key):
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
# OPTIONAL: CLUSTERING OF USER MOTION (for research)
# ------------------------
def cluster_user_movements(motion_data):
    if len(motion_data) < 10:
        print("Not enough data for clustering.")
        return None
    X = np.array(motion_data)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    print("User movement clusters:", labels)
    return labels

# ------------------------
# FLASK WEB APP (for mode toggling & manual confirmation)
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
# MAIN APPLICATION LOOP
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

    # Initialize MediaPipe solutions.
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    face_detector = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    gesture_classifier = initialize_classifier()
    sample_count = 0  # Count of training samples collected in this cycle
    pred_buffer = []  # Buffer for aggregating features for prediction
    last_action_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)  # Mirror the image for natural interaction
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using MediaPipe.
        hand_results = hands_detector.process(rgb_frame)
        face_results = face_detector.process(rgb_frame)
        pose_results = pose_detector.process(rgb_frame)

        # Save the current frame with timestamp into the frame buffer.
        frame_buffer.append((frame.copy(), time.time()))

        # Extract full features (for training the classifier).
        full_features = extract_full_features(hand_results, face_results, pose_results, mp_pose)

        # Use a window of frames to smooth predictions.
        if full_features:
            pred_buffer.append(full_features)
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
        # Process Key Press Events (for sample collection and performance logging)
        # -------------------------
        if key_press_buffer:
            for key_name, press_time in key_press_buffer:
                if shared_state["mode"] == "training":
                    if full_features:
                        training_samples.append((full_features, key_name))
                        samples_buffer.append((full_features, key_name))
                        sample_count += 1
                        print(f"Training sample added for key '{key_name}'. Total samples: {sample_count}")
                        save_augmented_sample(frame, key_name, sample_count)
                        if sample_count >= SAMPLE_THRESHOLD:
                            print(f"Collected {SAMPLE_THRESHOLD} samples. Updating classifier...")
                            gesture_classifier = update_classifier(gesture_classifier, samples_buffer)
                            acc = compute_training_accuracy(gesture_classifier)
                            print(f"Training accuracy: {acc*100:.2f}%")
                            with open("gesture_classifier_progress.pkl", "wb") as f:
                                pickle.dump(gesture_classifier, f)
                            sample_count = 0
                            samples_buffer = []
                elif shared_state["mode"] == "production":
                    actual_key = key_name
                    predicted_key = shared_state["predicted_action"]
                    conf = shared_state["confidence"]
                    update_performance_metrics(actual_key, predicted_key, conf, press_time)
                    print(f"Production mode: Actual: {actual_key}, Predicted: {predicted_key}, Confidence: {conf:.2f}")
                    total = performance_metrics["total_predictions"]
                    correct = performance_metrics["correct_predictions"]
                    if total > 0 and (correct / total) > 0.8 and conf > CONFIDENCE_THRESHOLD:
                        execute_system_action(predicted_key)
                        last_action_time = time.time()
                    save_performance_metrics()
            key_press_buffer.clear()

        # -------------------------
        # OPTIONAL: Cluster motion data (for research)
        # -------------------------
        motion_feat = extract_motion_features(hand_results, face_results, pose_results, mp_pose)
        if motion_feat:
            motion_data.append(motion_feat)
        if len(motion_data) > 50:
            cluster_user_movements(motion_data)
            motion_data.clear()

        # -------------------------
        # In production mode, auto-act if confidence is high and enough time has passed.
        # -------------------------
        if shared_state["mode"] == "production":
            if len(pred_buffer) == MOTION_WINDOW:
                if (shared_state["confidence"] > CONFIDENCE_THRESHOLD) and ((time.time() - last_action_time) > 2):
                    total = performance_metrics["total_predictions"]
                    correct = performance_metrics["correct_predictions"]
                    if total > 0 and (correct / total) > 0.8:
                        execute_system_action(shared_state["predicted_action"])
                        last_action_time = time.time()

        # -------------------------
        # Draw debug overlay (with pose info).
        # -------------------------
        debug_frame = draw_debug_info(frame, hand_results, face_results, pose_results, shared_state, mp_pose=mp_pose)
        cv2.imshow("Key Press Gesture Control", debug_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit if ESC is pressed.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()