#!/usr/bin/env python3
"""
integrated_keypress_gesture_pose_fuzzy.py

Workflow:
1. DATA COLLECTION & TRAINING:
   - Continuously monitors your video (hands, face, and body pose).
   - When you press a designated key (e.g. "volume_up", "volume_down", "next_song", "pause"),
     the current frame’s extracted features (from hands, face, and pose) are saved as a training sample
     with that key as its label. An augmented (fuzzy-distorted) image is also saved.
2. PRODUCTION:
   - In production mode the classifier predicts the key based on live features.
   - Key press events are compared with predictions and logged.
3. VISUALIZATION:
   - The background is removed (using MediaPipe SelfieSegmentation).
   - Left hand landmarks are drawn in blue; right hand landmarks in red; face landmarks in green.
   - Pose landmarks (upper-body) are drawn in yellow.
   - Recent key presses are overlaid on the video.
4. ACTIVITY MODEL:
   - Each key press event also saves a fixed-length movement feature vector (with time codes) into an activity log.
   - When enough entries are collected, these are clustered and saved as a secondary model called "activity_model.pkl".
5. WEB INTERFACE:
   - A Flask app (in a separate thread) allows toggling between training and production modes.
"""

import os
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
# GLOBAL STATE & BUFFERS
# ------------------------
shared_state = {
    "mode": "production",  # "training" or "production"
    "predicted_action": "none",
    "confidence": 0.0,
    "recent_key_presses": []  # list of (key, timestamp) to show on overlay
}

# Frame buffer (for keyframe extraction)
BUFFER_SECONDS = 5
FPS = 30
FRAME_BUFFER_SIZE = BUFFER_SECONDS * FPS
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)  # Each entry: (frame, timestamp)

# Key press events (for sample collection and logging)
key_press_buffer = []

# Training samples: list of tuples (feature_vector, key_label)
training_samples = []
samples_buffer = []  # temporary storage for current cycle

# Activity log: list of dicts with {"timestamp":..., "movement": fixed_length_vector}
activity_log = []

# Performance metrics for research logging
performance_metrics = {
    "total_predictions": 0,
    "correct_predictions": 0,
    "prediction_history": []  # each: {timestamp, actual, predicted, confidence, correct}
}

# ------------------------
# CONFIGURABLE CONSTANTS
# ------------------------
SAMPLE_THRESHOLD = 10         # samples before retraining classifier
CONFIDENCE_THRESHOLD = 0.7    # production confidence threshold
MOTION_WINDOW = 3             # window size for smoothing predictions
ACTIVITY_LOG_THRESHOLD = 50   # when to cluster activity log and update activity_model

# Designated keys for training/actions.
KEY_ACTIONS = {
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "next_song": "next_song",
    "pause": "pause"
}

# ------------------------
# BACKGROUND REMOVAL (SelfieSegmentation)
# ------------------------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# ------------------------
# DATA AUGMENTATION: FUZZY DISTORTION
# ------------------------
def fuzzy_distortion(image, strength=5, kernel_size=3):
    """Applies slight random displacement and blur to simulate variability."""
    rows, cols = image.shape[:2]
    dx = np.random.uniform(-strength, strength, size=(rows, cols))
    dy = np.random.uniform(-strength, strength, size=(rows, cols))
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = (map_x.astype(np.float32) + dx).clip(0, cols - 1)
    map_y = (map_y.astype(np.float32) + dy).clip(0, rows - 1)
    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    if kernel_size > 1:
        distorted = cv2.GaussianBlur(distorted, (kernel_size, kernel_size), 0)
    return distorted

def save_augmented_sample(image, label, count):
    """Saves a fuzzy-distorted image to disk."""
    aug_image = fuzzy_distortion(image, strength=5, kernel_size=3)
    filename = f"aug_sample_{label}_{count}.jpg"
    cv2.imwrite(filename, aug_image)
    print(f"Saved augmented sample: {filename}")

# ------------------------
# HELPER: BEEP
# ------------------------
def beep():
    os.system("afplay /System/Library/Sounds/Ping.aiff")

# ------------------------
# HELPER: FLATTEN LANDMARKS
# ------------------------
def flatten_hand_landmarks(hand_landmarks, expected_count=21):
    if hand_landmarks is None:
        return [0.0]*(expected_count*3)
    vec = []
    for lm in hand_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

def flatten_face_landmarks(face_landmarks, expected_count=468):
    if face_landmarks is None:
        return [0.0]*(expected_count*3)
    vec = []
    for lm in face_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    return vec

# ------------------------
# FEATURE EXTRACTION (Hands, Face, Pose)
# ------------------------
def extract_all_features(hand_results, face_results):
    """Concatenates hand (2x21x3) and face (468x3) features."""
    left_hand = [0.0]*(21*3)
    right_hand = [0.0]*(21*3)
    if hand_results.multi_hand_landmarks:
        hands = list(hand_results.multi_hand_landmarks)
        hands = sorted(hands, key=lambda h: h.landmark[mp.solutions.hands.HandLandmark.WRIST].x)
        if len(hands) >= 2:
            left_hand = flatten_hand_landmarks(hands[0])
            right_hand = flatten_hand_landmarks(hands[1])
        else:
            wrist = hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST]
            if wrist.x < 0.5:
                left_hand = flatten_hand_landmarks(hands[0])
            else:
                right_hand = flatten_hand_landmarks(hands[0])
    face = [0.0]*(468*3)
    if face_results.multi_face_landmarks:
        face = flatten_face_landmarks(face_results.multi_face_landmarks[0])
    return left_hand + right_hand + face

def extract_pose_features(pose_results, mp_pose):
    """Extracts pose features from eight key landmarks."""
    if pose_results and pose_results.pose_landmarks:
        landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
        ]
        feat = []
        for lm in landmarks:
            point = pose_results.pose_landmarks.landmark[lm]
            feat.extend([point.x, point.y, point.z])
        return feat
    else:
        return [0.0]*(8*3)

def extract_full_features(hand_results, face_results, pose_results, mp_pose):
    """Combines hand, face, and pose features into one vector."""
    return extract_all_features(hand_results, face_results) + extract_pose_features(pose_results, mp_pose)

# ------------------------
# FIXED-LENGTH MOTION FEATURES (for activity_model clustering)
# ------------------------
def extract_motion_features(hand_results, face_results, pose_results, mp_pose):
    """
    Returns a 15-dimensional vector:
      - Hands: left and right wrist positions (3 coords each)
      - Face: nose tip (3 coords)
      - Pose: left and right shoulders (3 coords each)
    """
    left_wrist = [0.0, 0.0, 0.0]
    right_wrist = [0.0, 0.0, 0.0]
    if hand_results.multi_hand_landmarks:
        hands = list(hand_results.multi_hand_landmarks)
        if len(hands) >= 2:
            hands = sorted(hands, key=lambda h: h.landmark[mp.solutions.hands.HandLandmark.WRIST].x)
            left_wrist = [hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                          hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                          hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST].z]
            right_wrist = [hands[1].landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                           hands[1].landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                           hands[1].landmark[mp.solutions.hands.HandLandmark.WRIST].z]
        else:
            wrist = hands[0].landmark[mp.solutions.hands.HandLandmark.WRIST]
            if wrist.x < 0.5:
                left_wrist = [wrist.x, wrist.y, wrist.z]
            else:
                right_wrist = [wrist.x, wrist.y, wrist.z]
    # Face: nose tip (index 1) – 3 values
    nose = [0.0, 0.0, 0.0]
    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        if len(face.landmark) > 1:
            nose_pt = face.landmark[1]
            nose = [nose_pt.x, nose_pt.y, nose_pt.z]
    # Pose: shoulders (each 3 values)
    left_shoulder = [0.0, 0.0, 0.0]
    right_shoulder = [0.0, 0.0, 0.0]
    if pose_results and pose_results.pose_landmarks:
        left_shoulder = [pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                         pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
        right_shoulder = [pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                          pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]
    return left_wrist + right_wrist + nose + left_shoulder + right_shoulder

# ------------------------
# KEY CLASSIFIER (SVM)
# ------------------------
class MyGestureClassifier:
    def __init__(self):
        self.samples = []
        self.labels = []
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
# PERFORMANCE METRICS HELPERS
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
# VISUALIZATION: HAND & FACE
# ------------------------
def draw_hand_and_face_info(frame, hand_results, face_results):
    h, w, _ = frame.shape
    # Draw hand landmarks: left in blue, right in red.
    if hand_results.multi_hand_landmarks:
        hands = list(hand_results.multi_hand_landmarks)
        # Sort by wrist x coordinate.
        hands = sorted(hands, key=lambda h: h.landmark[mp.solutions.hands.HandLandmark.WRIST].x)
        for idx, hand_landmarks in enumerate(hands):
            # Choose color: left hand = blue, right hand = red.
            color = (255, 0, 0) if idx == 0 else (0, 0, 255)
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, color, -1)
    # Draw face landmarks in green.
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        for lm in face_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
    return frame

# ------------------------
# VISUALIZATION: POSE
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
# DEBUG & OVERLAY DRAWING (including recent key presses)
# ------------------------
def draw_debug_info(frame, hand_results, face_results, pose_results, state, mp_pose=None):
    h, w, _ = frame.shape
    cv2.putText(frame, f"Mode: {state['mode']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Predicted: {state['predicted_action']} ({state['confidence']:.2f})", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # Display recent key presses.
    y0 = 110
    for i, (k, tstamp) in enumerate(state.get("recent_key_presses", [])):
        cv2.putText(frame, f"{k} @ {int(tstamp)}", (10, y0 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    frame = draw_hand_and_face_info(frame, hand_results, face_results)
    if pose_results and mp_pose is not None:
        frame = draw_pose_info(frame, pose_results, mp_pose)
    return frame

# ------------------------
# BACKGROUND REMOVAL
# ------------------------
def remove_background(frame, rgb_frame):
    """Uses MediaPipe SelfieSegmentation to remove the background (black background)."""
    seg_results = selfie_segmentation.process(rgb_frame)
    mask = seg_results.segmentation_mask
    condition = mask > 0.5
    # Create a black background.
    bg_image = np.zeros(frame.shape, dtype=np.uint8)
    # Composite: if condition true, use frame pixel; else use bg pixel.
    output_image = np.where(condition[..., None], frame, bg_image)
    return output_image

# ------------------------
# KEYBOARD LISTENER
# ------------------------
def on_key_press(key):
    try:
        key_name = key.char if hasattr(key, 'char') and key.char is not None else str(key)
        if key_name in KEY_ACTIONS:
            timestamp = time.time()
            key_press_buffer.append((key_name, timestamp))
            # Also update recent key presses (keep only last 5)
            shared_state.setdefault("recent_key_presses", []).append((key_name, timestamp))
            if len(shared_state["recent_key_presses"]) > 5:
                shared_state["recent_key_presses"] = shared_state["recent_key_presses"][-5:]
            print(f"Captured key press: {key_name} at {timestamp}")
    except Exception as e:
        print("Error capturing key press:", e)

keyboard_listener = keyboard.Listener(on_press=on_key_press)
keyboard_listener.daemon = True
keyboard_listener.start()

# ------------------------
# OPTIONAL: CLUSTERING OF USER MOTION (for activity_model)
# ------------------------
def cluster_user_movements(motion_data):
    if len(motion_data) < 10:
        print("Not enough data for clustering activity.")
        return None
    X = np.array(motion_data)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)
    print("Activity model clusters:", labels)
    return {"cluster_centers": kmeans.cluster_centers_.tolist(), "labels": labels.tolist()}

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
    # Start the Flask server.
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
    sample_count = 0
    pred_buffer = []
    last_action_time = time.time()

    # Global activity log for key presses with movement features.
    global activity_log

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip for natural interaction.
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Remove background.
        frame = remove_background(frame, rgb_frame)

        # Process MediaPipe detectors.
        hand_results = hands_detector.process(rgb_frame)
        face_results = face_detector.process(rgb_frame)
        pose_results = pose_detector.process(rgb_frame)

        # Save current frame with timestamp.
        frame_buffer.append((frame.copy(), time.time()))

        # Extract full features.
        full_features = extract_full_features(hand_results, face_results, pose_results, mp_pose)

        # Aggregate features for smoothing predictions.
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
        # Process Key Press Events.
        # -------------------------
        if key_press_buffer:
            for key_name, press_time in key_press_buffer:
                # Save the movement features for the activity model.
                movement_features = extract_motion_features(hand_results, face_results, pose_results, mp_pose)
                activity_log.append({"timestamp": press_time, "movement": movement_features})
                # If the log exceeds threshold, cluster and save as activity_model.
                if len(activity_log) >= ACTIVITY_LOG_THRESHOLD:
                    activity_model = cluster_user_movements([entry["movement"] for entry in activity_log])
                    # Also save time codes along with labels.
                    time_codes = [entry["timestamp"] for entry in activity_log]
                    with open("activity_model.pkl", "wb") as f:
                        pickle.dump({"activity_model": activity_model, "time_codes": time_codes}, f)
                    print("Saved activity_model with time codes.")
                    activity_log = []  # reset log

                if shared_state["mode"] == "training":
                    if full_features:
                        training_samples.append((full_features, key_name))
                        samples_buffer.append((full_features, key_name))
                        sample_count += 1
                        print(f"Training sample for key '{key_name}'. Total samples: {sample_count}")
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
                    print(f"Production: Actual: {actual_key}, Predicted: {predicted_key}, Confidence: {conf:.2f}")
                    total = performance_metrics["total_predictions"]
                    correct = performance_metrics["correct_predictions"]
                    if total > 0 and (correct / total) > 0.8 and conf > CONFIDENCE_THRESHOLD:
                        execute_system_action(predicted_key)
                        last_action_time = time.time()
                    save_performance_metrics()
            key_press_buffer.clear()

        # -------------------------
        # OPTIONAL: Cluster motion data (for research) if desired.
        # -------------------------
        motion_feat = extract_motion_features(hand_results, face_results, pose_results, mp_pose)
        if motion_feat:
            # (You can choose to also add to a separate motion log if needed.)
            pass

        # -------------------------
        # Production mode auto-act.
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
        # Draw debug overlay.
        # -------------------------
        debug_frame = draw_debug_info(frame, hand_results, face_results, pose_results, shared_state, mp_pose=mp_pose)
        cv2.imshow("Key Press Gesture Control", debug_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit.
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()