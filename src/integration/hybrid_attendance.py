import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
import tensorflow as tf
import csv
import time
from datetime import datetime
from collections import deque
from src.integration.emotion_detector import EmotionDetector
from src.integration.emotion_detector_efficientnet import EmotionDetectorEfficientNet
from src.integration.emotion_detector_transformer import EmotionDetectorTransformer
from src.integration.glasses_detector import GlassesDetector

class HybridAttendanceSystem:
    """
    Manages the integrated real-time identity recognition, automated enrollment, 
    spatiotemporal lip-reading, and emotion detection pipelines via webcam.
    """
    def __init__(self):
        """
        Initializes the HybridAttendanceSystem and loads required models.
        """
        self.attendance_dir = "src/attendance"
        self.db_path = os.path.join(self.attendance_dir, "faces_db")
        self.log_file = os.path.join(self.attendance_dir, "attendance_log.csv")
        
        self.enroll_after = 5.0
        self.exit_grace = 3.0
        self.seq_length = 15
        
        self.db_embeddings = {}
        self.tracked_faces = {}
        self.next_id = 0
        
        self.embedding_model = None
        self.lip_model = None
        self.emotion_detector = None
        self.glasses_detector = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.is_initialized = False

        self.config = {
            "identity_active": True,
            "similarity_threshold": 0.80,
            "spoofing_active": False,
            "emotion_active": False,
            "lip_active": False,
            "face_min_size": 100,
            "lip_movement_threshold": 8.0,
            "identity_model_path": 'models/checkpoints/mlp_best.h5',
            "lip_model_path": 'models/checkpoints/LipCNNLSTM_best.h5',
            "emotion_model_name": 'models/emotion_detection/residual/emotion_cnn_residual.keras',
            "glasses_model_path": 'models/glasses_detection/residual/glasses_detector_residual_cnn.keras'
        }

    def _make_emotion_detector(self, model_path):
        """Instantiate the correct detector class based on the model filename."""
        name = os.path.basename(model_path).lower()
        if "efficientnet" in name:
            return EmotionDetectorEfficientNet(model_path=model_path)
        if "transformer" in name:
            return EmotionDetectorTransformer(model_path=model_path)
        return EmotionDetector(model_path=model_path)

    def reload_models(self):
        """
        Dynamically unloads current models and loads the ones specified in self.config.
        """
        os.makedirs(self.db_path, exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(["name", "event", "spoken_statement", "emotion", "spoof_status", "timestamp"])

        if not os.path.exists(self.config["identity_model_path"]):
            print(f"Error: Identity model not found at {self.config['identity_model_path']}")
            self.embedding_model = None
        else:
            base_identity_model = tf.keras.models.load_model(self.config["identity_model_path"], compile=False)
            target_layer = None
            for layer in reversed(base_identity_model.layers):
                if hasattr(layer, 'output_shape') and layer.output_shape[-1] == 128:
                    target_layer = layer
                    break
            if target_layer is None:
                target_layer = base_identity_model.layers[-2]

            try:
                model_inputs = base_identity_model.inputs
            except AttributeError:
                model_inputs = base_identity_model.input

            self.embedding_model = tf.keras.Model(inputs=model_inputs, outputs=target_layer.output)

        if os.path.exists(self.config["lip_model_path"]):
            class LegacyBatchNormalization(tf.keras.layers.BatchNormalization):
                def __init__(self, axis=-1, **kwargs):
                    if isinstance(axis, list) and len(axis) == 1:
                        axis = axis[0]
                    super().__init__(axis=axis, **kwargs)

            class LegacyLSTM(tf.keras.layers.LSTM):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('time_major', None)
                    super().__init__(*args, **kwargs)

            custom_objs = {
                'Conv2D': tf.keras.layers.Conv2D,
                'MaxPooling2D': tf.keras.layers.MaxPooling2D,
                'Flatten': tf.keras.layers.Flatten,
                'Dense': tf.keras.layers.Dense,
                'Dropout': tf.keras.layers.Dropout,
                'LSTM': LegacyLSTM,
                'TimeDistributed': tf.keras.layers.TimeDistributed,
                'Sequential': tf.keras.Sequential,
                'BatchNormalization': LegacyBatchNormalization,
                'Activation': tf.keras.layers.Activation,
                'ZeroPadding2D': tf.keras.layers.ZeroPadding2D,
                'GlobalAveragePooling2D': tf.keras.layers.GlobalAveragePooling2D,
                'LeakyReLU': tf.keras.layers.LeakyReLU
            }
            self.lip_model = tf.keras.models.load_model(self.config["lip_model_path"], compile=False, custom_objects=custom_objs)
        else:
            print(f"Warning: Lip model not found at {self.config['lip_model_path']}. Lip reading disabled.")
            self.lip_model = None

        try:
            self.emotion_detector = self._make_emotion_detector(self.config["emotion_model_name"])
        except Exception as e:
            print(f"Warning: Failed to load Emotion Detector: {e}")
            self.emotion_detector = None

        try:
            self.glasses_detector = GlassesDetector(model_path=self.config["glasses_model_path"])
        except Exception as e:
            print(f"Warning: Failed to load Glasses Detector: {e}")
            self.glasses_detector = None

        self._update_db_cache()
        self.is_initialized = True
        return True

    def initialize_system(self):
        if not self.is_initialized:
            self.reload_models()

    def _update_db_cache(self):
        """
        Refreshes the dictionary of known identities and their embeddings from the database directory.
        """
        self.db_embeddings = {}
        if self.embedding_model is None:
            return
            
        for person_dir in os.listdir(self.db_path):
            p_path = os.path.join(self.db_path, person_dir)
            if os.path.isdir(p_path):
                embs = []
                for img_name in os.listdir(p_path):
                    img_path = os.path.join(p_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (112, 112))
                        img = np.expand_dims(img, axis=-1)
                        img = np.expand_dims(img, axis=0)
                        emb = self.embedding_model.predict(img, verbose=0)[0]
                        embs.append(emb)
                if embs:
                    self.db_embeddings[person_dir] = embs

    def _cosine_similarity(self, emb1, emb2):
        """
        Computes the cosine similarity between two feature vectors.
        """
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

    def _recognize_face(self, face_img):
        """
        Compares a detected face against the cached database embeddings.
        """
        if not self.db_embeddings or self.embedding_model is None:
            return None
        
        img = cv2.resize(face_img, (112, 112))
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        target_emb = self.embedding_model.predict(img, verbose=0)[0]
        
        best_match = None
        best_sim = 0
        
        for person, embs in self.db_embeddings.items():
            for e in embs:
                sim = self._cosine_similarity(target_emb, e)
                if sim > best_sim:
                    best_sim = sim
                    best_match = person
                    
        if best_sim > self.config["similarity_threshold"]:
            return best_match
        return None

    def _log_event(self, name: str, event: str, statement: str = "", emotion: str = "", spoof_status: str = ""):
        """
        Appends an event to the attendance CSV and prints cleanly to the console.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([name, event, statement, emotion, spoof_status, timestamp])
        
        log_str = f"[{timestamp}] {event}: {name}"
        if statement:
            log_str += f" | Spoke: '{statement}'"
        if emotion:
            log_str += f" | Emotion: {emotion}"
        if spoof_status:
            log_str += f" | Status: {spoof_status}"
            
        print(log_str)

    def process_frame(self, frame):
        """
        Processes a single BGR frame, tracking faces, inferencing all 3 models,
        and drawing annotations on the frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_size = int(self.config["face_min_size"])
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
        
        current_matched_ids = set()
        
        for (x, y, w, h) in faces:
            cx, cy = x + w//2, y + h//2
            best_id = None
            best_dist = 100
            
            for fid, info in self.tracked_faces.items():
                ox, oy = info["cx"], info["cy"]
                dist = abs(cx - ox) + abs(cy - oy)
                if dist < best_dist:
                    best_id = fid
                    best_dist = dist
                    
            if best_id is None:
                self.next_id += 1
                best_id = self.next_id
                self.tracked_faces[best_id] = {
                    "name": "Unknown",
                    "cx": cx, "cy": cy, "bbox": (x, y, w, h),
                    "unknown_since": time.time(),
                    "missing_since": None,
                    "lip_buffer": deque(maxlen=self.seq_length),
                    "last_lip_time": 0.0,
                    "spoken": "",
                    "emotion_label": "",
                    "emotion_emoji": "",
                    "spoof_status": ""
                }
            
            current_matched_ids.add(best_id)
            info = self.tracked_faces[best_id]
            info["cx"], info["cy"] = cx, cy
            info["bbox"] = (x, y, w, h)
            info["missing_since"] = None
            
            # Anti-spoofing (glasses) pathway
            if self.config["spoofing_active"] and self.glasses_detector is not None:
                face_crop_bgr = frame[y:y+h, x:x+w]
                if face_crop_bgr.size > 0:
                    try:
                        spoof_res = self.glasses_detector.predict_with_decision(face_crop_bgr)
                        if spoof_res["decision"] == "block":
                            info["spoof_status"] = "SPOOF (Sunglasses)"
                        else:
                            info["spoof_status"] = "Passed"
                    except Exception as e:
                        pass
            else:
                info["spoof_status"] = ""

            # Emotion pathway
            if self.config["emotion_active"] and self.emotion_detector is not None:
                face_crop_bgr = frame[y:y+h, x:x+w]
                if face_crop_bgr.size > 0:
                    emotion_result = self.emotion_detector.predict_from_face(face_crop_bgr)
                    if emotion_result.get("face_detected"):
                        emotion = emotion_result["emotion"]
                        emoji = emotion_result["emoji"]
                        info["emotion_label"] = f"{emotion} ({emotion_result['confidence']:.2f})"
                        info["emotion_emoji"] = emoji
            else:
                info["emotion_label"] = ""
                info["emotion_emoji"] = ""

            # Identity pathway
            if self.config["identity_active"] and self.embedding_model is not None:
                y_id = y + int(0.15 * h)
                h_id = int(0.70 * h)
                x_id = x + int(0.15 * w)
                w_id = int(0.70 * w)
                face_crop_gray = gray[y_id:y_id+h_id, x_id:x_id+w_id]
                
                if info["name"] == "Unknown" and face_crop_gray.size > 0:
                    name = self._recognize_face(face_crop_gray)
                    if name:
                        info["name"] = name
                        info["unknown_since"] = None
                        curr_emotion = info["emotion_label"].split(" ")[0] if info["emotion_label"] else ""
                        self._log_event(name, "ENTER", emotion=curr_emotion, spoof_status=info["spoof_status"])
                    elif time.time() - info["unknown_since"] > self.enroll_after:
                        nums = [int(d.split("_")[1]) for d in os.listdir(self.db_path) if d.startswith("Person_")]
                        new_num = max(nums, default=0) + 1
                        new_name = f"Person_{new_num:03d}"
                        os.makedirs(os.path.join(self.db_path, new_name), exist_ok=True)
                        cv2.imwrite(os.path.join(self.db_path, new_name, "auto_enroll.jpg"), face_crop_gray)
                        self._update_db_cache()
                        info["name"] = new_name
                        info["unknown_since"] = None
                        curr_emotion = info["emotion_label"].split(" ")[0] if info["emotion_label"] else ""
                        self._log_event(new_name, "ENTER (Auto-Enrolled)", emotion=curr_emotion, spoof_status=info["spoof_status"])
            else:
                info["name"] = "Unknown"

            # Lip reading pathway
            if self.config["lip_active"] and self.lip_model is not None:
                y_lip = y + int(0.50 * h)
                h_lip = int(0.50 * h)
                x_lip = x + int(0.10 * w)
                w_lip = int(0.80 * w)
                lip_crop = gray[y_lip:y_lip+h_lip, x_lip:x_lip+w_lip]
                
                if lip_crop.size > 0:
                    if time.time() - info["last_lip_time"] > 0.1:
                        lip_resized = cv2.resize(lip_crop, (64, 64))
                        info["lip_buffer"].append(lip_resized)
                        info["last_lip_time"] = time.time()
                    
                    if len(info["lip_buffer"]) == self.seq_length:
                        seq = np.array(info["lip_buffer"])
                        diffs = np.abs(seq[1:].astype(np.float32) - seq[:-1].astype(np.float32))
                        movement = np.mean(diffs)
                        
                        if movement > float(self.config["lip_movement_threshold"]):
                            seq_input = np.expand_dims(seq, axis=-1)
                            seq_input = np.expand_dims(seq_input, axis=0)
                            
                            pred = self.lip_model.predict(seq_input, verbose=0)[0]
                            class_id = np.argmax(pred)
                            statement = "Kids are talking" if class_id == 0 else "Dogs are sitting"
                            
                            if info["spoken"] != statement:
                                info["spoken"] = statement
                                curr_emotion = info["emotion_label"].split(" ")[0] if info["emotion_label"] else ""
                                self._log_event(info["name"], "SPOKE", statement=statement, emotion=curr_emotion)
                            
                            info["lip_buffer"].clear()
                        else:
                            info["spoken"] = ""
            else:
                info["spoken"] = ""
                info["lip_buffer"].clear()
            
            # Annotation
            color = (0, 255, 0) if info["name"] != "Unknown" else (0, 0, 255)
            if info["spoof_status"] and "SPOOF" in info["spoof_status"]:
                color = (0, 0, 255) # Red for spoof

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, info["name"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if info["spoof_status"]:
                cv2.putText(frame, info["spoof_status"], (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if info["emotion_label"]:
                cv2.putText(frame, info["emotion_label"], (x + w + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            if info["spoken"]:
                cv2.putText(frame, info["spoken"], (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
        # Cleanup missing faces
        for fid in list(self.tracked_faces.keys()):
            if fid not in current_matched_ids:
                info = self.tracked_faces[fid]
                if info["missing_since"] is None:
                    info["missing_since"] = time.time()
                elif time.time() - info["missing_since"] > self.exit_grace:
                    if info["name"] != "Unknown":
                        self._log_event(info["name"], "EXIT")
                    del self.tracked_faces[fid]

        return frame

    def get_processed_frame(self, cap):
        self.initialize_system()
        ret, frame = cap.read()
        if not ret:
            return False, None
        try:
            return True, self.process_frame(frame)
        except Exception as e:
            print(f"Frame processing error: {e}")
            return True, frame
