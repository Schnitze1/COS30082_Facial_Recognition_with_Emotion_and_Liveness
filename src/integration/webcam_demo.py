import cv2
import numpy as np
import tensorflow as tf
import argparse
import os

class WebcamDemo:
    """
    Manages the standalone live webcam demo for the face recognition model.
    """
    def __init__(self, model_path: str):
        """
        Initializes the WebcamDemo system.
        
        :param model_path: String path to the trained model .h5 file.
        """
        self.model_path = model_path
        self.model = None
        self.face_cascade = None

    def initialize_system(self):
        """
        Loads the trained model and Haar cascade classifier.
        
        :return: Boolean indicating successful initialization.
        """
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}")
            return False

        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully.")

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        return True

    def run(self):
        """
        Initiates the primary webcam feed and prediction loop.
        """
        if not self.initialize_system():
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting webcam. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                y_new = y + int(0.15 * h)
                h_new = int(0.70 * h)
                x_new = x + int(0.15 * w)
                w_new = int(0.70 * w)

                cv2.rectangle(frame, (x_new, y_new), (x_new+w_new, y_new+h_new), (255, 0, 0), 1)

                face_img = gray[y_new:y_new+h_new, x_new:x_new+w_new]
                
                if face_img.size == 0:
                    continue

                face_resized = cv2.resize(face_img, (112, 112))
                face_input = np.expand_dims(face_resized, axis=-1)
                face_input = np.expand_dims(face_input, axis=0)

                pred = self.model(face_input, training=False).numpy()
                class_id = np.argmax(pred[0])
                confidence = pred[0][class_id] * 100

                label_text = f"Actor: {class_id:02d} ({confidence:.1f}%)"
                color = (0, 255, 0) if confidence > 50 else (0, 0, 255)
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Face Recognition Live', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Entry point for the Webcam Demo.
    """
    parser = argparse.ArgumentParser(description="Live Webcam Demo for Face Recognition")
    parser.add_argument('--model', type=str, default='models/checkpoints/mlp_best.h5', help='Path to trained model .h5')
    args = parser.parse_args()

    demo = WebcamDemo(model_path=args.model)
    demo.run()

if __name__ == '__main__':
    main()
