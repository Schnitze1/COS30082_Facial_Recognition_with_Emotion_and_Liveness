# Intelligent Systems - Hybrid Facial Recognition Pipeline

This project is a high-definition, web-based hybrid attendance system capable of simultaneous identity tracking, automated enrollment, spatiotemporal lip reading, and emotion detection using live webcam feeds. It utilizes a suite of custom-trained Deep Learning models (CNN, MLP, LSTM, and 3D-CNNs) built from scratch.

## Instructions

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.8+
* pip (Python package installer)
* A local webcam

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your_username/COS30082_Facial_Recognition_with_Emotion_and_Liveness.git
    cd COS30082_Facial_Recognition_with_Emotion_and_Liveness
    ```

2.  **Create and activate a virtual environment (recommended):**
    * **Windows (PowerShell):**
        ```sh
        python -m venv .venv
        .\.venv\Scripts\activate
        ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the automated data & training pipeline (Optional):**
    ```sh
    python main.py --all
    ```
    *This will extract video sequences, generate cropped facial datasets, and train both Identity and Lip Reading models automatically.*

5.  **Run the Flask application:**
    ```sh
    cd src/gui
    python app.py
    ```

6.  Open your web browser and navigate to `http://127.0.0.1:5000` to view the live Hybrid Attendance dashboard.

## Project Architecture

The application is built with a Python Flask backend serving a dynamic MJPEG stream via OpenCV. The backend encapsulates multiple OOP classes dedicated to model inference, temporal tracking, and CSV event logging.

### Application Flow

1.  **Webcam Capture**: Frames are grabbed using OpenCV.
2.  **Face Detection**: A Haar Cascade classifier detects faces in the frame.
3.  **Temporal Tracking**: Faces are mapped frame-by-frame using Euclidean distance and matched to unique trackers. Unknown faces are automatically enrolled after 5 seconds of continuous tracking.
4.  **Model Inference**:
    - **Identity (MLP)**: Upper face crops (15% margin) are passed to an embedding model (MLP). Predictions are measured using Cosine Similarity.
    - **Emotion (CNN)**: Full facial crops are passed to the teammate's Emotion CNN to estimate emotional states (Happy, Neutral, Sad, Anger, Fear, Surprise, Disgust, Contempt).
    - **Lip Reading (CNN-LSTM)**: The lower 50% of the face is isolated, resized, and appended to a rolling 15-frame buffer. The 3D sequence is fed into a Spatiotemporal Lip-Reading network.
5.  **Display Result**: Bounding boxes, labels, emojis, and predicted sentences are overlaid on the frame, and the processed frame is encoded as a JPEG and sent to the Flask GUI via MJPEG streaming.

## Project Structure

The project is organized into distinct modules for data extraction, models, evaluation, integration, and the GUI.

```text
COS30082_Facial_Recognition_with_Emotion_and_Liveness
┣ data
┃ ┣ archive (4)                 # Raw video dataset
┃ ┣ classification_data         # Processed Face crops
┃ ┣ lip_sequence_data           # Processed Spatiotemporal numpy arrays
┃ ┗ verification_data           # Pairs used for AUC testing
┣ logs
┃ ┗ history                     # Saved JSON training histories
┣ models
┃ ┣ checkpoints                 # Saved model .h5 weights
┃ ┣ emotion_detection           # Teammate's Emotion models
┃ ┗ glasses_detection           # Teammate's Glasses models
┣ src
┃ ┣ attendance
┃ ┃ ┣ faces_db                  # Automatically enrolled actor faces
┃ ┃ ┗ attendance_log.csv        # Logged ENTER, EXIT, and SPOKEN events
┃ ┣ data
┃ ┃ ┣ build_dataset.py
┃ ┃ ┗ build_lip_sequence_dataset.py
┃ ┣ evaluation
┃ ┃ ┣ evaluate_lip_models.py
┃ ┃ ┣ evaluator.py
┃ ┃ ┗ verification.py
┃ ┣ gui
┃ ┃ ┣ templates
┃ ┃ ┃ ┗ index.html              # HD Glassmorphism UI
┃ ┃ ┗ app.py                    # Flask Web Server
┃ ┣ integration
┃ ┃ ┣ emotion_detector.py       # Teammate's OOP Emotion pipeline
┃ ┃ ┣ hybrid_attendance.py      # Core MJPEG streaming pipeline
┃ ┃ ┗ webcam_demo.py
┃ ┣ models
┃ ┃ ┣ base_model.py
┃ ┃ ┣ fnn.py
┃ ┃ ┣ lip_cnn_gru.py
┃ ┃ ┣ lip_cnn_lstm.py
┃ ┃ ┣ lip_conv3d.py
┃ ┃ ┗ mlp.py
┃ ┗ training
┃ ┃ ┣ base_trainer.py
┃ ┃ ┣ train_fnn.py
┃ ┃ ┣ train_lip_models.py
┃ ┃ ┗ train_mlp.py
┣ main.py                       # Pipeline Manager
┗ requirements.txt
```

## Features

- **Automated Enrolment**: Unrecognized actors are assigned an "Unknown" tag and automatically added to the face database (`faces_db`) if they remain in frame for > 5.0 seconds.
- **Vocal Movement Thresholding**: The Lip-Reading model computes the Mean Absolute Difference across frames to ensure inference only runs when the actor's mouth is actually moving.
- **Glassmorphism UI**: Uses a high-end, responsive dark-mode Web UI with dynamic glowing accents to view the webcam stream.
- **Event Logging**: Every entrance, exit, and spoken phrase is time-stamped and recorded automatically to `attendance_log.csv`.