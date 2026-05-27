import os
import sys
import subprocess
import threading
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import cv2
import FreeSimpleGUI as sg
from src.integration.hybrid_attendance import HybridAttendanceSystem

BG_COLOR = "#1A1414"
PAPER_COLOR = "#272121"
TEXT_COLOR = "#ffffff"
TEXT_SEC_COLOR = "#b3b3b3"
ACCENT_COLOR = "#90caf9"

sg.LOOK_AND_FEEL_TABLE['MaterialDark'] = {
    'BACKGROUND': BG_COLOR,
    'TEXT': TEXT_COLOR,
    'INPUT': PAPER_COLOR,
    'TEXT_INPUT': TEXT_COLOR,
    'SCROLL': PAPER_COLOR,
    'BUTTON': (TEXT_COLOR, PAPER_COLOR),
    'PROGRESS': (ACCENT_COLOR, PAPER_COLOR),
    'BORDER': 1,
    'SLIDER_DEPTH': 0,
    'PROGRESS_DEPTH': 0
}
sg.theme('MaterialDark')

IDENTITY_MODELS = {
    'MLP': 'models/checkpoints/mlp_best.h5',
    'FNN': 'models/checkpoints/fnn_best.h5'
}
EMOTION_MODELS = {
    'Residual CNN': 'models/checkpoints/emotion_cnn_residual.h5',
    'Vanilla CNN': 'models/checkpoints/emotion_cnn_vanilla.h5'
}
LIP_MODELS = {
    'CNN-LSTM': 'models/checkpoints/LipCNNLSTM_best.h5',
    'CNN-GRU': 'models/checkpoints/LipCNNGRU_best.h5',
    'Conv3D': 'models/checkpoints/LipConv3D_best.h5'
}
SPOOF_MODELS = {
    'Residual CNN (.keras)': 'models/checkpoints/glasses_detector_residual_cnn.keras',
    'Residual CNN (.h5)': 'models/checkpoints/glasses_detector_residual_cnn.h5'
}

TRAINING_SCRIPTS = {
    'MLP': 'src/training/train_mlp.py',
    'FNN': 'src/training/train_fnn.py',
    'Residual CNN': 'src/training/train_emotion_cnn_residual.py',
    'Vanilla CNN': 'src/training/train_emotion_cnn_vanilla.py',
    'CNN-LSTM': 'src/training/train_lip_models.py',
    'CNN-GRU': 'src/training/train_lip_models.py',
    'Conv3D': 'src/training/train_lip_models.py',
    'Residual CNN (.keras)': 'src/training/train_glasses_detector.py',
    'Residual CNN (.h5)': 'src/training/train_glasses_detector.py'
}

def create_sidebar_frame(title, layout):
    return sg.Frame(title, layout, font='Helvetica 12 bold', title_color=ACCENT_COLOR,
                    background_color=PAPER_COLOR, pad=(10, 10), border_width=0, expand_x=True)

def build_layout():
    face_rec_layout = [
        [sg.Checkbox("Active", default=True, key="-ID_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(IDENTITY_MODELS.keys()), default_value='MLP', key="-ID_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)],
    ]
    
    spoof_layout = [
        [sg.Checkbox("Glasses Active", default=False, key="-SPOOF_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(SPOOF_MODELS.keys()), default_value='Residual CNN (.keras)', key="-SPOOF_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)]
    ]
    
    emotion_layout = [
        [sg.Checkbox("Active", default=False, key="-EMOTION_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(EMOTION_MODELS.keys()), default_value='Residual CNN', key="-EMOTION_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)]
    ]
    
    lip_layout = [
        [sg.Checkbox("Active", default=False, key="-LIP_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(LIP_MODELS.keys()), default_value='CNN-LSTM', key="-LIP_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)],
    ]
    
    training_layout = [
        [sg.Text("Module to Train:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10')],
        [sg.Combo(['Identity', 'Anti-Spoofing', 'Emotion', 'Lip Reading'], default_value='Emotion', key='-TRAIN_MODULE-', background_color=BG_COLOR, text_color=TEXT_COLOR, size=(28, 1), readonly=True, enable_events=True)],
        [sg.Text("Model to Train:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10')],
        [sg.Combo(['Residual CNN', 'Vanilla CNN'], default_value='Residual CNN', key='-TRAIN_MODEL-', background_color=BG_COLOR, text_color=TEXT_COLOR, size=(28, 1), readonly=True)],
        [sg.Text("Epochs:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10'), 
         sg.InputText('120', key='-EPOCHS-', size=(10, 1), background_color=BG_COLOR, text_color=TEXT_COLOR)],
        [sg.Text("Learning Rate:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10'),
         sg.InputText('0.0003', key='-LR-', size=(10, 1), background_color=BG_COLOR, text_color=TEXT_COLOR)],
        [sg.Text("Batch Size:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10'),
         sg.Combo(['16', '32', '64', '128'], default_value='32', key='-BATCH-', size=(10, 1), background_color=BG_COLOR, text_color=TEXT_COLOR, readonly=True)],
        [sg.Button("Re-Train Selected Model", key="-RETRAIN-", size=(25, 1), button_color=(TEXT_COLOR, "#4CAF50"), pad=(0, 10))]
    ]

    sidebar = [
        [sg.Text("Project Dashboard", font='Helvetica 16 bold', background_color=PAPER_COLOR, text_color=TEXT_COLOR, pad=(10, 20))],
        [create_sidebar_frame("Face Recognition", face_rec_layout)],
        [create_sidebar_frame("Anti-Spoofing", spoof_layout)],
        [create_sidebar_frame("Emotion Detection", emotion_layout)],
        [create_sidebar_frame("Lip Reading", lip_layout)],
        [create_sidebar_frame("Hyperparameter Tuning", training_layout)],
        [sg.Push(background_color=PAPER_COLOR)],
        [sg.Button("Exit", size=(10, 1), button_color=(TEXT_COLOR, "#f44336"))]
    ]

    video_panel = [
        [sg.Text("Live Hybrid Feed", font='Helvetica 14 bold', background_color=BG_COLOR, text_color=TEXT_COLOR)],
        [sg.Image(filename='', key='-IMAGE-', background_color="#000000")],
        [sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS-', bar_color=(ACCENT_COLOR, PAPER_COLOR), visible=False)],
        [sg.Text("", key="-TRAIN_STATUS-", font='Helvetica 12', background_color=BG_COLOR, text_color=ACCENT_COLOR, visible=False)]
    ]
    
    layout = [
        [sg.Column(sidebar, background_color=PAPER_COLOR, expand_y=True, pad=(0,0), scrollable=True, vertical_scroll_only=True),
         sg.Column(video_panel, background_color=BG_COLOR, expand_x=True, expand_y=True, element_justification='center', pad=(20, 20))]
    ]
    return layout

def run_training_script(script_path, env_vars, window):
    """Runs the training script in a subprocess and parses stdout for progress."""
    env = os.environ.copy()
    env.update(env_vars)
    # Ensure stdout is unbuffered
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1
        )
        
        # Parse output line by line without blocking
        for line in process.stdout:
            print(line, end="") # Still output to terminal
            
            # Simple regex to catch Epoch 1/120 or similar
            match = re.search(r'Epoch (\d+)/(\d+)', line)
            if match:
                current_epoch = int(match.group(1))
                total_epochs = int(match.group(2))
                window.write_event_value('-UPDATE_PROGRESS-', (current_epoch, total_epochs))

        process.wait()
        window.write_event_value('-TRAINING_DONE-', process.returncode)
    except Exception as e:
        print(f"Failed to start training: {e}")
        window.write_event_value('-TRAINING_DONE-', -1)


def main():
    attendance_system = HybridAttendanceSystem()
    
    window = sg.Window('Hybrid Attendance System', build_layout(), margins=(0,0), 
                       background_color=BG_COLOR, finalize=True, resizable=True)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        sg.popup_error("Error", "Could not open webcam.")
        return

    training_thread = None

    while True:
        event, values = window.read(timeout=20)
        
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
            
        if event in ("-ID_ACTIVE-", "-SPOOF_ACTIVE-", "-EMOTION_ACTIVE-", "-LIP_ACTIVE-"):
            attendance_system.config["identity_active"] = values["-ID_ACTIVE-"]
            attendance_system.config["spoofing_active"] = values["-SPOOF_ACTIVE-"]
            attendance_system.config["emotion_active"] = values["-EMOTION_ACTIVE-"]
            attendance_system.config["lip_active"] = values["-LIP_ACTIVE-"]
            
        if event in ("-ID_MODEL-", "-SPOOF_MODEL-", "-EMOTION_MODEL-", "-LIP_MODEL-"):
            attendance_system.config["identity_model_path"] = IDENTITY_MODELS[values["-ID_MODEL-"]]
            attendance_system.config["glasses_model_path"] = SPOOF_MODELS[values["-SPOOF_MODEL-"]]
            attendance_system.config["emotion_model_name"] = EMOTION_MODELS[values["-EMOTION_MODEL-"]]
            attendance_system.config["lip_model_path"] = LIP_MODELS[values["-LIP_MODEL-"]]
            
            window['-IMAGE-'].update(data=b'') 
            print("Reloading models from dropdown change...")
            attendance_system.reload_models()
            
        if event == '-TRAIN_MODULE-':
            module = values['-TRAIN_MODULE-']
            if module == 'Identity':
                window['-TRAIN_MODEL-'].update(value='MLP', values=list(IDENTITY_MODELS.keys()))
            elif module == 'Anti-Spoofing':
                window['-TRAIN_MODEL-'].update(value='Residual CNN (.keras)', values=list(SPOOF_MODELS.keys()))
            elif module == 'Emotion':
                window['-TRAIN_MODEL-'].update(value='Residual CNN', values=list(EMOTION_MODELS.keys()))
            elif module == 'Lip Reading':
                window['-TRAIN_MODEL-'].update(value='CNN-LSTM', values=list(LIP_MODELS.keys()))
            
        if event == "-RETRAIN-":
            if training_thread is not None and training_thread.is_alive():
                sg.popup_error("Error", "A model is already training!")
                continue
                
            model_name = values['-TRAIN_MODEL-']

            script_path = TRAINING_SCRIPTS.get(model_name)
            if not script_path or not os.path.exists(script_path):
                sg.popup_error("Error", f"Training script for {model_name} not found!")
                continue
                
            env_vars = {
                "TRAIN_EPOCHS": str(values['-EPOCHS-']),
                "TRAIN_LR": str(values['-LR-']),
                "TRAIN_BATCH_SIZE": str(values['-BATCH-'])
            }
            
            # Show progress bar
            window['-PROGRESS-'].update(visible=True, current_count=0)
            window['-TRAIN_STATUS-'].update(f"Training {model_name}...", visible=True)
            
            training_thread = threading.Thread(
                target=run_training_script, 
                args=(script_path, env_vars, window), 
                daemon=True
            )
            training_thread.start()
            
        if event == '-UPDATE_PROGRESS-':
            current, total = values[event]
            percentage = (current / total) * 100
            window['-PROGRESS-'].update(current_count=percentage)
            window['-TRAIN_STATUS-'].update(f"Training Epoch {current}/{total}")
            
        if event == '-TRAINING_DONE-':
            code = values[event]
            if code == 0:
                window['-PROGRESS-'].update(current_count=100)
                window['-TRAIN_STATUS-'].update("Training Completed Successfully!", visible=True, text_color=ACCENT_COLOR)
                sg.popup("Success", "Training completed successfully. The new weights have been saved.")
                # Automatically reload models
                attendance_system.reload_models()
            else:
                window['-TRAIN_STATUS-'].update("Training Failed.", text_color="#f44336", visible=True)
                sg.popup_error("Error", "Training failed or was interrupted. Check terminal for details.")

        # Always read frame
        ret, processed_frame = attendance_system.get_processed_frame(cap)
        if ret:
            imgbytes = cv2.imencode('.png', processed_frame)[1].tobytes()
            window['-IMAGE-'].update(data=imgbytes)

    cap.release()
    window.close()

if __name__ == '__main__':
    main()
