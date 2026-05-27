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
    'FNN': 'models/face_recognition/FNN.h5',
}
EMOTION_MODELS = {
    'Residual CNN':       'models/emotion_detection/residual/emotion_cnn_residual.keras',
    'Vanilla CNN':        'models/emotion_detection/vanilla/emotion_cnn_vanilla.h5',
    'EfficientNet-B0':    'models/emotion_detection_2/efficientnet/emotion_efficientnet.h5',
    'Hybrid Transformer': 'models/emotion_detection_2/hybrid_transformer/emotion_hybrid_transformer.h5',
}
LIP_MODELS = {
    'CNN-LSTM': 'models/lip_reading/CNN LSTM.h5',
    'CNN-GRU':  'models/lip_reading/CNN GRU.h5',
    'Conv3D':   'models/lip_reading/Conv3d.h5',
}
GLASSES_MODELS = {
    'Residual CNN': 'models/glasses_detection/residual/glasses_detector_residual_cnn.keras',
}
ANTISPOOF_MODELS = {
    'MobileNetV2': 'models/anti-spoofing/antispoofing_model.h5',
}

TRAINING_SCRIPTS = {
    'FNN':                'src/training/train_fnn.py',
    'Residual CNN (Emotion)':  'src/training/emotion_detection/train_emotion_cnn_residual.py',
    'Vanilla CNN':        'src/training/emotion_detection/train_emotion_cnn_vanilla.py',
    'EfficientNet-B0':    'src/training/emotion_detection_2/train_emotion_efficientnet.py',
    'Hybrid Transformer': 'src/training/emotion_detection_2/train_emotion_hybrid_transformer.py',
    'CNN-LSTM':           'src/training/train_lip_models.py',
    'CNN-GRU':            'src/training/train_lip_models.py',
    'Conv3D':             'src/training/train_lip_models.py',
    'Residual CNN (Glasses)':  'src/training/train_glasses_detector.py',
    'MobileNetV2':        'src/training/anti_spoofing/train_antispoofing.py',
}


def create_sidebar_frame(title, layout):
    return sg.Frame(title, layout, font='Helvetica 12 bold', title_color=ACCENT_COLOR,
                    background_color=PAPER_COLOR, pad=(10, 10), border_width=0, expand_x=True)


def _status_text(key):
    return sg.Text("", key=key, font='Helvetica 9', background_color=PAPER_COLOR,
                   text_color="#f44336", pad=(0, 2))


def update_status_labels(window, attendance_system):
    checks = {
        "-ID_STATUS-":       attendance_system.config["identity_model_path"],
        "-GLASSES_STATUS-":  attendance_system.config["glasses_model_path"],
        "-SPOOF_STATUS-":    attendance_system.config["spoof_model_path"],
        "-EMOTION_STATUS-":  attendance_system.config["emotion_model_name"],
        "-LIP_STATUS-":      attendance_system.config["lip_model_path"],
    }
    for key, path in checks.items():
        if os.path.isfile(path):
            window[key].update("Model ready", text_color="#4CAF50")
        else:
            window[key].update("Model not found — train first", text_color="#f44336")


def build_layout():
    face_rec_layout = [
        [sg.Checkbox("Active", default=True, key="-ID_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(IDENTITY_MODELS.keys()), default_value='FNN', key="-ID_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)],
        [_status_text("-ID_STATUS-")],
    ]

    glasses_layout = [
        [sg.Checkbox("Active", default=False, key="-GLASSES_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(GLASSES_MODELS.keys()), default_value='Residual CNN', key="-GLASSES_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)],
        [_status_text("-GLASSES_STATUS-")],
    ]

    spoof_layout = [
        [sg.Checkbox("Active", default=False, key="-SPOOF_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(ANTISPOOF_MODELS.keys()), default_value='MobileNetV2', key="-SPOOF_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)],
        [_status_text("-SPOOF_STATUS-")],
    ]

    emotion_layout = [
        [sg.Checkbox("Active", default=False, key="-EMOTION_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True),
         sg.Checkbox("Temporal Smoothing", default=True, key="-SMOOTHING_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(EMOTION_MODELS.keys()), default_value='Residual CNN', key="-EMOTION_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)],
        [_status_text("-EMOTION_STATUS-")],
    ]

    lip_layout = [
        [sg.Checkbox("Active", default=False, key="-LIP_ACTIVE-", background_color=PAPER_COLOR, text_color=TEXT_COLOR, enable_events=True)],
        [sg.Combo(list(LIP_MODELS.keys()), default_value='CNN-LSTM', key="-LIP_MODEL-", background_color=BG_COLOR, text_color=TEXT_COLOR, size=(30, 1), readonly=True, enable_events=True)],
        [_status_text("-LIP_STATUS-")],
    ]

    training_layout = [
        [sg.Text("Module to Train:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10')],
        [sg.Combo(['Identity', 'Glasses Detection', 'Anti-Spoofing', 'Emotion', 'Lip Reading'],
                  default_value='Emotion', key='-TRAIN_MODULE-', background_color=BG_COLOR,
                  text_color=TEXT_COLOR, size=(28, 1), readonly=True, enable_events=True)],
        [sg.Text("Model to Train:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10')],
        [sg.Combo(['Residual CNN (Emotion)', 'Vanilla CNN'], default_value='Residual CNN (Emotion)',
                  key='-TRAIN_MODEL-', background_color=BG_COLOR, text_color=TEXT_COLOR, size=(28, 1), readonly=True)],
        [sg.Text("Epochs:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10'),
         sg.InputText('120', key='-EPOCHS-', size=(10, 1), background_color=BG_COLOR, text_color=TEXT_COLOR)],
        [sg.Text("Learning Rate:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10'),
         sg.InputText('0.0003', key='-LR-', size=(10, 1), background_color=BG_COLOR, text_color=TEXT_COLOR)],
        [sg.Text("Batch Size:", background_color=PAPER_COLOR, text_color=TEXT_SEC_COLOR, font='Helvetica 10'),
         sg.Combo(['16', '32', '64', '128'], default_value='32', key='-BATCH-', size=(10, 1),
                  background_color=BG_COLOR, text_color=TEXT_COLOR, readonly=True)],
        [sg.Button("Re-Train Selected Model", key="-RETRAIN-", size=(25, 1),
                   button_color=(TEXT_COLOR, "#4CAF50"), pad=(0, 10))]
    ]

    sidebar = [
        [sg.Text("Project Dashboard", font='Helvetica 16 bold', background_color=PAPER_COLOR,
                 text_color=TEXT_COLOR, pad=(10, 20))],
        [create_sidebar_frame("Face Recognition", face_rec_layout)],
        [create_sidebar_frame("Glasses Detection", glasses_layout)],
        [create_sidebar_frame("Anti-Spoofing (Liveness)", spoof_layout)],
        [create_sidebar_frame("Emotion Detection", emotion_layout)],
        [create_sidebar_frame("Lip Reading", lip_layout)],
        [create_sidebar_frame("Hyperparameter Tuning", training_layout)],
        [sg.Push(background_color=PAPER_COLOR)],
        [sg.Button("Exit", size=(10, 1), button_color=(TEXT_COLOR, "#f44336"))]
    ]

    video_panel = [
        [sg.Text("Live Hybrid Feed", font='Helvetica 14 bold', background_color=BG_COLOR, text_color=TEXT_COLOR)],
        [sg.Image(filename='', key='-IMAGE-', background_color="#000000")],
        [sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS-',
                        bar_color=(ACCENT_COLOR, PAPER_COLOR), visible=False)],
        [sg.Text("", key="-TRAIN_STATUS-", font='Helvetica 12', background_color=BG_COLOR,
                 text_color=ACCENT_COLOR, visible=False)]
    ]

    layout = [
        [sg.Column(sidebar, background_color=PAPER_COLOR, expand_y=True, pad=(0, 0),
                   scrollable=True, vertical_scroll_only=True),
         sg.Column(video_panel, background_color=BG_COLOR, expand_x=True, expand_y=True,
                   element_justification='center', pad=(20, 20))]
    ]
    return layout


def run_training_script(script_path, env_vars, window):
    env = os.environ.copy()
    env.update(env_vars)
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
        for line in process.stdout:
            print(line, end="")
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

    window = sg.Window('Hybrid Attendance System', build_layout(), margins=(0, 0),
                       background_color=BG_COLOR, finalize=True, resizable=True)

    update_status_labels(window, attendance_system)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sg.popup_error("Error", "Could not open webcam.")
        return

    training_thread = None

    while True:
        event, values = window.read(timeout=20)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event in ("-ID_ACTIVE-", "-GLASSES_ACTIVE-", "-SPOOF_ACTIVE-",
                     "-EMOTION_ACTIVE-", "-LIP_ACTIVE-", "-SMOOTHING_ACTIVE-"):
            attendance_system.config["identity_active"]  = values["-ID_ACTIVE-"]
            attendance_system.config["glasses_active"]   = values["-GLASSES_ACTIVE-"]
            attendance_system.config["spoofing_active"]  = values["-SPOOF_ACTIVE-"]
            attendance_system.config["emotion_active"]   = values["-EMOTION_ACTIVE-"]
            attendance_system.config["lip_active"]       = values["-LIP_ACTIVE-"]
            attendance_system.config["smoothing_active"] = values["-SMOOTHING_ACTIVE-"]
            if attendance_system.emotion_detector is not None:
                attendance_system.emotion_detector.smoothing_enabled = values["-SMOOTHING_ACTIVE-"]

        if event in ("-ID_MODEL-", "-GLASSES_MODEL-", "-SPOOF_MODEL-", "-EMOTION_MODEL-", "-LIP_MODEL-"):
            attendance_system.config["identity_model_path"] = IDENTITY_MODELS[values["-ID_MODEL-"]]
            attendance_system.config["glasses_model_path"]  = GLASSES_MODELS[values["-GLASSES_MODEL-"]]
            attendance_system.config["spoof_model_path"]    = ANTISPOOF_MODELS[values["-SPOOF_MODEL-"]]
            attendance_system.config["emotion_model_name"]  = EMOTION_MODELS[values["-EMOTION_MODEL-"]]
            attendance_system.config["lip_model_path"]      = LIP_MODELS[values["-LIP_MODEL-"]]
            window['-IMAGE-'].update(data=b'')
            print("Reloading models from dropdown change...")
            attendance_system.reload_models()
            update_status_labels(window, attendance_system)

        if event == '-TRAIN_MODULE-':
            module = values['-TRAIN_MODULE-']
            if module == 'Identity':
                window['-TRAIN_MODEL-'].update(value='FNN', values=list(IDENTITY_MODELS.keys()))
            elif module == 'Glasses Detection':
                window['-TRAIN_MODEL-'].update(value='Residual CNN (Glasses)', values=['Residual CNN (Glasses)'])
            elif module == 'Anti-Spoofing':
                window['-TRAIN_MODEL-'].update(value='MobileNetV2', values=list(ANTISPOOF_MODELS.keys()))
            elif module == 'Emotion':
                emotion_keys = [k for k in EMOTION_MODELS]
                window['-TRAIN_MODEL-'].update(value=emotion_keys[0], values=emotion_keys)
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
                "TRAIN_EPOCHS":     str(values['-EPOCHS-']),
                "TRAIN_LR":         str(values['-LR-']),
                "TRAIN_BATCH_SIZE": str(values['-BATCH-'])
            }
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
            window['-PROGRESS-'].update(current_count=(current / total) * 100)
            window['-TRAIN_STATUS-'].update(f"Training Epoch {current}/{total}")

        if event == '-TRAINING_DONE-':
            code = values[event]
            if code == 0:
                window['-PROGRESS-'].update(current_count=100)
                window['-TRAIN_STATUS-'].update("Training Completed Successfully!", visible=True, text_color=ACCENT_COLOR)
                sg.popup("Success", "Training completed successfully. The new weights have been saved.")
                attendance_system.reload_models()
                update_status_labels(window, attendance_system)
            else:
                window['-TRAIN_STATUS-'].update("Training Failed.", text_color="#f44336", visible=True)
                sg.popup_error("Error", "Training failed or was interrupted. Check terminal for details.")

        try:
            ret, processed_frame = attendance_system.get_processed_frame(cap)
            if ret and processed_frame is not None:
                imgbytes = cv2.imencode('.png', processed_frame)[1].tobytes()
                window['-IMAGE-'].update(data=imgbytes)
        except Exception as e:
            print(f"Frame error: {e}")

    cap.release()
    window.close()


if __name__ == '__main__':
    main()
