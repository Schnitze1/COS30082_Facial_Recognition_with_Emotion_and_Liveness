import os
import sys
import subprocess
import argparse

EMOTION_MODELS = [
    {
        "name": "Vanilla CNN",
        "train": "src/training/emotion_detection/train_emotion_cnn_vanilla.py",
        "model_keras": "models/emotion_detection/vanilla/emotion_cnn_vanilla.keras",
        "model_h5":    "models/emotion_detection/vanilla/emotion_cnn_vanilla.h5",
    },
    {
        "name": "Residual CNN",
        "train": "src/training/emotion_detection/train_emotion_cnn_residual.py",
        "model_keras": "models/emotion_detection/residual/emotion_cnn_residual.keras",
        "model_h5":    "models/emotion_detection/residual/emotion_cnn_residual.h5",
    },
    {
        "name": "EfficientNet-B0",
        "train": "src/training/emotion_detection_2/train_emotion_efficientnet.py",
        "model_keras": "models/emotion_detection_2/efficientnet/emotion_efficientnet.keras",
        "model_h5":    "models/emotion_detection_2/efficientnet/emotion_efficientnet.h5",
    },
    {
        "name": "Hybrid CNN-Transformer",
        "train": "src/training/emotion_detection_2/train_emotion_hybrid_transformer.py",
        "model_keras": "models/emotion_detection_2/hybrid_transformer/emotion_hybrid_transformer.keras",
        "model_h5":    "models/emotion_detection_2/hybrid_transformer/emotion_hybrid_transformer.h5",
    },
]


class PipelineRunner:
    """
    Manages the execution of the entire COS30082 Facial Recognition pipeline.
    """
    def __init__(self):
        """
        Initializes the PipelineRunner with the virtual environment python executable.
        """
        self.python_exe = sys.executable

    def _clear_screen(self):
        """
        Clears the terminal screen.
        """
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_menu(self):
        """
        Displays the main selection menu.
        """
        print("\nCOS30082 Facial Recognition & Emotion Detection Pipeline")
        print("1. Build Datasets")
        print("2. Train Identity Models")
        print("3. Train Lip Reading Models")
        print("4. Train Emotion Models")
        print("5. Evaluate Models")
        print("6. Launch Hybrid Attendance Demo")
        print("0. Exit")

    def _run_script(self, script_path: str, *args):
        """
        Executes a python script and halts on error.
        
        :param script_path: String path to the script to execute.
        :param args: Variable length argument list to pass to the script.
        """
        cmd = [self.python_exe, script_path] + list(args)
        print(f"\nExecuting: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Script {script_path} failed with exit code {e.returncode}")
            print("Halting the pipeline.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("Execution interrupted by user.")
            sys.exit(1)
        print("Done.\n")

    def _select_emotion_models_for_training(self):
        """
        Prompts the user to choose which emotion models to train.
        Returns a list of training script paths.
        """
        print("\nSelect emotion models to train:")
        for i, spec in enumerate(EMOTION_MODELS, start=1):
            print(f"  {i}. {spec['name']}")
        print(f"  {len(EMOTION_MODELS) + 1}. All models")

        choice = input(f"\nEnter choice (1-{len(EMOTION_MODELS) + 1}): ").strip()

        if choice == str(len(EMOTION_MODELS) + 1):
            return [spec["train"] for spec in EMOTION_MODELS]

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(EMOTION_MODELS):
                return [EMOTION_MODELS[idx]["train"]]
        except ValueError:
            pass

        print("Invalid choice. Training all emotion models.")
        return [spec["train"] for spec in EMOTION_MODELS]

    def run_all(self):
        """
        Executes the entire end-to-end pipeline automatically.
        """
        print("Starting automated pipeline execution.")
        
        print("\nPhase 1: Dataset Generation")
        self._run_script("src/data/build_dataset.py")
        self._run_script("src/data/build_lip_sequence_dataset.py")
        
        print("\nPhase 2: Identity Model Training")
        self._run_script("src/training/train_fnn.py")
        self._run_script("src/training/train_mlp.py")
        
        print("\nPhase 3: Lip Reading Model Training")
        self._run_script("src/training/train_lip_models.py")
        
        print("\nPhase 4: Emotion Model Training")
        self._run_script("src/training/emotion_detection/train_emotion_cnn_vanilla.py")
        self._run_script("src/training/emotion_detection/train_emotion_cnn_residual.py")
        self._run_script("src/training/emotion_detection_2/train_emotion_efficientnet.py")
        self._run_script("src/training/emotion_detection_2/train_emotion_hybrid_transformer.py")

        print("\nPhase 5: Evaluation")
        self._run_script("src/evaluation/verification.py")
        self._run_script("src/evaluation/evaluator.py")
        self._run_script("src/evaluation/evaluate_lip_models.py")
        self._run_script("src/evaluation/evaluate_emotion_models.py")
        self._run_script("src/evaluation/evaluate_emotion_efficientnet.py")
        self._run_script("src/evaluation/evaluate_emotion_hybrid_transformer.py")

        print("\nPhase 6: Hybrid Attendance Demo")
        self._run_script("src/gui/desktop_app.py")
        
        print("Pipeline execution complete.")

    def run_interactive(self):
        """
        Starts the interactive CLI menu for manual script execution.
        """
        while True:
            self._print_menu()
            choice = input("\nSelect a phase to execute (0-6): ")
            
            if choice == '1':
                self._run_script("src/data/build_dataset.py")
                self._run_script("src/data/build_lip_sequence_dataset.py")
            elif choice == '2':
                self._run_script("src/training/train_fnn.py")
                self._run_script("src/training/train_mlp.py")
            elif choice == '3':
                self._run_script("src/training/train_lip_models.py")
            elif choice == '4':
                for script in self._select_emotion_models_for_training():
                    self._run_script(script)
            elif choice == '5':
                self._run_script("src/evaluation/verification.py")
                self._run_script("src/evaluation/evaluator.py", "--model", "models/checkpoints/fnn_best.h5", "--history", "logs/history/fnn_history.json")
                self._run_script("src/evaluation/evaluator.py", "--model", "models/checkpoints/mlp_best.h5", "--history", "logs/history/mlp_history.json")
                self._run_script("src/evaluation/evaluate_lip_models.py")
                self._run_script("src/evaluation/evaluate_emotion_models.py")
                self._run_script("src/evaluation/evaluate_emotion_efficientnet.py")
                self._run_script("src/evaluation/evaluate_emotion_hybrid_transformer.py")
            elif choice == '6':
                self._run_script("src/gui/desktop_app.py")
            elif choice == '0':
                print("Exiting pipeline.")
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 6.")

def main():
    """
    Entry point for the application. Parses arguments to run automatically or interactively.
    """
    parser = argparse.ArgumentParser(description="COS30082 Pipeline Manager")
    parser.add_argument('--all', action='store_true', help='Run the entire pipeline automatically')
    args = parser.parse_args()
    
    runner = PipelineRunner()
    
    if args.all:
        runner.run_all()
    else:
        runner.run_interactive()

if __name__ == '__main__':
    main()
