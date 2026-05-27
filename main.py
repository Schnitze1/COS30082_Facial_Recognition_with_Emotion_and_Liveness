import os
import sys
import subprocess
import argparse

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
        print("\nCOS30082 Facial Recognition & Lip Reading Pipeline")
        print("1. Build Datasets")
        print("2. Train Identity Models")
        print("3. Train Lip Reading Models")
        print("4. Evaluate Models")
        print("5. Launch Hybrid Attendance Demo")
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
        
        print("\nPhase 4: Evaluation")
        self._run_script("src/evaluation/verification.py")
        self._run_script("src/evaluation/evaluator.py")
        self._run_script("src/evaluation/evaluate_lip_models.py")
        
        print("\nPhase 5: Hybrid Attendance Demo")
        self._run_script("src/integration/hybrid_attendance.py")
        
        print("Pipeline execution complete.")

    def run_interactive(self):
        """
        Starts the interactive CLI menu for manual script execution.
        """
        while True:
            self._print_menu()
            choice = input("\nSelect a phase to execute (0-5): ")
            
            if choice == '1':
                self._run_script("src/data/build_dataset.py")
                self._run_script("src/data/build_lip_sequence_dataset.py")
            elif choice == '2':
                self._run_script("src/training/train_fnn.py")
                self._run_script("src/training/train_mlp.py")
            elif choice == '3':
                self._run_script("src/training/train_lip_models.py")
            elif choice == '4':
                self._run_script("src/evaluation/verification.py")
                self._run_script("src/evaluation/evaluator.py", "--model", "models/checkpoints/fnn_best.h5", "--history", "logs/history/fnn_history.json")
                self._run_script("src/evaluation/evaluator.py", "--model", "models/checkpoints/mlp_best.h5", "--history", "logs/history/mlp_history.json")
                self._run_script("src/evaluation/evaluate_lip_models.py")
            elif choice == '5':
                self._run_script("src/integration/hybrid_attendance.py")
            elif choice == '0':
                print("Exiting pipeline.")
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 5.")

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
