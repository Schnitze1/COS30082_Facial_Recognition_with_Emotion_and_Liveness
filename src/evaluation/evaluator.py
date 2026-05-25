import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    """
    Evaluates a trained Face Recognition model on a test dataset.
    Provides methods to generate F1-scores, confusion matrices, and learning curves.
    """
    
    def __init__(self, model_path: str, test_dir: str, history_path: str):
        """
        Initializes the ModelEvaluator.
        
        :param model_path: String path to the trained model .h5 file.
        :param test_dir: String path to the test dataset directory.
        :param history_path: String path to the training history .json file.
        """
        self.model_path = model_path
        self.test_dir = test_dir
        self.history_path = history_path
        self.model = None
        self.test_ds = None
        self.y_true = None
        self.y_pred = None
        
        self._load_resources()

    def _load_resources(self):
        """
        Loads the model and the test dataset.
        """
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}. Please train the model first.")
            raise FileNotFoundError(f"Missing model: {self.model_path}")
            
        print("Loading test dataset (Grayscale)...")
        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_dir,
            labels='inferred',
            label_mode='int',
            color_mode='grayscale',
            image_size=(112, 112),
            batch_size=32,
            shuffle=False
        )
        
        print("Loading model weights...")
        self.model = tf.keras.models.load_model(self.model_path)
        
    def evaluate_metrics(self):
        """
        Generates predictions and prints the classification report.
        """
        print("Generating model predictions...")
        y_pred_probs = self.model.predict(self.test_ds)
        self.y_pred = np.argmax(y_pred_probs, axis=1)
        
        self.y_true = np.concatenate([y for x, y in self.test_ds], axis=0)
        
        print("Classification Report Generated:")
        print(classification_report(self.y_true, self.y_pred))
        
    def plot_confusion_matrix(self, output_path: str = 'src/evaluation/eval_confusion_matrix.png'):
        """
        Plots and saves the confusion matrix heatmap.
        
        :param output_path: String path to save the confusion matrix image.
        """
        if self.y_true is None or self.y_pred is None:
            print("Predictions not found. Run evaluate_metrics() first.")
            return
            
        print(f"Generating Confusion Matrix at {output_path}...")
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True Actor ID')
        plt.xlabel('Predicted Actor ID')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def plot_learning_curves(self, output_path: str = 'src/evaluation/eval_learning_curves.png'):
        """
        Plots and saves the learning curves from the history JSON.
        
        :param output_path: String path to save the learning curves image.
        """
        if not os.path.exists(self.history_path):
            print(f"History file not found at {self.history_path}. Skipping learning curves.")
            return
            
        print(f"Generating Learning Curves at {output_path}...")
        with open(self.history_path, 'r') as f:
            history = json.load(f)
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(history.get('accuracy', []), label='Train Accuracy')
        axes[0].plot(history.get('val_accuracy', []), label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(loc='lower right')
        
        axes[1].plot(history.get('loss', []), label='Train Loss')
        axes[1].plot(history.get('val_loss', []), label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    """
    Entry point for the Model Evaluator.
    """
    parser = argparse.ArgumentParser(description="Evaluate a face recognition model.")
    parser.add_argument('--model', type=str, default='models/checkpoints/fnn_best.h5', help='Path to the trained model .h5 file')
    parser.add_argument('--history', type=str, default='logs/history/fnn_history.json', help='Path to the training history .json file')
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model,
        test_dir='data/classification_data/test_data',
        history_path=args.history
    )
    evaluator.evaluate_metrics()
    
    base_name = os.path.splitext(os.path.basename(args.model))[0]
    evaluator.plot_confusion_matrix(output_path=f'src/evaluation/{base_name}_confusion_matrix.png')
    evaluator.plot_learning_curves(output_path=f'src/evaluation/{base_name}_learning_curves.png')
    print("Evaluation complete. Artifacts saved.")

if __name__ == '__main__':
    main()
