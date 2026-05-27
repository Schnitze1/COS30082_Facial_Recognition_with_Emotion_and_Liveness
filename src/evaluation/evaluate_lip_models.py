import os
import json
import matplotlib.pyplot as plt

class LipModelEvaluator:
    """
    Evaluates and compares the performance of spatiotemporal lip reading models.
    """
    def __init__(self, history_dir='logs/history', output_path='src/evaluation/lip_models_comparison.png'):
        """
        Initializes the LipModelEvaluator.
        
        :param history_dir: String path to the directory containing history JSON files.
        :param output_path: String path to save the comparison chart.
        """
        self.history_dir = history_dir
        self.output_path = output_path
        self.models = ["LipCNNLSTM", "LipCNNGRU", "LipConv3D"]

    def evaluate_and_plot(self):
        """
        Reads the history files and generates a validation accuracy comparison chart.
        """
        if not os.path.exists(self.history_dir):
            print("No history logs found.")
            return
            
        plt.figure(figsize=(12, 6))
        
        for model_name in self.models:
            hist_path = os.path.join(self.history_dir, f"{model_name}_history.json")
            if os.path.exists(hist_path):
                with open(hist_path, 'r') as f:
                    history = json.load(f)
                    
                val_acc = history.get('val_accuracy', [])
                plt.plot(val_acc, label=f"{model_name} (Max: {max(val_acc):.4f})")
                
        plt.title('Spatiotemporal Models - Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        plt.savefig(self.output_path)
        plt.close()
        print(f"Comparison chart saved to {self.output_path}")

def main():
    """
    Entry point for Lip Model Evaluator.
    """
    evaluator = LipModelEvaluator()
    evaluator.evaluate_and_plot()

if __name__ == '__main__':
    main()
