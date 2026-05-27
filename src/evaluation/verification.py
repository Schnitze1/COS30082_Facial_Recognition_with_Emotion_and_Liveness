import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class VerificationEvaluator:
    """
    Evaluates a trained model's ability to verify if two face images belong to the same person.
    Uses Cosine Similarity on extracted 128D embeddings to generate ROC/AUC metrics.
    """
    def __init__(self, model_path: str, pairs_file: str):
        """
        Initializes the VerificationEvaluator.
        
        :param model_path: String path to the trained model .h5 file.
        :param pairs_file: String path to the text file containing image pairs.
        """
        self.model_path = model_path
        self.pairs_file = pairs_file
        self.embedding_model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the trained model and strips the classification head to output 128D embeddings.
        """
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            raise FileNotFoundError()
            
        print("Loading full model...")
        base_model = tf.keras.models.load_model(self.model_path)
        
        target_layer = None
        for layer in reversed(base_model.layers):
            if hasattr(layer, 'output_shape') and layer.output_shape[-1] == 128:
                target_layer = layer
                break
                
        if target_layer is None:
            print("Could not automatically find a 128D layer. Falling back to second-to-last layer.")
            target_layer = base_model.layers[-2]
            
        print(f"Embedding layer selected: {target_layer.name} (Output Shape: {target_layer.output_shape})")
        self.embedding_model = tf.keras.Model(inputs=base_model.input, outputs=target_layer.output)

    def _preprocess_image(self, img_path: str):
        """
        Loads and preprocesses an image for the embedding model.
        
        :param img_path: String path to the image file.
        :return: A preprocessed numpy array.
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.resize(img, (112, 112))
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255.0
        return img

    def _compute_cosine_similarity(self, emb1, emb2):
        """
        Computes the cosine similarity between two vectors.
        
        :param emb1: First embedding vector.
        :param emb2: Second embedding vector.
        :return: Float value of cosine similarity.
        """
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return dot_product / (norm1 * norm2)

    def evaluate(self):
        """
        Runs the verification evaluation across all pairs and generates the ROC curve.
        """
        if not os.path.exists(self.pairs_file):
            print(f"Pairs file not found at {self.pairs_file}")
            raise FileNotFoundError()

        print("Evaluating verification pairs...")
        y_true = []
        y_scores = []
        
        with open(self.pairs_file, 'r') as f:
            content = f.read()
        
        # Handle literal '\n' text strings and normal newlines
        content = content.replace('\\n', '\n').replace('\r', '')
        lines = content.split('\n')
            
        for line in lines:
            line_str = line.strip()
            if ',' in line_str:
                parts = line_str.split(',')
            else:
                parts = line_str.split()
            if len(parts) != 3:
                continue
                
            img1_path, img2_path, label = parts
            label = int(label)
            
            try:
                img1 = self._preprocess_image(img1_path)
                img2 = self._preprocess_image(img2_path)
                
                emb1 = self.embedding_model.predict(img1, verbose=0)[0]
                emb2 = self.embedding_model.predict(img2, verbose=0)[0]
                
                sim = self._compute_cosine_similarity(emb1, emb2)
                
                y_true.append(label)
                y_scores.append(sim)
            except Exception as e:
                print(f"Error processing pair {img1_path}, {img2_path}: {e}")

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        print(f"Verification Evaluation Complete. AUC: {roc_auc:.4f}")
        
        self._plot_roc(fpr, tpr, roc_auc)

    def _plot_roc(self, fpr, tpr, roc_auc):
        """
        Plots and saves the Receiver Operating Characteristic curve.
        
        :param fpr: False positive rate array.
        :param tpr: True positive rate array.
        :param roc_auc: Computed AUC float value.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Face Verification')
        plt.legend(loc="lower right")
        
        base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        output_path = f'src/evaluation/{base_name}_verification_roc.png'
        plt.savefig(output_path)
        plt.close()
        print(f"ROC curve saved to {output_path}")

def main():
    """
    Entry point for the Verification Evaluator.
    """
    parser = argparse.ArgumentParser(description="Evaluate a face verification model using ROC/AUC.")
    parser.add_argument('--model', type=str, default='models/checkpoints/fnn_best.h5', help='Path to the trained model .h5 file')
    parser.add_argument('--pairs', type=str, default='data/classification_data/verification_pairs_val.txt', help='Path to the verification pairs txt file')
    args = parser.parse_args()

    evaluator = VerificationEvaluator(model_path=args.model, pairs_file=args.pairs)
    evaluator.evaluate()

if __name__ == '__main__':
    main()
