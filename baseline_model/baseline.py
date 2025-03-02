import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
import cv2
from pathlib import Path
import time

class BaselineKNN:
    """
    A baseline K-Nearest Neighbors classifier for mechanical parts identification.
    This model works directly on raw pixel values and serves as a performance benchmark.
    """
    
    def __init__(self, n_neighbors=5, img_size=(64, 64), use_grayscale=True):
        """
        Initialize the baseline KNN model.
        
        Args:
            n_neighbors (int): Number of neighbors to use for classification
            img_size (tuple): Size to resize all images to (width, height)
            use_grayscale (bool): Whether to convert images to grayscale
        """
        self.n_neighbors = n_neighbors
        self.img_size = img_size
        self.use_grayscale = use_grayscale
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.classes = None
        self.training_time = None
        
    def load_and_preprocess_image(self, img_path):
        """
        Load and preprocess a single image.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            np.array: Preprocessed image as a flattened vector
        """
        # Read the image
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        
        # Resize to common dimensions
        img = cv2.resize(img, self.img_size)
        
        # Convert to grayscale if specified
        if self.use_grayscale:
            if len(img.shape) == 3:  # Check if the image has color channels
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Flatten the image to a 1D vector
        img_flattened = img.flatten()
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_flattened / 255.0
        
        return img_normalized
    
    def load_dataset(self, data_dir):
        """
        Load and preprocess all images from the dataset directory.
        
        Args:
            data_dir (str): Path to the dataset directory
            
        Returns:
            tuple: X (features) and y (labels)
        """
        data_path = Path(data_dir)
        X = []
        y = []
        
        # Get all class folders
        class_folders = [f for f in data_path.iterdir() if f.is_dir()]
        self.classes = [folder.name for folder in class_folders]
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        # Process each class folder
        for class_idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            print(f"Processing class: {class_name}")
            
            # Get all image files in the class folder
            img_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
            
            for img_path in img_files:
                try:
                    # Load and preprocess the image
                    img_vector = self.load_and_preprocess_image(img_path)
                    X.append(img_vector)
                    y.append(class_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """
        Train the KNN model on the provided data.
        
        Args:
            X (np.array): Feature vectors (flattened images)
            y (np.array): Class labels
            
        Returns:
            self: The trained model instance
        """
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Training KNN model with k={self.n_neighbors}...")
        start_time = time.time()
        self.model.fit(X_scaled, y)
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.array): Test feature vectors
            y_test (np.array): Test labels
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("Evaluating model on test data...")
        X_test_scaled = self.scaler.transform(X_test)
        
        # Measure prediction time
        start_time = time.time()
        y_pred = self.model.predict(X_test_scaled)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.classes)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Prediction time for {len(X_test)} samples: {prediction_time:.2f} seconds")
        print("\nClassification Report:")
        print(report)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'prediction_time': prediction_time,
            'training_time': self.training_time
        }
    
    def predict(self, img_path):
        """
        Predict the class of a single image.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            tuple: Predicted class index and class name
        """
        # Load and preprocess the image
        img_vector = self.load_and_preprocess_image(img_path)
        img_vector = img_vector.reshape(1, -1)
        
        # Scale the features
        img_scaled = self.scaler.transform(img_vector)
        
        # Make prediction
        class_idx = self.model.predict(img_scaled)[0]
        class_name = self.classes[class_idx]
        
        return class_idx, class_name
    
    def visualize_results(self, X_test, y_test, num_samples=5):
        """
        Visualize some test samples and their predictions.
        
        Args:
            X_test (np.array): Test feature vectors
            y_test (np.array): Test labels
            num_samples (int): Number of samples to visualize
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Select random samples
        indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        plt.figure(figsize=(15, 3 * num_samples))
        for i, idx in enumerate(indices):
            # Reshape the flattened image back to 2D
            if self.use_grayscale:
                img = X_test[idx].reshape(self.img_size)
                cmap = 'gray'
            else:
                img = X_test[idx].reshape((*self.img_size, 3))
                cmap = None
            
            true_label = self.classes[y_test[idx]]
            pred_label = self.classes[y_pred[idx]]
            
            plt.subplot(num_samples, 1, i + 1)
            plt.imshow(img, cmap=cmap)
            plt.title(f"True: {true_label}, Predicted: {pred_label}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('baseline_predictions.png')
        plt.show()


def main():
    """
    Main function to run the baseline model.
    """
    # Set parameters
    data_dir = "./images_folder" 
    img_size = (64, 64)
    use_grayscale = True
    test_size = 0.15
    k_values = [3, 5, 7]  # Try different k values
    
    # Create baseline model
    best_accuracy = 0
    best_k = None
    best_results = None
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    try:
        # Initialize with any k value for loading
        baseline = BaselineKNN(n_neighbors=3, img_size=img_size, use_grayscale=use_grayscale)
        X, y = baseline.load_dataset(data_dir)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"Dataset split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # Try different k values
        for k in k_values:
            print(f"\n--- Testing with k={k} ---")
            baseline = BaselineKNN(n_neighbors=k, img_size=img_size, use_grayscale=use_grayscale)
            baseline.classes = baseline.classes  
            # Train and evaluate
            baseline.train(X_train, y_train)
            results = baseline.evaluate(X_test, y_test)
            
            # Keep track of the best model
            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_k = k
                best_results = results
        
        print(f"\n--- Best model: k={best_k} with accuracy {best_accuracy:.4f} ---")
        
        # Visualize results with the best model
        baseline = BaselineKNN(n_neighbors=best_k, img_size=img_size, use_grayscale=use_grayscale)
        baseline.classes = baseline.classes  
        baseline.train(X_train, y_train)
        baseline.visualize_results(X_test, y_test)
        
        # Save results to a file
        with open('baseline_results.txt', 'w') as f:
            f.write(f"Baseline KNN Model Results\n")
            f.write(f"=======================\n\n")
            f.write(f"Parameters:\n")
            f.write(f"- Image size: {img_size}\n")
            f.write(f"- Grayscale: {use_grayscale}\n")
            f.write(f"- Best k value: {best_k}\n\n")
            f.write(f"Performance:\n")
            f.write(f"- Accuracy: {best_results['accuracy']:.4f}\n")
            f.write(f"- Training time: {best_results['training_time']:.2f} seconds\n")
            f.write(f"- Prediction time: {best_results['prediction_time']:.2f} seconds\n\n")
            f.write(f"Classification Report:\n")
            f.write(f"{best_results['classification_report']}\n")
        
        print("Results saved to baseline_results.txt")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
