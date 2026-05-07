import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import cv2

def get_tabular_data():
    """Loads and scales breast cancer dataset for ML classification."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    target = data.target
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.2, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # We return the scaled arrays and the feature names for SHAP
    return X_train_scaled, X_test_scaled, y_train, y_test, data.feature_names

def preprocess_image(image, target_size=(224, 224)):
    """Prepares images for the Deep Learning / Transfer Learning model."""
    # Convert PIL image to CV2 format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img