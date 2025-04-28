# src/preprocess.py

import os
import numpy as np
import cv2

def load_data(data_dir, img_size=(128, 128)):
    X = []
    y = []
    
    for label, folder in enumerate(['good', 'bad']):  # good -> 0, bad -> 1
        folder_path = os.path.join(data_dir, folder)
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    X = np.array(X, dtype="float32") / 255.0  # Normalize to 0-1
    y = np.array(y)
    return X, y
