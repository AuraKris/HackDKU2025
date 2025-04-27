import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224

model = load_model("EffNetV2S.h5")

class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']

def preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)  # Load and resize
    # img_array = img_to_array(img)  # Convert to array
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img)  # Normalize (same as training)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def predict_and_visualize(img_path, model, class_names, threshold=0.5):
    # Preprocess
    processed_img = preprocess_image(img_path)
    
    # Predict
    predictions = model.predict(processed_img)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    # Display results
    img = load_img(img_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    
    if confidence > threshold:
        title = f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}"
    else:
        title = "Uncertain - Low Confidence"
    
    plt.title(title, fontsize=14)
    plt.axis('off')
    
    # Print detailed probabilities
    print("\nClass Probabilities:")
    for class_name, prob in zip(class_names, predictions):
        print(f"{class_name}: {prob:.4f}")
    
    plt.show()

# Test with multiple images
test_images = [
    'test_images/cardboard_box.jpg',
    'test_images/plastic_bottle.jpg',
    'test_images/glass_jar.jpg',
    'test_images/paper_trash.jpg',
    'test_images/biological.jpg',
    'test_images/battery.jpg',
    'test_images/clothes.jpg',
    'test_images/shoes.jpg',
    'test_images/metal.jpg',
]

for img_path in test_images:
    predict_and_visualize(img_path, model, class_names)