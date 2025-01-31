import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load pre-trained MobileNetV2 model (either from .h5 or SavedModel format)
# Ensure you provide the correct path for the model file
model = tf.keras.models.load_model(r'C:\Users\Micha\Desktop\Snowy Defence System\mobilenet_v2_model.h5')  # For .h5 file
# model = tf.keras.models.load_model(r'C:\Users\Micha\Desktop\Snowy Defence System\mobilenet_v2_model')  # For SavedModel

# Path to the folder containing images
folder_path = r'C:\Users\Micha\Desktop\Snowy Defence System\images'  # Path to the folder with images

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Loop through each image file
for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    print(f"Processing image: {img_path}")
    
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)

        # Predict the class of the image
        preds = model.predict(img_array)

        # Decode and print the prediction
        decoded_preds = decode_predictions(preds, top=3)[0]
        print(f"Predictions for {img_file}:")
        for i, (imagenet_id, label, score) in enumerate(decoded_preds):
            print(f"{i + 1}: {label} ({score:.2f})")
        print("-" * 40)
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
