import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# File paths for models
cat_detector_path = "cat_vs_not_cat.h5"
cat_identifier_path = "my_cat_vs_other_cats.h5"

# === FUNCTION TO CREATE A MODEL ===
def create_model(output_units=1):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base model layers
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')(x)  # Binary or multi-class classification

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === TRAIN MODEL IF IT DOESN’T EXIST ===
def train_model(model, dataset_path, save_path):
    if os.path.exists(save_path):
        print(f"Model {save_path} already exists. Skipping training.")
        return load_model(save_path)
    
    print(f"Training model: {save_path}")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary', subset='training')
    val_generator = train_datagen.flow_from_directory(
        dataset_path, target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation')

    model.fit(train_generator, validation_data=val_generator, epochs=10)
    model.save(save_path)
    print(f"Model saved as {save_path}")
    return model

# === CHECK AND TRAIN MODELS ===
cat_detector = train_model(create_model(1), "dataset/cat_vs_not_cat/", cat_detector_path)
cat_identifier = train_model(create_model(1), "dataset/my_cat_vs_other_cats/", cat_identifier_path)

# === IMAGE CLASSIFICATION ===
folder_path = r'C:\Users\Micha\Desktop\Snowy-Defence-Force\Snowy Defence System\images'
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    print(f"Processing image: {img_path}")
    
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Step 1: Detect if it's a cat
        is_cat = cat_detector.predict(img_array)[0][0]
        
        if is_cat > 0.5:
            # Step 2: Identify if it's your cat
            is_my_cat = cat_identifier.predict(img_array)[0][0]
            if is_my_cat > 0.5:
                print(f"{img_file}: ✅ This is YOUR CAT! 🐱 (Confidence: {is_my_cat:.2f})")
            else:
                print(f"{img_file}: ❌ This is another cat. (Confidence: {1 - is_my_cat:.2f})")
        else:
            print(f"{img_file}: ❌ Not a cat. (Confidence: {1 - is_cat:.2f})")

        print("-" * 40)

    except Exception as e:
        print(f"Error processing {img_file}: {e}")
