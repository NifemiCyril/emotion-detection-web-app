# import cv2
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os

# def train_emotion_model():
#     # Simplified CNN model for emotion detection
#     model = Sequential([
#         Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
#         MaxPooling2D(2,2),
#         Conv2D(64, (3,3), activation='relu'),
#         MaxPooling2D(2,2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(7, activation='softmax')
#     ])

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Placeholder training — for demo, no dataset included
#     print("✅ Model compiled successfully (use your dataset for training).")
#     model.save("emotion_model.h5")
#     print("💾 Model saved as emotion_model.pkl")

# if __name__ == "__main__":
#     train_emotion_model()


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Image Preprocessing and Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# ⚠️ Replace 'dataset/' with your emotion dataset folder path
train_generator = train_datagen.flow_from_directory(
    'train/',
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'test/',
    target_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Step 2: Build a Simple CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

# Step 3: Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train Model
model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Step 5: Save Model in Keras Format
model.save("emotion_model.h5")

print("✅ Model training complete and saved as emotion_model.h5")
