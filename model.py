# This code is creating a convolutional neural network (CNN) model using TensorFlow and Keras to
# classify images of fresh and rotten eggs. It uses an ImageDataGenerator to load and augment the
# training and validation data, and then trains the model using the augmented data. Finally, it saves
# the trained model as a .h5 file.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the training and validation data directories
train_dir = 'data/train'
valid_dir = 'data/valid'

# Set the image size and batch size
img_width, img_height = 224, 224
batch_size = 20

# Create the data generator for training data with augmentation
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create the data generator for validation data
valid_data_gen = ImageDataGenerator(rescale=1./255)

# Load the training and validation data from directories with augmentation
train_data = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['fresh', 'rotten']
)

valid_data = valid_data_gen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['fresh', 'rotten']
)

# Define the CNN model architecture
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',
          input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model with augmented data
model.fit(
    train_data,
    epochs=50,
    validation_data=valid_data
)

# Save the trained model
model.save('model.h5')