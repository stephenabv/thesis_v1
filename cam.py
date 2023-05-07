import numpy as np
import cv2
import tensorflow as tf
import os
import sys

model = tf.keras.models.load_model('model.h5')

# Capture an image from the camera
cap = cv2.VideoCapture(0)

stop_signal_received = False

while not stop_signal_received:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Preprocess the image for input to the model
    img_width, img_height = 224, 224
    processed_frame = cv2.resize(frame, (img_width, img_height))
    processed_frame = np.array(processed_frame, dtype='float32') / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)

    # Predict the freshness of the egg
    prediction = model.predict(processed_frame)

    # Show a preview of the camera feed
    if prediction[0] < 0.5:
        cv2.putText(frame, 'Fresh egg', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Rotten egg', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Egg freshness detection', frame)

    # Check if the 'q' key is pressed in the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if the stop signal is received from the main application
    if os.path.exists('stop_signal.txt'):
        stop_signal_received = True
        os.remove('stop_signal.txt')

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
