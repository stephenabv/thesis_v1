import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import tkinter as tk
import threading
from hx711 import HX711
import serial
import picamera
from picamera.array import PiRGBArray

# def measure_weight(EMULATE_HX711=False, ref_unit=1):
#     if not EMULATE_HX711:
#         hx = HX711(5, 6)
#         hx.set_reading_format("MSB", "MSB")
#         hx.set_reference_unit(ref_unit)
#         hx.reset()
#         hx.tare()
#     else:
#         hx = HX711(None, None, gain=128, bits=24)
#         hx.set_reference_unit(ref_unit)
#         hx.reset()
#         hx.tare()

#     while True:
#         try:
#             weight = hx.get_weight(5)
#             hx.power_down()
#             hx.power_up()
#             time.sleep(0.1)
#             print("Weight: {}".format(weight))

#         except (KeyboardInterrupt, SystemExit):
#             if not EMULATE_HX711:
#                 GPIO.cleanup()
#             sys.exit()


def measure_weight(EMULATE_HX711=False, ref_unit=1):
    # initialize serial port for Arduino communication
    ser = serial.Serial('/dev/ttyACM0', 9600)

    # rest of the function code...

    while True:
        try:
            weight = hx.get_weight(5)
            hx.power_down()
            hx.power_up()
            time.sleep(0.1)
            print("Weight: {}".format(weight))

            # send message to Arduino if weight is 20g, 40g, or 60g
            if weight == 20:
                ser.write(b"20g\n")
            elif weight == 40:
                ser.write(b"40g\n")
            elif weight == 60:
                ser.write(b"60g\n")

        except (KeyboardInterrupt, SystemExit):
            if not EMULATE_HX711:
                GPIO.cleanup()
            sys.exit()

# The EggDetector class uses a pre-trained model to detect the freshness of eggs in real-time video
# feed and can be started and stopped using threads.


class EggDetector:

    def __init__(self, model_path='model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.camera = picamera.PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 24
        self.camera.rotation = 180
        self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
        self.stop_signal = False
        self.lock = threading.Lock()
        # replace with your Arduino serial port and baud rate
        self.serial_port = serial.Serial('/dev/ttyACM0', 9600)

    def detect_egg_freshness(self):
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            img_width, img_height = 224, 224
            processed_frame = cv2.resize(frame.array, (img_width, img_height))
            processed_frame = np.array(
                processed_frame, dtype='float32') / 255.0
            processed_frame = np.expand_dims(processed_frame, axis=0)
            prediction = self.model.predict(processed_frame)

            if prediction[0] < 0.5:
                cv2.putText(frame, 'Fresh egg', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.send_serial('Fresh')
            else:
                cv2.putText(frame, 'Rotten egg', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Egg freshness detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def send_serial(self, message):
        self.serial_port.write(message.encode())

    def start(self):
        self.stop_signal = False
        if not hasattr(self, 'thread') or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.detect_egg_freshness)
            self.thread.start()

        if not hasattr(self, 'weight_thread') or not self.weight_thread.is_alive():
            self.weight_thread = threading.Thread(target=measure_weight)
            self.weight_thread.start()

    def stop(self):
        with self.lock:
            self.stop_signal = True
        self.thread.join()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.serial_port.close()


# The class creates a GUI application with buttons to start, stop, and quit a camera feed for an egg
# detector.
class App:
    def __init__(self, master):
        self.master = master
        self.master.geometry("480x320")
        self.frame = tk.Frame(self.master)
        self.frame.pack()

        self.start_button = tk.Button(
            self.frame, text="Start", width=10, command=self.start_cam)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(
            self.frame, text="Stop", width=10, command=self.stop_cam)
        self.stop_button.pack(side=tk.LEFT)

        self.quit_button = tk.Button(
            self.frame, text="Quit", width=10, command=self.quit)
        self.quit_button.pack(side=tk.LEFT)

        self.egg_detector = EggDetector()

    def start_cam(self):
        self.egg_detector.start()

    def stop_cam(self):
        self.egg_detector.stop()

    def quit(self):
        self.egg_detector.release()
        self.master.destroy()


root = tk.Tk()
app = App(root)
root.mainloop()
