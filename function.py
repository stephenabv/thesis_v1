import pigpio
import time


def rotate_servo(servo_pin):
    servo = servo_pin
    pwm = pigpio.pi()

    try:
        pwm.set_mode(servo, pigpio.OUTPUT)
        pwm.set_PWM_frequency(servo, 50)

        print("90 deg")
        pwm.set_servo_pulsewidth(servo, 1500)
        time.sleep(3)

        pwm.set_PWM_dutycycle(servo, 0)
        pwm.set_PWM_frequency(servo, 0)

        time.sleep(1)  # add a delay after turning off the servo

    finally:
        pwm.stop()