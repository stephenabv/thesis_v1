# Connect 
import time
import sys
import RPi.GPIO as GPIO
from hx711 import HX711


def measure_weight(EMULATE_HX711=False, ref_unit=1):
    if not EMULATE_HX711:
        hx = HX711(5, 6)
        hx.set_reading_format("MSB", "MSB")
        hx.set_reference_unit(ref_unit)
        hx.reset()
        hx.tare()
    else:
        hx = HX711(None, None, gain=128, bits=24)
        hx.set_reference_unit(ref_unit)
        hx.reset()
        hx.tare()

    while True:
        try:
            weight = hx.get_weight(5)
            hx.power_down()
            hx.power_up()
            time.sleep(0.1)
            return weight

        except (KeyboardInterrupt, SystemExit):
            if not EMULATE_HX711:
                GPIO.cleanup()
            sys.exit()
