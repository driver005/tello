from djitellopy import Tello
import time

tello = Tello()
tello.connect()
tello.takeoff()
print("takeoff")
tello.send_rc_control(0, 30, 0, 0)
print("flying")
time.sleep(2.5)
tello.send_rc_control(0, -10, 0, 0)
print("landing")
tello.land()
