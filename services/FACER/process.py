import traceback
import time
import imagezmq
import cv2
import os
import facer


try:

    while True:

        dt = time.perf_counter()

        information, frame = image_reciever.recv_image()

        print("recieved_fps=", 1 / (time.perf_counter() - dt))




        cv2.imshow("a", frame)
        cv2.waitKey(1)

except:
    traceback.print_exc()
finally:
    image_reciever.close()

