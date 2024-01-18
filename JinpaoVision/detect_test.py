import cv2
import numpy as np

from func.Realsense import RealSense
from func.Display import DISPLAY
from func.ColorCalibrate import CALIRBRATE
import my_Function as ff
import torch
import time

class BoxDetect:
    def __init__(self):
        self.box_cam = RealSense(1280,720, "Box")
            
        self.model = torch.hub.load('WongKinYiu/yolov7','custom','model/yolov7_tiny.pt')
        self.model.eval()  # Set the model to evaluation mode

        self.calibrate = CALIRBRATE()
        self.display = DISPLAY()

        self.BoxClass = np.array(['red_box', 'red_strip', 'green_box', 'green_strip', 'blue_box', 'blue_strip'])

    def test_blocking(self):
        while(1):
            depth_data, color_data =  self.box_cam.get_frame()
            cv2.imshow('con', color_data)
            key = cv2.waitKey(1) & 0xFF


    # def test_con(self):
    #     depth_data, color_data =  self.box_cam.get_frame()
    #     cv2.imshow('con', color_data)
    #     cv2.waitKey(1)
    #     return "ss"



boxDetect = BoxDetect()

boxDetect.test()

# while(1):
#     x, y, the = (boxDetect.test_con())
#     f




    