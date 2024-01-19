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

    # def test_blocking(self):
    #     while(1):
    #         depth_data, color_data =  self.box_cam.get_frame()
    #         cv2.imshow('con', color_data)
    #         key = cv2.waitKey(1) & 0xFF


    # def test_con(self):
    #     depth_data, color_data =  self.box_cam.get_frame()
    #     cv2.imshow('con', color_data)
    #     cv2.waitKey(1)
    #     return "ss"
    def findPath(self):
        timestamp = time.time() + 4
        Color = []
        Position = []

        while(time.time() < timestamp):
            depth_data, color_data = self.box_cam.get_frame()

            contrast_factor = 2.5  # You can adjust this value accordingly
            image_with_adjusted_contrast = self.calibrate.adjust_contrast(color_data, contrast_factor)
            
            # Adjust exposure
            exposure_factor = 3  # You can adjust this value accordingly
            image_with_adjusted_exposure = self.calibrate.adjust_exposure(image_with_adjusted_contrast, exposure_factor)
            # cv2.imshow('con', image_with_adjusted_exposure)

            roi_mask, contour_area = ff.create_ROI(0.3,0.8,image_with_adjusted_exposure, depth_data)
            # cv2.imshow("Roi",contour_area)      

            pred = self.model(contour_area)
            # print(time.time_ns()- start)
            # print(pred.xyxy)
            pred_list = np.array(pred.xyxy[0][:].tolist()).astype(object)


            # Create an array of indices for replacement
            if(len(pred_list)):
                pred_list[:, -1] = self.BoxClass[pred_list[:, -1].astype(int)]
                # print(pred_list)
                self.display.show_detect(pred_list, contour_area)
            
                for pred in pred_list:
                    x1, y1, x2, y2, _ , c = pred
                    if str(c).split('_')[1] == "box":
                        w = x2 - x1
                        h = y2 - y1

                        # print(c)
                        pos = ff.find_pos(contour_area,w ,x1+int(w)//2 ,y1+int(h)//2)
                        # depth_value = depth_data[y1+int(h)//2,x1+int(w)//2]
                        # print([y1+int(h)//2, x1+int(w)//2])
                        depth_value = depth_data[int(y1+int(h)//2), int(x1+int(w)//2)]
                        Position.append([pos[0], pos[1], depth_value])
                        Color.append(str(c).split('_')[0])

            # cv2.imshow("Detect", contour_area)
            cv2.waitKey(1)
        # print(Position)
        # print(Color)

        Position, Color = ff.positionFilter(Position,Color)

        print(ff.BoxPath([2,1], Color))


boxDetect = BoxDetect()
print("fin init")
boxDetect.findPath()





    