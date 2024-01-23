import cv2
import numpy as np

from func.Realsense import RealSense
from func.Display import DISPLAY
from func.ColorCalibrate import CALIRBRATE
import my_Function as ff
import torch
import time

def xyxy2center(coordinates):
    result = []
    for box in coordinates:
        x1, y1, x2, y2, confidence, color = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        result.append([center_x, center_y, color])

    return np.array(result)


def boxbox(pred_list):  
    output_coordinates = xyxy2center(pred_list)
    
    up = output_coordinates[output_coordinates[:, 1].astype(float) < 720 / 2]
    down = output_coordinates[output_coordinates[:, 1].astype(float) > 720 / 2]
    

    # Get the indices that would sort the array based on the first column
    sorted_indices_up = np.argsort(up[:, 0].astype(float))
    sorted_indices_down = np.argsort(down[:, 0].astype(float))

    # Use the indices to sort the array
    sorted_data_up = up[sorted_indices_up]
    sorted_data_down = down[sorted_indices_down]

    return np.array([sorted_data_up[:,-1],sorted_data_down[:,-1]])
class BoxDetect:
    def __init__(self):
        self.box_cam = RealSense(1280,720, "Box")
        self.box_cam.light_level = 100
        self.box_cam.light_add()
        self.model = torch.hub.load('WongKinYiu/yolov7','custom','model/yolov7_tiny.pt')
        self.model.eval()  # Set the model to evaluation mode

        self.calibrate = CALIRBRATE()
        self.display = DISPLAY()

        self.BoxClass = np.array(['red_box', 'red_strip', 'green_box', 'green_strip', 'blue_box', 'blue_strip'])

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

            roi_mask, contour_area = ff.create_ROI(0.5,1.5,image_with_adjusted_exposure, depth_data)
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
            # cv2.waitKey(1)
        # print(Position)
        # print(Color)

        Position, self.Color = ff.positionFilter(Position,Color)
        self.min_path, self.color, min_cost, CostA = ff.BoxPath([2,1], self.Color)
        return self.color, self.min_path

    def findPickShelf(self):
        depth_data, color_data = self.box_cam.get_frame()

        contrast_factor = 2.5  # You can adjust this value accordingly
        image_with_adjusted_contrast = self.calibrate.adjust_contrast(color_data, contrast_factor)
            
        # Adjust exposure
        exposure_factor = 3  # You can adjust this value accordingly
        image_with_adjusted_exposure = self.calibrate.adjust_exposure(image_with_adjusted_contrast, exposure_factor)
        roi_mask, contour_area = ff.create_ROI(0.3,0.8,image_with_adjusted_exposure, depth_data)

        pred = self.model(contour_area)
        pred_list = np.array(pred.xyxy[0][:].tolist()).astype(object)
        # cv2.imshow("Sss",contour_area)

        if(len(pred_list)):
            pred_list[:, -1] = self.BoxClass[pred_list[:, -1].astype(int)]
            # pred_list = pred_list[pred_list[:,4] > 0.7]
            # print(pred_list)

            for pred in pred_list:
                    pred[-1] = str(pred[-1]).split("_")[0]
                    
            # print(pred_list)
            self.display.show_detect(pred_list, contour_area)
            # cv2.imshow("Ss",contour_area)
            # print(boxbox(pred_list))

            array_3x3 = [self.Color[3:6],self.Color[6:9]]
            # print(array_3x3)
            arr = np.array(array_3x3)
            pattern = boxbox(pred_list)
            pat = np.array(pattern)
            # print(pat)
            dir = ['Righttt',"asdsfgdfnghm",'Lefttt']
            if np.size(pat) == 2: 
                for i in range(len(arr.T)):
                    if (np.all(arr.T[i] == pat.T) and i != 1):
                        return (dir[i])

            elif np.size(pat) == 4:
                dir = ['Right','Left']
                for i in range(len(arr.T)-1):
                    # print(arr.T[i])
                    if(np.all(arr.T[i] == pat.T[0]) and np.all(arr.T[i+1] == pat.T[1])):
                        return (dir[i])
            elif np.size(pat) == 6:
                dir = 'center'
                if(np.all(arr == pat)):
                    return (dir)
        # cv2.waitKey(1)



boxDetect = BoxDetect()
print("fin init")
print(boxDetect.findPath())
print("fin findpath")
Go = ""
while 1:
    go = boxDetect.findPickShelf()
    if go != None:
        Go = go
    print(Go)
    if go == "center":
        break




    