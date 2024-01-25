import cv2
import numpy as np

from func.Realsense import RealSense
from func.Display import DISPLAY
from func.ColorCalibrate import CALIRBRATE
import func.my_Function as ff
import torch
import time

class BoxDetect:
    def __init__(self):
        self.box_cam = RealSense(1280,720, "Box")
        self.box_cam.light_set(219)
        self.model = torch.hub.load('WongKinYiu/yolov7','custom','model/yolov7_tiny.pt')
        self.model.eval()  # Set the model to evaluation mode
        self.y_desire = 300 #mm
        self.x = 999
        self.y = 999
        self.omega = 999
        self.Error = 999
        self.calibrate = CALIRBRATE()
        self.display = DISPLAY()

        self.C = []
        self.P = []

        self.BoxClass = np.array(['red_box', 'red_strip', 'green_box', 'green_strip', 'blue_box', 'blue_strip'])
        self.is_debug  = False
    def findPath_INIT(self,current_time,runtime ,min_distance, max_distance):

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.findpath_time = current_time + runtime

    def findPath(self,current_time):
        if(current_time < self.findpath_time):
            depth_data, color_data = self.box_cam.get_frame()

            contrast_factor = 2.5  # You can adjust this value accordingly
            image_with_adjusted_contrast = self.calibrate.adjust_contrast(color_data, contrast_factor)
            
            # Adjust exposure
            exposure_factor = 3  # You can adjust this value accordingly
            image_with_adjusted_exposure = self.calibrate.adjust_exposure(image_with_adjusted_contrast, exposure_factor)
            # cv2.imshow('con', image_with_adjusted_exposure)

            roi_mask, contour_area = ff.create_ROI(self.min_distance,self.max_distance,image_with_adjusted_exposure, depth_data)
            # cv2.imshow("Roi",contour_area)      

            pred = self.model(contour_area)
            pred_list = np.array(pred.xyxy[0][:].tolist()).astype(object)


            # Create an array of indices for replacement
            if(len(pred_list)):
                pred_list[:, -1] = self.BoxClass[pred_list[:, -1].astype(int)]
                # print(pred_list)
                if(self.is_debug):
                    self.display.show_detect(pred_list, contour_area)
            
                for pred in pred_list:
                    x1, y1, x2, y2, _ , c = pred
                    if str(c).split('_')[1] == "box":
                        w = x2 - x1
                        h = y2 - y1

                        # print(c)
                        pos = ff.find_pos(contour_area,w ,x1+int(w)//2 ,y1+int(h)//2)
                        depth_value = depth_data[int(y1+int(h)//2), int(x1+int(w)//2)]
                        self.P.append([pos[0], pos[1], depth_value])
                        self.C.append(str(c).split('_')[0])
            if(self.is_debug):
                cv2.imshow("Detect", contour_area)
                cv2.waitKey(1)
            return 1
        else:
            # print(self.P,self.C)
            self.Position, self.Color = ff.positionFilter(self.P,self.C)
            self.min_path, self.color, min_cost, CostA = ff.BoxPath([2,1], self.Color)

            # print(self.color, self.min_path)
            self.min_path = np.array([self.min_path[0] - np.array([2,1]),self.min_path[1] - self.min_path[0],self.min_path[2]- self.min_path[1]])
            self.min_path[:, 0] = self.min_path[:, 0] * (-1) * 20
            self.min_path[:, 1] = self.min_path[:, 1]* 25

            return 0

    def findPickShelf(self,min_distance, max_distance):
        weight = [-100,-50,0,50,100]
        depth_data, color_data = self.box_cam.get_frame()

        contrast_factor = 2.5  # You can adjust this value accordingly
        image_with_adjusted_contrast = self.calibrate.adjust_contrast(color_data, contrast_factor)
            
        # Adjust exposure
        exposure_factor = 3  # You can adjust this value accordingly
        image_with_adjusted_exposure = self.calibrate.adjust_exposure(image_with_adjusted_contrast, exposure_factor)
        roi_mask, contour_area = ff.create_ROI(min_distance,max_distance,image_with_adjusted_exposure, depth_data)

        pred = self.model(contour_area)
        pred_list = np.array(pred.xyxy[0][:].tolist()).astype(object)
        

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
            pattern = ff.boxbox(pred_list)
            pat = np.array(pattern)
            # print(pat)
            dir = [weight[0],99,weight[4]]
            error = None
            if np.size(pat) == 2: 
                for i in range(len(arr.T)):
                    if (np.all(arr.T[i] == pat.T) and i != 1):
                        error = dir[i]

            elif np.size(pat) == 4:
                dir = [weight[1],weight[3]]
                for i in range(len(arr.T)-1):
                    # print(arr.T[i])
                    if(np.all(arr.T[i] == pat.T[0]) and np.all(arr.T[i+1] == pat.T[1])):
                        error = dir[i]
            elif np.size(pat) == 6:
                dir = weight[2]
                if(np.all(arr == pat)):
                    error = dir
                
        
            if error != None:
                self.Error = error
        if(self.is_debug):
            cv2.imshow("ss",contour_area)
            cv2.waitKey(1)
        return self.Error

    def finetune(self):
        depth_data, color_data = self.box_cam.get_frame()

        contrast_factor = 2.5  # You can adjust this value accordingly
        image_with_adjusted_contrast = self.calibrate.adjust_contrast(color_data, contrast_factor)
            
        # Adjust exposure
        exposure_factor = 3  # You can adjust this value accordingly
        image_with_adjusted_exposure = self.calibrate.adjust_exposure(image_with_adjusted_contrast, exposure_factor)
        roi_mask, contour_area = ff.create_ROI(0.3,0.8,image_with_adjusted_exposure, depth_data)

        pred = self.model(contour_area)
        pred_list = np.array(pred.xyxy[0][:].tolist()).astype(object)
        # Create an array of indices for replacement
        if(len(pred_list)):
            pred_list[:, -1] = self.BoxClass[pred_list[:, -1].astype(int)]

            for pred in pred_list:
                    pred[-1] = str(pred[-1]).split("_")[0]
                    
            cclist = ff.boxy(pred_list)
            avgX = np.mean(cclist[:,0].astype(np.float32))
            avgY = np.mean(cclist[:,1].astype(np.float32))

            cv2.circle(contour_area, (int(avgX),int(avgY)), 10, (255,255,255), thickness=-1)
            cv2.circle(contour_area, (1280//2,720//2), 2, (0,0,255), thickness=-1)
            if len(cclist) == 6:
                x1 ,y1,_= cclist[3]
                x_cen ,y_cen,_= cclist[4]
                x2 ,y2,_= cclist[5]
                z1 = (depth_data[int(float(y1)),int(float(x1))])
                z2 = (depth_data[int(float(y2)),int(float(x2))])
                dif_x = (1280//2) - float(x_cen)
                
                theta = np.arcsin((int(z2)-int(z1))/400)
                y_measure = (int(z2)+int(z1))/2
                fov = 37.5

                x_measure = np.tan(np.deg2rad(fov))*y_measure
                x_px = x_measure/640
                x_des = x_px*dif_x
                if  np.isnan(theta):
                    theta = self.omega

                self.x, self.y, self.omega = x_des, y_measure-self.y_desire,theta
        return self.x, self.y, self.omega

    def findPlaceShelf(self,min_distance, max_distance):
        weight = [-100,-50,0,50,100]
        depth_data, color_data = self.box_cam.get_frame()

        contrast_factor = 2.5  # You can adjust this value accordingly
        image_with_adjusted_contrast = self.calibrate.adjust_contrast(color_data, contrast_factor)
            
        # Adjust exposure
        exposure_factor = 3  # You can adjust this value accordingly
        image_with_adjusted_exposure = self.calibrate.adjust_exposure(image_with_adjusted_contrast, exposure_factor)
        roi_mask, contour_area = ff.create_ROI(min_distance,max_distance,image_with_adjusted_exposure, depth_data)

        pred = self.model(contour_area)
        pred_list = np.array(pred.xyxy[0][:].tolist()).astype(object)
        
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

            array_3x3 = [['red','green','blue'],['red','green','blue']]
            # print(array_3x3)
            arr = np.array(array_3x3)
            pattern = ff.boxbox(pred_list)
            pat = np.array(pattern)
            # print(pat)
            dir = [weight[0],99,weight[4]]
            error = None
            if np.size(pat) == 2: 
                for i in range(len(arr.T)):
                    if (np.all(arr.T[i] == pat.T) and i != 1):
                        error = dir[i]

            elif np.size(pat) == 4:
                dir = [weight[1],weight[3]]
                for i in range(len(arr.T)-1):
                    # print(arr.T[i])
                    if(np.all(arr.T[i] == pat.T[0]) and np.all(arr.T[i+1] == pat.T[1])):
                        error = dir[i]
            elif np.size(pat) == 6:
                dir = weight[2]
                if(np.all(arr == pat)):
                    error = dir
            if error != None:
                self.Error = error
        if (self.is_debug):  
            cv2.imshow("Sss",contour_area)
            cv2.waitKey(1)
        return self.Error






    