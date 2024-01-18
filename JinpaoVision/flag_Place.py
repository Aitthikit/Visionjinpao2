import numpy as np
import cv2
import time
from utils.Realsense import RealSense
from utils.KalmanFilter import KF

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.filtered_value = None
    def update(self, new_value):
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
class Getimage:
    def __init__(self, frame):
        self.frame = frame
    def find_Depth(self):
        depth_roi_mask = np.logical_and(self.frame >= min_distance * 1000, depth_data <= max_distance * 1000)
        depth_roi_mask2 = np.logical_and(self.frame >= min_distance2 * 1000, depth_data <= max_distance2 * 1000)
        # Apply the mask to the depth data
        depth_roi = np.where(depth_roi_mask, self.frame, 0)
        depth_roi2 = np.where(depth_roi_mask2, self.frame, 0)
        # Create a grayscale image from the ROI data
        depth_roi_image = np.uint8(255-(depth_roi / np.max(depth_roi) * 255))
        depth_roi_image2 = np.uint8((depth_roi2 / np.max(depth_roi2) * 255))
        _, binary_image = cv2.threshold(depth_roi_image, 128, 255, cv2.THRESH_BINARY)
        # binary_image = create_hatty(binary_image)
        contours_black, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_roi_image, contours_black, -1, (255), thickness=cv2.FILLED)
        _, binary_image2 = cv2.threshold(depth_roi_image2, 230, 255, cv2.THRESH_BINARY)
        binary_image2 = create_hatty(binary_image2)
        contours_white, _ = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("asd",depth_roi_image2)
        return depth_roi_image,contours_black,contours_white
def findTheta(center,posX,posY):
    disX = center[0]-posX 
    disY = center[1]-posY
    theta = np.arctan2(disY,disX)
    return np.mod(theta + np.pi,np.pi)
def findPos(r, theta , center):
    r = r*1.164
    x = r*np.cos(theta) + center[0]
    y = r*np.sin(theta) + center[1]
    return int(x), int(y)
def pixel_convert(mid_pixel,pixel):
    x = pixel[0]-mid_pixel[0]
    y = pixel[1]-mid_pixel[1]
    return x*scale,y*scale
def create_hatty(mask):
    kernel = np.ones((10,10),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.erode(mask,kernel,iterations = 1)
    return mask



frame_count = 0
x_sum = 0
y_sum = 0
slice_multi = 0.2
slice_multi2 = 0.3
min_distance = 0.47  # in meters
max_distance = 0.57  # in meters
min_distance2 = 0.0  # in meters
max_distance2 = 0.67  # in meters
scale = 1.164
scale2 = 1.173
scalex = ((600*np.tan(np.radians(34.5))))/((470*np.tan(np.radians(34.5))))
scaley = ((600*np.tan(np.radians(21))))/((470*np.tan(np.radians(21))))
posX = 0
posY = 0
radius = 0
theta = 0
camera_L = 333
center = (100,100)
mid_pixel = (480,270)

highlight1 = (255,0,0)
highlight2 = (255,0,0)
highlight3 = (255,0,0)

INIT = 0
WAIT = 1
DETECT = 2
SEND = 3
state = INIT
prev_theta = 0
dt = 1/10

start_time = time.time()

real = RealSense(1280,720,"Flag")

depth_data =  cv2.resize(real.get_frame()[0],(960,540))
color_data =  cv2.resize(real.get_frame()[1],(960,540))
lowpass_filter_x = LowPassFilter(alpha=0.5)
lowpass_filter_y = LowPassFilter(alpha=0.5)

