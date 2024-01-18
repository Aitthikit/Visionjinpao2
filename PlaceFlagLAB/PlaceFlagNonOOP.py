import numpy as np
import pyrealsense2 as rs
import cv2
import time
from KalmanFilter import *
import Laser

pipeline = rs.pipeline()
config = rs.config()
frame_count = 0
x_sum = 0
y_sum = 0
# x3 = 0
# y3 = 0
# h3 = 0
# w3 = 0
# x = 0
# y = 0
# h = 0
# w = 0
# min_distance = 0.47  # in meters
# max_distance = 0.68  # in meters
# min_distance2 = 0.6  # in meters
# max_distance2 = 0.68  # in meters
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

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
align = rs.align(rs.stream.color)
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

lowpass_filter_x = LowPassFilter(alpha=0.5)
lowpass_filter_y = LowPassFilter(alpha=0.5)
while True:
    frame_count += 1
    timestamp = time.time()
    # Wait for a new frame
    frames = pipeline.wait_for_frames()
    # Align the depth frame with the color frame
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    # Access depth data as a numpy array and resize
    depth_data = cv2.resize(np.asanyarray(depth_frame.get_data()),(960,540))
    # Access color data as a numpy array and resize
    color_data = cv2.resize(np.asanyarray(color_frame.get_data()),(960,540))
    depth1 = Getimage(depth_data)
    
    for cnt in depth1.find_Depth()[2]: 
        contour_area = cv2.contourArea(cnt)
        if contour_area > 1500:#limit lower BB
            x3, y3, w3, h3 = cv2.boundingRect(cnt)
            center = int(x3+(w3/2)), int(h3-(w3/2)) #center of place (000,000)
            cv2.rectangle(color_data, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 2)
            cv2.circle(color_data, center, int((w3/4)+(h3/2)), (0, 0, 255), 5)
            cv2.circle(color_data, center, 1, (0, 255, 0), 5)
            if center[1] not in range(0,30):
                None
                # if center[1] >= 0:
                #     # print("move to left",0-center[1])
                # else:
                #     # print("move to right",0-center[1])
            else:
                break
    for cnt in depth1.find_Depth()[1]:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 300 and contour_area < 5000:#limit lower BB
            x, y, w, h = cv2.boundingRect(cnt) # พื้นที่ของแท่งวางธงที่สามารถอยู่ได้ x = 000 , y = 000 , w = 000 , h = 000
            cv2.rectangle(color_data, (x, y), (x + w, y + h), (0, 0, 255), 2)
            posX = int(((x+w//2)-480)/scalex)+480
            posY = int(((y+10)-270)/scaley)+270
            cv2.circle(color_data, (posX,posY), 2, (0, 255, 0), 2)
            # theta = findTheta(center,posX,posY) 
            # cv2.circle(color_data,(int(x+w/2),int(y+h/2)), 1, (0, 255, 255), 5)
            radius = np.sqrt(((center[0]-posX)**2)+((center[1]-posY)**2))
    # Find Hand of Robot From Center and Scale
    # print(center,radius)
    handPosX = (180*scale)
    handPosY = (270-center[1]) + 40*scale
    #Show hand position
    cv2.circle(color_data, (center[0]-int(handPosX),center[1]+int(handPosY)), 2, (0, 0, 0), 2) 
    # print(handPosY)

    #Find theta from center and hand position
    des_theta = np.abs(np.arctan2(handPosX,handPosY)) + np.pi/2
    # print(np.rad2deg(des_theta))
    theta = findTheta(center, posX, posY)
    # print(theta,des_theta,radius)
    #Start state for kalman estimate and place
    r_offset = 25
    if (radius < 100 + r_offset and radius > 80 - r_offset):
        radius = 100
    elif (radius < 150 + r_offset and radius > 150 - r_offset):
        radius = 150
    elif (radius < 200 + r_offset and radius > 200 - r_offset):
        radius = 200
    else : radius =0
    # print(radius)
    Y = [theta, (theta - prev_theta)/dt]
    if(state == INIT):
        state_count = 0
        prev_radius = 0
        is_repeat = 0
        

        state = WAIT
    elif(state == WAIT ):
        # print(theta)
        if theta <= np.pi/6 and theta > 0:
            #init kalman
            
            # print(Y)
            kf = KF(Y)
            state = DETECT
    elif(state == DETECT):

        kf.update(Y,dt)
        xk, yk = findPos(radius, kf.X[0] , center)
        cv2.circle(color_data, (xk,yk), 2, (0, 0, 255), 2)



        # print(theta)
        if (kf.X[0] >= np.pi/6) and not is_repeat:
            if radius == prev_radius:
                is_repeat = True
            
            prev_radius = radius
            # print(is_repeat)
            state = WAIT
        elif(kf.X[0] >= 2*np.pi/3 and theta <= des_theta and is_repeat):
            count_time = (des_theta - kf.X[0])/0.785
            print(count_time*1000)
            # Laser.send_time(count_time*1500)
            Laser.send_time(int(count_time*1000)-333)
            # print("sss")
            state = SEND
        
    elif(state == SEND):
        # print("place")
        pass
        
    prev_theta = theta
    dt = time.time() - timestamp

    #Visual
    # cv2.imshow('gray', edges1.find_Edge()[1])
    # # cv2.imshow('gray_bb', gray_blurred)
    cv2.circle(color_data, (480,270),1, (0, 0, 255), 5)
    cv2.imshow("RGB Frame with ROI", color_data)
    # print(scale)
    cv2.imshow("ROI Frame", depth1.find_Depth()[0])
    # print(100/scale,150/scale,200/scale)
    # Wait for a key press, and exit the loop if 'q' is pressed
    # out.write(color_data)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # out.release()
        break

