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
x3 = 0
y3 = 0
h3 = 0
w3 = 0
x = 0
y = 0
h = 0
w = 0
place = 0
wait = 0
cap = 0
dep = 1
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
scale = ((700*np.sin(np.radians(34.5)))*2.0)/960.0
scale2 = ((485*np.tan(np.radians(34.5)))*2.0)/960.0
scalex = ((600*np.tan(np.radians(34.5))))/((470*np.tan(np.radians(34.5))))
scaley = ((600*np.tan(np.radians(21))))/((470*np.tan(np.radians(21))))
posX = 0
posY = 0
radius = 0
theta = 0
center = (100,100)
mid_pixel = (480,270)
gap = 130
state_Gap1 = 10 #10cm
state_Gap2 = 50 #15cm
state_Gap3 = 100 #20cm
rect_list = []
highlight1 = (255,0,0)
highlight2 = (255,0,0)
highlight3 = (255,0,0)
ellipse_axes_lengths = (int(90/scale),int(30/scale))
ellipse_axes_lengths2 = (int(140/scale),int(90/scale))
ellipse_axes_lengths3 = (int(200/scale),int(150/scale))
start_time = time.time()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
align = rs.align(rs.stream.color)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outWithtrack.avi', fourcc, 30.0, (1280, 720))
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

def align(pipeline):
        # Wait for a new frame
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_data = cv2.resize(np.asanyarray(depth_frame.get_data()),(960,540))
    color_data = cv2.resize(np.asanyarray(color_frame.get_data()),(960,540))
    return depth_data, color_data
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
def get_ellipse_contour(center, axes_lengths, angle):
    # Create a black image
    mask = np.zeros((540, 960), dtype=np.uint8)
    # Draw the ellipse on the mask
    cv2.ellipse(mask, center, axes_lengths, angle, 0, 180, 255, thickness=cv2.FILLED)
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
def is_point_inside_circle(point, circle_center, circle_radius):
    distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)
    return distance <= circle_radius
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
lowpass_filter_x = LowPassFilter(alpha=0.5)
lowpass_filter_y = LowPassFilter(alpha=0.5)
class Detection:
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
    def find_Edge(self):
        color_image = self.frame
        # cv2.imshow("RGB Frame with ROI2", color_image)
        # cv2.imshow("RGB Frame with ROI", color_data)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray_image, (9, 9), 10)
        edges = cv2.Canny(gray_blurred, 150,100)
        contours_edge, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges, contours_edge, -1, (255), thickness=cv2.FILLED)
        return color_image,edges,contours_edge
    def find_flag(self):
        color_image = self.frame
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray_image, (11, 11), 20)
        edges = cv2.Canny(gray_blurred, 50,70)
        return edges,gray_blurred

INIT = 0
WAIT = 1
DETECT = 2
SEND = 3

state = INIT
prev_theta = 0
dt = 1/10
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
    # Access depth data as a numpy array
    depth_data = cv2.resize(np.asanyarray(depth_frame.get_data()),(960,540))
    # Access color data as a numpy array
    color_data = cv2.resize(np.asanyarray(color_frame.get_data()),(960,540))
    depth1 = Detection(depth_data)
    edges1 = Detection(color_data)
    # Define the distance range for your ROI
    # Create a mask for the ROI
    # depth_roi_mask = np.logical_and(depth_data >= min_distance * 1000, depth_data <= max_distance * 1000)
    # # Apply the mask to the depth data
    # depth_roi = np.where(depth_roi_mask, depth_data, 0)
    # # Create a grayscale image from the ROI data
    # depth_roi_image = np.uint8(255-(depth_roi / np.max(depth_roi) * 255))
    # depth_roi_image2 = np.uint8((depth_roi / np.max(depth_roi) * 255))
    # _, binary_image = cv2.threshold(depth_roi_image, 128, 255, cv2.THRESH_BINARY)
    # contours_black, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, binary_image2 = cv2.threshold(depth_roi_image2, 128, 255, cv2.THRESH_BINARY)
    # contours_white, _ = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Display the RGB frame with the ROI
    # color_image = np.asanyarray(color_frame.get_data())
    # gray_image = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
    # gray_blurred = cv2.GaussianBlur(gray_image, (11, 11), 10)
    # edges = cv2.Canny(gray_blurred, 50,100)
    # contours_edge, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(edges, contours_edge, -1, (255), thickness=cv2.FILLED)
    # gray_image2 = cv2.cvtColor(depth_roi, cv2.COLOR_BGR2GRAY)
    # gray_blurred2 = cv2.GaussianBlur(gray_image2, (9, 9), 2)
    # edges2 = cv2.Canny(depth_roi_image, 50,100)
    # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
    #                            param1=50, param2=30, minRadius=1, maxRadius=40)
    # circles2 = cv2.HoughCircles(edges2, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
    #                            param1=50, param2=30, minRadius=200, maxRadius=400)
    # Create a black image for each frame
    # Convert RealSense color frame to OpenCV format
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         for contour in contours_white:
    #             result = cv2.pointPolygonTest(contour,(i[0], i[1]), False)
    #             if result > 0:
    #                 cv2.circle(color_data, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #                 cv2.circle(color_data, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #                 if is_point_inside_circle((i[0], i[1]),(480,0), 125):
    #                     print("1",i[0], i[1])
    #                 elif is_point_inside_circle((i[0], i[1]),(480,0), 190):
    #                     print("2",i[0], i[1])
    #                 elif is_point_inside_circle((i[0], i[1]),(480,0), 250):
    #                     print("3",i[0], i[1])
    # if circles2 is not None:
    #     circles = np.uint16(np.around(circles2))
    #     for i in circles[0, :]:
    #         cv2.circle(depth_roi_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #         cv2.circle(depth_roi_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #         for contour in contours_white:
    #             result = cv2.pointPolygonTest(contour,(i[0], i[1]), False)
    #             if result > 0:
    #                 cv2.circle(depth_roi_image, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #                 cv2.circle(depth_roi_image, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #                 cv2.circle(white_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #                 cv2.circle(white_frame, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #                 if is_point_inside_circle((i[0], i[1]),(480,0), 125):
    #                     print("1",i[0], i[1])
    #                 elif is_point_inside_circle((i[0], i[1]),(480,0), 190):
    #                     print("2",i[0], i[1])
    #                 elif is_point_inside_circle((i[0], i[1]),(480,0), 250):
    #                     print("3",i[0], i[1])
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
    # ellipse_center = (center[0]-int((480-center[0])*slice_multi),center[1]+int((center[1])*slice_multi2))
    ellipse_center = center
            
    # cv2.ellipse(color_data, ellipse_center, (int(90/scale),int(65/scale)), 0, 0, 180, highlight1, 5)
    # cv2.ellipse(color_data, ellipse_center, (int(160/scale),int(125/scale)), 0, 0, 180, highlight2, 5)
    # cv2.ellipse(color_data, ellipse_center, (int(215/scale),int(185/scale)), 0, 0, 180, highlight3, 5)
    cv2.ellipse(color_data, ellipse_center, (int(100*scalex),int(100*scaley)), 0, 0, 180, highlight1, 5)
    cv2.ellipse(color_data, ellipse_center, (int(150*scalex),int(150*scaley)), 0, 0, 180, highlight2, 5)
    cv2.ellipse(color_data, ellipse_center, (int(200*scalex),int(200*scaley)), 0, 0, 180, highlight3, 5)
    cv2.circle(color_data, (center[0],int(ellipse_center[1]+(45*scale))), 2, (0, 255, 0), 2)  # outer circle
    cv2.circle(color_data, (center[0],int(ellipse_center[1]+(135*scale))), 2, (0, 0, 255), 2)
    cv2.circle(color_data, (center[0],int(ellipse_center[1]+(225*scale))), 2, (255, 0, 0), 3)  # center
    contours = get_ellipse_contour(ellipse_center, ellipse_axes_lengths, 0)
    contours2 = get_ellipse_contour(ellipse_center, ellipse_axes_lengths2, 0)
    contours3 = get_ellipse_contour(ellipse_center, ellipse_axes_lengths3, 0)
    # Draw the contours on a white image
    contour_image = np.ones((540, 960, 3), dtype=np.uint8) * 255
    # cv2.drawContours(contour_image, contours3, -1, (255, 0, 0), -1)
    # cv2.drawContours(contour_image, contours2, -1, (0, 255, 0), -1)
    # cv2.drawContours(contour_image, contours, -1, (0, 0, 255), -1)
    # cv2.imshow("sd",contour_image)
    for cnt in depth1.find_Depth()[1]:
        contour_area = cv2.contourArea(cnt)
        if contour_area > 300 and contour_area < 5000:#limit lower BB
            x, y, w, h = cv2.boundingRect(cnt) # พื้นที่ของแท่งวางธงที่สามารถอยู่ได้ x = 000 , y = 000 , w = 000 , h = 000
            cv2.rectangle(color_data, (x, y), (x + w, y + h), (0, 0, 255), 2)
            posX = int(((x+w//2)-480)/scalex)+480
            posY = int(((y+10)-270)/scaley)+270
            cv2.circle(color_data, (posX,posY), 2, (0, 255, 0), 2)
            theta = findTheta(center,posX,posY) 
            # cv2.circle(color_data,(int(x+w/2),int(y+h/2)), 1, (0, 255, 255), 5)
            radius = np.sqrt(((center[0]-posX)**2)+((center[1]-posY)**2))
            # print(center[0])
            if len(contours) != 0:
                distance = cv2.pointPolygonTest(contours[0], ((x+w/2), y), measureDist=True)
                distance2 = cv2.pointPolygonTest(contours2[0], ((x+w/2), y), measureDist=True)
                distance3 = cv2.pointPolygonTest(contours3[0], ((x+w/2), y), measureDist=True)
            # elif len(contours) != 0 and x<480:
            #     distance = cv2.pointPolygonTest(contours[0], (x+w, y), measureDist=True)
            #     distance2 = cv2.pointPolygonTest(contours2[0], (x+w, y), measureDist=True)
            #     distance3 = cv2.pointPolygonTest(contours3[0], (x+w, y), measureDist=True)
            # if y > 5:
            #     if distance >= 0: #(state 1)
            #         start_time = time.time()
            #         # print("Point is inside 1",y)
            #         highlight1 = (0,255,0)
            #         highlight2 = (255,0,0)
            #         highlight3 = (255,0,0)
            #         # print("a")
            #         if wait == 1 and place == 1:
            #             print(place)
            #             wait = 0
            #             cap = 1
            #         else:
            #             wait = 0
            #         place = 1
            #     elif distance < 0 and distance2 >= 0:  #(state 2)
            #         start_time = time.time()
            #         # print("Point is inside 2",y)
            #         highlight1 = (255,0,0)
            #         highlight2 = (0,255,0)
            #         highlight3 = (255,0,0)
            #         # print("b")
            #         if wait == 1 and place == 2:
            #             print(place)
            #             wait = 0
            #             cap = 2
            #         else:
            #             wait = 0
            #         place = 2
            #     elif distance2 < 0 and distance3 >= 0: #(state 3)
            #         start_time = time.time()
            #         # print("Point is inside 3",y) 
            #         highlight1 = (255,0,0)
            #         highlight2 = (255,0,0)
            #         highlight3 = (0,255,0)
            #         # print("c")
            #         if wait == 1 and place == 3:
            #             if dep:
            #                 deploy_time = time.time()
            #                 dep = 0
            #             # print(place)
            #             wait = 0
            #             cap = 3
            #         else:
            #             wait = 0
            #         place = 3
            #     else: #(init state)
            #         # print("Point is outside",y)
            #         highlight1 = (255,0,0)
            #         highlight2 = (255,0,0)
            #         highlight3 = (255,0,0)
            #         # print(start_time,time.time() - start_time)
            #         if time.time() - start_time > 0.9:  
            #             wait = 1
            #     if center[0] in range(x,x+w) :
            #         if cap !=  0:
            #             if cap == 1 and place == 1:
            #                 cv2.imshow("Deploy",color_data)
            #                 cap = 0
            #     if center[0] in range(x,x+w) :
            #         if cap !=  0:
            #             if cap == 2 and place == 2:
            #                 cv2.imshow("Deploy",color_data)
            #                 cap = 0
            #     if center[0] in range(x,x+w) :
            #         if cap !=  0:
            #             if cap == 3 and place == 3:
            #                 cv2.imshow("Deploy",color_data)
            #                 cap = 0
    # state_count = 0
    # r = 0
    # prev_r = 4
    # print(theta,radius)
    # if theta <= -2.4:
    #     if theta > -2.4:
    #         if r == prev_r:
    #             state_count = 1
    #         prev_r = r
    #     if theta > -1 and state_count == 1:
    #         None    
    # print("r ",radius)
    handPosX = (180*1.164)
    handPosY = (270-center[1]) + 40*1.164
    cv2.circle(color_data, (center[0]-int(handPosX),center[1]+int(handPosY)), 2, (0, 0, 0), 2) 
    # print(handPosY)
    des_theta = np.abs(np.arctan2(handPosX,handPosY)) + np.pi/2
    # print(np.rad2deg(des_theta))
    theta = findTheta(center, posX, posY)

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
    # for cnt in edges1.find_Edge()[2]:
    #     contour_area = cv2.contourArea(cnt)
    #     if contour_area < 500:#limit lower BB
    #         x2, y2, w2, h2 = cv2.boundingRect(cnt)
    #         if x2 in range(x,x+w) and y2 in range(y,y+h):
    #             # cv2.rectangle(color_data, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
    #             rect_list.append((x2,y2,w2,h2))
    #             min_x = min(rect[0] for rect in rect_list)
    #             min_y = min(rect[1] for rect in rect_list)
    #             max_x = max(rect[0] + rect[2] for rect in rect_list)
    #             max_y = max(rect[1] + rect[3] for rect in rect_list)
    #             mid_x = int(max_x-((max_x-min_x)/2)) # ตำแหน่งของจุดศูนย์กลางรูในแกน X
    #             mid_y = int(max_y-((max_y-min_y)/2)) # ตำแหน่งของจุดศูนย์กลางรูในแกน Y
    #             mid_x = int(lowpass_filter_x.update(mid_x))
    #             mid_y = int(lowpass_filter_y.update(mid_y))
    #             if len(rect_list) > 2:
    #                 rect_list = rect_list[-2:-1]
    #                 distance = cv2.pointPolygonTest((np.array([center], dtype=np.int32)), (mid_x, mid_y), True)
    #                 cv2.circle(color_data, center, abs(int(distance)), (0, 255, 255), 5)
    #                 if mid_x in range(x,x+w) and mid_y in range(y,y+h) and (mid_y-y)<17:
    #                     cv2.circle(color_data, (mid_x, mid_y), 1, (0, 255, 255), 5)
    #                     print(pixel_convert(mid_pixel,(mid_x,mid_y)),distance)
                        # if mid_x in range(center[0]-gap,center[0]+gap):
                        #     if distance > state_Gap1:
                        #         print("1")
                        #     elif distance <= state_Gap1 and distance > state_Gap2:
                        #         print("2")
                        #     elif distance <= state_Gap2 and distance > state_Gap3:
                        #         print("3")
    # circles = cv2.HoughCircles(
    #         edges1.find_flag()[1],
    #         cv2.HOUGH_GRADIENT,
    #         dp=1,
    #         minDist=100,
    #         param1=7,
    #         param2=25,
    #         minRadius=10,
    #         maxRadius=27
    #     )
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         # Draw circles on the original frame
    #         cv2.circle(color_data, (i[0], i[1]), i[2], (0, 255, 0), 2)  # outer circle
    #         cv2.circle(color_data, (i[0], i[1]), 2, (0, 0, 255), 3)  # center
    #         print(i[0], i[1])
            # Draw circles on the black image
          # center
    # cv2.drawContours(edges,contours_red,-1,(255,0,0),2)
    # cv2.drawContours(color_data,depth1.find_Depth()[2],-1,(255,0,0),2) 
    # if not dep:
    #     if (time.time() - deploy_time) >= 1.35:
    #         cv2.imshow("ddfddd",color_data) 
    #         dep = 1    

    cv2.imshow('gray', edges1.find_Edge()[1])
    # # cv2.imshow('gray_bb', gray_blurred)
    cv2.circle(color_data, (480,270),1, (0, 0, 255), 5)
    cv2.imshow("RGB Frame with ROI", color_data)
    # print(scale)
    cv2.imshow("ROI Frame", depth1.find_Depth()[0])
    # print(100/scale,150/scale,200/scale)
    # Wait for a key press, and exit the loop if 'q' is pressed
    out.write(color_data)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        out.release()
        break

# except KeyboardInterrupt:
#     pass
# finally:
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     fps = frame_count / elapsed_time
#     print(f"Frames Per Second (FPS): {fps}")
#     pipeline.stop()
#     cv2.destroyAllWindows()