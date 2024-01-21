import numpy as np
import cv2
import time
from func.Realsense import RealSense
from func.KalmanFilter import KF
from func.Sumfunc import *

frame_count = 0
min_distance = 0  # in meters
max_distance = 0.30  # in meters
min_distance2 = 0.5  # in meters
max_distance2 = 0.65  # in meters
scale = 1.164
mid_pixel = (480,270)
lowpass_filter_x = LowPassFilter(alpha=0.5)
lowpass_filter_y = LowPassFilter(alpha=0.5)
start_time = time.time()

class flagPick:
    def __init__(self):
        self.real = RealSense(1280,720,"Flag")
    def flag_Pos(self):
        #Open to test FPS
        # start_time = time.time()
        depth_data =  cv2.resize(self.real.get_frame()[0],(960,540))
        color_data =  cv2.resize(self.real.get_frame()[1],(960,540))
        depth1 = Getimage(depth_data)
        for cnt in depth1.find_Depth(min_distance,max_distance,min_distance2,max_distance2)[1]: 
            contour_area = cv2.contourArea(cnt)
            if contour_area > 600 and contour_area < 5000:#limit lower BB
                x3, y3, w3, h3 = cv2.boundingRect(cnt)
                center = int(x3+(w3/2)), int(y3+(h3/2)) #center of place (000,000)
                cv2.rectangle(color_data, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 2)
                cv2.circle(color_data, center, int((w3/4)+(h3/2)), (0, 0, 255), 5)
                cv2.circle(color_data, center, 1, (0, 255, 0), 5)
                target_X,target_Y = pixel_convert(mid_pixel,center,scale)
                # cv2.imshow("RGB Frame with ROI", color_data)
                #Open to test FPS
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(1/elapsed_time)q

                return target_X,target_Y
                # return target_X,target_Y
        # cv2.circle(color_data, (480,270),1, (0, 0, 255), 5)
        # cv2.imshow("RGB Frame with ROI", color_data)
        # # print(scale)
        # cv2.imshow("ROI Frame", depth1.find_Depth(min_distance,max_distance,min_distance2,max_distance2)[0])
        # print(100/scale,150/scale,200/scale)
        # Wait for a key press, and exit the loop if 'q' is pressed
        # out.write(color_data)
    def another_F(self):
        vertical_offset = 10
        frame = cv2.resize(self.real.get_frame()[1],(960,540))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=45,
            maxRadius=55
        )
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=200, maxLineGap=5)
        cv2.imshow("e",edges)        

        # Create a black image for each frame
        black_image = np.zeros_like(frame)

        # If circles are found, draw them on the frame and the black image
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0],i[1])
                # Draw circles on the original frame
                cv2.circle(frame, center, i[2], (0, 255, 0), 2)  # outer circle
                # print(i[0],i[1],i[2])
                cv2.circle(frame, center, 2, (0, 0, 255), 3)  # center

                # Draw circles on the black image
                cv2.circle(black_image, center, i[2], (0, 255, 0), 2)  # outer circle
                cv2.circle(black_image, center, 2, (0, 0, 255), 3)  # center
                target_X,target_Y = pixel_convert(mid_pixel,center,scale)
                return target_X,target_Y

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < vertical_offset:
                    cv2.line(black_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # resized_frame = cv2.resize(frame, (960, 540))
        # resized_black_image = cv2.resize(black_image, (960, 540))
        # cv2.imshow('Webcam Circles Detection', resized_frame)
        # cv2.imshow('Detected Circles and Lines', resized_black_image)
    # mainPl()
    # while(1):  
    #     print(mainCa())
    #     key = cv2.waitKey(1) & 0xFF
FlagDet = flagPick()
# FlagDet.place_blocking()
while(1):
    # pos = FlagDet.another_F()
    pos = FlagDet.flag_Pos()
    print(pos)
    key = cv2.waitKey(1) & 0xFF