import numpy as np
import cv2
import time
from func.Realsense import RealSense
from func.KalmanFilter import KF
from func.Sumfunc import *

frame_count = 0
x_sum = 0
y_sum = 0
scale_Place = 1.164
scale_Pick = 1.164
scalex = ((600*np.tan(np.radians(34.5))))/((470*np.tan(np.radians(34.5))))
scaley = ((600*np.tan(np.radians(21))))/((470*np.tan(np.radians(21))))
theta = 0
camera_L = 333
center = (100,100)
mid_pixel = (480,270)

highlight1 = (255,0,0)
highlight2 = (255,0,0)
highlight3 = (255,0,0)


lowpass_filter_x = LowPassFilter(alpha=0.5)
lowpass_filter_y = LowPassFilter(alpha=0.5)
start_time = time.time()

class flagDetect:
    def __init__(self):
        self.real = RealSense(1280,720,"Flag")
        self.center = (0,0)
        self.posX = 0
        self.posY = 0
        self.radius = 0
        self.INIT = 0
        self.WAIT = 1
        self.DETECT = 2
        self.SEND = 3
        self.state = self.INIT
        self.prev_theta = 0
        self.dt = 1/10
        self.kf = None
        self.is_repeat = 0
        self.prev_radius = 0
    def flag_Pos(self):
        #Open to test FPS
        # start_time = time.time()
        depth_data =  cv2.resize(self.real.get_frame()[0],(960,540))
        color_data =  cv2.resize(self.real.get_frame()[1],(960,540))
        depth1 = Getimage(min_distance = 0,
                          max_distance = 0.30,
                          min_distance2 = 0.5,
                          max_distance2 = 0.65)
        for cnt in depth1.find_Depth(depth_data)[1]: 
            contour_area = cv2.contourArea(cnt)
            if contour_area > 600 and contour_area < 5000:#limit lower BB
                x3, y3, w3, h3 = cv2.boundingRect(cnt)
                center = int(x3+(w3/2)), int(y3+(h3/2)) #center of place (000,000)
                cv2.rectangle(color_data, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 2)
                cv2.circle(color_data, center, int((w3/4)+(h3/2)), (0, 0, 255), 5)
                cv2.circle(color_data, center, 1, (0, 255, 0), 5)
                target_X,target_Y = pixel_convert(mid_pixel,center,scale_Pick)
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
                target_X,target_Y = pixel_convert(mid_pixel,center,scale_Pick)
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
    def place_blocking(self,depth1):
        # while True:
        timestamp = time.time()
        # if aa == 1:
        #     time.sleep(count_time)
        depth_data =  cv2.resize(self.real.get_frame()[0],(960,540))
        color_data =  cv2.resize(self.real.get_frame()[1],(960,540))
        # if aa == 1:
        #     cv2.imshow("fireeeee", color_data)
        #     aa = 0
        for cnt in depth1.find_Depth(depth_data)[2]: 
            contour_area = cv2.contourArea(cnt)
            if contour_area > 1500:#limit lower BB
                x3, y3, w3, h3 = cv2.boundingRect(cnt)
                self.center = int(x3+(w3/2)), int(h3-(w3/2)) #center of place (000,000)
                cv2.rectangle(color_data, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 2)
                cv2.circle(color_data, self.center, int((w3/4)+(h3/2)), (0, 0, 255), 5)
                cv2.circle(color_data, self.center, 1, (0, 255, 0), 5)
                # target_X,target_Y = pixel_convert(mid_pixel,center,scale)
                # print(target_X,target_Y)
        for cnt in depth1.find_Depth(depth_data)[1]:
            contour_area = cv2.contourArea(cnt)
            if contour_area > 300 and contour_area < 5000:#limit lower BB
                x, y, w, h = cv2.boundingRect(cnt) # พื้นที่ของแท่งวางธงที่สามารถอยู่ได้ x = 000 , y = 000 , w = 000 , h = 000
                cv2.rectangle(color_data, (x, y), (x + w, y + h), (0, 0, 255), 2)
                self.posX = int(((x+w//2)-480)/scalex)+480
                self.posY = int(((y+10)-270)/scaley)+270
                cv2.circle(color_data, (self.posX,self.posY), 2, (0, 255, 0), 2)
                # theta = findTheta(center,posX,posY) 
                # cv2.circle(color_data,(int(x+w/2),int(y+h/2)), 1, (0, 255, 255), 5)
                self.radius = np.sqrt(((self.center[0]-self.posX)**2)+((self.center[1]-self.posY)**2))
        # handPosX = (180*scale)
        # handPosY = (270-center[1]) + 40*scale
        handPosX = 0*scale_Place
        handPosY = (270-self.center[1])
        #Show hand position
        cv2.circle(color_data,(305,200), 4, (0, 255, 0), 2)
        cv2.line(color_data,(305,0),(305,540),(255,0,0),2)
        cv2.circle(color_data, (self.center[0]-int(handPosX),self.center[1]+int(handPosY)), 2, (0, 0, 0), 2) 
        # print(handPosY)

        #Find theta from center and hand position
        des_theta = np.abs(np.arctan2(handPosX,handPosY)) + np.pi/2
        # print(np.rad2deg(des_theta))
        theta = findTheta(self.center, self.posX, self.posY)
        # print(theta,des_theta,radius)
        #Start state for kalman estimate and place
        r_offset = 25
        if (self.radius < 100 + r_offset and self.radius > 80 - r_offset):
            self.radius = 100
        elif (self.radius < 150 + r_offset and self.radius > 150 - r_offset):
            self.radius = 150
        elif (self.radius < 200 + r_offset and self.radius > 200 - r_offset):
            self.radius = 200
        else : self.radius =0
        # print(radius) #Gripper Value
        Y = [theta, (theta - self.prev_theta)/self.dt]
        if(self.state == self.INIT):
            state_count = 0
            self.prev_radius = 0
            self.is_repeat = 0
            

            self.state = self.WAIT
        elif(self.state == self.WAIT ):
            # print(theta)
            if theta <= np.pi/6 and theta > 0:
                #init kalman
                
                # print(Y)
                self.kf = KF(Y)
                self.state = self.DETECT
        elif(self.state == self.DETECT):

            self.kf.update(Y,self.dt)
            xk, yk = findPos(self.radius, self.kf.X[0] , self.center)
            cv2.circle(color_data, (xk,yk), 2, (0, 0, 255), 2)



            # print(theta)
            if (self.kf.X[0] >= np.pi/6) and not self.is_repeat:
                if self.radius == self.prev_radius:
                    self.is_repeat = True
                
                self.prev_radius = self.radius
                # print(is_repeat)
                self.state = self.WAIT
            elif(self.kf.X[0] >= np.pi/4 and theta <= des_theta and self.is_repeat):
                count_time = (des_theta - self.kf.X[0])/0.785
                print((count_time*1000)+camera_L)
                print("End")
                return (count_time*1000)+(camera_L/1000) # Wait Time Value
                aa = 1
                # Laser.send_time(count_time*1500)
                # Laser.send_time(int(count_time*1000)-camera_L)
                # print("sss")
                self.state = self.SEND
        elif(self.state == self.SEND):
            # print("place")
            pass
            
        self.prev_theta = theta
        self.dt = time.time() - timestamp

        #Open to test FPS
        # end_time = time.time()
        # elapsed_time = end_time - timestamp
        # print(1/elapsed_time)
        #Visual
        cv2.circle(color_data, (480,270),1, (0, 0, 255), 5)
        cv2.imshow("RGB Frame with ROI", color_data)
        # print(scale)
        cv2.imshow("ROI Frame", depth1.find_Depth(depth_data)[0])
        # print(100/scale,150/scale,200/scale)
        # Wait for a key press, and exit the loop if 'q' is pressed
        # out.write(color_data)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     # out.release()
        #     break
    def calibration_(self):
        #Open to test FPS
        # start_time = time.time()
        depth_data =  cv2.resize(self.real.get_frame()[0],(960,540))
        color_data =  cv2.resize(self.real.get_frame()[1],(960,540))
        depth1 = Getimage(min_distance = 0.47,
                          max_distance = 0.57,
                          min_distance2 = 0.0,
                          max_distance2 = 0.67)
        for cnt in depth1.find_Depth(depth_data)[2]: 
            contour_area = cv2.contourArea(cnt)
            if contour_area > 1500:#limit lower BB
                x3, y3, w3, h3 = cv2.boundingRect(cnt)
                center = int(x3+(w3/2)), int(h3-(w3/2)) #center of place (000,000)
                cv2.rectangle(color_data, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 2)
                cv2.circle(color_data, center, int((w3/4)+(h3/2)), (0, 0, 255), 5)
                cv2.circle(color_data, center, 1, (0, 255, 0), 5)
                # if center[1] not in range(0,30):
                #     None
                #     if center[1] >= 0:
                #         print("move to left",0-center[1])
                #     else:
                #         print("move to right",0-center[1])
                # else:
                #     break
                target_X,target_Y = pixel_convert(mid_pixel,center,scale_Place)
                
                #Open to test FPS
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(1/elapsed_time)

                return target_X,target_Y
                # return target_X,target_Y
        # cv2.circle(color_data, (480,270),1, (0, 0, 255), 5)
        # cv2.imshow("RGB Frame with ROI", color_data)
        # # print(scale)
        # cv2.imshow("ROI Frame", depth1.find_Depth(min_distance,max_distance,min_distance2,max_distance2)[0])
        # print(100/scale,150/scale,200/scale)
        # Wait for a key press, and exit the loop if 'q' is pressed
        # out.write(color_data)
FlagDet = flagDetect()
depth1 = Getimage(min_distance = 0.47,
                    max_distance = 0.57,
                    min_distance2 = 0.0,
                    max_distance2 = 0.67)
while(1):
    pos = FlagDet.flag_Pos()
    print(pos)
    key = cv2.waitKey(1) & 0xF
while(1):
    hold = FlagDet.place_blocking(depth1)
    print(hold)
    if hold != None:
        break
    key = cv2.waitKey(1) & 0xF

# while(1):
#     pos = FlagDet.another_F()
#     print(pos)
#     key = cv2.waitKey(1) & 0xFF