import numpy as np
import cv2
import time
from func.Realsense import RealSense
from func.KalmanFilter import KF
from func.Sumfunc import *

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
    def place_blocking(self):
        center = (0,0)
        posX = 0
        posY = 0
        radius = 0
        INIT = 0
        WAIT = 1
        DETECT = 2
        SEND = 3
        state = INIT
        prev_theta = 0
        dt = 1/10
        while True:
            timestamp = time.time()
            depth_data =  cv2.resize(self.real.get_frame()[0],(960,540))
            color_data =  cv2.resize(self.real.get_frame()[1],(960,540))
            depth1 = Getimage(depth_data)
            for cnt in depth1.find_Depth(min_distance,max_distance,min_distance2,max_distance2)[2]: 
                contour_area = cv2.contourArea(cnt)
                if contour_area > 1500:#limit lower BB
                    x3, y3, w3, h3 = cv2.boundingRect(cnt)
                    center = int(x3+(w3/2)), int(h3-(w3/2)) #center of place (000,000)
                    cv2.rectangle(color_data, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 2)
                    cv2.circle(color_data, center, int((w3/4)+(h3/2)), (0, 0, 255), 5)
                    cv2.circle(color_data, center, 1, (0, 255, 0), 5)
                    # target_X,target_Y = pixel_convert(mid_pixel,center,scale)
                    # print(target_X,target_Y)
            for cnt in depth1.find_Depth(min_distance,max_distance,min_distance2,max_distance2)[1]:
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
            # print(radius) #Gripper Value
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
                    print("End")
                    # Laser.send_time(count_time*1500)
                    # Laser.send_time(int(count_time*1000)-camera_L)
                    # print("sss")
                    state = SEND
            elif(state == SEND):
                # print("place")
                pass
                
            prev_theta = theta
            dt = time.time() - timestamp

            #Open to test FPS
            # end_time = time.time()
            # elapsed_time = end_time - timestamp
            # print(1/elapsed_time)
            #Visual
            cv2.circle(color_data, (480,270),1, (0, 0, 255), 5)
            cv2.imshow("RGB Frame with ROI", color_data)
            # print(scale)
            cv2.imshow("ROI Frame", depth1.find_Depth(min_distance,max_distance,min_distance2,max_distance2)[0])
            # print(100/scale,150/scale,200/scale)
            # Wait for a key press, and exit the loop if 'q' is pressed
            # out.write(color_data)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # out.release()
                break


    def mainCa(self):
        #Open to test FPS
        # start_time = time.time()
        depth_data =  cv2.resize(self.real.get_frame()[0],(960,540))
        color_data =  cv2.resize(self.real.get_frame()[1],(960,540))
        depth1 = Getimage(depth_data)
        for cnt in depth1.find_Depth(min_distance,max_distance,min_distance2,max_distance2)[2]: 
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
                target_X,target_Y = pixel_convert(mid_pixel,center,scale)
                
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
    # mainPl()
    # while(1):  
    #     print(mainCa())
    #     key = cv2.waitKey(1) & 0xFF
FlagDet = flagDetect()
# FlagDet.place_blocking()
while(1):
    pos = FlagDet.mainCa()
    print(pos)
    key = cv2.waitKey(1) & 0xFF