import numpy as np
import pyrealsense2 as rs
import cv2
import time
import my_Function as ff
import math
from my_Function import BOXDETECTION
from ColorCalibrate import CALIRBRATE
# import glob
import os
def get_hsv_range(frame, point1, point2):
    # Extract the region of interest (ROI) around the points
    roi = frame[min(point1[1], point2[1]):max(point1[1], point2[1]),
                min(point1[0], point2[0]):max(point1[0], point2[0])]

    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate the min and max HSV values
    min_hue = np.min(hsv_roi[:, :, 0])
    max_hue = np.max(hsv_roi[:, :, 0])
    min_s = np.min(hsv_roi[:, :, 1])
    max_s = np.max(hsv_roi[:, :, 1])
    min_v = np.min(hsv_roi[:, :, 2])
    max_v = np.max(hsv_roi[:, :, 2])
    print(min_s,max_s,min_v,max_v)

    return min_hue, max_hue
points = []
points_r = (0,0)
def on_mouse_click(event, x, y, flags, param):
    global points
    global points_r
    if event == cv2.EVENT_LBUTTONUP:
        points_r = (x, y)
        points.append((x, y))

def adjust_contrast(image, alpha):
    """
    Adjusts the contrast of an image.

    Parameters:
    - image: Input image (numpy array).
    - alpha: Contrast adjustment factor (float).

    Returns:
    - adjusted_image: Image with adjusted contrast.
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted_image

def adjust_exposure(image, gamma):
    """
    Adjusts the exposure of an image.

    Parameters:
    - image: Input image (numpy array).
    - gamma: Exposure adjustment factor (float).

    Returns:
    - adjusted_image: Image with adjusted exposure.
    """
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    adjusted_image = cv2.LUT(image, table)
    return adjusted_image






# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)


# model.eval()  # Set the model to evaluation mode
color_sensor = profile.get_device().query_sensors()[1]
# Disable auto exposure and auto white balance
color_sensor.set_option(rs.option.enable_auto_exposure, True)
color_sensor.set_option(rs.option.enable_auto_white_balance, True)
# color_sensor.set_option(rs.option.enable_auto_color_balance, False)

# color_sensor.set_option(rs.option.exposure,900)  # Adjus  
calibrate = CALIRBRATE()

target_image = cv2.imread('target.jpg')
Position = []
timestamp = 0

BoxDetect = BOXDETECTION()



time.sleep(2)

state = 'Idle'
frame_count = 0
start_time = time.time()
def update_mask(x):
    global min_h
    global max_h
    min_h = cv2.getTrackbarPos('min','HueR')
    max_h = cv2.getTrackbarPos('max','HueR')

cv2.namedWindow("HueR")
min_h = 0
max_h = 255
# i=0

cv2.createTrackbar('min','HueR',min_h,255,update_mask)
cv2.createTrackbar('max','HueR',max_h,255,update_mask)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_550mm_place2.avi', fourcc, 20.0, (1280, 720))
stack_weight=0.1
depth_data, color_data = ff.align(pipeline)
result = np.float32(color_data)

# Create a window and set the mouse callback function
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", on_mouse_click)
save_folder = 'foto_jinpao_precision'

try:

    while True:
        BoxDetect.lower_green =  np.array([min_h, 100, 75])
        BoxDetect.upper_green = np.array([max_h, 255, 255])

        # fps count
        frame_count += 2
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Real sense Align
        depth_data, color_data = ff.align(pipeline)
        

        cv2.imshow('rgb',color_data)
        cd = color_data.copy()

        color_data = calibrate.matchHistogram_bgr(color_data)

        contrast_factor = 1.4  # You can adjust this value accordingly
        image_with_adjusted_contrast = adjust_contrast(color_data, contrast_factor)
        # cv2.imshow('con', image_with_adjusted_contrast)
        # Adjust exposure
        exposure_factor = 2.4  # You can adjust this value accordingly
        image_with_adjusted_exposure = adjust_exposure(image_with_adjusted_contrast, exposure_factor)
        cv2.imshow("ecp", image_with_adjusted_exposure)
        # print("a")
        # frame = calibrate.matchHistogram_tf(frame)
        # print("b")
        # Display the frame
        cv2.circle(color_data, points_r, 1,(0, 255, 0))
        cv2.imshow("Frame", color_data)
        # matched_image = ff.histogram_matching(color_data,target_image)
        # cv2.imshow("Matched Image",matched_image)
        # color_data = ff.auto_exposure_adjustment(color_data)
        # cv2.imshow("Exp",color_data)
        # Convert frame to float32
        # frame_float32 = np.float32(color_data)

        # Update the stack with a weighted average
        # result = cv2.addWeighted(result, 1 - stack_weight, frame_float32, stack_weight, 0)

        # Convert result to uint8 for display
        # color_data = np.uint8(result)

        # Display the result
        # cv2.imshow("Real-Time Stacking", color_data)
        # Create ROI from depth cameara
        roi_mask, contour_area = ff.create_ROI(0.3,0.8,color_data, depth_data)
        # out.write(contour_area)
        cv2.imshow("Roi",contour_area)
        # find box
        Box_Pos, Box_Color = BoxDetect.HSV_filtering(contour_area, depth_data)
        # print(Box_Pos)
        # print(Box_Color)
        BoxDetect.mask_show()
        key = cv2.waitKey(1) & 0xFF
        # press q for exit
        if key == ord('q'):
            break
        
        # press T for capture sensor
        # Check if the 'c' key is pressed
        if key == ord('c'):
            if len(points) >= 2:
                # Get the min and max hue values in the specified area
                min_hue, max_hue = get_hsv_range(color_data, points[0],points[1])
                
                # Print the results
                print(f"Min Hue: {min_hue}, Max Hue: {max_hue}")
                # Clear the points list
                
                points.clear()

        # # init Cap
        if key == ord('t'):
            Color = []
            Position = []
            print("Cap jaa")
            # i +=1 
            # cv2.imwrite(f'captured_frame_550mm_Place_{i}.jpg', contour_area)

            t_cap = 5
            timestamp = time.time() + t_cap
            state = 'Capture'
        elif key == ord('b'):
            if len(points) >= 2:
                # Get the min and max hue values in the specified area
                roi = color_data[min(points[0][1], points[1][1]):max(points[0][1], points[1][1]),
                min(points[0][0], points[1][0]):max(points[0][0], points[1][0])]

                calibrate.calHist_bgr(roi)
                
                cv2.imshow('roi',roi)

                points.clear()
                # Check if the 'l' key is pressed
        if key == ord('y'):
            light = color_sensor.get_option(rs.option.exposure)
            color_sensor.set_option(rs.option.enable_auto_exposure, False)
            color_sensor.set_option(rs.option.enable_auto_white_balance, True)
            color_sensor.set_option(rs.option.exposure,light)  # Adjus

        elif key == ord('s'):
            # Save the frame to a file in the specified folder
            filename = os.path.join(save_folder, f"captured_image_{len(os.listdir(save_folder)) + 1}.png")
            cv2.imwrite(filename, color_data)
            filename = os.path.join(save_folder, f"captured_image_{len(os.listdir(save_folder)) + 1}.png")
            cv2.imwrite(filename, cd)
            print(f"Image saved as {filename}")

        # finish Cap
        if time.time() > timestamp and state == 'Capture':
            Color = [element for sublist in Color for element in sublist]
            # print(Color)
            Position = [element for sublist in Position for element in sublist]
            # print(Position)
            # print(Color)
            position, color = ff.positionFilter(Position,Color)
            # print(position)
            # print(color)

            print(ff.BoxPath([2,1], color))
    
            state = 'Idle'

        elif state == 'Capture':
            Color.append(Box_Color)
            Position.append(Box_Pos)

        # anglemode
        if key == ord('a'):
            state = 'Angle'

        if state == 'Angle':
            z_plot = []
            x_plot = []
        
            # for x in range(1280):
            #     for z in depth_data[:,x]:
            #         z_plot.append(z)
            #         x_plot.append(x)
            
            roi_d_data = np.where(roi_mask, depth_data, 0)
            nzi = np.nonzero(roi_d_data)
            x_coor = nzi[1]
            for x in x_coor:
                d = roi_d_data[nzi[0][x], x]
                z_plot.append(d)
                x_plot.append(x)
            x_plot = np.array(x_plot)
            z_plot = np.array(z_plot)
            nonzero_indices = np.nonzero(z_plot)
            
            # Use the indices to filter x and y
            x_cut = x_plot[nonzero_indices]
            z_cut = z_plot[nonzero_indices]
            # Calculate mean and standard deviation
            mean_value = np.mean(z_cut)
            std_dev = np.std(z_cut)

           
            # Create a DataFrame
            import pandas as pd
            # Create a DataFrame
            # data = pd.DataFrame({'x_plot': x_plot, 'z_plot': z_plot})

            # # Calculate the min and max for each unique value of x_plot
            # min_max_z_plot = data.groupby('x_plot')['z_plot'].agg(['min', 'max']).reset_index()

            # # Merge the original data with the min_max values based on x_plot
            # new_data = pd.merge(data, min_max_z_plot, on='x_plot')

            # # Calculate the mean of min and max values
            # new_data['z_plot'] = (new_data['min'] + new_data['max']) / 2

            # # Drop unnecessary columns
            # new_data = new_data.drop(columns=[ 'min', 'max'])
            # # print(new_data)
            # z_cut  = new_data['z_plot'].values
            # x_cut = new_data['x_plot'].values

    

            import numpy as np
            from sklearn.linear_model import LinearRegression
            import matplotlib.pyplot as plt

            # Perform linear fit using ordinary least squares
            regressor = LinearRegression()
            regressor.fit(x_cut.reshape(-1, 1), z_cut)

            # Generate y values for the fitted line
            y_fit_ols = regressor.predict(x_cut.reshape(-1, 1))

            # # # Plot the original points
            # plt.scatter(z_plot, x_plot, color='red', marker='o', label='Points')
            plt.scatter(z_cut, x_cut, color='blue', marker='o', label='Points')

            # Plot the linear fit using ordinary least squares
            plt.grid('on')
            # plt.axis('equal')
            # plt.plot(y_fit_ols, x_cut)

            # # Display the plot
            # plt.legend()
            # plt.show()

            slope = regressor.coef_[0]

            # Calculate the angle in degrees
            angle_degrees = np.degrees(np.arctan(slope))
            print(angle_degrees)
            state = 'Idle'
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()