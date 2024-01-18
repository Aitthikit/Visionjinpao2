import cv2
import numpy as np

from func.Realsense import RealSense
from func.Display import DISPLAY
from func.ColorCalibrate import CALIRBRATE
import my_Function as ff
import torch
import time

points = []
points_r = (0,0)

def on_mouse_click(event, x, y, flags, param):
    global points
    global points_r

    if event == cv2.EVENT_LBUTTONUP:
        points_r = (x, y)
        points.append((x, y))

def main():
    #camera config
    rs = RealSense(1280,720, "Box")
    
    model = torch.hub.load('WongKinYiu/yolov7','custom','model/yolov7_tiny.pt')
    model.eval()  # Set the model to evaluation mode
   
    calibrate = CALIRBRATE()
    display = DISPLAY()

    # Create a window and set the mouse callback function
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", on_mouse_click)

    while True:
        # Calculate and display frame rate
        start_time = time.time()
        BoxClass = np.array(['red_box', 'red_strip', 'green_box', 'green_strip', 'blue_box', 'blue_strip'])
        
        # Read a frame from the camera
        depth_data, color_data = rs.get_frame()

        frame = calibrate.matchHistogram_bgr(color_data)

        contrast_factor = 2.5  # You can adjust this value accordingly
        image_with_adjusted_contrast = calibrate.adjust_contrast(frame, contrast_factor)
        
        # Adjust exposure
        exposure_factor = 3  # You can adjust this value accordingly
        image_with_adjusted_exposure = calibrate.adjust_exposure(image_with_adjusted_contrast, exposure_factor)
        cv2.imshow('con', image_with_adjusted_exposure)


        roi_mask, contour_area = ff.create_ROI(0.3,0.8,image_with_adjusted_exposure, depth_data)
        cv2.imshow("Roi",contour_area)        
        # Display the frame
        cv2.circle(frame, points_r, 1,(0, 255, 0))
        cv2.imshow("Frame", frame)
        # start = time.time_ns()
        pred = model(contour_area)
        # print(time.time_ns()- start)
        # print(pred.xyxy)
        pred_list = np.array(pred.xyxy[0][:].tolist()).astype(object)


        # Create an array of indices for replacement
        if(len(pred_list)):
            pred_list[:, -1] = BoxClass[pred_list[:, -1].astype(int)]
            # print(pred_list)
            display.show_detect(pred_list, contour_area)
        

        end_time = time.time()
        frame_rate = 1 / (end_time - start_time)
        
        # # Overlay frame rate on the video frame
        cv2.putText(contour_area, f"FPS: {frame_rate:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("Detect", contour_area)
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # Check if the 'l' key is pressed
        if key == ord('y'):
            rs.light_add()
        if key == ord('u'):
            rs.light_sub()
            # pass
        if key == ord('l'):
            if len(points) >= 2:
                # Get the min and max hue values in the specified area
                roi = frame[min(points[0][1], points[1][1]):max(points[0][1], points[1][1]),
                min(points[0][0], points[1][0]):max(points[0][0], points[1][0])]

                calibrate.calHist_tf(roi)
                cv2.imshow('roi',roi)

                points.clear()
        elif key == ord('b'):
            if len(points) >= 2:
                # Get the min and max hue values in the specified area
                roi = frame[min(points[0][1], points[1][1]):max(points[0][1], points[1][1]),
                min(points[0][0], points[1][0]):max(points[0][0], points[1][0])]

                calibrate.calHist_bgr(roi)
                
                cv2.imshow('roi',roi)

                points.clear()
         
        # Check if the 'q' key is pressed
        elif key == ord('q'):
            break

    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
