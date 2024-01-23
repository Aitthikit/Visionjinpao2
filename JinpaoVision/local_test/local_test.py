import cv2 as cv
import numpy as np

img = cv.imread('local_test/template.jpg')
cap = cv.VideoCapture(0)
while 1:
    _, img = cap.read()
    img = img[300:,:]
    key = cv.waitKey(1) & 0xFF
    cv.imshow('cap', img)
    if key == ord('q'):  # Press 'Esc' to exit
        img2 = img.copy()
        break
cv.destroyAllWindows()





# All the 6 methods for comparison in a list
# methods = [cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED]


method = cv.TM_CCOEFF_NORMED
method = cv.TM_SQDIFF_NORMED
while 1:
    ret, template = cap.read()
    # print(template.shape)
    

    img = img2.copy()
    template = cv.resize(template, (320//6, 240//6))
    h, w, _ = template.shape
    # Apply template Matching


    res = cv.matchTemplate(img, template, method)
    cv.imshow('Matching Result', res)



    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = min_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # Use (0, 255, 0) for green color

    
    cv.imshow('Detected Point', img)  # Convert BGR to RGB for displaying with cv.imshow
    cv.imshow('cc', template)  # Convert BGR to RGB for displaying with cv.imshow

    # cv.waitKey(1)  # Adjust the waitKey value if necessary for proper display speed

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'Esc' to exit
        break


# Release video capture object and close windows
cap.release()
cv.destroyAllWindows()
