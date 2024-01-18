import cv2
import numpy as np

# def calHist_yuv(img):
#     # Compute histogram
#     img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     hist_y = cv2.calcHist([img_yuv], [0], None, [256], [0, 256])
#     hist_u = cv2.calcHist([img_yuv], [1], None, [256], [0, 256])
#     hist_v = cv2.calcHist([img_yuv], [2], None, [256], [0, 256])
#     return hist_y, hist_u, hist_v

# def matchHistogram_yuv(hist_y, hist_u, hist_v, img):
#     img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#     cdf_y = hist_y.cumsum()
#     cdf_u = hist_u.cumsum()
#     cdf_v = hist_v.cumsum()

#     # Normalize CDF to the range [0, 255]
#     cdf_normalized_y = (cdf_y * 255) / cdf_y[-1]
#     cdf_normalized_u = (cdf_u * 255) / cdf_u[-1]
#     cdf_normalized_v = (cdf_v * 255) / cdf_v[-1]

#     # Map the intensity values using the CDF
#     equalized_channel_y = np.interp(img_yuv[:,:,0], range(256), cdf_normalized_y)
#     equalized_channel_u = np.interp(img_yuv[:,:,1], range(256), cdf_normalized_u)
#     equalized_channel_v = np.interp(img_yuv[:,:,2], range(256), cdf_normalized_v)

#     # Combine equalized channels into a 3D array
#     equalized_img_yuv = np.stack([equalized_channel_y, equalized_channel_u, equalized_channel_v], axis=-1)

#     # Convert back to BGR after converting to uint8
#     equalized_img_bgr = cv2.cvtColor(equalized_img_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)

#     return equalized_img_bgr




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

   
def calHist_y(img):
    # Compute histogram
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    hist_y, _ = np.histogram(img_yuv[:,:,0].flatten(), 256, [0, 256])
    return hist_y
def matchHistogram_y(hist_y, img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    cdf_y = hist_y.cumsum()

    # Normalize CDF to the range [0, 255]
    cdf_normalized_y = (cdf_y * 255) / cdf_y.max()


    # Map the intensity values using the CDF
    equalized_channel_y = cdf_normalized_y[img_yuv[:,:,0]]

    # Combine equalized channels into a 3D array
    img_yuv[:,:,0] = np.uint8(equalized_channel_y)

    # Convert back to BGR after converting to uint8
    equalized_img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return equalized_img_bgr

class CALIRBRATE():
    def __init__(self):
        self.bgr = 0
        self.matching = False
        self.target_cdf = None
        self.mapping = None
        self.tf = None
        self.r = range(256)


    
    def calHist_bgr(self,img):
        # Compute histogram
        hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

        cdf_b = hist_b.cumsum()
        cdf_g = hist_g.cumsum()
        cdf_r = hist_r.cumsum()

        # Normalize CDF to the range [0, 255]
        self.cdf_normalized_b = (cdf_b * 255) / cdf_b[-1]
        self.cdf_normalized_g = (cdf_g * 255) / cdf_g[-1]
        self.cdf_normalized_r = (cdf_r * 255) / cdf_r[-1]
        
        print("cal")
        # Normalize histograms to the range [0, thresh]
        self.bgr = 1
    # def calHist_bgr(self,img):
    #     # # Compute histogram
    #     # hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    #     # hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    #     # hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])

    #     # cdf_b = hist_b.cumsum()
    #     # cdf_g = hist_g.cumsum()
    #     # cdf_r = hist_r.cumsum()

    #     # # Normalize CDF to the range [0, 255]
    #     # self.cdf_normalized_b = (cdf_b * 255) / cdf_b[-1]
    #     # self.cdf_normalized_g = (cdf_g * 255) / cdf_g[-1]
    #     # self.cdf_normalized_r = (cdf_r * 255) / cdf_r[-1]
        
    #     # print("cal")
    #     # Normalize histograms to the range [0, thresh]
    #     # Split the image into its channels
    #     # blue_channel, green_channel, red_channel = cv2.split(img)

    #     # Find the minimum and maximum values for each channel
    #     min_val_blue, max_val_blue, _, _ = cv2.minMaxLoc(blue_channel)
    #     min_val_green, max_val_green, _, _ = cv2.minMaxLoc(green_channel)
    #     min_val_red, max_val_red, _, _ = cv2.minMaxLoc(red_channel)

    #     # Print the results
    #     print("Blue channel - Minimum pixel value:", min_val_blue, "Maximum pixel value:", max_val_blue)
    #     print("Green channel - Minimum pixel value:", min_val_green, "Maximum pixel value:", max_val_green)
    #     print("Red channel - Minimum pixel value:", min_val_red, "Maximum pixel value:", max_val_red)
    #     self.bgr = 1
    def matchHistogram_bgr(self, img):
        if self.bgr == 0:
            return img
        else:
            img_bgr = img.copy()
            b, g, r = cv2.split(img_bgr)
           # Map the intensity values using the CDF

            # Map the intensity values using the CDF
            equalized_channel_b = np.interp(b, self.r, self.cdf_normalized_b)
            equalized_channel_g = np.interp(g, self.r, self.cdf_normalized_g)
            equalized_channel_r = np.interp(r, self.r, self.cdf_normalized_r)

            
            # equalized_channel_b = cv2.equalizeHist(b)
            # equalized_channel_g = cv2.equalizeHist(g)
            # equalized_channel_r = cv2.equalizeHist(r)



            # Combine equalized channels into a 3D array
            equalized_img_bgr = np.stack([equalized_channel_b, equalized_channel_g, equalized_channel_r], axis=-1)

            return equalized_img_bgr.astype(np.uint8)
    def contrast_cal(input_image, ref_image):
        # Convert the images to LAB color space
        lab_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2Lab)
        lab_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2Lab)

        # Calculate the mean and standard deviation of the L channel in the input and reference images
        l_mean_input, l_std_input = cv2.meanStdDev(lab_input[:, :, 0])
        l_mean_ref, l_std_ref = cv2.meanStdDev(lab_ref[:, :, 0])

        # Calculate the scaling factors for mean and standard deviation
        scale_factor_mean = l_mean_ref / l_mean_input
        scale_factor_std = l_std_ref / l_std_input

        # Scale the L channel
        lab_input[:, :, 0] = np.clip(lab_input[:, :, 0] * scale_factor_mean, 0, 255)
        lab_input[:, :, 0] = np.clip(lab_input[:, :, 0] * scale_factor_std, 0, 255)

        # Convert the LAB image back to BGR
        result_image = cv2.cvtColor(lab_input, cv2.COLOR_Lab2BGR)

        return result_image

    # def matchHistogram_bgr(self, img):
    #     if self.bgr == 0:
    #         return img
    #     else:
    #         img_bgr = img.copy()
    #         b, g, r = cv2.split(img_bgr)
    #        # Map the intensity values using the CDF

    #         # Map the intensity values using the CDF
    #         # equalized_channel_b = np.interp(b, range(256), self.cdf_normalized_b)
    #         # equalized_channel_g = np.interp(g, range(256), self.cdf_normalized_g)
    #         # equalized_channel_r = np.interp(r, range(256), self.cdf_normalized_r)

            
    #         equalized_channel_b = cv2.equalizeHist(b)
    #         equalized_channel_g = cv2.equalizeHist(g)
    #         equalized_channel_r = cv2.equalizeHist(r)



    #         # Combine equalized channels into a 3D array
    #         equalized_img_bgr = np.stack([equalized_channel_b, equalized_channel_g, equalized_channel_r], axis=-1)

    #         return equalized_img_bgr.astype(np.uint8)
    def calHist_tf(self,img):
        # Load two images
        image1 = cv2.imread('image1.jpg') 
        image2 = cv2.imread('image2.jpg')

        # Calculate histograms
        hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)

        # Calculate cumulative histograms
        cumulative_hist1 = np.cumsum(hist1)
        cumulative_hist2 = np.cumsum(hist2)

        # Create a LUT
        lut = np.zeros(256, dtype=np.uint8)

        # Map pixel values based on cumulative histograms
        for i in range(256):
            lut[i] = np.argmin(np.abs(cumulative_hist1 - cumulative_hist2[i]))
    def matchHistogram_tf(self, img):
        # print(self.bgr)
        if self.tf == 0:
            return img
        else:
             # Compute histogram
            hist_b = np.multiply(cv2.calcHist([img], [0], None, [256], [0, 256]),self.h_b)
            hist_g = np.multiply(cv2.calcHist([img], [1], None, [256], [0, 256]),self.h_g)
            hist_r = np.multiply(cv2.calcHist([img], [2], None, [256], [0, 256]),self.h_r)

        
            height, width = 1280, 720  # Adjust the size as needed
            result_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Generate random pixel values based on the histograms
            result_image[:, :, 0] = np.random.choice(256, size=(height, width), p=hist_b.flatten() / hist_b.sum())
            result_image[:, :, 1] = np.random.choice(256, size=(height, width), p=hist_g.flatten() / hist_g.sum())
            result_image[:, :, 2] = np.random.choice(256, size=(height, width), p=hist_r.flatten() / hist_r.sum())


            return result_image.astype(np.uint8)
    def adjust_contrast(self,image, alpha):
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

    def adjust_exposure(self,image, gamma):
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
        return  adjusted_image
        # def histMatchCal(self, roi):
        #     try:
        #         # Load the reference image
        #         reference_image = cv2.imread('ref.png')

        #         if reference_image is None:
        #             raise Exception("Error loading reference image.")

        #         # Convert images to BGR color space
        #         reference_bgr = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        #         target_bgr = roi

        #         # Calculate histograms for the reference and target images
        #         reference_hist = cv2.calcHist([reference_bgr], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        #         target_hist = cv2.calcHist([target_bgr], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

        #         # Normalize histograms
        #         cv2.normalize(reference_hist, reference_hist, 0, 1, cv2.NORM_MINMAX)
        #         cv2.normalize(target_hist, target_hist, 0, 1, cv2.NORM_MINMAX)

        #         # Calculate the cumulative distribution functions (CDF) for the histograms
        #         reference_cdf = reference_hist.cumsum()
        #         self.target_cdf = target_hist.cumsum()

        #         target_bgr = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        #         # Map the pixel values in the target image to the corresponding values in the reference image
        #         self.mapping = np.zeros(256, dtype=np.uint8)
        #         for i in range(256):
        #             print(f"Calibrating {int(i/256.0*100)} %                       ", end = '\r')
        #             diff = np.abs(self.target_cdf[i] - reference_cdf)
        #             min_diff_index = np.argmin(diff)
        #             self.mapping[i] = min_diff_index

        #         self.matching = True

        #     except Exception as e:
        #         print(f"Error in histMatchCal: {e}")
        #         self.matching = False

        # def histMatching(self, image):
        #     try:
        #         if self.matching:
        #             target_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #             # Apply the mapping to the target image
        #             matched_image = cv2.LUT(target_bgr, self.mapping)
        #             return matched_image
        #         else:
        #             return image

        #     except Exception as e:
        #         print(f"Error in histMatching: {e}")
        #         return image