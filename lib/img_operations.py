import numpy as np
import cv2


class ImgOperator:
    def __init__(self):
        self.description = "Image Threshold and Image Operation Holder"
        self.sobel_thresh_x = (20, 100)
        self.sobel_thresh_y = (50, 100)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.sobel_kernel = 3
        self.mag_thresh = (30, 240)
        self.dir_thresh = (0.7, 1.3)
        self.thresh_h = (0, 60)
        self.thresh_l = (0, 60)
        self.thresh_s = (100, 255)
        self.hcr = 230  # Horizontal Row Crop (Selecting only a part of image to neglect unnecessary area in Image processing)
        self.vcc = 0
        self.channel_select = 2  # 0-Blue, 1-Green, 2 - Red in BGR Format (Change According to the way of image ingestion)

    def histogram_equalizer(self, img):
        # Apply Histogram Equalization using Clahe
        # Check on Skimage exposure - Diff b/w Clahe
        return self.clahe.apply(img)

    def warp_image(self, img, src, dst, size):
        """Perspective Transform - Req, IMG, SRC, DST, Size"""
        M = cv2.getPerspectiveTransform(src, dst)  # Transform
        Minv = cv2.getPerspectiveTransform(dst, src)  # Inverse transform
        warped_img = cv2.warpPerspective(
            img, M, size, flags=cv2.INTER_LINEAR
        )  # Warped Image output

        return warped_img, M, Minv

    def sobel_xy(self, img, orient="x"):
        """
        Function which can apply Sobel x or y.
        Gradient in the x-direction ==> Edges closer to vertical.
        Gradient in the y-direction ==> Edges closer to horizontal.
        Default - X direction
        """
        if orient == "x":
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
            sobel_thresh = self.sobel_thresh_x
        if orient == "y":
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
            sobel_thresh = self.sobel_thresh_y
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[
            (scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])
        ] = 255
        return binary_output

    def mag_dir_thresh(self, img):
        """
        Function to Calculate Magnitude of the Gradient using x and y
        """

        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output_mag = np.zeros_like(gradmag)
        binary_output_mag[
            (gradmag >= self.mag_thresh[0]) & (gradmag <= self.mag_thresh[1])
        ] = 255

        # Calculate Absolute Gradient
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output_dir = np.zeros_like(absgraddir)
        binary_output_dir[
            (absgraddir >= self.dir_thresh[0]) & (absgraddir <= self.dir_thresh[1])
        ] = 255

        binary_output_dir = binary_output_dir.astype(np.uint8)

        # Return the binary image
        return binary_output_mag, binary_output_dir

    def channel_thresh(self, channel, thresh):
        binary = np.zeros_like(channel)
        binary[(channel > thresh[0]) & (channel <= thresh[1])] = 255
        return binary

    def gradient_combine_threshold(self, img):
        """
        Lane finding from Gradient information of the selected channel
        if the Ingested image is BGR
        CV2 base format - BGR (B-0, G-1, R-2)
        """
        rows, cols = img.shape[:2]
        select_channel = img[self.hcr : rows - 10, 0:cols, self.channel_select]

        sobelx = self.sobel_xy(select_channel, "x")
        sobely = self.sobel_xy(select_channel, "y")
        mag_img, dir_img = self.mag_dir_thresh(select_channel)

        # Combine Gradient Measurements
        gradient_comb = np.zeros_like(dir_img).astype(np.uint8)
        gradient_comb[
            ((sobelx > 1) & (mag_img > 1) & (dir_img > 1))
            | ((sobelx > 1) & (sobely > 1))
        ] = 255

        return gradient_comb

    def hls_combined_threshold(self, img):
        # Conversion to HLS Color Space
        # Hue, Luminance (Brightness), Saturation - Helps to define colors more naturally
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        rows, cols = img.shape[:2]
        H = hls[self.hcr : rows - 10, 0:cols, 0]
        L = hls[self.hcr : rows - 10, 0:cols, 1]
        S = hls[self.hcr : rows - 10, 0:cols, 2]

        h_img = self.channel_thresh(H, self.thresh_h)
        # cv2.imshow('HLS (H) threshold', h_img)
        l_img = self.channel_thresh(L, self.thresh_l)
        # cv2.imshow('HLS (L) threshold', l_img)
        s_img = self.channel_thresh(S, self.thresh_s)
        # cv2.imshow('HLS (S) threshold', s_img)

        # Two cases - lane lines in shadow or not
        hls_comb = np.zeros_like(s_img).astype(np.uint8)

        R = img[self.hcr : rows - 10, 0:cols, 2]
        _, R = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY)
        hls_comb[
            ((s_img > 1) & (l_img == 0))
            | ((s_img == 0) & (h_img > 1) & (l_img > 1))
            | (R > 1)
        ] = 255

        return hls_comb

    def combined_hls_sobel_threshold(self, grad, hls):

        result = np.zeros_like(hls).astype(np.uint8)
        # result[((grad > 1) | (hls > 1))] = 255
        # Give different values to represent different_color
        result[(grad > 1)] = 100
        result[(hls > 1)] = 255

        return result
