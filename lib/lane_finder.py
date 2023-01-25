import numpy as np
import cv2
import matplotlib.pyplot as plt

# Class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype="float")
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # Window margin - Helps in Giving min and max in Horizontal direction
        self.window_margin = (
            50  # Not Window height - Window height is based on num_of_windows
        )

        # Previous x
        self.prevx = []


class LaneFinder:
    def __init__(self):
        self.description = "Class for Lane finder"
        self.nwindows = 10  # No of moving windows for Line detection
        self.road_color = (0, 255, 0)
        self.lane_color = (0, 0, 255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fonttitle = 1
        self.fonttext = 0.5
        self.titlecolor = (0, 0, 255)
        self.textcolor = (0, 0, 0)
        self.titlethickness = 3
        self.textthickness = 2

    def smoothing(self, lines, pre_lines=3):
        # collect lines & print average line
        lines = np.squeeze(lines)
        avg_line = np.zeros((720))

        for ii, line in enumerate(reversed(lines)):
            if ii == pre_lines:
                break
            avg_line += line
        avg_line = avg_line / pre_lines

        return avg_line

    def radius_of_curvature(self, left_line, right_line):
        """Function to measure Radius"""

        ploty = left_line.ally
        leftx, rightx = left_line.allx, right_line.allx

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Define conversions in x and y from pixels space to meters
        width_lanes = abs(right_line.startx - left_line.startx)
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 * (720 / 1280) / width_lanes  # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = (
            (1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2)
            ** 1.5
        ) / np.absolute(2 * left_fit_cr[0])
        right_curverad = (
            (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2)
            ** 1.5
        ) / np.absolute(2 * right_fit_cr[0])
        # radius of curvature result
        left_line.radius_of_curvature = left_curverad
        right_line.radius_of_curvature = right_curverad

    def blind_window_search(self, warped_img, left_line, right_line):
        """
        Blind search Using Histogram & Sliding Window
        Process:
        1. Find the Histogram (Pixel Intensity Sum) for the first half of the image vertically
            this is to ensure we get correct points
        2. Find and left and right maximum around horizontal centre of image
        3. Take the Line Width with the help of Search Margin
        4. Take the Line height using the Image vertical shape/ Window Count
        5. Iterate through the windows to fit the Curve
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped_img[int(warped_img.shape[0] / 2) :, :], axis=0)

        # Create an output image to draw on and  visualize the result
        output = np.dstack((warped_img, warped_img, warped_img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        start_leftX = np.argmax(histogram[:midpoint])
        start_rightX = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        num_windows = self.nwindows
        # Set height of windows
        window_height = np.int(warped_img.shape[0] / num_windows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        current_leftX = start_leftX
        current_rightX = start_rightX

        # Set minimum number of pixels found to recenter window
        min_num_pixel = 50

        # Create empty lists to receive left and right lane pixel indices
        win_left_lane = []
        win_right_lane = []

        window_margin = left_line.window_margin

        # Step through the windows one by one and get the non zero pixels
        # and append the list
        for window in range(num_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_img.shape[0] - (window + 1) * window_height
            win_y_high = warped_img.shape[0] - window * window_height
            win_leftx_min = current_leftX - window_margin
            win_leftx_max = current_leftX + window_margin
            win_rightx_min = current_rightX - window_margin
            win_rightx_max = current_rightX + window_margin

            # Draw the windows on the visualization image
            cv2.rectangle(
                output,
                (win_leftx_min, win_y_low),
                (win_leftx_max, win_y_high),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                output,
                (win_rightx_min, win_y_low),
                (win_rightx_max, win_y_high),
                (0, 255, 0),
                2,
            )

            # Identify the nonzero pixels in x and y within the window
            left_window_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy <= win_y_high)
                & (nonzerox >= win_leftx_min)
                & (nonzerox <= win_leftx_max)
            ).nonzero()[0]
            right_window_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy <= win_y_high)
                & (nonzerox >= win_rightx_min)
                & (nonzerox <= win_rightx_max)
            ).nonzero()[0]
            # Append these indices to the lists
            win_left_lane.append(left_window_inds)
            win_right_lane.append(right_window_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(left_window_inds) > min_num_pixel:
                current_leftX = np.int(np.mean(nonzerox[left_window_inds]))
            if len(right_window_inds) > min_num_pixel:
                current_rightX = np.int(np.mean(nonzerox[right_window_inds]))

        # Concatenate the arrays of indices
        win_left_lane = np.concatenate(win_left_lane)
        win_right_lane = np.concatenate(win_right_lane)

        # Extract left and right line pixel positions
        leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
        rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [0, 0, 255]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_img.shape[0] - 1, warped_img.shape[0])

        # Plot in form of polynomial - a*(x**2) + b*x+c
        left_plotx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_plotx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        if len(left_line.prevx) > 10:
            left_avg_line = self.smoothing(left_line.prevx, 10)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = (
                left_avg_fit[0] * ploty**2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            )
            left_line.current_fit = left_avg_fit
            left_line.allx, left_line.ally = left_fit_plotx, ploty
        else:
            left_line.current_fit = left_fit
            left_line.allx, left_line.ally = left_plotx, ploty

        if len(right_line.prevx) > 10:
            right_avg_line = self.smoothing(right_line.prevx, 10)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = (
                right_avg_fit[0] * ploty**2
                + right_avg_fit[1] * ploty
                + right_avg_fit[2]
            )
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        left_line.startx, right_line.startx = (
            left_line.allx[len(left_line.allx) - 1],
            right_line.allx[len(right_line.allx) - 1],
        )
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

        left_line.detected, right_line.detected = True, True
        # print radius of curvature
        self.radius_of_curvature(left_line, right_line)
        return output

    def draw_road_info(self, img, left_line, right_line):

        curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

        direction = (
            (left_line.endx - left_line.startx) + (right_line.endx - right_line.startx)
        ) / 2

        if curvature > 2000 and abs(direction) < 100:
            road_inf = "Straight Line"
            curvature = float("inf")
        elif curvature <= 2000 and direction < -100:
            road_inf = "Left Curve"
        elif curvature <= 2000 and direction > 100:
            road_inf = "Right Curve"
        else:
            road_inf = "None"
            curvature = curvature

        center_lane = (right_line.startx + left_line.startx) / 2
        lane_width = right_line.startx - left_line.startx

        center_car = img.shape[1] / 2
        if center_lane > center_car:
            deviation = "Left {} %".format(
                str(round(abs(center_lane - center_car) / (lane_width / 2) * 100, 3))
            )
        elif center_lane < center_car:
            deviation = "Right {} %".format(
                str(round(abs(center_lane - center_car) / (lane_width / 2) * 100, 3))
            )
        else:
            deviation = "Center"

        cv2.putText(
            img,
            "Road Info",
            (20, 30),
            self.font,
            self.fonttitle,
            self.titlecolor,
            self.titlethickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Radius of Curvature: {} (m)".format(round(curvature, 2)),
            (20, 60),
            self.font,
            self.fonttext,
            self.textcolor,
            self.textthickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Direction of Curvature: {}".format(road_inf),
            (20, 80),
            self.font,
            self.fonttext,
            self.textcolor,
            self.textthickness,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            "Car Position and Deviation: {}".format(deviation),
            (20, 100),
            self.font,
            self.fonttext,
            self.textcolor,
            self.textthickness,
            cv2.LINE_AA,
        )

    def draw_lane(self, img, left_line, right_line):
        """Draw Lines and Driving Space"""
        window_img = np.zeros_like(img)

        window_margin = left_line.window_margin
        left_plotx, right_plotx = left_line.allx, right_line.allx
        ploty = left_line.ally

        # Generate a polygon to illustrate the search window area
        left_pts_l = np.array(
            [np.transpose(np.vstack([left_plotx - window_margin / 5, ploty]))]
        )
        left_pts_r = np.array(
            [
                np.flipud(
                    np.transpose(np.vstack([left_plotx + window_margin / 5, ploty]))
                )
            ]
        )
        left_pts = np.hstack((left_pts_l, left_pts_r))
        right_pts_l = np.array(
            [np.transpose(np.vstack([right_plotx - window_margin / 5, ploty]))]
        )
        right_pts_r = np.array(
            [
                np.flipud(
                    np.transpose(np.vstack([right_plotx + window_margin / 5, ploty]))
                )
            ]
        )
        right_pts = np.hstack((right_pts_l, right_pts_r))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_pts]), self.lane_color)
        cv2.fillPoly(window_img, np.int_([right_pts]), self.lane_color)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array(
            [np.transpose(np.vstack([left_plotx + window_margin / 5, ploty]))]
        )
        pts_right = np.array(
            [
                np.flipud(
                    np.transpose(np.vstack([right_plotx - window_margin / 5, ploty]))
                )
            ]
        )
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([pts]), self.road_color)
        result = cv2.addWeighted(img, 1, window_img, 0.3, 0)

        return result, window_img
