import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
        Calculate directional gradient
    """
    # Apply x or y gradient
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute values
    sobel = np.absolute(sobel)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
        Calculate magnitude tresholde
    """
    # grayscaling
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # takes the gradient x and y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # calculate the magnitude
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

    # scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    # create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return sxbinary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
        Directional threshold
    """
    # grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # error statement to ignore division and invalid errors
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely / sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1

    # return the binary image
    return binary_output


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def thresholded(image, mtx, dist):
    """
        Threshold image using chanels: L,S,G and B
    """
    kernel_size = 7

    copy = np.copy(image)

    img = cv2.undistort(copy, mtx, dist, None, mtx)

    # Gaussian Blur
    # img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    L = hls[:, :, 1]
    S = hls[:, :, 2]

    # R & G thresholds so that yellow lanes are detected well.
    color_threshold = 150

    R = img[:, :, 0]
    G = img[:, :, 1]

    color_combined = np.zeros_like(R)
    g_r_condition = (G > color_threshold) & (R > color_threshold)

    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    # Define sobel kernel size
    ksize = 3

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(10, 200))

    # mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(70, 100))

    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(np.pi / 6, np.pi / 2))

    combined_condition = ((gradx == 1) & (dir_binary == 1))
    # combined_condition = ((mag_binary == 1) | (gradx == 1) & (dir_binary == 1))

    # Threshold S color channel
    s_thresh = (100, 255)
    s_condition = (S > s_thresh[0]) & (S <= s_thresh[1])

    # Threshold L color channel
    l_thresh = (120, 255)
    l_condition = (L > l_thresh[0]) & (L <= l_thresh[1])

    combined = np.zeros_like(dir_binary)

    # Combine all the thresholding information
    combined[(g_r_condition & l_condition) & (s_condition | combined_condition)] = 1

    mask = np.zeros_like(combined)
    region_of_interest_vertices = np.array([[0, height - 1], [width / 2, int(0.5 * height)], [width - 1, height - 1]],
                                           dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv2.bitwise_and(combined, mask)

    return thresholded


def warped_perspective(img):
    """
        Perform perspective transform
    """

    copy = np.copy(img)

    image_shape = copy.shape
    # Vertices extracted manually for performing a perspective transform
    bottom_left = [220, 720]
    bottom_right = [1110, 720]
    top_left = [570, 470]
    top_right = [722, 470]

    src = np.float32([bottom_left, bottom_right, top_right, top_left])

    # Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
    bottom_left = [320, 720]
    bottom_right = [920, 720]
    top_left = [320, 1]
    top_right = [920, 1]

    dst = np.float32([bottom_left, bottom_right, top_right, top_left])

    img_size = (image_shape[1], image_shape[0])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return cv2.warpPerspective(copy, M, img_size, flags=cv2.INTER_LINEAR), M, M_inv

def measure_radius_of_curvature(x_values, num_rows):
    """
        Radius of Curvature
    """
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # If no pixels were found return None
    y_points = np.linspace(0, num_rows-1, num_rows)
    y_eval = np.max(y_points)

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y_points*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

def average_radius(left_x_predictions, right_x_predictions, num_rows, ):
    """
        Calculates an average radius of the road
    """
    left_curve_rad = measure_radius_of_curvature(left_x_predictions, num_rows, )
    right_curve_rad = measure_radius_of_curvature(right_x_predictions, num_rows)
    average_curve_rad = (left_curve_rad + right_curve_rad)/2
    return average_curve_rad

def offset(left_x_predictions, right_x_predictions, img_size):
    """
        Callculates the center offset in meters
    """
    lane_center = (right_x_predictions[719] + left_x_predictions[719])/2
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    center_offset_pixels = abs(img_size[0]/2 - lane_center)
    center_offset_mtrs = xm_per_pix*center_offset_pixels
    return center_offset_mtrs


def paintRoad(image, warped, left_x_predictions, right_x_predictions, y_points, M_inv, img_size):
    """
        The function paints the space between the lines.
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_x_predictions, y_points]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x_predictions, y_points])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, img_size)
    # Combine the result with the original image
    return cv2.addWeighted(image, 1.0, newwarp, 0.3, 0.0)


def line_predictions(non_zeros_x, non_zeros_y, left_coordinates, right_coordinates, num_rows):
    """
        Given ncoordinates of non-zeros pixels and coordinates of non-zeros pixels within the sliding windows,
        this function generates a prediction for the lane line.
    """
    left_x = non_zeros_x[left_coordinates]
    left_y = non_zeros_y[left_coordinates]

    # If no pixels were found return None
    if (left_y.size == 0 or left_x.size == 0):
        return None, None

    # Fit the polynomial
    polyfit_left = np.polyfit(left_y, left_x, 2)

    right_x = non_zeros_x[right_coordinates]
    right_y = non_zeros_y[right_coordinates]

    # If no pixels were found return None
    if (right_y.size == 0 or right_x.size == 0):
        return None, None

    # Fit the polynomial
    polyfit_right = np.polyfit(right_y, right_x, 2)

    # If no pixels were found return None
    y_points = np.linspace(0, num_rows - 1, num_rows)

    # Generate the lane lines from the polynomial fit
    left_x_predictions = polyfit_left[0] * y_points ** 2 + polyfit_left[1] * y_points + polyfit_left[2]
    right_x_predictions = polyfit_right[0] * y_points ** 2 + polyfit_right[1] * y_points + polyfit_right[2]

    return left_x_predictions, right_x_predictions


def brute_search(warped):
    """
        This function searches for lane lines from scratch.
        Thresholding & performing a sliding window search.
    """
    non_zeros = warped.nonzero()
    non_zeros_y = non_zeros[0]
    non_zeros_x = non_zeros[1]

    num_rows = warped.shape[0]

    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)

    half_width = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:half_width])
    rightx_base = np.argmax(histogram[half_width:]) + half_width

    num_windows = 10
    window_height = np.int(num_rows / num_windows)
    window_half_width = 50

    min_pixels = 100

    left_coordinates = []
    right_coordinates = []

    for window in range(num_windows):
        y_max = num_rows - window * window_height
        y_min = num_rows - (window + 1) * window_height

        left_x_min = leftx_base - window_half_width
        left_x_max = leftx_base + window_half_width

        left_window_coordinates = (
        (non_zeros_x >= left_x_min) & (non_zeros_x <= left_x_max) & (non_zeros_y >= y_min) & (
        non_zeros_y <= y_max)).nonzero()[0]
        left_coordinates.append(left_window_coordinates)

        if len(left_window_coordinates) > min_pixels:
            leftx_base = np.int(np.mean(non_zeros_x[left_window_coordinates]))

        right_x_min = rightx_base - window_half_width
        right_x_max = rightx_base + window_half_width

        right_window_coordinates = (
        (non_zeros_x >= right_x_min) & (non_zeros_x <= right_x_max) & (non_zeros_y >= y_min) & (
        non_zeros_y <= y_max)).nonzero()[0]
        right_coordinates.append(right_window_coordinates)

        if len(right_window_coordinates) > min_pixels:
            rightx_base = np.int(np.mean(non_zeros_x[right_window_coordinates]))

    left_coordinates = np.concatenate(left_coordinates)
    right_coordinates = np.concatenate(right_coordinates)

    left_x_predictions, right_x_predictions = line_predictions(non_zeros_x, non_zeros_y, left_coordinates,
                                                               right_coordinates, num_rows)
    return left_x_predictions, right_x_predictions


def averaged_line(previous_lines, new_line):
    """
        This function computes an averaged lane line by averaging over previous good frames.
    """

    # Number of frames to average over
    num_frames = 12

    if new_line is None:
        # No line was detected

        if len(previous_lines) == 0:
            # If there are no previous lines, return None
            return previous_lines, None
        else:
            # Else return the last line
            return previous_lines, previous_lines[-1]
    else:
        if len(previous_lines) < num_frames:
            # we need at least num_frames frames to average over
            previous_lines.append(new_line)
            return previous_lines, new_line
        else:
            # average over the last num_frames frames
            previous_lines[0:num_frames - 1] = previous_lines[1:]
            previous_lines[num_frames - 1] = new_line
            new_line = np.zeros_like(new_line)
            for i in range(num_frames):
                new_line += previous_lines[i]
            new_line /= num_frames
            return previous_lines, new_line