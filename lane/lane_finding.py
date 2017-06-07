import numpy as np
import cv2
import glob
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lane.utils as ut


def init_all():
    global mtx, dist, M, M_inv, wrp

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./lane/camera_cal/calibration*.jpg')


    # Step through the list and search for chessboard corners
    for fname in images:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)



    fname = './lane/test1.jpg'
    img1 = mpimg.imread(fname)

    img_size = (img1.shape[1], img1.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

    copy = np.copy(img1)

    image_shape = copy.shape

    thr = ut.thresholded(copy, mtx, dist)
    wrp, M, M_inv = ut.warped_perspective(thr)

    # Reinitialize some global variables.
    global polyfit_left
    global polyfit_right
    global running_average

    global past_good_left_lines
    global past_good_right_lines

    polyfit_left = None
    polyfit_right = None
    running_average = 0

    past_good_left_lines = []
    past_good_right_lines = []




def pipeline(img):
    """
        The main pipline of the project:
        1. image distorsing
        2. thresholding
        3. warping
        4. finding lines
        5. painting road
    """
    # global variables to store the polynomial coefficients of the line detected in the last frame
    global polyfit_right
    global polyfit_left

    # global variables to store the line coordinates in previous n (=4) frames
    global past_good_right_lines
    global past_good_left_lines

    # global variable which contains running average of the mean difference between left and right lanes
    global running_average

    img_size = (img.shape[1], img.shape[0])

    # get thresholded image
    thr = ut.thresholded(img, mtx, dist)

    # perform a perspective transform
    warped = cv2.warpPerspective(thr, M, img_size, flags=cv2.INTER_LINEAR)

    out_img = np.dstack((warped, warped, warped)) * 255

    non_zeros = warped.nonzero()
    non_zeros_y = non_zeros[0]
    non_zeros_x = non_zeros[1]

    num_rows = warped.shape[0]
    y_points = np.linspace(0, num_rows - 1, num_rows)

    if (polyfit_left is None) or (polyfit_right is None):
        # If the polynomial coefficients of the previous frames are None then perform a brute force search
        brute = True
        left_x_predictions, right_x_predictions = ut.brute_search(warped)
    else:
        # Else search in a margin of 100 pixels on each side of the pervious polynomial fit
        brute = False
        margin = 100

        left_x_predictions, right_x_predictions = ut.line_predictions(non_zeros_x, non_zeros_y, left_coordinates,
                                                                   right_coordinates, num_rows)

    if (left_x_predictions is None or right_x_predictions is None):
        if not brute:
            left_x_predictions, right_x_predictions = brute_search(warped)

    bad_lines = False

    if (left_x_predictions is None or right_x_predictions is None):
        bad_lines = True
    else:
        mean_difference = np.mean(right_x_predictions - left_x_predictions)

        if running_average == 0:
            running_average = mean_difference

        if (mean_difference < 0.7 * running_average or mean_difference > 1.3 * running_average):
            bad_lines = True
            if not brute:
                left_x_predictions, right_x_predictions = brute_search(warped)
                if (left_x_predictions is None or right_x_predictions is None):
                    bad_lines = True
                else:
                    mean_difference = np.mean(right_x_predictions - left_x_predictions)
                    if (mean_difference < 0.7 * running_average or mean_difference > 1.3 * running_average):
                        bad_lines = True
                    else:
                        bad_lines = False
        else:
            bad_lines = False

    if bad_lines:
        polyfit_left = None
        polyfit_right = None
        if len(past_good_left_lines) == 0 and len(past_good_right_lines) == 0:
            return img
        else:
            left_x_predictions = past_good_left_lines[-1]
            right_x_predictions = past_good_right_lines[-1]
    else:
        past_good_left_lines, left_x_predictions = ut.averaged_line(past_good_left_lines, left_x_predictions)
        past_good_right_lines, right_x_predictions = ut.averaged_line(past_good_right_lines, right_x_predictions)

        mean_difference = np.mean(right_x_predictions - left_x_predictions)
        running_average = 0.9 * running_average + 0.1 * mean_difference

    curvature_string = "Radius of curvature: %.2f m" % ut.average_radius(left_x_predictions, right_x_predictions, num_rows)
    offset_string = "Center offset: %.2f m" % ut.offset(left_x_predictions, right_x_predictions, img_size)
    result = ut.paintRoad(img, wrp, left_x_predictions, right_x_predictions, y_points, M_inv, img_size)

    cv2.putText(result, curvature_string, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=2)
    cv2.putText(result, offset_string, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=2)

    return result

